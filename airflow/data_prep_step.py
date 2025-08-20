import os
import gc
import logging
import numpy as np
import pandas as pd
from airflow.providers.google.cloud.hooks.gcs import GCSHook

BUCKET = "europe-west1-airflow-a054c263-bucket"
BASE_PREFIX = "demand_forecasting"
INPUT_DIRS = [f"{BASE_PREFIX}/input", f"{BASE_PREFIX}/input_parquet"]  # input öncelikli
OUTPUT_PREFIX = f"{BASE_PREFIX}/output"
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "180"))

def _pick_parquet_engine():
    try:
        import pyarrow  # noqa: F401
        return "pyarrow"
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return "fastparquet"
        except Exception:
            raise ImportError("Parquet için pyarrow veya fastparquet kurulu olmalı (pyarrow>=12 önerilir).")

ENGINE = _pick_parquet_engine()

def _download_first_found(gcs: GCSHook, filename: str, local_path: str) -> str:
    last_err = None
    for pref in INPUT_DIRS:
        obj = f"{pref}/{filename}"
        try:
            gcs.download(bucket_name=BUCKET, object_name=obj, filename=local_path)
            logging.info("✅ Downloaded: gs://%s/%s", BUCKET, obj)
            return obj
        except Exception as e:
            last_err = e
            logging.info("Not found at gs://%s/%s", BUCKET, obj)
    raise FileNotFoundError(f"{filename} not found under {INPUT_DIRS}. Last error: {last_err}")

def _to_cat(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df

def _to_float32(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    return df

def run_data_prep(**context):
    gcs = GCSHook(gcp_conn_id="google_cloud_default")

    # 1) Girdi dosyalarını indir
    local = {}
    for key, fname in {
        "product": "dim_product.parquet",
        "sales": "history_sales.parquet",
        "kmeans_store": "k_means_store.parquet",
    }.items():
        local[key] = f"/tmp/{fname}"
        _download_first_found(gcs, fname, local[key])

    # 2) Okuma (yalnızca gerekli kolonlar)
    sales_cols = ["date","store_code","product_id","discount_amount","net_amount_wovat","unit_price","net_quantity"]
    sales = pd.read_parquet(local["sales"], columns=sales_cols, engine=ENGINE)
    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    start_date = (pd.Timestamp("now", tz="UTC") - pd.Timedelta(days=LOOKBACK_DAYS)).tz_localize(None)
    sales = sales[sales["date"] >= start_date].copy()
    sales["week_start_date"] = (sales["date"] - pd.to_timedelta(sales["date"].dt.weekday, unit="D")).dt.floor("D")
    sales["product_id"] = sales["product_id"].astype(str)
    sales["store_code"] = sales["store_code"].astype(str)
    _to_float32(sales, ["discount_amount","net_amount_wovat","unit_price","net_quantity"])

    product_cols = [
        "product_id","marka_aciklama",
        "product_att_01","product_att_02","product_att_03",
        "product_att_04","product_att_05","product_att_06",
        "product_att_01_desc","product_att_02_desc","product_att_03_desc",
        "product_att_04_desc","product_att_05_desc","product_att_06_desc",
    ]
    product = pd.read_parquet(local["product"], columns=product_cols, engine=ENGINE)
    product["product_id"] = product["product_id"].astype(str)

    kmeans = pd.read_parquet(local["kmeans_store"], engine=ENGINE)[["store_code","store_cluster"]]
    kmeans["store_code"] = kmeans["store_code"].astype(str)
    kmeans["store_cluster"] = kmeans["store_cluster"].astype("category")

    # 3) Join’ler
    df = sales.merge(product, on="product_id", how="left")
    del product; gc.collect()
    df = df.merge(kmeans, on="store_code", how="left")
    del kmeans; gc.collect()

    # 4) Temizlik/özellikler
    if "product_att_02" in df.columns:
        df["product_att_02"] = df["product_att_02"].astype(str).str.replace("D","9").str.replace("N","9")
        df["product_att_02"] = pd.to_numeric(df["product_att_02"], errors="coerce")

    _to_cat(df, ["marka_aciklama","product_att_01_desc","product_att_02_desc","product_att_05_desc","store_cluster"])
    if "product_att_01" in df.columns:
        df = df[~(df["product_att_01"] == 9.0)]
    if "product_att_02" in df.columns:
        df = df[~(df["product_att_02"] == 9.0)]
    df = df.copy()

    denom = (df["discount_amount"].fillna(0) + df["net_amount_wovat"].fillna(0)).astype("float32")
    df["discount_frac"] = np.where(denom > 0, (df["discount_amount"].fillna(0) / denom), 0.0).astype("float32")
    df["discount_frac"] = df["discount_frac"].clip(0, 0.9)
    df["rev_pre_disc"] = denom
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").astype("float32")
    df["net_quantity"] = pd.to_numeric(df["net_quantity"], errors="coerce").fillna(0).astype("float32")

    granularity = [
        "product_att_01","product_att_02","product_att_05",
        "product_att_01_desc","product_att_02_desc","product_att_05_desc",
        "store_cluster","week_start_date"
    ]
    df["w_num"] = (df["discount_frac"] * df["rev_pre_disc"]).astype("float32")
    df["w_den"] = df["rev_pre_disc"].astype("float32")

    g_sum = df.groupby(granularity, observed=True)[["net_quantity","w_num","w_den"]].sum().rename(
        columns={"net_quantity":"total_quantity"}
    )
    g_mean = df.groupby(granularity, observed=True)[["unit_price","discount_frac"]].mean().rename(
        columns={"unit_price":"unit_price_mean","discount_frac":"discount_frac_mean"}
    )
    g_median = df.groupby(granularity, observed=True)[["unit_price"]].median().rename(
        columns={"unit_price":"unit_price_median"}
    )

    agg = g_sum.join(g_mean, how="left").join(g_median, how="left").reset_index()
    agg["discount_frac_wavg"] = np.where(agg["w_den"] > 0, agg["w_num"]/agg["w_den"], np.nan).astype("float32")
    agg.drop(columns=["w_num","w_den"], inplace=True)
    agg["total_quantity"] = agg["total_quantity"].round().astype("int32")
    agg = agg.sort_values("total_quantity", ascending=False)

    # 5) Çıkış
    out_local = "/tmp/weekly_sales_with_cluster_125.parquet"
    agg.to_parquet(out_local, index=False, engine=ENGINE, compression="snappy")
    out_obj = f"{OUTPUT_PREFIX}/weekly_sales_with_cluster_125.parquet"
    logging.info("Uploading result to gs://%s/%s", BUCKET, out_obj)
    gcs.upload(bucket_name=BUCKET, object_name=out_obj, filename=out_local)
