# dags/df_data_prep_dag.py
import gc
import logging
from datetime import timedelta

import pendulum
import numpy as np
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.google.cloud.hooks.gcs import GCSHook

# ====== ORTAM SABİTLERİ ======
BUCKET = "europe-west1-airflow-a054c263-bucket"

BASE_PREFIX = "demand_forecasting"
INPUT_DIRS = [f"{BASE_PREFIX}/input", f"{BASE_PREFIX}/input_parquet"]   # input öncelikli
OUTPUT_PREFIX = f"{BASE_PREFIX}/output"

# Kaç günlük veri işlensin? (hafızayı korumak için kısabiliriz)
LOOKBACK_DAYS = 180

default_args = {
    "owner": "genboost", 
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2)
}


# ====== YARDIMCILAR ======
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
    """INPUT_DIRS içinde sırayla arayıp ilk bulduğunu indirir."""
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

def _to_float32(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    return df


# ====== ASIL İŞ ======
def run_data_prep(**context):
    gcs = GCSHook(gcp_conn_id="google_cloud_default")

    # 1) Girdi Parquetlerini indir
    local_paths = {}
    for key, fname in {
        "product":      "dim_product.parquet",
        "sales":        "history_sales.parquet",
        "kmeans_store": "k_means_store.parquet",
    }.items():
        local_paths[key] = f"/tmp/{fname}"
        _download_first_found(gcs, fname, local_paths[key])

    # 2) Sadece gereken kolonları oku
    sales_cols = [
        "date", "store_code", "product_id",
        "discount_amount", "net_amount_wovat",
        "unit_price", "net_quantity",
    ]
    sales = pd.read_parquet(local_paths["sales"], columns=sales_cols, engine=ENGINE)

    # Tarih filtresi
    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    start_date = (pd.Timestamp("now", tz="UTC") - pd.Timedelta(days=LOOKBACK_DAYS)).tz_localize(None)
    sales = sales[sales["date"] >= start_date].copy()

    # Haftabaşı (Pazartesi)
    sales["week_start_date"] = (sales["date"] - pd.to_timedelta(sales["date"].dt.weekday, unit="D")).dt.floor("D")

    # Tip indirgeme
    sales["product_id"] = sales["product_id"].astype(str)
    sales["store_code"] = sales["store_code"].astype(str)
    _to_float32(sales, ["discount_amount", "net_amount_wovat", "unit_price", "net_quantity"])

    product_cols = [
        "product_id","marka_aciklama",
        "product_att_01","product_att_02","product_att_03",
        "product_att_04","product_att_05","product_att_06",
        "product_att_01_desc","product_att_02_desc","product_att_03_desc",
        "product_att_04_desc","product_att_05_desc","product_att_06_desc",
    ]
    product = pd.read_parquet(local_paths["product"], columns=product_cols, engine=ENGINE)
    product["product_id"] = product["product_id"].astype(str)

    kmeans = pd.read_parquet(local_paths["kmeans_store"], engine=ENGINE)[["store_code","store_cluster"]]
    kmeans["store_code"] = kmeans["store_code"].astype(str)

    # 3) Join'ler
    df = sales.merge(product, on="product_id", how="left")
    del product; gc.collect()
    df = df.merge(kmeans, on="store_code", how="left")
    del kmeans; gc.collect()

    # product_att_02: D/N -> 9, sonra numerik
    if "product_att_02" in df.columns:
        df["product_att_02"] = df["product_att_02"].astype(str).str.replace("D","9").str.replace("N","9")
        df["product_att_02"] = pd.to_numeric(df["product_att_02"], errors="coerce")

    # 9’lar hariç
    if "product_att_01" in df.columns:
        df = df[~(df["product_att_01"] == 9.0)]
    if "product_att_02" in df.columns:
        df = df[~(df["product_att_02"] == 9.0)]
    df = df.copy()

    # 4) İndirim oranı & ağırlıklar
    denom = (df["discount_amount"].fillna(0) + df["net_amount_wovat"].fillna(0)).astype("float32")
    df["discount_frac"] = np.where(denom > 0, (df["discount_amount"].fillna(0) / denom), 0.0).astype("float32")
    df["discount_frac"] = df["discount_frac"].clip(0, 0.9)
    df["rev_pre_disc"] = denom
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").astype("float32")
    df["net_quantity"] = pd.to_numeric(df["net_quantity"], errors="coerce").fillna(0).astype("float32")

    # 5) Agregasyon
    granularity = [
        "product_att_01","product_att_02","product_att_05",
        "product_att_01_desc","product_att_02_desc","product_att_05_desc",
        "store_cluster","week_start_date",
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
    agg["discount_frac_wavg"] = np.where(agg["w_den"] > 0, agg["w_num"] / agg["w_den"], np.nan).astype("float32")
    agg.drop(columns=["w_num","w_den"], inplace=True)
    agg["total_quantity"] = agg["total_quantity"].round().astype("int32")
    agg = agg.sort_values("total_quantity", ascending=False)

    # 6) Çıkış
    out_local = "/tmp/weekly_sales_with_cluster_125.parquet"
    agg.to_parquet(out_local, index=False, engine=ENGINE, compression="snappy")

    out_obj = f"{OUTPUT_PREFIX}/weekly_sales_with_cluster_125.parquet"
    logging.info("Uploading result to gs://%s/%s", BUCKET, out_obj)
    gcs.upload(bucket_name=BUCKET, object_name=out_obj, filename=out_local)


# ====== DAG TANIMI ======
with DAG(
    dag_id="df_data_prep_dag",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    schedule_interval="0 7 * * 0",    # Her Pazar 07:00 UTC = 10:00 TRT
    catchup=False,
    default_args=default_args,
    description="Data preparation DAG for demand forecasting - runs every Sunday at 10:00 AM TRT",
    tags=["demand_forecasting", "data_prep"],
) as dag:

    run = PythonOperator(
        task_id="run_data_prep",
        python_callable=run_data_prep,
    )

    # data_prep bittiğinde feature_eng DAG'ını tetikle
    trigger_feature_eng = TriggerDagRunOperator(
        task_id="trigger_feature_eng",
        trigger_dag_id="df_feature_eng_dag",   # DAG ID birebir aynı olmalı
        wait_for_completion=True,              # tamamlanmasını bekle
        allowed_states=["success"],
        failed_states=["failed"],
        poke_interval=30,                      # 30 saniyede bir kontrol et
        reset_dag_run=True,                    # aynı run varsa sıfırla
        trigger_run_id="feature_eng__{{ ts_nodash }}",
        execution_timeout=timedelta(hours=3),  # 3 saat timeout
        conf={
            "prep_output": f"gs://{BUCKET}/{OUTPUT_PREFIX}/weekly_sales_with_cluster_125.parquet"
        },
    )

    run >> trigger_feature_eng
