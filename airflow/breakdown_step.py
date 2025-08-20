import gc
import logging
import numpy as np
import pandas as pd
from airflow.providers.google.cloud.hooks.gcs import GCSHook

BUCKET = "europe-west1-airflow-a054c263-bucket"

BASE_PREFIX   = "demand_forecasting"
INPUT_DIRS    = [f"{BASE_PREFIX}/input_parquet", f"{BASE_PREFIX}/input"]  # ham girdiler burada
OUTPUT_PREFIX = f"{BASE_PREFIX}/output"

# Girdi/çıktı nesneleri
SALES_OBJ     = "history_sales.parquet"
PROD_OBJ      = "dim_product.parquet"
CAL_OBJ       = "dim_calendar_pivot.parquet"
CLUSTER_OBJ   = "k_means_store.parquet"

FORECAST_OBJ  = f"{OUTPUT_PREFIX}/future_12_weeks_hierarchical_forecast.parquet"
BREAKDOWN_OBJ = f"{OUTPUT_PREFIX}/breakdown_store.parquet"   # <- ÇIKTI: PARQUET

# Satışlardan kaç hafta geriye bakılsın? (son 8 hafta)
LOOKBACK_WEEKS = 8

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
    """INPUT_DIRS içinde sırayla arayıp ilk bulduğunu indirir (ham girdiler için)."""
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

def run_breakdown(**context):
    gcs = GCSHook(gcp_conn_id="google_cloud_default")

    # 1) Tahmin tablosu (output'tan)
    fc_local = "/tmp/future_12_weeks_hierarchical_forecast.parquet"
    gcs.download(bucket_name=BUCKET, object_name=FORECAST_OBJ, filename=fc_local)
    logging.info("✅ Downloaded forecast: gs://%s/%s", BUCKET, FORECAST_OBJ)
    future_forecast = pd.read_parquet(fc_local, engine=ENGINE)
    future_forecast["week_start_date"] = pd.to_datetime(future_forecast["week_start_date"], errors="coerce")

    # 2) Ham girdiler (input veya input_parquet altından)
    sales_local   = "/tmp/history_sales.parquet"
    prod_local    = "/tmp/dim_product.parquet"
    cal_local     = "/tmp/dim_calendar_pivot.parquet"
    cluster_local = "/tmp/k_means_store.parquet"

    _download_first_found(gcs, SALES_OBJ,   sales_local)
    _download_first_found(gcs, PROD_OBJ,    prod_local)
    _download_first_found(gcs, CAL_OBJ,     cal_local)
    _download_first_found(gcs, CLUSTER_OBJ, cluster_local)

    sales   = pd.read_parquet(sales_local, engine=ENGINE)
    product = pd.read_parquet(prod_local,  engine=ENGINE)
    calendar= pd.read_parquet(cal_local,   engine=ENGINE)
    cluster = pd.read_parquet(cluster_local, engine=ENGINE)

    # 3) Dönüşümler / filtreler
    sales["date"] = pd.to_datetime(sales.get("date"), errors="coerce")

    # Negatif adetleri sıfırla
    if "net_quantity" in sales.columns:
        sales.loc[sales["net_quantity"] < 0, "net_quantity"] = 0
    else:
        raise KeyError("history_sales içinde 'net_quantity' kolonu bekleniyor.")

    # Son tarih ve LOOKBACK
    last_date = pd.to_datetime(sales["date"].max())
    cutoff = last_date - pd.Timedelta(weeks=LOOKBACK_WEEKS)
    sales = sales[sales["date"] >= cutoff].copy()

    # Takvimden hafta başlangıcı (Pazartesi)
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
    calendar["week_start_date"] = (calendar["date"] - pd.to_timedelta(calendar["date"].dt.weekday, unit="D")).dt.floor("D")
    sales = sales.merge(calendar[["date","week_start_date"]], on="date", how="left")

    # Mağaza cluster
    if not {"store_code","store_cluster"} <= set(cluster.columns):
        raise KeyError("k_means_store içinde 'store_code' ve 'store_cluster' bekleniyor.")
    cluster["store_code"] = cluster["store_code"].astype(str)
    sales["store_code"] = sales["store_code"].astype(str)
    sales = sales.merge(cluster[["store_code","store_cluster"]], on="store_code", how="left")

    # 4) Ürün özellikleri ile join (esnek anahtar seçimi)
    # product tablosunda product_id / product_code olabilir; sales tarafında stok_kodu / product_id / product_code olabilir
    product_keys = [c for c in ["product_id","product_code"] if c in product.columns]
    if not product_keys:
        raise KeyError("dim_product içinde 'product_id' veya 'product_code' bulunamadı.")
    # Ürün özellik kolonları
    prod_feat_cols = ["product_att_01","product_att_02","product_att_05"]
    prod_use_cols = product_keys + [c for c in prod_feat_cols if c in product.columns]
    product_use = product[prod_use_cols].drop_duplicates().copy()

    # sales tarafında hangi ürün kolonu var?
    sales_key_col = None
    for cand in ["stok_kodu","product_id","product_code"]:
        if cand in sales.columns:
            sales_key_col = cand
            break
    if sales_key_col is None:
        raise KeyError("history_sales içinde 'stok_kodu' veya 'product_id' ya da 'product_code' bekleniyor.")

    # Join anahtarı eşlemesi
    # Eğer sales 'stok_kodu' taşıyorsa product'ta 'product_code' ile eşle
    if sales_key_col == "stok_kodu" and "product_code" in product_use.columns:
        left_on, right_on = "stok_kodu", "product_code"
    else:
        # Aynı isim varsa direkt kullan
        left_on = right_on = sales_key_col if sales_key_col in product_use.columns else None
        if left_on is None:
            # sales 'product_id', product 'product_code' ise bunu desteklemiyoruz -> kullanıcı datası gerekir
            raise KeyError(f"Eşleşen ürün anahtarı bulunamadı (sales:{sales_key_col}, product cols:{product_use.columns.tolist()}).")

    sales = sales.merge(product_use, left_on=left_on, right_on=right_on, how="left")

    # product_att_02: 'D'/'N' -> '9' ve numerik
    if "product_att_02" in sales.columns:
        sales["product_att_02"] = sales["product_att_02"].apply(
            lambda x: int(str(x).replace("D","9").replace("N","9")) if isinstance(x,str) and (("D" in x) or ("N" in x)) else x
        )
        sales["product_att_02"] = pd.to_numeric(sales["product_att_02"], errors="coerce")

    # 9’ları filtrele
    for col in ["product_att_01","product_att_02"]:
        if col in sales.columns:
            sales = sales[~(sales[col] == 9.0)]

    # 5) Mağaza oranları (cluster içi pay)
    group_keys = ["store_cluster","product_att_01","product_att_02","product_att_05","store_code"]
    missing = [k for k in group_keys if k not in sales.columns]
    if missing:
        raise KeyError(f"Mağaza oranı için eksik kolon(lar): {missing}")

    # (opsiyonel) ürün anahtarını da taşı (stok_kodu varsa onu, yoksa ortak anahtar)
    id_col = "stok_kodu" if "stok_kodu" in sales.columns else ("product_code" if "product_code" in sales.columns else ("product_id" if "product_id" in sales.columns else None))
    if id_col:
        group_keys_with_id = group_keys + [id_col]
    else:
        group_keys_with_id = group_keys

    store_ratio_df = (
        sales.groupby(group_keys_with_id, observed=True)["net_quantity"]
             .sum()
             .reset_index()
             .rename(columns={"net_quantity":"net_quantity"})
    )

    # cluster toplamı ve oran
    cluster_keys = ["store_cluster","product_att_01","product_att_02","product_att_05"]
    totals = store_ratio_df.groupby(cluster_keys, observed=True)["net_quantity"].transform("sum")
    store_ratio_df["cluster_total"] = totals
    store_ratio_df["store_ratio"] = np.where(store_ratio_df["cluster_total"] > 0,
                                             store_ratio_df["net_quantity"] / store_ratio_df["cluster_total"],
                                             0.0)

    # 6) Forecast ile merge
    merge_keys = ["store_cluster","product_att_01","product_att_02","product_att_05"]
    # tip güvenliği: numerik/int'e çevir
    for col in merge_keys:
        future_forecast[col] = pd.to_numeric(future_forecast[col], errors="coerce")
        store_ratio_df[col] = pd.to_numeric(store_ratio_df[col], errors="coerce")

    # NA key'leri düşür
    future_forecast = future_forecast.dropna(subset=merge_keys).copy()
    store_ratio_df   = store_ratio_df.dropna(subset=merge_keys).copy()

    # İç birleşim
    forecast_store = future_forecast.merge(store_ratio_df, on=merge_keys, how="inner")

    # 7) Mağaza bazlı tahmin
    if "predicted_quantity" not in forecast_store.columns:
        raise KeyError("Forecast tablosunda 'predicted_quantity' kolonu bekleniyor.")
    forecast_store["store_predicted_quantity"] = (forecast_store["predicted_quantity"] * forecast_store["store_ratio"]).astype(float)

    # 8) Çıkış kolonları
    out_cols = ["week_start_date","store_cluster","store_code",
                "product_att_01","product_att_02","product_att_05",
                "store_predicted_quantity","predicted_quantity","store_ratio"]
    if id_col and id_col not in out_cols:
        out_cols.insert(6, id_col)  # ürün anahtarını da ekle

    final_forecast = forecast_store[out_cols].copy()

    # 9) Yaz & yükle (PARQUET)
    out_local = "/tmp/breakdown_store.parquet"
    final_forecast.to_parquet(out_local, index=False, engine=ENGINE, compression="snappy")
    gcs.upload(bucket_name=BUCKET, object_name=BREAKDOWN_OBJ, filename=out_local)
    logging.info("✅ Uploaded: gs://%s/%s", BUCKET, BREAKDOWN_OBJ)

    # temizlik
    del (future_forecast, sales, product, calendar, cluster, store_ratio_df, forecast_store, final_forecast)
    gc.collect()
