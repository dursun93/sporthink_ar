import gc
import logging
import numpy as np
import pandas as pd
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.providers.postgres.hooks.postgres import PostgresHook

BUCKET = "europe-west1-airflow-a054c263-bucket"

BASE_PREFIX   = "demand_forecasting"
OUTPUT_PREFIX = f"{BASE_PREFIX}/output"

# Girdi/çıktı nesneleri
FORECAST_OBJ  = f"{OUTPUT_PREFIX}/future_12_weeks_hierarchical_forecast.parquet"
BREAKDOWN_OBJ = f"{OUTPUT_PREFIX}/breakdown_store.parquet"   

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

def _fetch_from_db(conn_id: str, query: str) -> pd.DataFrame:
    """Fetch data from database using the specified connection and query"""
    try:
        # Try MySQL first
        hook = MySqlHook(mysql_conn_id=conn_id)
        df = hook.get_pandas_df(query)
        logging.info(f"Successfully fetched data from MySQL using connection: {conn_id}")
        return df
    except Exception as mysql_err:
        try:
            # Try PostgreSQL if MySQL fails
            hook = PostgresHook(postgres_conn_id=conn_id)
            df = hook.get_pandas_df(query)
            logging.info(f"Successfully fetched data from PostgreSQL using connection: {conn_id}")
            return df
        except Exception as postgres_err:
            logging.error(f"Failed to fetch data from both MySQL and PostgreSQL. MySQL error: {mysql_err}, PostgreSQL error: {postgres_err}")
            raise

def run_breakdown(**context):
    gcs = GCSHook(gcp_conn_id="google_cloud_default")

    # Database connection ID
    db_conn_id = "sporthink_mysql"

    # Download forecast from GCS (this is output from modelling step)
    fc_local = "/tmp/future_12_weeks_hierarchical_forecast.parquet"
    gcs.download(bucket_name=BUCKET, object_name=FORECAST_OBJ, filename=fc_local)
    logging.info("✅ Downloaded forecast: gs://%s/%s", BUCKET, FORECAST_OBJ)
    future_forecast = pd.read_parquet(fc_local, engine=ENGINE)
    future_forecast["week_start_date"] = pd.to_datetime(future_forecast["week_start_date"], errors="coerce")

    # SQL queries - optimized for breakdown step (only 8 weeks of sales data needed)
    history_sales_query = """
        SELECT 
            date, store_code, product_id, stok_kodu, net_quantity
        FROM Genboost.history_sales
        WHERE `date` >= DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 8 WEEK), '%Y-%m-01')
        AND net_quantity >= 0
    """
    dim_product_query = "SELECT * FROM Genboost.dim_product"
    dim_calendar_query = "SELECT * FROM Genboost.dim_calendar"
    store_clustering_query = "SELECT * FROM Genboost.store_clustering"

    # Fetch data from database
    logging.info("Fetching history_sales data from database (last 8 weeks)...")
    sales = _fetch_from_db(db_conn_id, history_sales_query)
    
    logging.info("Fetching dim_product data from database...")
    product = _fetch_from_db(db_conn_id, dim_product_query)
    
    logging.info("Fetching dim_calendar data from database...")
    calendar = _fetch_from_db(db_conn_id, dim_calendar_query)
    
    logging.info("Fetching store_clustering data from database...")
    cluster = _fetch_from_db(db_conn_id, store_clustering_query)

    # Process sales data
    sales["date"] = pd.to_datetime(sales.get("date"), errors="coerce")

    # Calendar processing
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
    calendar["week_start_date"] = (calendar["date"] - pd.to_timedelta(calendar["date"].dt.weekday, unit="D")).dt.floor("D")
    sales = sales.merge(calendar[["date","week_start_date"]], on="date", how="left")

    # Cluster processing
    if not {"store_code","store_cluster"} <= set(cluster.columns):
        raise KeyError("k_means_store içinde 'store_code' ve 'store_cluster' bekleniyor.")
    cluster["store_code"] = cluster["store_code"].astype(str)
    sales["store_code"] = sales["store_code"].astype(str)
    sales = sales.merge(cluster[["store_code","store_cluster"]], on="store_code", how="left")

    # Product processing
    product_keys = [c for c in ["product_id","product_code"] if c in product.columns]
    if not product_keys:
        raise KeyError("dim_product içinde 'product_id' veya 'product_code' bulunamadı.")

    prod_feat_cols = ["product_att_01","product_att_02","product_att_05"]
    prod_use_cols = product_keys + [c for c in prod_feat_cols if c in product.columns]
    product_use = product[prod_use_cols].drop_duplicates().copy()

    # Sales key identification
    sales_key_col = None
    for cand in ["stok_kodu","product_id","product_code"]:
        if cand in sales.columns:
            sales_key_col = cand
            break
    if sales_key_col is None:
        raise KeyError("history_sales içinde 'stok_kodu' veya 'product_id' ya da 'product_code' bekleniyor.")

    # Merge sales with product
    if sales_key_col == "stok_kodu" and "product_code" in product_use.columns:
        left_on, right_on = "stok_kodu", "product_code"
    else:
        left_on = right_on = sales_key_col if sales_key_col in product_use.columns else None
        if left_on is None:
            raise KeyError(f"Eşleşen ürün anahtarı bulunamadı (sales:{sales_key_col}, product cols:{product_use.columns.tolist()}).")

    sales = sales.merge(product_use, left_on=left_on, right_on=right_on, how="left")

    # Product attribute processing
    if "product_att_02" in sales.columns:
        sales["product_att_02"] = sales["product_att_02"].apply(
            lambda x: int(str(x).replace("D","9").replace("N","9")) if isinstance(x,str) and (("D" in x) or ("N" in x)) else x
        )
        sales["product_att_02"] = pd.to_numeric(sales["product_att_02"], errors="coerce")

    # Filter out invalid product attributes
    for col in ["product_att_01","product_att_02"]:
        if col in sales.columns:
            sales = sales[~(sales[col] == 9.0)]

    # Group keys for store ratio calculation
    group_keys = ["store_cluster","product_att_01","product_att_02","product_att_05","store_code"]
    missing = [k for k in group_keys if k not in sales.columns]
    if missing:
        raise KeyError(f"Mağaza oranı için eksik kolon(lar): {missing}")

    # Add product ID to group keys if available
    id_col = "stok_kodu" if "stok_kodu" in sales.columns else ("product_code" if "product_code" in sales.columns else ("product_id" if "product_id" in sales.columns else None))
    if id_col:
        group_keys_with_id = group_keys + [id_col]
    else:
        group_keys_with_id = group_keys

    # Calculate store ratios
    store_ratio_df = (
        sales.groupby(group_keys_with_id, observed=True)["net_quantity"]
             .sum()
             .reset_index()
             .rename(columns={"net_quantity":"net_quantity"})
    )

    # Calculate cluster totals and store ratios
    cluster_keys = ["store_cluster","product_att_01","product_att_02","product_att_05"]
    totals = store_ratio_df.groupby(cluster_keys, observed=True)["net_quantity"].transform("sum")
    store_ratio_df["cluster_total"] = totals
    store_ratio_df["store_ratio"] = np.where(store_ratio_df["cluster_total"] > 0,
                                             store_ratio_df["net_quantity"] / store_ratio_df["cluster_total"],
                                             0.0)

    # Merge with forecast
    merge_keys = ["store_cluster","product_att_01","product_att_02","product_att_05"]

    for col in merge_keys:
        future_forecast[col] = pd.to_numeric(future_forecast[col], errors="coerce")
        store_ratio_df[col] = pd.to_numeric(store_ratio_df[col], errors="coerce")

    future_forecast = future_forecast.dropna(subset=merge_keys).copy()
    store_ratio_df   = store_ratio_df.dropna(subset=merge_keys).copy()

    forecast_store = future_forecast.merge(store_ratio_df, on=merge_keys, how="inner")

    # Calculate store-level predictions
    if "predicted_quantity" not in forecast_store.columns:
        raise KeyError("Forecast tablosunda 'predicted_quantity' kolonu bekleniyor.")
    forecast_store["store_predicted_quantity"] = (forecast_store["predicted_quantity"] * forecast_store["store_ratio"]).astype(float)

    # Prepare final output
    out_cols = ["week_start_date","store_cluster","store_code",
                "product_att_01","product_att_02","product_att_05",
                "store_predicted_quantity","predicted_quantity","store_ratio"]
    if id_col and id_col not in out_cols:
        out_cols.insert(6, id_col)  

    final_forecast = forecast_store[out_cols].copy()

    # Save output
    out_local = "/tmp/breakdown_store.parquet"
    final_forecast.to_parquet(out_local, index=False, engine=ENGINE, compression="snappy")
    gcs.upload(bucket_name=BUCKET, object_name=BREAKDOWN_OBJ, filename=out_local)
    logging.info("✅ Uploaded: gs://%s/%s", BUCKET, BREAKDOWN_OBJ)

    # Cleanup
    del (future_forecast, sales, product, calendar, cluster, store_ratio_df, forecast_store, final_forecast)
    gc.collect()
