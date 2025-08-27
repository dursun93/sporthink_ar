import os
import gc
import logging
import numpy as np
import pandas as pd
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.providers.postgres.hooks.postgres import PostgresHook

BUCKET = "europe-west1-airflow-a054c263-bucket"
BASE_PREFIX = "demand_forecasting"
OUTPUT_PREFIX = f"{BASE_PREFIX}/output"

def _pick_parquet_engine():
    try:
        import pyarrow  
        return "pyarrow"
    except Exception:
        try:
            import fastparquet  
            return "fastparquet"
        except Exception:
            raise ImportError("pyarrow veya fastparquet hatası")

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

    # Database connection ID - using the same connection as other scripts
    db_conn_id = "sporthink_mysql"

    # SQL queries as specified by the user
    dim_product_query = "SELECT * FROM Genboost.dim_product"
    
    # SQL queries as specified by the user
    
    history_sales_query = """
        SELECT 
            date, store_code, product_id,
            SUM(discount_amount) as discount_amount,
            SUM(net_amount_wovat) as net_amount_wovat,
            SUM(net_quantity) as net_quantity,
            SUM(net_amount_wovat) / SUM(net_quantity) as unit_price
        FROM Genboost.history_sales
        WHERE `date` >= DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 24 MONTH), '%Y-%m-01')
        GROUP BY date, store_code, product_id
    """
    k_means_query = "SELECT * FROM Genboost.store_clustering"

    # Fetch data from database
    logging.info("Fetching dim_product data from database...")
    product = _fetch_from_db(db_conn_id, dim_product_query)
    
    logging.info("Fetching history_sales data from database...")
    sales = _fetch_from_db(db_conn_id, history_sales_query)
    
    logging.info("Fetching store_clustering data from database...")
    kmeans = _fetch_from_db(db_conn_id, k_means_query)

    # Process sales data
    sales_cols = ["date","store_code","product_id","discount_amount","net_amount_wovat","unit_price","net_quantity"]
    sales = sales[sales_cols]  # Select only needed columns
    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    # Removed 180-day limit - using full data from SQL query
    sales = sales.copy()
    sales["week_start_date"] = (sales["date"] - pd.to_timedelta(sales["date"].dt.weekday, unit="D")).dt.floor("D")
    sales["product_id"] = sales["product_id"].astype(str)
    sales["store_code"] = sales["store_code"].astype(str)
    _to_float32(sales, ["discount_amount","net_amount_wovat","unit_price","net_quantity"])

    # Process product data
    product_cols = [
        "product_id","marka_aciklama",
        "product_att_01","product_att_02","product_att_03",
        "product_att_04","product_att_05","product_att_06",
        "product_att_01_desc","product_att_02_desc","product_att_03_desc",
        "product_att_04_desc","product_att_05_desc","product_att_06_desc",
    ]
    product = product[product_cols]  # Select only needed columns
    product["product_id"] = product["product_id"].astype(str)

    # Process kmeans data
    kmeans = kmeans[["store_code","store_cluster"]]  # Select only needed columns
    kmeans["store_code"] = kmeans["store_code"].astype(str)
    kmeans["store_cluster"] = kmeans["store_cluster"].astype("category")

    # Merge dataframes
    df = sales.merge(product, on="product_id", how="left")
    del product; gc.collect()
    df = df.merge(kmeans, on="store_code", how="left")
    del kmeans; gc.collect()

    # Data processing
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

    # Save output
    out_local = "/tmp/weekly_sales_with_cluster_125.parquet"
    agg.to_parquet(out_local, index=False, engine=ENGINE, compression="snappy")
    out_obj = f"{OUTPUT_PREFIX}/weekly_sales_with_cluster_125.parquet"
    gcs.upload(bucket_name=BUCKET, object_name=out_obj, filename=out_local)
