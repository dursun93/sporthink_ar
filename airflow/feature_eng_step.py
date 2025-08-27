import gc
import logging
import numpy as np
import pandas as pd
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.providers.postgres.hooks.postgres import PostgresHook

BUCKET = "europe-west1-airflow-a054c263-bucket"
BASE_PREFIX = "demand_forecasting"
INPUT_DIRS = [f"{BASE_PREFIX}/input"]
OUTPUT_PREFIX = f"{BASE_PREFIX}/output"
DEFAULT_PREP_INPUT = f"{OUTPUT_PREFIX}/weekly_sales_with_cluster_125.parquet"
FEATURE_OUTPUT_OBJ = f"{OUTPUT_PREFIX}/weekly_sales_with_features.parquet"

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

def add_shifted_rolled_features(data: pd.DataFrame, date_col: str, granularity_cols: list,
                                target_col: str, shifts: list, rolls: dict, compute_diffs: list = None):
    data = data.sort_values(granularity_cols + [date_col]).reset_index(drop=True)
    for s in shifts:
        data[f"{target_col}_shifted_{s}"] = data.groupby(granularity_cols, observed=True)[target_col].shift(s)
    if compute_diffs:
        for d in compute_diffs:
            c1 = f"{target_col}_shifted_{d}"
            c2 = f"{target_col}_shifted_{d+1}"
            if c1 not in data.columns:
                data[c1] = data.groupby(granularity_cols, observed=True)[target_col].shift(d)
            if c2 not in data.columns:
                data[c2] = data.groupby(granularity_cols, observed=True)[target_col].shift(d+1)
            base = data[c2].replace(0, np.nan)
            data[f"pct_change_{d}"] = (data[c1] - data[c2]) / base
    for shift, windows in rolls.items():
        shifted = data.groupby(granularity_cols, observed=True)[target_col].shift(shift)
        for w in windows:
            data[f"min_{target_col}_roll{w}_shift{shift}"]  = shifted.rolling(w, min_periods=2).min()
            data[f"mean_{target_col}_roll{w}_shift{shift}"] = shifted.rolling(w, min_periods=2).mean()
            data[f"max_{target_col}_roll{w}_shift{shift}"]  = shifted.rolling(w, min_periods=2).max()
    return data

def _to_cat(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df

def run_feature_eng(**context):
    gcs = GCSHook(gcp_conn_id="google_cloud_default")

    dag_run = context.get("dag_run")
    conf = dag_run.conf or {} if dag_run else {}
    prep_obj = conf.get("prep_output", DEFAULT_PREP_INPUT)
    logging.info("Using prep_output: gs://%s/%s", BUCKET, prep_obj)

    agg_local = "/tmp/weekly_sales_with_cluster_125.parquet"
    gcs.download(bucket_name=BUCKET, object_name=prep_obj, filename=agg_local)

    # Database connection ID - using the same connection as other scripts
    db_conn_id = "sporthink_mysql"
    
    # SQL query for dim_calendar
    dim_calendar_query = "SELECT * FROM Genboost.dim_calendar"
    
    # Fetch calendar data from database
    logging.info("Fetching dim_calendar data from database...")
    calendar = _fetch_from_db(db_conn_id, dim_calendar_query)

    cols_needed = [
        "week_start_date","total_quantity",
        "discount_frac_wavg","discount_frac_mean","unit_price_mean","unit_price_median",
        "product_att_01","product_att_02","product_att_05",
        "product_att_01_desc","product_att_02_desc","product_att_05_desc",
        "store_cluster",
    ]
    df = pd.read_parquet(agg_local, columns=[c for c in cols_needed if c], engine=ENGINE)
    cal_cols = ["date","month","special_day_tag","ramazan_bayrami","kurban_bayrami","kara_cuma"]
    calendar = calendar[cal_cols]  # Select only needed columns

    df["week_start_date"] = pd.to_datetime(df["week_start_date"], errors="coerce")
    df["week_of_year"] = df["week_start_date"].dt.isocalendar().week.astype("int16")
    df.loc[df["total_quantity"] < 0, "total_quantity"] = 0
    df["total_quantity"] = pd.to_numeric(df["total_quantity"], errors="coerce")
    segment_cols = ["product_att_01","product_att_02","product_att_05","store_cluster"]
    _to_cat(df, ["product_att_01_desc","product_att_02_desc","product_att_05_desc","store_cluster"])

    for col in ["discount_frac_wavg","discount_frac_mean","unit_price_mean","unit_price_median"]:
        if col in df.columns:
            df[f"{col}_l1"] = df.groupby(segment_cols, observed=True)[col].shift(1)
            df[f"{col}_ma4_l1"] = (
                df.groupby(segment_cols, observed=True)[col].shift(1).rolling(4, min_periods=2).mean()
            )

    df["total_quantity_last_year"] = df.groupby(segment_cols, observed=True)["total_quantity"].shift(53)

    df = add_shifted_rolled_features(
        data=df,
        date_col="week_start_date",
        granularity_cols=segment_cols,
        target_col="total_quantity",
        shifts=[1,2,3,4,5,6,8,12,24,48,49,50,51,52],
        rolls={1:[4,8], 2:[4,8], 3:[4,8], 8:[4], 12:[4]},
        compute_diffs=[1,2,3,4,8,12],
    )

    df["segment_mean"] = df.groupby(segment_cols, observed=True)["total_quantity"].transform("mean")

    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
    calendar["week_start_date"] = (calendar["date"] - pd.to_timedelta(calendar["date"].dt.weekday, unit="D")).dt.floor("D")
    calendar_weekly = (
        calendar.groupby("week_start_date", observed=True)
        .agg({
            "month": "first",
            "special_day_tag": lambda x: x.dropna().iloc[0] if x.notna().any() else "YOK",
            "ramazan_bayrami": "max",
            "kurban_bayrami": "max",
            "kara_cuma": "max",
        })
        .reset_index()
    )
    calendar_weekly["is_special_day"] = (calendar_weekly["special_day_tag"] != "YOK").astype("int8")
    calendar_weekly = calendar_weekly.drop(columns=["special_day_tag"])

    df = df.merge(calendar_weekly, on="week_start_date", how="left")
    if "month" in df.columns:
        df["month"] = df["month"].astype("category")

    out_local = "/tmp/weekly_sales_with_features.parquet"
    df.to_parquet(out_local, index=False, engine=ENGINE, compression="snappy")
    gcs.upload(bucket_name=BUCKET, object_name=FEATURE_OUTPUT_OBJ, filename=out_local)

    del df, calendar, calendar_weekly
    gc.collect()
