# =============================
# dags/df_feature_eng_dag.py
# =============================
import gc
import logging
from datetime import timedelta

import pendulum
import numpy as np
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator, get_current_context
from airflow.providers.google.cloud.hooks.gcs import GCSHook

BUCKET = "europe-west1-airflow-a054c263-bucket"

BASE_PREFIX = "demand_forecasting"
INPUT_DIRS = [f"{BASE_PREFIX}/input", f"{BASE_PREFIX}/input_parquet"]  # input öncelikli
OUTPUT_PREFIX = f"{BASE_PREFIX}/output"

DEFAULT_PREP_INPUT = f"{OUTPUT_PREFIX}/weekly_sales_with_cluster_125.parquet"
FEATURE_OUTPUT_OBJ = f"{OUTPUT_PREFIX}/weekly_sales_with_features.parquet"

default_args = {
    "owner": "genboost", 
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2)
}


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


def add_shifted_rolled_features(
    data: pd.DataFrame,
    date_col: str,
    granularity_cols: list,
    target_col: str,
    shifts: list,
    rolls: dict,
    compute_diffs: list = None,
):
    logging.info(f"Starting feature engineering for {len(data)} rows...")
    data = data.sort_values(granularity_cols + [date_col]).reset_index(drop=True)

    # LAG
    logging.info(f"Adding {len(shifts)} shift features...")
    for i, s in enumerate(shifts):
        if i % 5 == 0:  # Her 5 shift'te bir log
            logging.info(f"Processing shift {i+1}/{len(shifts)}: shift={s}")
        data[f"{target_col}_shifted_{s}"] = data.groupby(granularity_cols, observed=True)[target_col].shift(s)

    # DIFF/PCT CHANGE
    if compute_diffs:
        logging.info(f"Adding {len(compute_diffs)} difference features...")
        for i, d in enumerate(compute_diffs):
            if i % 3 == 0:  # Her 3 diff'te bir log
                logging.info(f"Processing diff {i+1}/{len(compute_diffs)}: diff={d}")
            c1 = f"{target_col}_shifted_{d}"
            c2 = f"{target_col}_shifted_{d+1}"
            if c1 not in data.columns:
                data[c1] = data.groupby(granularity_cols, observed=True)[target_col].shift(d)
            if c2 not in data.columns:
                data[c2] = data.groupby(granularity_cols, observed=True)[target_col].shift(d + 1)
            base = data[c2].replace(0, np.nan)
            data[f"pct_change_{d}"] = (data[c1] - data[c2]) / base

    # ROLLING
    total_rolls = sum(len(windows) for windows in rolls.values())
    logging.info(f"Adding {total_rolls} rolling features...")
    roll_count = 0
    for shift, windows in rolls.items():
        shifted = data.groupby(granularity_cols, observed=True)[target_col].shift(shift)
        for w in windows:
            roll_count += 1
            if roll_count % 3 == 0:  # Her 3 rolling'te bir log
                logging.info(f"Processing rolling {roll_count}/{total_rolls}: shift={shift}, window={w}")
            data[f"min_{target_col}_roll{w}_shift{shift}"] = shifted.rolling(w, min_periods=2).min()
            data[f"mean_{target_col}_roll{w}_shift{shift}"] = shifted.rolling(w, min_periods=2).mean()
            data[f"max_{target_col}_roll{w}_shift{shift}"] = shifted.rolling(w, min_periods=2).max()

    logging.info("✅ Feature engineering completed")
    return data


def _to_cat(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def run_feature_eng():
    try:
        context = get_current_context()
        dag_run = context.get("dag_run")
        conf = dag_run.conf or {} if dag_run else {}
        prep_obj = conf.get("prep_output", DEFAULT_PREP_INPUT)
        logging.info("Using prep_output: gs://%s/%s", BUCKET, prep_obj)

        gcs = GCSHook(gcp_conn_id="google_cloud_default")

        # 1) data_prep çıktısı
        agg_local = "/tmp/weekly_sales_with_cluster_125.parquet"
        logging.info("Downloading prep output from gs://%s/%s", BUCKET, prep_obj)
        gcs.download(bucket_name=BUCKET, object_name=prep_obj, filename=agg_local)
        logging.info("✅ Downloaded prep output")

        # 2) Takvim
        cal_local = "/tmp/dim_calendar_pivot.parquet"
        logging.info("Downloading calendar data...")
        _download_first_found(gcs, "dim_calendar_pivot.parquet", cal_local)
        logging.info("✅ Downloaded calendar data")

        # 3) Parquet oku (yalnız gerekli kolonlar)
        cols_needed = [
            "week_start_date",
            "total_quantity",
            "discount_frac_wavg",
            "discount_frac_mean",
            "unit_price_mean",
            "unit_price_median",
            "product_att_01",
            "product_att_02",
            "product_att_05",
            "product_att_01_desc",
            "product_att_02_desc",
            "product_att_05_desc",
            "store_cluster",
        ]
        logging.info("Reading aggregated data...")
        df = pd.read_parquet(agg_local, columns=[c for c in cols_needed if c], engine=ENGINE)
        logging.info(f"✅ Loaded {len(df)} rows from aggregated data")

        cal_cols = ["date", "month", "special_day_tag", "ramazan_bayrami", "kurban_bayrami", "kara_cuma"]
        logging.info("Reading calendar data...")
        calendar = pd.read_parquet(cal_local, columns=cal_cols, engine=ENGINE)
        logging.info(f"✅ Loaded {len(calendar)} rows from calendar data")

        # 4) Feature engineering
        logging.info("Starting feature engineering...")
        df["week_start_date"] = pd.to_datetime(df["week_start_date"], errors="coerce")
        df["week_of_year"] = df["week_start_date"].dt.isocalendar().week.astype("int16")

        df.loc[df["total_quantity"] < 0, "total_quantity"] = 0
        df["total_quantity"] = pd.to_numeric(df["total_quantity"], errors="coerce")

        segment_cols = ["product_att_01", "product_att_02", "product_att_05", "store_cluster"]
        _to_cat(df, ["product_att_01_desc", "product_att_02_desc", "product_att_05_desc", "store_cluster"])

        # Fiyat/indirim LAG & MA
        logging.info("Adding price/discount features...")
        for col in ["discount_frac_wavg", "discount_frac_mean", "unit_price_mean", "unit_price_median"]:
            if col in df.columns:
                df[f"{col}_l1"] = df.groupby(segment_cols, observed=True)[col].shift(1)
                df[f"{col}_ma4_l1"] = (
                    df.groupby(segment_cols, observed=True)[col].shift(1).rolling(4, min_periods=2).mean()
                )

        # Geçen yıl aynı hafta
        logging.info("Adding last year features...")
        df["total_quantity_last_year"] = df.groupby(segment_cols, observed=True)["total_quantity"].shift(53)

        # Hedef için geniş lag/rolling
        logging.info("Adding shifted and rolling features...")
        df = add_shifted_rolled_features(
            data=df,
            date_col="week_start_date",
            granularity_cols=segment_cols,
            target_col="total_quantity",
            shifts=[1, 2, 3, 4, 5, 6, 8, 12, 24, 48, 49, 50, 51, 52],
            rolls={1: [4, 8], 2: [4, 8], 3: [4, 8], 8: [4], 12: [4]},
            compute_diffs=[1, 2, 3, 4, 8, 12],
        )

        df["segment_mean"] = df.groupby(segment_cols, observed=True)["total_quantity"].transform("mean")

        # Takvimi haftalık özetle
        logging.info("Processing calendar features...")
        calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
        calendar["week_start_date"] = (calendar["date"] - pd.to_timedelta(calendar["date"].dt.weekday, unit="D")).dt.floor("D")
        calendar_weekly = (
            calendar.groupby("week_start_date", observed=True)
            .agg(
                {
                    "month": "first",
                    "special_day_tag": lambda x: x.dropna().iloc[0] if x.notna().any() else "YOK",
                    "ramazan_bayrami": "max",
                    "kurban_bayrami": "max",
                    "kara_cuma": "max",
                }
            )
            .reset_index()
        )
        calendar_weekly["is_special_day"] = (calendar_weekly["special_day_tag"] != "YOK").astype("int8")
        calendar_weekly = calendar_weekly.drop(columns=["special_day_tag"])

        # Merge
        logging.info("Merging calendar data...")
        df = df.merge(calendar_weekly, on="week_start_date", how="left")
        if "month" in df.columns:
            df["month"] = df["month"].astype("category")

        # 5) Çıkış
        logging.info("Saving results...")
        out_local = "/tmp/weekly_sales_with_features.parquet"
        df.to_parquet(out_local, index=False, engine=ENGINE, compression="snappy")

        logging.info("Uploading to GCS...")
        gcs.upload(bucket_name=BUCKET, object_name=FEATURE_OUTPUT_OBJ, filename=out_local)
        logging.info("✅ Uploaded: gs://%s/%s", BUCKET, FEATURE_OUTPUT_OBJ)

        del df, calendar, calendar_weekly
        gc.collect()
        
        logging.info("🎉 Feature engineering completed successfully!")
        
    except Exception as e:
        logging.error(f"❌ Error in feature engineering: {str(e)}")
        raise e


with DAG(
    dag_id="df_feature_eng_dag",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    schedule_interval=None,          # yalnız tetiklenince çalışır
    catchup=False,
    dagrun_timeout=timedelta(hours=3),  # 3 saat timeout (daha uzun)
    max_active_runs=1,                     # aynı anda tek run
    default_args=default_args,
    description="Feature engineering DAG for demand forecasting - triggered by data prep DAG",
    tags=["demand_forecasting", "feature_engineering"],
) as dag:
    run = PythonOperator(
        task_id="run_feature_eng",
        python_callable=run_feature_eng,
        execution_timeout=timedelta(hours=2),  # Task timeout
        retries=1,
        retry_delay=timedelta(minutes=5),
    )
