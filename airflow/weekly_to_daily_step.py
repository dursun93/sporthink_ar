import os
import gc
import logging
import psutil
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from airflow.providers.google.cloud.hooks.gcs import GCSHook


BUCKET = "europe-west1-airflow-a054c263-bucket"

BASE_PREFIX   = "demand_forecasting"
INPUT_DIRS    = [f"{BASE_PREFIX}/input"]
OUTPUT_PREFIX = f"{BASE_PREFIX}/output"

BREAKDOWN_OBJ        = f"{OUTPUT_PREFIX}/breakdown_store.parquet"
CHUNK_PREFIX = f"{OUTPUT_PREFIX}/chunks"


def log_memory_usage(stage: str):
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    logging.info(f"Memory usage at {stage}: {memory_mb:.1f} MB")


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


def run_weekly_to_daily(**context):
    try:
        log_memory_usage("weekly_to_daily_start")
        gcs = GCSHook(gcp_conn_id="google_cloud_default")

        breakdown_local = "/tmp/breakdown_store.parquet"
        gcs.download(bucket_name=BUCKET, object_name=BREAKDOWN_OBJ, filename=breakdown_local)
        logging.info("Downloaded breakdown")

        weekly_pred = pd.read_parquet(breakdown_local, engine=ENGINE)
        weekly_pred["week_start_date"] = pd.to_datetime(weekly_pred["week_start_date"], errors="coerce")
        log_memory_usage("after_loading_breakdown")

        daily_store_pred = weekly_to_daily(weekly_pred)
        logging.info("Weekly->Daily done; daily rows=%d", len(daily_store_pred))
        
        daily_local = "/tmp/daily_store_pred.parquet"
        daily_store_pred.to_parquet(daily_local, index=False, engine=ENGINE, compression="snappy")
        
        daily_obj = f"{OUTPUT_PREFIX}/daily_store_pred.parquet"
        gcs.upload(bucket_name=BUCKET, object_name=daily_obj, filename=daily_local)

        del weekly_pred, daily_store_pred
        gc.collect()
        for temp_file in [breakdown_local, daily_local]:
            try:
                os.remove(temp_file)
            except FileNotFoundError:
                pass
        
        log_memory_usage("weekly_to_daily_end")
        
    except Exception as e:
        logging.error(f"Error in weekly_to_daily: {str(e)}")
        raise


def weekly_to_daily(weekly_pred: pd.DataFrame) -> pd.DataFrame:
    w = weekly_pred.copy()
    w["week_start_date"] = pd.to_datetime(w["week_start_date"], errors="coerce")
    w["store_code"] = w["store_code"].astype(str).str.strip()
    w["stok_kodu"] = w["stok_kodu"].astype(str).str.strip()
    w["store_predicted_quantity"] = pd.to_numeric(w["store_predicted_quantity"], errors="coerce").fillna(0.0)

    w = w.sort_values(["store_code", "stok_kodu", "week_start_date"]).reset_index(drop=True)
    w["next_week_start"] = (
        w.groupby(["store_code", "stok_kodu"])["week_start_date"]
         .shift(-1)
         .fillna(w["week_start_date"] + pd.Timedelta(days=7))
    )

    w["n_days"] = (w["next_week_start"] - w["week_start_date"]).dt.days
    w.loc[w["n_days"] <= 0, "n_days"] = 7
    w["n_days"] = w["n_days"].astype(int)

    rep = w.index.repeat(w["n_days"])
    out = w.loc[rep, ["store_code", "stok_kodu", "week_start_date", "store_predicted_quantity", "n_days"]].copy()
    out["offset"] = out.groupby(level=0).cumcount()
    out["date"] = (out["week_start_date"] + pd.to_timedelta(out["offset"], unit="D")).dt.normalize()
    out["prediction_store_daily"] = out["store_predicted_quantity"] / out["n_days"]

    daily_store_pred = (
        out.groupby(["store_code", "stok_kodu", "date"], as_index=False)["prediction_store_daily"]
           .sum()
    )
    return daily_store_pred


if __name__ == "__main__":
    run_weekly_to_daily()
