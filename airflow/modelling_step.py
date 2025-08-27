import os
import gc
import logging
import numpy as np
import pandas as pd
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from lightgbm import LGBMRegressor

BUCKET = "europe-west1-airflow-a054c263-bucket"

BASE_PREFIX = "demand_forecasting"
INPUT_DIRS = [f"{BASE_PREFIX}/output"]  
OUTPUT_PREFIX = f"{BASE_PREFIX}/output"

FUTURE_WEEKS = int(os.environ.get("FUTURE_WEEKS", "12"))
FEATURES_OBJ = f"{OUTPUT_PREFIX}/weekly_sales_with_features.parquet"
FORECAST_OBJ = f"{OUTPUT_PREFIX}/future_12_weeks_hierarchical_forecast.parquet"

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

def run_forecast(**context):
    gcs = GCSHook(gcp_conn_id="google_cloud_default")

    feat_local = "/tmp/weekly_sales_with_features.parquet"
    gcs.download(bucket_name=BUCKET, object_name=FEATURES_OBJ, filename=feat_local)

    df = pd.read_parquet(feat_local, engine=ENGINE)
    df["week_start_date"] = pd.to_datetime(df["week_start_date"], errors="coerce")

    target = "total_quantity"
    hier_cols = ["product_att_01", "product_att_02", "product_att_05", "store_cluster"]

    # Check if all hierarchical columns exist
    missing_hier_cols = [col for col in hier_cols if col not in df.columns]
    if missing_hier_cols:
        logging.error(f"Missing hierarchical columns: {missing_hier_cols}")
        logging.info(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"Missing hierarchical columns: {missing_hier_cols}")

    drop_cols = [
        "week_start_date", "total_quantity",
        "product_att_01_desc", "product_att_02_desc", "product_att_05_desc",
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    for c in feature_cols:
        if df[c].dtype == "object":
            df[c] = df[c].astype("category")
    cat_features = [c for c in feature_cols if str(df[c].dtype) == "category"]

    agg_map = {target: "sum"}
    for c in feature_cols:
        if c not in hier_cols:
            agg_map[c] = "last"
    
    # Add error handling for groupby
    try:
        df_agg = df.groupby(hier_cols + ["week_start_date"], as_index=False, observed=True).agg(agg_map)
    except Exception as e:
        logging.error(f"Groupby error: {e}")
        logging.info(f"DataFrame shape: {df.shape}")
        logging.info(f"DataFrame columns: {df.columns.tolist()}")
        logging.info(f"Hierarchical columns: {hier_cols}")
        logging.info(f"Feature columns: {feature_cols}")
        logging.info(f"Aggregation map: {agg_map}")
        raise


    X_train = df_agg[feature_cols]
    y_train = df_agg[target]

    model = LGBMRegressor(
        objective="regression",
        learning_rate=0.1,
        max_depth=2,
        num_leaves=15,
        n_estimators=100,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        categorical_feature=cat_features if len(cat_features) > 0 else None,
    )


    last_date = pd.to_datetime(df_agg["week_start_date"].max())
    future_dates = [last_date + pd.Timedelta(weeks=i + 1) for i in range(FUTURE_WEEKS)]

    base_keys = df_agg[hier_cols].drop_duplicates().reset_index(drop=True)
    future_df = (
        base_keys.assign(key=1)
        .merge(pd.DataFrame({"week_start_date": future_dates, "key": 1}), on="key")
        .drop(columns="key")
    )


    if "month" in feature_cols:
        future_df["month"] = future_df["week_start_date"].dt.month
    if "week_of_year" in feature_cols:
        future_df["week_of_year"] = future_df["week_start_date"].dt.isocalendar().week.astype(int)


    special_flags = [c for c in feature_cols if c.lower() in
                    ["ramazan_bayrami", "kurban_bayrami", "kara_cuma", "is_special_day"]]
    for c in special_flags:
        future_df[c] = 0


    for col in feature_cols:
        if col in hier_cols:
            continue
        if col in ["month", "week_of_year"] + special_flags:
            continue
        mapping = df_agg.groupby(hier_cols)[col].last()
        future_df[col] = future_df.set_index(hier_cols).index.map(lambda idx: mapping.get(idx, np.nan))


    preds = model.predict(future_df[feature_cols])
    future_df["predicted_quantity"] = np.clip(preds, a_min=0, a_max=None).astype(float)


    out_cols = hier_cols + ["week_start_date", "predicted_quantity"]
    out_local = "/tmp/future_12_weeks_hierarchical_forecast.parquet"
    future_df[out_cols].to_parquet(out_local, index=False, engine=ENGINE, compression="snappy")

    gcs.upload(bucket_name=BUCKET, object_name=FORECAST_OBJ, filename=out_local)

    del df, df_agg, future_df, model
    gc.collect()
