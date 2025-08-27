import logging
import pendulum
import pandas as pd
import os
import gc
from datetime import timedelta

from airflow import DAG
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.operators.python import PythonOperator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

default_args = {
    "owner": "genorthink",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

BUCKET = "europe-west1-airflow-a054c263-bucket"
BASE_PREFIX = "demand_forecasting"
INPUT_PREFIX = f"{BASE_PREFIX}/input"


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


def log_memory_usage(stage: str):
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        logging.info("Memory usage at %s: %.1f MB", stage, memory_mb)
    except Exception:
        logging.info("Stage: %s", stage)


def load_open_order(**context):
    logging.info("Open order verileri yükleniyor...")
    log_memory_usage("open_order_start")

    try:
        mysql_hook = MySqlHook(mysql_conn_id="sporthink_mysql")

        sql_query = """
        SELECT 
            urun_kodu AS product_id,
            cikis_depo_kodu AS depo_kodu,
            SUM(toplam_yoldaki_miktar) AS toplam_yoldaki_miktar,
            SUM(toplanmayan_urun) AS toplanmayan_urun
        FROM Genboost.realtime_stock_movement
        GROUP BY urun_kodu, cikis_depo_kodu
        """

        df = mysql_hook.get_pandas_df(sql_query)
        logging.info("Open order verisi çekildi. Shape: %s", df.shape)
        log_memory_usage("open_order_loaded")

        local_file = "/tmp/open_order.parquet"
        df.to_parquet(local_file, index=False, engine=ENGINE, compression="snappy")
        logging.info("Open order parquet dosyası oluşturuldu: %s", local_file)

        gcs = GCSHook(gcp_conn_id="google_cloud_default")
        gcs_obj = f"{INPUT_PREFIX}/open_order.parquet"
        gcs.upload(bucket_name=BUCKET, object_name=gcs_obj, filename=local_file)
        logging.info("Open order GCS'e yüklendi: gs://%s/%s", BUCKET, gcs_obj)

        try:
            os.remove(local_file)
        except FileNotFoundError:
            pass
        del df
        gc.collect()
        log_memory_usage("open_order_end")

        return f"gs://{BUCKET}/{gcs_obj}"

    except Exception as e:
        logging.error("Open order yükleme hatası: %s", str(e))
        raise


def load_store_stock(**context):
    logging.info("Store stock verileri yükleniyor...")
    log_memory_usage("store_stock_start")

    try:
        mysql_hook = MySqlHook(mysql_conn_id="sporthink_mysql")

        sql_query = "SELECT * FROM Genboost.realtime_store_stock"

        df = mysql_hook.get_pandas_df(sql_query)
        logging.info("Store stock verisi çekildi. Shape: %s", df.shape)
        log_memory_usage("store_stock_loaded")

        local_file = "/tmp/store_stock.parquet"
        df.to_parquet(local_file, index=False, engine=ENGINE, compression="snappy")
        logging.info("Store stock parquet dosyası oluşturuldu: %s", local_file)

        gcs = GCSHook(gcp_conn_id="google_cloud_default")
        gcs_obj = f"{INPUT_PREFIX}/store_stock.parquet"
        gcs.upload(bucket_name=BUCKET, object_name=gcs_obj, filename=local_file)
        logging.info("Store stock GCS'e yüklendi: gs://%s/%s", BUCKET, gcs_obj)

        try:
            os.remove(local_file)
        except FileNotFoundError:
            pass
        del df
        gc.collect()
        log_memory_usage("store_stock_end")

        return f"gs://{BUCKET}/{gcs_obj}"

    except Exception as e:
        logging.error("Store stock yükleme hatası: %s", str(e))
        raise


def load_warehouse_stock(**context):
    logging.info("Warehouse stock verileri yükleniyor...")
    log_memory_usage("warehouse_stock_start")

    try:
        mysql_hook = MySqlHook(mysql_conn_id="sporthink_mysql")

        sql_query = "SELECT * FROM Genboost.warehouse_stock"

        df = mysql_hook.get_pandas_df(sql_query)
        logging.info("Warehouse stock verisi çekildi. Shape: %s", df.shape)
        log_memory_usage("warehouse_stock_loaded")

        local_file = "/tmp/warehouse_stock.parquet"
        df.to_parquet(local_file, index=False, engine=ENGINE, compression="snappy")
        logging.info("Warehouse stock parquet dosyası oluşturuldu: %s", local_file)

        gcs = GCSHook(gcp_conn_id="google_cloud_default")
        gcs_obj = f"{INPUT_PREFIX}/warehouse_stock.parquet"
        gcs.upload(bucket_name=BUCKET, object_name=gcs_obj, filename=local_file)
        logging.info("Warehouse stock GCS'e yüklendi: gs://%s/%s", BUCKET, gcs_obj)

        try:
            os.remove(local_file)
        except FileNotFoundError:
            pass
        del df
        gc.collect()
        log_memory_usage("warehouse_stock_end")

        return f"gs://{BUCKET}/{gcs_obj}"

    except Exception as e:
        logging.error("Warehouse stock yükleme hatası: %s", str(e))
        raise


def load_history_sales(**context):
    logging.info("History sales verileri yükleniyor...")
    log_memory_usage("history_sales_start")

    try:
        mysql_hook = MySqlHook(mysql_conn_id="sporthink_mysql")

        sql_query = """
        SELECT *
        FROM Genboost.history_sales
        WHERE `date` >= DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 12 MONTH), '%Y-%m-01')
        """

        df = mysql_hook.get_pandas_df(sql_query)
        logging.info("History sales verisi çekildi. Shape: %s", df.shape)
        log_memory_usage("history_sales_loaded")

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            logging.info("Tarih dönüştürüldü. Aralık: %s - %s", df["date"].min(), df["date"].max())

        local_file = "/tmp/history_sales.parquet"
        df.to_parquet(local_file, index=False, engine=ENGINE, compression="snappy")
        logging.info("History sales parquet dosyası oluşturuldu: %s", local_file)

        gcs = GCSHook(gcp_conn_id="google_cloud_default")
        gcs_obj = f"{INPUT_PREFIX}/history_sales.parquet"
        gcs.upload(bucket_name=BUCKET, object_name=gcs_obj, filename=local_file)
        logging.info("History sales GCS'e yüklendi: gs://%s/%s", BUCKET, gcs_obj)

        try:
            os.remove(local_file)
        except FileNotFoundError:
            pass
        del df
        gc.collect()
        log_memory_usage("history_sales_end")

        return f"gs://{BUCKET}/{gcs_obj}"

    except Exception as e:
        logging.error("History sales yükleme hatası: %s", str(e))
        raise





# DAG tanımı
with DAG(
    dag_id="sporthink_input_data_loader",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule_interval="0 2 * * *",  # Her gün UTC 02:00
    catchup=False,
    default_args=default_args,
    description="Sporthink input tablolarını database'den yükler ve parquet formatında GCS'e kaydeder",
    tags=["sporthink", "data-loading", "etl"],
) as dag:

    task_open_order = PythonOperator(
        task_id="load_open_order",
        python_callable=load_open_order,
        doc_md="Open order verilerini yükler.",
    )

    task_store_stock = PythonOperator(
        task_id="load_store_stock",
        python_callable=load_store_stock,
        doc_md="Store stock verilerini yükler.",
    )

    task_warehouse_stock = PythonOperator(
        task_id="load_warehouse_stock",
        python_callable=load_warehouse_stock,
        doc_md="Warehouse stock verilerini yükler.",
    )

    task_history_sales = PythonOperator(
        task_id="load_history_sales",
        python_callable=load_history_sales,
        doc_md="History sales verilerini yükler (son 12 ay).",
    )


