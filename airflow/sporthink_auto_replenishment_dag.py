import pendulum
import sys
import os
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.hooks.gcs import GCSHook

def load_step_from_gcs(step_name):
    """GCS'den step dosyasını yükle ve import et"""
    gcs = GCSHook(gcp_conn_id="google_cloud_default")
    bucket = "europe-west1-airflow-a054c263-bucket"
    gcs_path = f"demand_forecasting/df_steps/{step_name}.py"
    local_path = f"/tmp/{step_name}.py"
    
    try:
        gcs.download(bucket_name=bucket, object_name=gcs_path, filename=local_path)
        
        # Dosyayı import et
        import importlib.util
        spec = importlib.util.spec_from_file_location(step_name, local_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        return module
    except Exception as e:
        raise ImportError(f"GCS'den {step_name} yüklenemedi: {e}")

auto_replenishment_module = load_step_from_gcs("auto_replenishment_step")
db_output_module = load_step_from_gcs("db_output_step")

default_args = {
    "owner": "genboost", 
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2)
}

with DAG(
    dag_id="sporthink_auto_replenishment",
    start_date=pendulum.datetime(2025, 8, 1, tz="UTC"),
    schedule_interval="0 6 * * *",  # Her gün 06:00 UTC = 09:00 TRT (UTC+3)
    catchup=False,
    default_args=default_args,
    description="Auto replenishment pipeline - runs daily at 09:00 Istanbul time",
    tags=["auto_replenishment", "daily"],
) as dag:

    t_auto_replenishment = PythonOperator(
        task_id="auto_replenishment_step",
        python_callable=auto_replenishment_module.run_auto_replenishment,
        execution_timeout=timedelta(hours=2),
        retries=2,
        retry_delay=timedelta(minutes=5),
    )

    t_db_output = PythonOperator(
        task_id="db_output_step",
        python_callable=db_output_module.run_db_output,
        execution_timeout=timedelta(hours=1),
        retries=2,
        retry_delay=timedelta(minutes=3),
    )

    t_auto_replenishment >> t_db_output
