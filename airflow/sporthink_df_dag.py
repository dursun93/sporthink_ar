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

data_prep_module = load_step_from_gcs("data_prep_step")
feature_eng_module = load_step_from_gcs("feature_eng_step")
modelling_step_module = load_step_from_gcs("modelling_step")
breakdown_step_module = load_step_from_gcs("breakdown_step")
weekly_to_daily_module = load_step_from_gcs("weekly_to_daily_step")


default_args = {
    "owner": "genboost", 
    "retries": 3,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=6)
}

with DAG(
    dag_id="sporthink_demand_forecast",
    start_date=pendulum.datetime(2025, 8, 1, tz="UTC"),
    schedule_interval="0 6 * * 1",  # Her Pazartesi 06:00 UTC = 09:00 TRT (UTC+3)
    catchup=False,
    default_args=default_args,
    description="Demand forecasting pipeline with GCS step files",
    tags=["demand_forecasting"],
) as dag:

    t_data_prep = PythonOperator(
        task_id="data_prep",
        python_callable=data_prep_module.run_data_prep,
        execution_timeout=timedelta(hours=2),
    )

    t_feature_eng = PythonOperator(
        task_id="feature_eng",
        python_callable=feature_eng_module.run_feature_eng,
        execution_timeout=timedelta(hours=2),
    )

    t_modelling_step = PythonOperator(
        task_id="modelling_step",
        python_callable=modelling_step_module.run_forecast,
        execution_timeout=timedelta(hours=3),
    )

    t_breakdown_step = PythonOperator(
        task_id="breakdown_step",
        python_callable=breakdown_step_module.run_breakdown,
        execution_timeout=timedelta(hours=2),
    )

    t_weekly_to_daily = PythonOperator(
        task_id="weekly_to_daily_step",
        python_callable=weekly_to_daily_module.run_weekly_to_daily,
        execution_timeout=timedelta(hours=1),
        retries=1,
        retry_delay=timedelta(minutes=3),
    )



    t_data_prep >> t_feature_eng >> t_modelling_step >> t_breakdown_step >> t_weekly_to_daily
