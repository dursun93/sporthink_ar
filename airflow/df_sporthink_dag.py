import pendulum
import sys
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.hooks.gcs import GCSHook

# GCS'den step dosyalarını dinamik olarak import et
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

# Step modüllerini yükle
data_prep_module = load_step_from_gcs("data_prep_step")
feature_eng_module = load_step_from_gcs("feature_eng_step")
modelling_step_module = load_step_from_gcs("modelling_step")
breakdown_step_module = load_step_from_gcs("breakdown_step")

default_args = {"owner": "genboost", "retries": 1}

with DAG(
    dag_id="sporthink_demand_forecast_dag",
    start_date=pendulum.datetime(2025, 8, 1, tz="UTC"),
    schedule_interval="0 7 * * 0",  # Her Pazar 07:00 UTC = 10:00 TRT
    catchup=False,
    default_args=default_args,
    description="Demand forecasting pipeline with GCS step files",
    tags=["demand_forecasting"],
) as dag:

    t_data_prep = PythonOperator(
        task_id="data_prep",
        python_callable=data_prep_module.run_data_prep,
    )

    t_feature_eng = PythonOperator(
        task_id="feature_eng",
        python_callable=feature_eng_module.run_feature_eng,
    )

    t_modelling_step = PythonOperator(
        task_id="modelling_step",
        python_callable=modelling_step_module.run_forecast,

    )

    t_breakdown_step = PythonOperator(
        task_id="breakdown_step",
        python_callable=breakdown_step_module.run_breakdown,
    )



    t_data_prep >> t_feature_eng >> t_modelling_step >> t_breakdown_step
