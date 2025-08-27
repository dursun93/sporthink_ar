import os
import logging
import pandas as pd
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.providers.postgres.hooks.postgres import PostgresHook

BUCKET = "europe-west1-airflow-a054c263-bucket"
BASE_PREFIX = "demand_forecasting"
OUTPUT_PREFIX = f"{BASE_PREFIX}/output"

def _fetch_from_db(conn_id: str, query: str) -> pd.DataFrame:
    """Fetch data from database using the specified connection and query"""
    try:
        # Try MySQL first
        hook = MySqlHook(mysql_conn_id=conn_id)
        df = hook.get_pandas_df(query)
        return df
    except Exception as mysql_err:
        try:
            # Try PostgreSQL if MySQL fails
            hook = PostgresHook(postgres_conn_id=conn_id)
            df = hook.get_pandas_df(query)
            return df
        except Exception as postgres_err:
            logging.error(f"Failed to fetch data from both MySQL and PostgreSQL. MySQL error: {mysql_err}, PostgreSQL error: {postgres_err}")
            raise

def _execute_db_query(conn_id: str, query: str):
    """Execute a query on the database"""
    try:
        # Try MySQL first
        hook = MySqlHook(mysql_conn_id=conn_id)
        hook.run(query)
    except Exception as mysql_err:
        try:
            # Try PostgreSQL if MySQL fails
            hook = PostgresHook(postgres_conn_id=conn_id)
            hook.run(query)
        except Exception as postgres_err:
            logging.error(f"Failed to execute query on both MySQL and PostgreSQL. MySQL error: {mysql_err}, PostgreSQL error: {postgres_err}")
            raise

def _insert_dataframe_to_db(df: pd.DataFrame, conn_id: str, table_name: str, if_exists: str = 'replace'):
    """Insert DataFrame to database table"""
    try:
        # Try MySQL first
        hook = MySqlHook(mysql_conn_id=conn_id)
        hook.insert_rows(table=table_name, rows=df.values.tolist(), target_fields=df.columns.tolist(), replace=if_exists=='replace')
    except Exception as mysql_err:
        try:
            # Try PostgreSQL if MySQL fails
            hook = PostgresHook(postgres_conn_id=conn_id)
            hook.insert_rows(table=table_name, rows=df.values.tolist(), target_fields=df.columns.tolist(), replace=if_exists=='replace')
        except Exception as postgres_err:
            logging.error(f"Failed to insert data to both MySQL and PostgreSQL. MySQL error: {mysql_err}, PostgreSQL error: {postgres_err}")
            raise

def truncate_and_insert_replenishment_results(gcs: GCSHook, db_conn_id: str):
    """Truncate and insert replenishment results to database"""
    
    # Download replenishment results CSV from GCS
    csv_local = "/tmp/replenishment_results.csv"
    csv_obj = f"{OUTPUT_PREFIX}/replenishment_results.csv"
    
    try:
        gcs.download(bucket_name=BUCKET, object_name=csv_obj, filename=csv_local)
        
        # Read CSV file
        df = pd.read_csv(csv_local)
        
        # Add model_date column
        from datetime import date
        today = date.today()
        df.insert(0, 'model_date', today)
        
        # Filter replenishment results to only include ihtiyac_adet > 0
        if 'ihtiyac_adet' in df.columns:
            original_count = len(df)
            df = df[df['ihtiyac_adet'] > 0]
            filtered_count = len(df)
            logging.info(f"📊 Filtered replenishment results: {original_count} → {filtered_count} records (removed {original_count - filtered_count} records)")
        else:
            logging.warning("⚠️ Warning: 'ihtiyac_adet' column not found, no filtering applied")
        
        # Truncate table
        truncate_query = "TRUNCATE TABLE Genboost.replenishment_results"
        _execute_db_query(db_conn_id, truncate_query)
        
        # Insert new data
        _insert_dataframe_to_db(df, db_conn_id, "Genboost.replenishment_results", if_exists='append')
        
    except Exception as e:
        logging.error(f"❌ Error processing replenishment results: {e}")
        raise

def truncate_and_insert_warehouse_distribution(gcs: GCSHook, db_conn_id: str):
    """Truncate and insert warehouse distribution to database"""
    
    # Download warehouse distribution CSV from GCS
    csv_local = "/tmp/warehouse_distribution.csv"
    csv_obj = f"{OUTPUT_PREFIX}/warehouse_distribution.csv"
    
    try:
        gcs.download(bucket_name=BUCKET, object_name=csv_obj, filename=csv_local)
        
        # Read CSV file
        df = pd.read_csv(csv_local)
        
        # Add model_date column
        from datetime import date
        today = date.today()
        df.insert(0, 'model_date', today)
        
        # Truncate table
        truncate_query = "TRUNCATE TABLE Genboost.warehouse_distribution"
        _execute_db_query(db_conn_id, truncate_query)
        
        # Insert new data
        _insert_dataframe_to_db(df, db_conn_id, "Genboost.warehouse_distribution", if_exists='append')
        
    except Exception as e:
        logging.error(f"❌ Error processing warehouse distribution: {e}")
        raise

def run_db_output(**context):
    """Main function to write replenishment outputs to database"""
    
    gcs = GCSHook(gcp_conn_id="google_cloud_default")
    db_conn_id = "sporthink_mysql"
    
    try:
        # Process replenishment results
        truncate_and_insert_replenishment_results(gcs, db_conn_id)
        
        # Process warehouse distribution
        truncate_and_insert_warehouse_distribution(gcs, db_conn_id)
        
    except Exception as e:
        logging.error(f"❌ Database output process failed: {e}")
        raise

if __name__ == "__main__":
    run_db_output()
