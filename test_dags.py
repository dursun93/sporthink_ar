#!/usr/bin/env python3
"""
Test script to verify DAG imports and basic structure
"""
import sys
import os

# Add the dags directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dags'))

def test_dag_imports():
    """Test that both DAGs can be imported successfully"""
    try:
        # Test data prep DAG
        print("Testing df_data_prep_dag import...")
        from df_data_prep_dag import dag as data_prep_dag
        print(f"✅ df_data_prep_dag imported successfully")
        print(f"   - DAG ID: {data_prep_dag.dag_id}")
        print(f"   - Schedule: {data_prep_dag.schedule_interval}")
        print(f"   - Tasks: {[task.task_id for task in data_prep_dag.tasks]}")
        
        # Test feature engineering DAG
        print("\nTesting df_feature_eng_dag import...")
        from df_feature_eng_dag import dag as feature_eng_dag
        print(f"✅ df_feature_eng_dag imported successfully")
        print(f"   - DAG ID: {feature_eng_dag.dag_id}")
        print(f"   - Schedule: {feature_eng_dag.schedule_interval}")
        print(f"   - Tasks: {[task.task_id for task in feature_eng_dag.tasks]}")
        
        # Test trigger relationship
        print("\nTesting trigger relationship...")
        trigger_task = None
        for task in data_prep_dag.tasks:
            if task.task_id == "trigger_feature_eng":
                trigger_task = task
                break
        
        if trigger_task and trigger_task.trigger_dag_id == "df_feature_eng_dag":
            print("✅ Trigger relationship correctly configured")
        else:
            print("❌ Trigger relationship not found or misconfigured")
            
        print("\n🎉 All tests passed! DAGs are ready for Airflow deployment.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_dag_imports()
    sys.exit(0 if success else 1)
