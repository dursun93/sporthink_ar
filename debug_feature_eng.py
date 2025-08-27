#!/usr/bin/env python3
"""
Debug script for feature engineering DAG
"""
import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the dags directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dags'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_data():
    """Create small test data to debug the feature engineering"""
    print("Creating test data...")
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='W-MON')
    product_attrs = [1, 2, 3, 4, 5]
    store_clusters = ['A', 'B', 'C']
    
    data = []
    for date in dates:
        for att1 in product_attrs:
            for att2 in product_attrs:
                for att5 in product_attrs:
                    for cluster in store_clusters:
                        data.append({
                            'week_start_date': date,
                            'total_quantity': np.random.randint(0, 100),
                            'discount_frac_wavg': np.random.random(),
                            'discount_frac_mean': np.random.random(),
                            'unit_price_mean': np.random.uniform(10, 100),
                            'unit_price_median': np.random.uniform(10, 100),
                            'product_att_01': att1,
                            'product_att_02': att2,
                            'product_att_05': att5,
                            'product_att_01_desc': f'Desc_{att1}',
                            'product_att_02_desc': f'Desc_{att2}',
                            'product_att_05_desc': f'Desc_{att5}',
                            'store_cluster': cluster,
                        })
    
    df = pd.DataFrame(data)
    print(f"Created test data with {len(df)} rows")
    return df

def test_feature_engineering():
    """Test the feature engineering function with small data"""
    try:
        from df_feature_eng_dag import add_shifted_rolled_features, _to_cat
        
        print("Testing feature engineering with small dataset...")
        
        # Create test data
        df = create_test_data()
        
        # Test basic operations
        print("Testing basic operations...")
        df["week_start_date"] = pd.to_datetime(df["week_start_date"], errors="coerce")
        df["week_of_year"] = df["week_start_date"].dt.isocalendar().week.astype("int16")
        
        df.loc[df["total_quantity"] < 0, "total_quantity"] = 0
        df["total_quantity"] = pd.to_numeric(df["total_quantity"], errors="coerce")
        
        segment_cols = ["product_att_01", "product_att_02", "product_att_05", "store_cluster"]
        _to_cat(df, ["product_att_01_desc", "product_att_02_desc", "product_att_05_desc", "store_cluster"])
        
        print("✅ Basic operations completed")
        
        # Test feature engineering with smaller parameters
        print("Testing feature engineering with reduced parameters...")
        df = add_shifted_rolled_features(
            data=df,
            date_col="week_start_date",
            granularity_cols=segment_cols,
            target_col="total_quantity",
            shifts=[1, 2, 3, 4],  # Reduced from original
            rolls={1: [4], 2: [4]},  # Reduced from original
            compute_diffs=[1, 2],  # Reduced from original
        )
        
        print("✅ Feature engineering completed successfully")
        print(f"Final dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in feature engineering test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_usage():
    """Test memory usage with different data sizes"""
    print("\nTesting memory usage...")
    
    sizes = [1000, 5000, 10000]
    
    for size in sizes:
        print(f"\nTesting with {size} rows...")
        try:
            # Create data of specified size
            dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='W-MON')
            data = []
            
            for i in range(size):
                data.append({
                    'week_start_date': np.random.choice(dates),
                    'total_quantity': np.random.randint(0, 100),
                    'product_att_01': np.random.randint(1, 6),
                    'product_att_02': np.random.randint(1, 6),
                    'product_att_05': np.random.randint(1, 6),
                    'store_cluster': np.random.choice(['A', 'B', 'C']),
                })
            
            df = pd.DataFrame(data)
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            print(f"Error with size {size}: {e}")

if __name__ == "__main__":
    print("🔍 Debugging feature engineering DAG...")
    
    success = test_feature_engineering()
    
    if success:
        test_memory_usage()
        print("\n🎉 All tests passed! The feature engineering should work in Airflow.")
    else:
        print("\n❌ Tests failed. Check the errors above.")
        sys.exit(1)

