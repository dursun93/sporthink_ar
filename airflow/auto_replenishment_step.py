import os
import gc
import logging
import psutil
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.providers.postgres.hooks.postgres import PostgresHook


BUCKET = "europe-west1-airflow-a054c263-bucket"
BASE_PREFIX = "demand_forecasting"
OUTPUT_PREFIX = f"{BASE_PREFIX}/output"
DAILY_STORE_PRED_OBJ = f"{OUTPUT_PREFIX}/daily_store_pred.parquet"





def _pick_parquet_engine():
    try:
        import pyarrow
        return "pyarrow"
    except Exception:
        try:
            import fastparquet
            return "fastparquet"
        except Exception:
            raise ImportError("Parquet için pyarrow veya fastparquet kurulu olmalı.")

ENGINE = _pick_parquet_engine()


def _fetch_from_db(conn_id: str, query: str) -> pd.DataFrame:
    """Fetch data from database using the specified connection and query"""
    try:
        # Try MySQL first
        hook = MySqlHook(mysql_conn_id=conn_id)
        df = hook.get_pandas_df(query)
        logging.info(f"Successfully fetched data from MySQL using connection: {conn_id}")
        return df
    except Exception as mysql_err:
        try:
            # Try PostgreSQL if MySQL fails
            hook = PostgresHook(postgres_conn_id=conn_id)
            df = hook.get_pandas_df(query)
            logging.info(f"Successfully fetched data from PostgreSQL using connection: {conn_id}")
            return df
        except Exception as postgres_err:
            logging.error(f"Failed to fetch data from both MySQL and PostgreSQL. MySQL error: {mysql_err}, PostgreSQL error: {postgres_err}")
            raise


def get_optimal_lookback_period() -> int:
    available_memory_mb = psutil.virtual_memory().available / 1024 / 1024
    total_memory_mb = psutil.virtual_memory().total / 1024 / 1024
    
    logging.info(f"Memory status: {available_memory_mb:.1f} MB available / {total_memory_mb:.1f} MB total")
    
    if available_memory_mb > 12000:
        return 180
    elif available_memory_mb > 8000:
        return 120
    elif available_memory_mb > 4000:
        return 90
    elif available_memory_mb > 2000:
        return 60
    else:
        return 30


def process_auto_replenishment(product, store_stock, warehouse_stock, predictions, sales, orders):
    sales['date'] = pd.to_datetime(sales['date'])
    sales['store_code'] = sales['store_code'].astype(str)
    sales['product_id'] = sales['product_id'].astype(str)
    sales['net_quantity'] = pd.to_numeric(sales['net_quantity'], errors='coerce').fillna(0)
    start_date = sales.date.max()

    orders['acik_siparis'] = orders['toplam_yoldaki_miktar'] + orders['toplanmayan_urun']
    orders.rename(columns={'depo_kodu' : 'store_code'}, inplace=True)
    orders['store_code'] = orders['store_code'].astype(str)
    orders['product_id'] = orders['product_id'].astype(str)
    
    predictions.rename(columns={'prediction_store_daily':'store_predicted_quantity'}, inplace=True)

    product_segments_cols = [
        'product_id',
        'product_att_01', 'product_att_02', 'product_att_03'
    ]
    
    product_subset = product[product_segments_cols].copy()
    product_subset['product_id'] = product_subset['product_id'].astype(str)
    sales_with_segment = sales.merge(product_subset, on='product_id', how='left')
    del product_subset
    gc.collect()

    sales_with_segment['product_att_02'] = sales_with_segment['product_att_02'].astype(str)
    sales_with_segment['product_att_02'] = sales_with_segment['product_att_02'].str.replace('D', '9').str.replace('N', '9')
    sales_with_segment['product_att_02'] = pd.to_numeric(sales_with_segment['product_att_02'], errors='coerce')

    mask = (
        (sales_with_segment['product_att_01'] != 9.0) &
        (sales_with_segment['product_att_02'] != 9.0) &
        (sales_with_segment['product_att_01'] != 999) &
        (sales_with_segment['product_att_02'] != 999) &
        (sales_with_segment["net_quantity"] >= 0)
    )
    
    filtered_sales = sales_with_segment[mask].copy()
    del sales_with_segment
    gc.collect()
    
    filtered_sales['date'] = pd.to_datetime(filtered_sales['date'])
    sales = filtered_sales
    del filtered_sales
    gc.collect()

    store_stock['stok_tarihi'] = pd.to_datetime(store_stock['stok_tarihi'])
    store_stock['magaza'] = store_stock['magaza'].astype(str)
    store_stock['stok_kodu'] = store_stock['stok_kodu'].astype(str)
    store_stock['product_id'] = store_stock['product_id'].astype(str)
    store_stock['toplam_miktar'] = pd.to_numeric(store_stock['toplam_miktar'], errors='coerce').fillna(0)
    
    store_stock_agg = store_stock.groupby(
        ['stok_tarihi','magaza','stok_kodu','product_id']
    ).agg({'toplam_miktar':'sum'}).reset_index()
    
    store_stock_agg.rename(columns={'toplam_miktar': 'stok_miktar'}, inplace=True)
    store_stock_agg['stok_miktar'] = store_stock_agg['stok_miktar'].astype(int)
    
    del store_stock
    gc.collect()

    max_date = pd.to_datetime(sales['date'].max())
    
    sales['period_7'] = (sales['date'] >= max_date - pd.Timedelta(days=6)) & (sales['date'] <= max_date)
    sales['period_14'] = (sales['date'] >= max_date - pd.Timedelta(days=13)) & (sales['date'] <= max_date)
    sales['period_28'] = (sales['date'] >= max_date - pd.Timedelta(days=27)) & (sales['date'] <= max_date)
    
    sales_7 = (sales[sales['period_7']]
               .groupby(['product_id','store_code'], as_index=False)['net_quantity']
               .sum()
               .rename(columns={'net_quantity':'net_quantity_7'}))
    sales_7['net_quantity_7'] = sales_7['net_quantity_7'].clip(lower=0)
    
    sales_14 = (sales[sales['period_14']]
               .groupby(['product_id','store_code'], as_index=False)['net_quantity']
               .sum()
               .rename(columns={'net_quantity':'net_quantity_14'}))
    sales_14['net_quantity_14'] = sales_14['net_quantity_14'].clip(lower=0)
    
    sales_28 = (sales[sales['period_28']]
               .groupby(['product_id','store_code'], as_index=False)['net_quantity']
               .sum()
               .rename(columns={'net_quantity':'net_quantity_28'}))
    sales_28['net_quantity_28'] = sales_28['net_quantity_28'].clip(lower=0)
    
    sales = sales.drop(columns=['period_7', 'period_14', 'period_28'])
    gc.collect()

    store_sku_pairs = get_store_sku_pairs_union(store_stock_agg)
    store_sku_pairs["magaza"] = store_sku_pairs["magaza"].astype(str)
    store_sku_pairs["stok_kodu"] = store_sku_pairs["stok_kodu"].astype(str)

    product["product_code"] = product["product_code"].astype(str)
    product["product_id"] = product["product_id"].astype(str)

    pm_active = product.copy()
    if "is_blocked" in pm_active.columns:
        pm_active = pm_active.loc[pm_active["is_blocked"].fillna(0).astype(int) == 0]

    store_product_pairs = (
        store_sku_pairs
        .merge(pm_active[["product_code", "product_id"]], left_on="stok_kodu", right_on="product_code", how="left")
        .drop(columns=["product_code"])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    store_product_pairs.rename({'magaza':'store_code'}, axis=1, inplace=True)

    ss_df = seasonal_safety_stock(
        sales, 
        store_col="store_code", 
        month_window=1,
        use_weighted_calculation=True,
        weight_strategy="exponential_decay",
        weight_decay_factor=0.95,
        recent_days_weight=0.6
    )
    
    weight_map = build_weight_map(sales, store_stock_agg, store_product_pairs)
    
    predictions = predictions[predictions["store_predicted_quantity"] >= 0]
    pred = predictions.copy()  
    pred["date"] = pd.to_datetime(pred["date"])
    pred.rename({'date':'sales_date'}, axis=1, inplace=True)
    pred["store_code"] = pred["store_code"].astype(str)
    pred["stok_kodu"] = pred["stok_kodu"].astype(str)
    pred["store_predicted_quantity"] = pd.to_numeric(pred["store_predicted_quantity"], errors="coerce").fillna(0)
    
    weight_map_subset = weight_map[["store_code", "stok_kodu", "product_id", "weight"]].copy()
    
    if len(pred) > 100000:
        chunk_size = 50000
        pred_chunks = []
        
        for i in range(0, len(pred), chunk_size):
            chunk = pred.iloc[i:i+chunk_size].copy()
            chunk_merged = chunk.merge(weight_map_subset, on=["store_code", "stok_kodu"], how="inner")
            chunk_merged["prediction_df"] = chunk_merged["store_predicted_quantity"] * chunk_merged["weight"]
            pred_chunks.append(chunk_merged[["store_code", "stok_kodu", "product_id", "sales_date", "prediction_df"]])
            
            del chunk, chunk_merged
            gc.collect()
        
        pred_pid = pd.concat(pred_chunks, ignore_index=True)
        del pred_chunks
    else:
        pred_pid = pred.merge(weight_map_subset, on=["store_code", "stok_kodu"], how="inner")
        pred_pid["prediction_df"] = pred_pid["store_predicted_quantity"] * pred_pid["weight"]
        pred_pid = pred_pid[["store_code", "stok_kodu", "product_id", "sales_date", "prediction_df"]]

    pred_pid['prediction_df'] = pred_pid['prediction_df'].fillna(0).astype(float)
    
    del pred, weight_map_subset
    gc.collect()

    ss_subset = ss_df[["product_id", "store_code", "safety_stock"]].copy()
    
    base_calc_table_0 = store_product_pairs.merge(
        ss_subset,
        on=["product_id", "store_code"],
        how="left"
    )
    base_calc_table_0["safety_stock"] = pd.to_numeric(base_calc_table_0["safety_stock"], errors="coerce").fillna(0)
    base_calc_table_0['store_code'] = base_calc_table_0['store_code'].astype(str)
    
    del ss_subset
    gc.collect()

    max_date = pd.to_datetime(sales['date'].max())
    end_date = max_date + timedelta(14)
    
    date_mask = (pred_pid['sales_date'] <= end_date) & (pred_pid['sales_date'] > max_date)
    prediction_daily = pred_pid[date_mask].groupby(
        ['store_code','stok_kodu','product_id']
    )['prediction_df'].sum().reset_index()
    prediction_daily['store_code'] = prediction_daily['store_code'].astype(str)

    base_calc_table_1 = base_calc_table_0.merge(
        prediction_daily[["product_id", "store_code", "prediction_df"]],
        on=["product_id", "store_code"],
        how="left"
    )
    base_calc_table_1['prediction_df'] = base_calc_table_1['prediction_df'].fillna(0).astype(float)
    
    del prediction_daily
    gc.collect()

    store_stock_agg.rename(columns={'magaza':'store_code'}, inplace=True)
    
    store_stock_subset = store_stock_agg[["product_id", "store_code", "stok_miktar"]].copy()
    store_stock_subset['store_code'] = store_stock_subset['store_code'].astype(str)
    store_stock_subset['product_id'] = store_stock_subset['product_id'].astype(str)
    
    base_calc_table_2 = base_calc_table_1.merge(
        store_stock_subset,
        on=["product_id", "store_code"],
        how="left"
    )
    base_calc_table_2['stok_miktar'] = base_calc_table_2['stok_miktar'].fillna(0).astype(int)
    
    del store_stock_subset
    gc.collect()

    base_calc_table_2['store_code'] = base_calc_table_2['store_code'].astype(str)
    base_calc_table_2['product_id'] = base_calc_table_2['product_id'].astype(str)
    
    sales_7['store_code'] = sales_7['store_code'].astype(str)
    sales_7['product_id'] = sales_7['product_id'].astype(str)
    
    sales_14['store_code'] = sales_14['store_code'].astype(str)
    sales_14['product_id'] = sales_14['product_id'].astype(str)
    
    sales_28['store_code'] = sales_28['store_code'].astype(str)
    sales_28['product_id'] = sales_28['product_id'].astype(str)
    
    base_calc_table_3 = (base_calc_table_2
        .merge(sales_7,  on=['store_code','product_id'], how='left')
        .merge(sales_14, on=['store_code','product_id'], how='left')
        .merge(sales_28, on=['store_code','product_id'], how='left')
    )

    for c in ['net_quantity_7','net_quantity_14','net_quantity_28']:
        base_calc_table_3[c] = base_calc_table_3[c].fillna(0).clip(lower=0).astype(int)

    base_calc_table_3['prediction_ma'] = (
        (base_calc_table_3['net_quantity_7']/7)*0.5 + 
        (base_calc_table_3['net_quantity_14']/14)*0.3 + 
        (base_calc_table_3['net_quantity_28']/28)*0.2
    )*14

    orders_subset = orders[['product_id','store_code', 'acik_siparis']].copy()
    
    orders_subset['store_code'] = orders_subset['store_code'].astype(str)
    orders_subset['product_id'] = orders_subset['product_id'].astype(str)
    
    base_calc_table_4 = (
        base_calc_table_3
        .merge(
            orders_subset, 
            on=['store_code','product_id'], 
            how='left'
        )
    )
    base_calc_table_4['acik_siparis'] = base_calc_table_4['acik_siparis'].fillna(0).astype(int)
    
    del orders_subset
    gc.collect()

    result = need_with_14day_coverage(base_calc_table_4, lead_time_days=2)
    
    distribution_result = distribute_warehouse_stock(
        replenishment_needs=result,
        warehouse_stock=warehouse_stock,
        store_col="store_code",
        product_col="product_id",
        need_col="ihtiyac_adet",
        warehouse_col="alt_yer",
        warehouse_qty_col="miktar"
    )
    
    return result, distribution_result


def get_store_sku_pairs_union(
    store_stock: pd.DataFrame,
    prediction: pd.DataFrame | None = None,
    stock_date_col="stok_tarihi",
    store_col="magaza",
    sku_col="stok_kodu",
    stock_qty_col="stok_miktar",
    lookback_stock_days=30,
    pred_date_col="date",
    pred_store_col="store_code",
    pred_sku_col="stok_kodu",
    pred_qty_col="store_predicted_quantity",
    lookahead_pred_days=30,
    min_pred=0.0
):
    ss = store_stock.copy()
    ss[stock_date_col] = pd.to_datetime(ss[stock_date_col])
    ss[store_col] = ss[store_col].astype(str)
    ss[sku_col] = ss[sku_col].astype(str)
    ss[stock_qty_col] = pd.to_numeric(ss[stock_qty_col], errors="coerce").fillna(0)

    max_dates = [ss[stock_date_col].max()]
    if prediction is not None:
        max_dates.append(pd.to_datetime(prediction[pred_date_col]).max())
    today = max([d for d in max_dates if pd.notna(d)])

    start_stock = today - pd.Timedelta(days=lookback_stock_days-1)
    
    date_mask = (ss[stock_date_col] >= start_stock) & (ss[stock_date_col] <= today)
    qty_mask = ss[stock_qty_col] > 0
    combined_mask = date_mask & qty_mask
    
    stock_pairs = (
        ss.loc[combined_mask, [store_col, sku_col]]
        .drop_duplicates()
    )

    if prediction is not None:
        pr = prediction.copy()
        pr[pred_date_col] = pd.to_datetime(pr[pred_date_col])
        pr[pred_store_col] = pr[pred_store_col].astype(str)
        pr[pred_sku_col] = pr[pred_sku_col].astype(str)
        pr[pred_qty_col] = pd.to_numeric(pr[pred_qty_col], errors="coerce").fillna(0)

        end_pred = today + pd.Timedelta(days=lookahead_pred_days-1)

        pred_date_mask = pr[pred_date_col] <= end_pred
        pred_qty_mask = pr[pred_qty_col] > min_pred
        pred_combined_mask = pred_date_mask & pred_qty_mask

        pred_pairs = (
            pr.loc[pred_combined_mask, [pred_store_col, pred_sku_col]]
            .drop_duplicates()
            .rename(columns={pred_store_col: store_col, pred_sku_col: sku_col})
        )
        
        pairs = pd.concat([stock_pairs, pred_pairs], ignore_index=True).drop_duplicates()
    else:
        pairs = stock_pairs

    result = pairs.sort_values([store_col, sku_col]).reset_index(drop=True)
    
    return result


def seasonal_safety_stock(
    sales: pd.DataFrame,
    date_col="date",
    store_col="store_code",          
    product_col="product_id",
    qty_col="net_quantity",
    lead_time_days=2,
    service_level=0.9,
    month_window=1,
    use_weighted_calculation=True,
    weight_strategy="exponential_decay",
    weight_decay_factor=0.95,
    recent_days_weight=0.7
):
    df = sales.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[product_col] = df[product_col].astype(str)
    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)
    df[store_col] = df[store_col].astype(str)

    today = df[date_col].max()
    current_month = today.month

    months_to_keep = [(current_month + i - 1) % 12 + 1 for i in range(-month_window, month_window + 1)]
    df_seasonal = df[df[date_col].dt.month.isin(months_to_keep)]

    if use_weighted_calculation:
        daily_demand = (
            df_seasonal.groupby([store_col, product_col, date_col], as_index=False)[qty_col]
                       .sum()
        )
        
        daily_demand['days_from_today'] = (today - daily_demand[date_col]).dt.days
        
        if weight_strategy == "exponential_decay":
            daily_demand['weight'] = weight_decay_factor ** daily_demand['days_from_today']
            
        elif weight_strategy == "linear_decay":
            max_days = daily_demand['days_from_today'].max()
            daily_demand['weight'] = 1 - (daily_demand['days_from_today'] / max_days)
            daily_demand['weight'] = daily_demand['weight'].clip(lower=0.1)
            
        elif weight_strategy == "recent_heavy":
            recent_days = 7
            daily_demand['weight'] = np.where(
                daily_demand['days_from_today'] <= recent_days,
                recent_days_weight,
                1 - recent_days_weight
            )
            
        else:
            daily_demand['weight'] = weight_decay_factor ** daily_demand['days_from_today']
        
        total_weight = daily_demand['weight'].sum()
        if total_weight > 0:
            daily_demand['weight'] = daily_demand['weight'] / total_weight
        
        weighted_stats = []
        
        for (store, product), group in daily_demand.groupby([store_col, product_col]):
            if len(group) == 0:
                continue
                
            weighted_avg = np.average(group[qty_col], weights=group['weight'])
            
            if len(group) > 1:
                weighted_variance = np.average(
                    (group[qty_col] - weighted_avg) ** 2, 
                    weights=group['weight']
                )
                weighted_std = np.sqrt(weighted_variance)
            else:
                weighted_std = 0.0
            
            weighted_stats.append({
                store_col: store,
                product_col: product,
                'daily_avg': weighted_avg,
                'daily_std': weighted_std
            })
        
        demand_stats = pd.DataFrame(weighted_stats)
        
    else:
        daily_demand = (
            df_seasonal.groupby([store_col, product_col, date_col], as_index=False)[qty_col]
                       .sum()
        )

        demand_stats = (
            daily_demand.groupby([store_col, product_col])
            .agg(
                daily_avg=(qty_col, "mean"),
                daily_std=(qty_col, "std")
            )
            .fillna(0)
            .reset_index()
        )

    z_value = norm.ppf(service_level)

    demand_stats["safety_stock"] = (
        z_value * demand_stats["daily_std"] * np.sqrt(lead_time_days)
    ).round(2)

    return demand_stats


def compare_weighting_strategies(sales_data, store_code="store_code", product_id="product_id"):
    strategies = {
        "no_weight": {"use_weighted_calculation": False},
        "exponential_decay": {
            "use_weighted_calculation": True,
            "weight_strategy": "exponential_decay",
            "weight_decay_factor": 0.95
        },
        "linear_decay": {
            "use_weighted_calculation": True,
            "weight_strategy": "linear_decay"
        },
        "recent_heavy": {
            "use_weighted_calculation": True,
            "weight_strategy": "recent_heavy",
            "recent_days_weight": 0.6
        }
    }
    
    results = {}
    
    for strategy_name, params in strategies.items():
        logging.info(f"Testing strategy: {strategy_name}")
        try:
            result = seasonal_safety_stock(sales_data, **params)
            results[strategy_name] = result
            logging.info(f"Strategy {strategy_name}: {len(result)} records calculated")
        except Exception as e:
            logging.error(f"Error in strategy {strategy_name}: {e}")
            results[strategy_name] = None
    
    return results


def build_weight_map(sales, store_stock_agg, store_product_pairs):
    LOOKBACK_SALES = 90
    LOOKBACK_STOCK = 60

    end_date = pd.to_datetime(sales["date"]).max()

    sales = sales.copy()
    sales["date"] = pd.to_datetime(sales["date"])
    for c in ["store_code", "stok_kodu", "product_id"]:
        sales[c] = sales[c].astype(str)

    pairs = store_product_pairs.rename(columns={"magaza": "store_code"}).copy()
    for c in ["store_code", "stok_kodu", "product_id"]:
        pairs[c] = pairs[c].astype(str)

    stock = store_stock_agg.copy()
    stock["stok_tarihi"] = pd.to_datetime(stock["stok_tarihi"])
    for c in ["magaza", "stok_kodu", "product_id"]:
        stock[c] = stock[c].astype(str)

    start_sales = end_date - pd.Timedelta(days=LOOKBACK_SALES - 1)
    s_win = sales[(sales["date"] >= start_sales) & (sales["date"] <= end_date)]
    s_win = s_win[s_win["net_quantity"] > 0]

    s_win = s_win.merge(pairs[["store_code", "stok_kodu", "product_id"]],
                        on=["store_code", "stok_kodu", "product_id"], how="inner")

    sales_agg = (
        s_win.groupby(["store_code", "stok_kodu", "product_id"], as_index=False)["net_quantity"]
             .sum()
             .rename(columns={"net_quantity": "sales_90d"})
    )
    sales_agg["sales_90d"] = sales_agg["sales_90d"].clip(lower=0)

    sales_agg["weight_sales"] = (
        sales_agg["sales_90d"] /
        sales_agg.groupby(["store_code", "stok_kodu"])["sales_90d"].transform("sum")
    )

    start_stock = end_date - pd.Timedelta(days=LOOKBACK_STOCK - 1)
    st_win = stock[(stock["stok_tarihi"] >= start_stock) & (stock["stok_tarihi"] <= end_date)]
    st_win = st_win[st_win["stok_miktar"] > 0]

    st_win = st_win.rename(columns={"magaza": "store_code"})

    st_win = st_win.merge(pairs[["store_code", "stok_kodu", "product_id"]],
                          on=["store_code", "stok_kodu", "product_id"], how="inner")

    stock_agg = (
        st_win.groupby(["store_code", "stok_kodu", "product_id"], as_index=False)["stok_miktar"]
             .sum()
             .rename(columns={"stok_miktar": "onhand_30d"})
    )

    stock_agg["weight_onhand"] = (
        stock_agg["onhand_30d"] /
        stock_agg.groupby(["store_code", "stok_kodu"])["onhand_30d"].transform("sum")
    )

    children = pairs.copy()

    weights = (children
        .merge(sales_agg[["store_code","stok_kodu","product_id","weight_sales"]], 
               on=["store_code","stok_kodu","product_id"], how="left")
        .merge(stock_agg[["store_code","stok_kodu","product_id","weight_onhand"]],
               on=["store_code","stok_kodu","product_id"], how="left")
    )

    weights["weight"] = weights["weight_sales"].where(
        weights["weight_sales"].notna(), weights["weight_onhand"]
    )

    weights["n_option"] = weights.groupby(["store_code","stok_kodu"])["product_id"].transform("count")
    grp_has_any = weights.groupby(["store_code","stok_kodu"])["weight"].transform(lambda s: s.notna().any())
    weights.loc[~grp_has_any, "weight"] = 1.0 / weights["n_option"]

    sum_w = weights.groupby(["store_code","stok_kodu"])["weight"].transform("sum")
    zero_sum = (sum_w == 0)
    weights.loc[zero_sum, "weight"] = 1.0 / weights.loc[zero_sum, "n_option"]
    sum_w = weights.groupby(["store_code","stok_kodu"])["weight"].transform("sum")
    weights["weight"] = weights["weight"] / sum_w

    return weights[["store_code","stok_kodu","product_id","weight"]].copy()


def need_with_14day_coverage(
    df: pd.DataFrame,
    pred_df_col="prediction_df",
    pred_ma_col="prediction_ma",
    ss_col="safety_stock",
    onhand_col="stok_miktar",
    last14_col="net_quantity_14",
    last28_col="net_quantity_28",
    lead_time_days=2,
    in_transit_col='acik_siparis',
    w_min=0.0, w_max=0.20
) -> pd.DataFrame:
    out = df.copy()

    for c in [pred_df_col, pred_ma_col, ss_col, onhand_col, last14_col, last28_col]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        else:
            out[c] = 0.0
    if in_transit_col and in_transit_col in out.columns:
        out[in_transit_col] = pd.to_numeric(out[in_transit_col], errors="coerce").fillna(0.0)
    else:
        in_transit_col = None

    avg14_from_28 = out[last28_col] / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        momentum = out[last14_col] / avg14_from_28.replace(0, np.nan)
    momentum = momentum.fillna(1.0).clip(0.0, 2.0)

    w_model = (0.1 + 0.1 * (momentum - 1.0)).clip(w_min, w_max)
    w_model = np.where((out[pred_ma_col] <= 0) & (out[pred_df_col] > 0), 1.0, w_model)
    w_model = np.where((out[pred_df_col] <= 0) & (out[pred_ma_col] > 0) & (out[last28_col] > 0), 0.0, w_model)
    # Son 7 gün satışı 4'ten küçükse sadece MA'dan hesapla
    w_model = np.where(out['net_quantity_7'] < 4, 0.0, w_model)
    out["w_model"] = w_model.astype(float)

    out["pred_14_blended"] = out["w_model"] * out[pred_df_col] + (1.0 - out["w_model"]) * out[pred_ma_col]

    out["pred_LT"] = out["pred_14_blended"] * (lead_time_days / 14.0)

    out["OUL_14"] = out[ss_col] + out["pred_14_blended"]
    expected_onhand_at_arrival = out[onhand_col] - out["pred_LT"]
    if in_transit_col:
        expected_onhand_at_arrival = expected_onhand_at_arrival + out[in_transit_col]
    out["ihtiyac_adet"] = np.round(np.maximum(0.0, out["OUL_14"] - expected_onhand_at_arrival)).astype(int)
    
    # Son 28 günde hiç satış yoksa ihtiyaç adetini sıfır yap
    out["ihtiyac_adet"] = np.where(out[last28_col] == 0, 0, out["ihtiyac_adet"])
    out["prediction_df"] = np.where(out["prediction_df"] > out[last14_col], 0, out["prediction_df"])
    out["safety_stock"] = np.where(out["safety_stock"] > 1.15*out["prediction_ma"], out["safety_stock"]*0.8, out["safety_stock"])
    out.drop(columns=['OUL_14'],axis=1,inplace=True)

    return out


def distribute_warehouse_stock(
    replenishment_needs: pd.DataFrame,
    warehouse_stock: pd.DataFrame,
    store_col="store_code",
    product_col="product_id",
    need_col="ihtiyac_adet",
    warehouse_col="alt_yer",
    warehouse_qty_col="miktar"
) -> pd.DataFrame:
    needs = replenishment_needs.copy()
    warehouse = warehouse_stock.copy()
    
    needs[store_col] = needs[store_col].astype(str)
    needs[product_col] = needs[product_col].astype(str)
    warehouse[warehouse_col] = warehouse[warehouse_col].astype(str)
    warehouse[product_col] = warehouse[product_col].astype(str)
    
    needs = needs[needs[need_col] > 0].copy()
    
    if needs.empty:
        return pd.DataFrame(columns=[
            "gonderen_depo", "alan_magaza", "urun", "baslangic_depo_stok", 
            "ihtiyac_adet", "karsilanan_adet", "kalan_ihtiyac", "depo_sevk_toplam", "depo_kalan_nihai"
        ])
    
    products_with_needs = needs[product_col].unique()
    
    warehouse_by_product = {}
    for product in products_with_needs:
        product_warehouse = warehouse[
            (warehouse[product_col] == product) & 
            (warehouse[warehouse_qty_col] > 0)
        ].copy()
        warehouse_by_product[product] = product_warehouse
    
    distribution_results = []
    
    for product in products_with_needs:
        product_needs = needs[needs[product_col] == product].copy()
        product_needs = product_needs.sort_values(need_col, ascending=False)
        
        product_warehouse = warehouse_by_product[product]
        
        if product_warehouse.empty:
            for _, need_row in product_needs.iterrows():
                distribution_results.append({
                    "gonderen_depo": "STOK_YOK",
                    "alan_magaza": need_row[store_col],
                    "urun": product,
                    "baslangic_depo_stok": 0,
                    "ihtiyac_adet": need_row[need_col],
                    "karsilanan_adet": 0,
                    "kalan_ihtiyac": need_row[need_col],
                    "depo_sevk_toplam": 0,
                    "depo_kalan_nihai": 0
                })
            continue
        
        initial_warehouse_stocks = {}
        for _, warehouse_row in product_warehouse.iterrows():
            warehouse_code = warehouse_row[warehouse_col]
            warehouse_qty = warehouse_row[warehouse_qty_col]
            initial_warehouse_stocks[warehouse_code] = warehouse_qty
        
        remaining_needs = product_needs.copy()
        remaining_warehouse = product_warehouse.copy()
        
        store_warehouse_distributions = {}
        
        while not remaining_needs.empty and not remaining_warehouse.empty:
            current_need = remaining_needs.iloc[0]
            current_warehouse = remaining_warehouse.iloc[0]
            
            need_store = current_need[store_col]
            need_qty = current_need[need_col]
            warehouse_code = current_warehouse[warehouse_col]
            warehouse_qty = current_warehouse[warehouse_qty_col]
            
            key = (need_store, warehouse_code)
            if key not in store_warehouse_distributions:
                store_warehouse_distributions[key] = {
                    'initial_need': need_qty,
                    'distributed': 0,
                    'initial_warehouse_stock': initial_warehouse_stocks[warehouse_code]
                }
            
            distributed_qty = 1
            remaining_qty = need_qty - distributed_qty
            warehouse_remaining = warehouse_qty - distributed_qty
            
            store_warehouse_distributions[key]['distributed'] += distributed_qty
            
            if remaining_qty <= 0:
                remaining_needs = remaining_needs.iloc[1:].reset_index(drop=True)
            else:
                remaining_needs.iloc[0, remaining_needs.columns.get_loc(need_col)] = remaining_qty
                temp_need = remaining_needs.iloc[0:1]
                remaining_needs = pd.concat([
                    remaining_needs.iloc[1:], 
                    temp_need
                ], ignore_index=True)
            
            if warehouse_remaining <= 0:
                remaining_warehouse = remaining_warehouse.iloc[1:].reset_index(drop=True)
            else:
                remaining_warehouse.iloc[0, remaining_warehouse.columns.get_loc(warehouse_qty_col)] = warehouse_remaining
        
        warehouse_total_shipments = {}
        for (store, warehouse_code), dist_info in store_warehouse_distributions.items():
            if warehouse_code not in warehouse_total_shipments:
                warehouse_total_shipments[warehouse_code] = 0
            warehouse_total_shipments[warehouse_code] += dist_info['distributed']
        
        for (store, warehouse_code), dist_info in store_warehouse_distributions.items():
            total_shipped_from_warehouse = warehouse_total_shipments[warehouse_code]
            distribution_results.append({
                "gonderen_depo": warehouse_code,
                "alan_magaza": store,
                "urun": product,
                "baslangic_depo_stok": dist_info['initial_warehouse_stock'],
                "ihtiyac_adet": dist_info['initial_need'],
                "karsilanan_adet": dist_info['distributed'],
                "kalan_ihtiyac": dist_info['initial_need'] - dist_info['distributed'],
                "depo_sevk_toplam": total_shipped_from_warehouse,
                "depo_kalan_nihai": dist_info['initial_warehouse_stock'] - total_shipped_from_warehouse
            })
    
    if distribution_results:
        result_df = pd.DataFrame(distribution_results)
    else:
        result_df = pd.DataFrame(columns=[
            "gonderen_depo", "alan_magaza", "urun", "baslangic_depo_stok", 
            "ihtiyac_adet", "karsilanan_adet", "kalan_ihtiyac", "depo_sevk_toplam", "depo_kalan_nihai"
        ])
    
    return result_df


def run_auto_replenishment(**context):
    gcs = GCSHook(gcp_conn_id="google_cloud_default")

    # Database connection ID
    db_conn_id = "sporthink_mysql"

    # Download daily store predictions from GCS (this is output from weekly_to_daily step)
    pred_local = "/tmp/daily_store_pred.parquet"
    gcs.download(bucket_name=BUCKET, object_name=DAILY_STORE_PRED_OBJ, filename=pred_local)
    logging.info("✅ Downloaded predictions: gs://%s/%s", BUCKET, DAILY_STORE_PRED_OBJ)
    predictions = pd.read_parquet(pred_local, engine=ENGINE)

    # SQL queries - optimized for auto replenishment step
    lookback_days = get_optimal_lookback_period()
    
    # Sales data - only recent data needed for replenishment
    sales_query = f"""
        SELECT 
            date, store_code, product_id, net_quantity
        FROM Genboost.history_sales
        WHERE `date` >= DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL {lookback_days} DAY), '%Y-%m-01')
        AND net_quantity >= 0
    """
    
    # Store stock - only recent data
    store_stock_query = f"""
        SELECT 
            stok_tarihi, magaza, stok_kodu, product_id, SUM(toplam_miktar) as toplam_miktar
        FROM Genboost.realtime_store_stock
        WHERE stok_tarihi >= DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 30 DAY), '%Y-%m-01')
        AND toplam_miktar > 0
        GROUP BY stok_tarihi, magaza, stok_kodu, product_id
    """
    
    # Warehouse stock - only current stock
    warehouse_stock_query = """
        SELECT 
            alt_yer, product_id, miktar
        FROM Genboost.warehouse_stock
        WHERE miktar > 0
    """
    
    # Open orders - only current orders
    orders_query = """
        SELECT 
            urun_kodu as product_id,
            LEFT(giris_depo_kodu,4) as depo_kodu,
            SUM(toplam_yoldaki_miktar) as toplam_yoldaki_miktar,
            SUM(toplanmayan_urun) as toplanmayan_urun
        FROM Genboost.realtime_stock_movement
        WHERE (toplam_yoldaki_miktar > 0 OR toplanmayan_urun > 0)
        GROUP BY urun_kodu, LEFT(giris_depo_kodu,4)
    """
    
    # Product data - only needed columns
    product_query = """
        SELECT 
            product_id, product_code, product_att_01, product_att_02, product_att_03, is_blocked
        FROM Genboost.dim_product
    """

    sales = _fetch_from_db(db_conn_id, sales_query)
    
    if "stok_kodu" not in sales.columns:
        sales["stok_kodu"] = sales["product_id"]
    
    store_stock = _fetch_from_db(db_conn_id, store_stock_query)
    warehouse_stock = _fetch_from_db(db_conn_id, warehouse_stock_query)
    orders = _fetch_from_db(db_conn_id, orders_query)
    product = _fetch_from_db(db_conn_id, product_query)

    result, distribution_result = process_auto_replenishment(product, store_stock, warehouse_stock, predictions, sales, orders)

    # Save replenishment results as parquet
    out_local = "/tmp/replenishment_results.parquet"
    result.to_parquet(out_local, index=False, engine=ENGINE, compression="snappy")
    out_obj = f"{OUTPUT_PREFIX}/replenishment_results.parquet"
    gcs.upload(bucket_name=BUCKET, object_name=out_obj, filename=out_local)

    # Save replenishment results as CSV
    out_csv_local = "/tmp/replenishment_results.csv"
    result.to_csv(out_csv_local, index=False)
    out_csv_obj = f"{OUTPUT_PREFIX}/replenishment_results.csv"
    gcs.upload(bucket_name=BUCKET, object_name=out_csv_obj, filename=out_csv_local)

    # Save warehouse distribution as parquet
    dist_out_local = "/tmp/warehouse_distribution.parquet"
    distribution_result.to_parquet(dist_out_local, index=False, engine=ENGINE, compression="snappy")
    dist_out_obj = f"{OUTPUT_PREFIX}/warehouse_distribution.parquet"
    gcs.upload(bucket_name=BUCKET, object_name=dist_out_obj, filename=dist_out_local)

    # Save warehouse distribution as CSV
    dist_csv_local = "/tmp/warehouse_distribution.csv"
    distribution_result.to_csv(dist_csv_local, index=False)
    dist_csv_obj = f"{OUTPUT_PREFIX}/warehouse_distribution.csv"
    gcs.upload(bucket_name=BUCKET, object_name=dist_csv_obj, filename=dist_csv_local)


if __name__ == "__main__":
    run_auto_replenishment()
