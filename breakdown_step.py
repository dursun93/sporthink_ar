import pandas as pd
from datetime import timedelta

product = pd.read_csv('data/dim_product.csv')
calendar = pd.read_csv('data/dim_calendar_pivot.csv')
sales = pd.read_csv('data/history_sales.csv')
cluster = pd.read_csv('k_means_store.csv')

future_forecast = pd.read_csv(
    "output/future_12_weeks_hierarchical_forecast.csv",
    sep=";",           
    decimal=",",       
    thousands="."      
)


sales['date'] = pd.to_datetime(sales['date'])

last_date =  sales['date'].max()

# Bu yıl: son 4 hafta
cutoff_this_year = last_date - timedelta(weeks=8)
sales = sales[sales['date'] >= cutoff_this_year]




calendar['date'] = pd.to_datetime(calendar['date'])
calendar['week_start_date'] = calendar['date'].dt.to_period('W').apply(lambda r: r.start_time)

sales.loc[sales['net_quantity'] < 0, 'net_quantity'] = 0

sales = sales.merge(calendar[['date', 'week_start_date']], on='date', how='left')

product=product[['product_code','product_att_01', 'product_att_02', 'product_att_05']].drop_duplicates() #
sales = sales.merge(product, left_on="stok_kodu",
    right_on="product_code", how='left')


sales['product_att_02'] = sales['product_att_02'].apply(
    lambda x: int(str(x).replace('D', '9').replace('N', '9')) if isinstance(x, str) and ('D' in x or 'N' in x) else x
)

sales['product_att_02'] = pd.to_numeric(sales['product_att_02'], errors='coerce')


sales = sales.merge(cluster, on='store_code', how='left')

sales = sales[
    (~(sales['product_att_01'] == 9.0)) &
    (~(sales['product_att_02'] == 9.0))
]


store_ratio_df = (
    sales.groupby(['store_cluster', 'product_att_01', 'product_att_02', 'product_att_05', 'store_code','stok_kodu'], as_index=False)['net_quantity'] #
    .sum()
    .rename(columns={'net_quantity': 'net_quantity'})
)


store_ratio_df['cluster_total'] = store_ratio_df.groupby(['store_cluster', 'product_att_01', 'product_att_02', 'product_att_05'])['net_quantity'].transform('sum') #, 'product_att_05'
store_ratio_df['store_ratio'] = store_ratio_df['net_quantity'] / store_ratio_df['cluster_total']




future_forecast['week_start_date'] = pd.to_datetime(future_forecast['week_start_date'])


merge_keys = ['store_cluster', 'product_att_01', 'product_att_02' , 'product_att_05']  #, 'product_att_05'
for col in merge_keys:
    future_forecast[col] = future_forecast[col].astype(int)
    store_ratio_df[col] = store_ratio_df[col].astype(int)

store_ratio_df['store_code'] = store_ratio_df['store_code'].astype(str)

forecast_store = future_forecast.merge(store_ratio_df, on=merge_keys, how='inner')
forecast_store['store_predicted_quantity'] = forecast_store['predicted_quantity'] * forecast_store['store_ratio']

final_forecast = forecast_store[[
    'week_start_date', 'store_cluster', 'store_code',
    'product_att_01', 'product_att_02', 'product_att_05','stok_kodu',
    'store_predicted_quantity','predicted_quantity','store_ratio'
]]


final_forecast['store_predicted_quantity'] = final_forecast['store_predicted_quantity'].fillna(0).astype(float)
final_forecast["store_predicted_quantity"] = final_forecast["store_predicted_quantity"].astype(float)
#final_forecast.to_csv("output/breakdown_store.csv", sep=";", index=False, encoding="utf-8-sig")  # 6 ondalık

final_forecast.to_csv(
    "output/breakdown_store.csv",
    index=False,
    sep=";",             
    encoding="utf-8-sig",
    float_format="%.6f"  
)

