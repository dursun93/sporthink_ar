import pandas as pd
from datetime import datetime,timedelta
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

product = pd.read_csv('data/dim_product.csv',index_col=None)
store_stock =  pd.read_csv('data/store_stock.csv',index_col=None)
warehouse_stock =  pd.read_csv('data/warehouse_stock.csv',index_col=None)
predictions = pd.read_parquet('daily_store_pred.parquet')
sales =  pd.read_csv('data/history_sales.csv',index_col=None)
orders = pd.read_csv('data/open_order.csv',index_col=None)


sales['date'] = pd.to_datetime(sales['date'])

start_date = sales.date.max()


orders['acik_siparis'] = orders['toplam_yoldaki_miktar'] + orders['toplanmayan_urun']
orders.rename(columns={'depo_kodu' : 'store_code'},inplace=True)
predictions.rename(columns={'prediction_store_daily':'store_predicted_quantity'},inplace=True)

product_segments_cols = [
    'product_id',
    'product_att_01', 'product_att_02', 'product_att_03'
]
sales_with_segment = sales.merge(product[product_segments_cols], on='product_id', how='left')

sales_with_segment['product_att_02'] = sales_with_segment['product_att_02'].apply(
    lambda x: int(str(x).replace('D', '9').replace('N', '9')) if isinstance(x, str) and ('D' in x or 'N' in x) else x
)

sales_with_segment['product_att_02'] = pd.to_numeric(sales_with_segment['product_att_02'], errors='coerce')


# Filtrele: 9.0'lar hariç
filtered_sales = sales_with_segment[
    (~(sales_with_segment['product_att_01'] == 9.0)) &
    (~(sales_with_segment['product_att_02'] == 9.0))
]
filtered_sales = sales_with_segment[
    (~(sales_with_segment['product_att_01'] == 999)) &
    (~(sales_with_segment['product_att_02'] == 999))
]

filtered_sales = filtered_sales[filtered_sales["net_quantity"] >= 0]
filtered_sales['date'] = pd.to_datetime(filtered_sales['date'])
sales = filtered_sales


store_stock['product_id'] = store_stock['stok_kodu'] + store_stock['color_code'] + store_stock['size'] 
store_stock_agg = store_stock.groupby(
    ['stok_tarihi','magaza','stok_kodu','product_id']
).agg({'toplam_miktar':'sum'}).reset_index()
store_stock_agg.rename(columns={'toplam_miktar': 'stok_miktar'}, inplace=True)


store_stock_agg['stok_miktar'] = store_stock_agg['stok_miktar'].astype(int)



max_date = pd.to_datetime(sales['date'].max())
start_7 = max_date - pd.Timedelta(days=6)  

mask = (sales['date'] >= start_7) & (sales['date'] <= max_date)
sales_agg_7 = sales.loc[mask]

sales_7 = (sales_agg_7
           .groupby(['product_id','store_code'], as_index=False)['net_quantity']
           .sum()
           .rename(columns={'net_quantity':'net_quantity_7'}))

start_14 = max_date - pd.Timedelta(days=13)  

mask = (sales['date'] >= start_14) & (sales['date'] <= max_date)
sales_agg_14 = sales.loc[mask]

sales_14 = (sales_agg_14
           .groupby(['product_id','store_code'], as_index=False)['net_quantity']
           .sum()
           .rename(columns={'net_quantity':'net_quantity_14'}))

start_28 = max_date - pd.Timedelta(days=27)  

mask = (sales['date'] >= start_28) & (sales['date'] <= max_date)
sales_agg_28 = sales.loc[mask]

sales_28 = (sales_agg_28
           .groupby(['product_id','store_code'], as_index=False)['net_quantity']
           .sum()
           .rename(columns={'net_quantity':'net_quantity_28'}))


def get_store_sku_pairs_union(
    store_stock: pd.DataFrame,
    prediction: pd.DataFrame | None = None,
    # stok tarafı
    stock_date_col="stok_tarihi",
    store_col="magaza",
    sku_col="stok_kodu",
    stock_qty_col="stok_miktar",
    lookback_stock_days=30,
    # prediction tarafı 
    pred_date_col="date",
    pred_store_col="store_code",
    pred_sku_col="stok_kodu",
    pred_qty_col="store_predicted_quantity",
    lookahead_pred_days=30,
    min_pred=0.0
):

    ss = store_stock_agg.copy()
    ss[stock_date_col] = pd.to_datetime(ss[stock_date_col])
    ss[store_col] = ss[store_col].astype(str)
    ss[sku_col] = ss[sku_col].astype(str)
    ss[stock_qty_col] = pd.to_numeric(ss[stock_qty_col], errors="coerce").fillna(0)


    max_dates = [ss[stock_date_col].max()]
    if prediction is not None:
        max_dates.append(pd.to_datetime(prediction[pred_date_col]).max())
    today = max([d for d in max_dates if pd.notna(d)])

    # son 30 günde pozitif stok
    start_stock = today - pd.Timedelta(days=lookback_stock_days-1)
    stock_pairs = (
        ss[(ss[stock_date_col] >= start_stock) & (ss[stock_date_col] <= today) & (ss[stock_qty_col] > 0)]
        [[store_col, sku_col]]
        .drop_duplicates()
    )

    # prediction’dan gelen ikililer 
    if prediction is not None:
        pr = prediction.copy()
        pr[pred_date_col] = pd.to_datetime(pr[pred_date_col])
        pr[pred_store_col] = pr[pred_store_col].astype(str)
        pr[pred_sku_col] = pr[pred_sku_col].astype(str)
        pr[pred_qty_col] = pd.to_numeric(pr[pred_qty_col], errors="coerce").fillna(0)


        end_pred   = today + pd.Timedelta(days=lookahead_pred_days-1)

        pred_pairs = (
            pr[(pr[pred_date_col] <= end_pred) & (pr[pred_qty_col] > min_pred)]
            [[pred_store_col, pred_sku_col]]
            .drop_duplicates()
            .rename(columns={pred_store_col: store_col, pred_sku_col: sku_col})
        )
        pairs = pd.concat([stock_pairs, pred_pairs], ignore_index=True).drop_duplicates()
    else:
        pairs = stock_pairs

    return pairs.sort_values([store_col, sku_col]).reset_index(drop=True)
store_sku_pairs = get_store_sku_pairs_union(store_stock)

store_sku_pairs["magaza"]  = store_sku_pairs["magaza"].astype(str)
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
import pandas as pd
import numpy as np
from scipy.stats import norm

def seasonal_safety_stock(
    sales: pd.DataFrame,
    date_col="date",
    store_col="store_code",          
    product_col="product_id",
    qty_col="net_quantity",
    lead_time_days=2,
    service_level=0.9,
    month_window=1                   
):
    df = sales.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[product_col] = df[product_col].astype(str)
    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)
    df[store_col] = df[store_col].astype(str)   # <-- mağaza tipi

    today = df[date_col].max()
    current_month = today.month

    # Mevsimsel filtre
    months_to_keep = [(current_month + i - 1) % 12 + 1 for i in range(-month_window, month_window + 1)]
    df_seasonal = df[df[date_col].dt.month.isin(months_to_keep)]

    # Günlük talep (STORE + PRODUCT + DATE)
    daily_demand = (
        df_seasonal.groupby([store_col, product_col, date_col], as_index=False)[qty_col]
                   .sum()
    )

    # Standart sapma ve ortalama (STORE + PRODUCT)
    demand_stats = (
        daily_demand.groupby([store_col, product_col])
        .agg(
            daily_avg=(qty_col, "mean"),
            daily_std=(qty_col, "std")
        )
        .fillna(0)
        .reset_index()
    )

    # Z değeri
    z_value = norm.ppf(service_level)

    # Safety stock
    demand_stats["safety_stock"] = (
        z_value * demand_stats["daily_std"] * np.sqrt(lead_time_days)
    ).round(2)

    return demand_stats

ss_df = seasonal_safety_stock(sales, store_col="store_code", month_window=1)
LOOKBACK_SALES = 90
LOOKBACK_STOCK = 60

# prediction varsa ondan, yoksa sales'ten "bugün"
end_date = pd.to_datetime(sales["date"]).max()

# Tipler
sales = sales.copy()
sales["date"] = pd.to_datetime(sales["date"])
for c in ["store_code", "stok_kodu", "product_id"]:
    sales[c] = sales[c].astype(str)

pairs = store_product_pairs.rename(columns={"magaza": "store_code"}).copy()
for c in ["store_code", "stok_kodu", "product_id"]:
    pairs[c] = pairs[c].astype(str)

stock = store_stock.copy()
stock["stok_tarihi"] = pd.to_datetime(stock["stok_tarihi"])
for c in ["magaza", "stok_kodu", "product_id"]:
    stock[c] = stock[c].astype(str)

start_sales = end_date - pd.Timedelta(days=LOOKBACK_SALES - 1)
s_win = sales[(sales["date"] >= start_sales) & (sales["date"] <= end_date)]
s_win = s_win[s_win["net_quantity"] > 0]

# Sadece mevcut pair + product_id 
s_win = s_win.merge(pairs[["store_code", "stok_kodu", "product_id"]],
                    on=["store_code", "stok_kodu", "product_id"], how="inner")

sales_agg = (
    s_win.groupby(["store_code", "stok_kodu", "product_id"], as_index=False)["net_quantity"]
         .sum()
         .rename(columns={"net_quantity": "sales_90d"})
)

sales_agg["weight_sales"] = (
    sales_agg["sales_90d"] /
    sales_agg.groupby(["store_code", "stok_kodu"])["sales_90d"].transform("sum")
)

start_stock = end_date - pd.Timedelta(days=LOOKBACK_STOCK - 1)
st_win = stock[(stock["stok_tarihi"] >= start_stock) & (stock["stok_tarihi"] <= end_date)]
st_win = st_win[st_win["toplam_miktar"] > 0]

# store_code isim eşitlemesi
st_win = st_win.rename(columns={"magaza": "store_code"})

st_win = st_win.merge(pairs[["store_code", "stok_kodu", "product_id"]],
                      on=["store_code", "stok_kodu", "product_id"], how="inner")

stock_agg = (
    st_win.groupby(["store_code", "stok_kodu", "product_id"], as_index=False)["toplam_miktar"]
         .sum()
         .rename(columns={"toplam_miktar": "onhand_30d"})
)

stock_agg["weight_onhand"] = (
    stock_agg["onhand_30d"] /
    stock_agg.groupby(["store_code", "stok_kodu"])["onhand_30d"].transform("sum")
)
# (pair + product_id)
children = pairs.copy()  # store_code, stok_kodu, product_id

weights = (children
    .merge(sales_agg[["store_code","stok_kodu","product_id","weight_sales"]], 
           on=["store_code","stok_kodu","product_id"], how="left")
    .merge(stock_agg[["store_code","stok_kodu","product_id","weight_onhand"]],
           on=["store_code","stok_kodu","product_id"], how="left")
)

# Öncelik: sales -> onhand
weights["weight"] = weights["weight_sales"].where(
    weights["weight_sales"].notna(), weights["weight_onhand"]
)

# Tamamı NaN olan grupları eşit böl
weights["n_option"] = weights.groupby(["store_code","stok_kodu"])["product_id"].transform("count")
grp_has_any = weights.groupby(["store_code","stok_kodu"])["weight"].transform(lambda s: s.notna().any())
weights.loc[~grp_has_any, "weight"] = 1.0 / weights["n_option"]

# Normalizasyon (toplam = 1)
sum_w = weights.groupby(["store_code","stok_kodu"])["weight"].transform("sum")
# Eğer nadiren sum_w=0 kalırsa, eşit böl:
zero_sum = (sum_w == 0)
weights.loc[zero_sum, "weight"] = 1.0 / weights.loc[zero_sum, "n_option"]
sum_w = weights.groupby(["store_code","stok_kodu"])["weight"].transform("sum")
weights["weight"] = weights["weight"] / sum_w

weight_map = weights[["store_code","stok_kodu","product_id","weight"]].copy()
predictions = predictions[predictions["store_predicted_quantity"] >= 0]
pred = predictions.copy()  
pred["date"] = pd.to_datetime(pred["date"])
pred.rename({'date':'sales_date'},axis=1, inplace=True)
pred["store_code"] = pred["store_code"].astype(str)
pred["stok_kodu"] = pred["stok_kodu"].astype(str)
pred["store_predicted_quantity"] = pd.to_numeric(pred["store_predicted_quantity"], errors="coerce").fillna(0)
pred_pid = (
    pred.merge(weight_map, on=["store_code", "stok_kodu"], how="inner")
)
pred_pid["prediction_df"] = pred_pid["store_predicted_quantity"] * pred_pid["weight"]

pred_pid = pred_pid[["store_code", "stok_kodu", "product_id", "sales_date", "prediction_df"]]
pred_pid['prediction_df'] = pred_pid['prediction_df'].fillna(0).astype(float)

base_calc_table_0 = store_product_pairs.merge(
    ss_df[["product_id", "store_code", "safety_stock"]],
    on=["product_id", "store_code"],
    how="left"

)
base_calc_table_0["safety_stock"] = pd.to_numeric(base_calc_table_0["safety_stock"], errors="coerce").fillna(0)
base_calc_table_0['store_code']=base_calc_table_0['store_code'].astype(int)

max_date = pd.to_datetime(filtered_sales['date'].max())
end_date = max_date + timedelta(14)
prediction_daily = pred_pid[(pred_pid['sales_date']<= end_date)&(pred_pid['sales_date'] > max_date)]
prediction_daily = prediction_daily.groupby(
    ['store_code','stok_kodu','product_id']
)['prediction_df'].sum().reset_index()
prediction_daily['store_code'] = prediction_daily['store_code'].astype(int)

base_calc_table_1 = base_calc_table_0.merge(
    prediction_daily[["product_id", "store_code", "prediction_df"]],
    on=["product_id", "store_code"],
    how="left"

)
base_calc_table_1['prediction_df'] = base_calc_table_1['prediction_df'].fillna(0).astype(float)

store_stock_agg.rename(columns={'magaza':'store_code'}, inplace=True)

base_calc_table_2 = base_calc_table_1.merge(
    store_stock_agg[["product_id", "store_code", "stok_miktar"]],
    on=["product_id", "store_code"],
    how="left"

)
base_calc_table_2['stok_miktar'] = base_calc_table_2['stok_miktar'].fillna(0).astype(int)

base_calc_table_3 = (base_calc_table_2
    .merge(sales_7,  on=['store_code','product_id'], how='left')
    .merge(sales_14, on=['store_code','product_id'], how='left')
    .merge(sales_28, on=['store_code','product_id'], how='left')
)

for c in ['net_quantity_7','net_quantity_14','net_quantity_28']:
    base_calc_table_3[c] = base_calc_table_3[c].fillna(0).astype(int)

base_calc_table_3['prediction_ma'] = (
    (base_calc_table_3['net_quantity_7']/7)*0.5 + 
    (base_calc_table_3['net_quantity_14']/14)*0.3 + 
    (base_calc_table_3['net_quantity_28']/28)*0.2
)*14
base_calc_table_4 = (
    base_calc_table_3
    .merge(
        orders[['product_id','store_code', 'acik_siparis']], 
        on=['store_code','product_id'], 
        how='left'
    )
)
base_calc_table_4['acik_siparis'] = base_calc_table_4['acik_siparis'].fillna(0).astype(int)

def need_with_14day_coverage(
    df: pd.DataFrame,
    pred_df_col="prediction_df",     # 14g model tahmini
    pred_ma_col="prediction_ma",     # 14g MA tahmini
    ss_col="safety_stock",
    onhand_col="stok_miktar",
    last14_col="net_quantity_14",    # son 14 gün satış
    last28_col="net_quantity_28",    # son 28 gün satış
    lead_time_days=2,
    in_transit_col='acik_siparis',             # varsa "in_transit" gibi bir kolon adı ver
    w_min=0.25, w_max=0.75           # model ağırlığı sınırı
) -> pd.DataFrame:
    out = df.copy()

    # sayısal kolonları temizle
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

    w_model = (0.5 + 0.25 * (momentum - 1.0)).clip(w_min, w_max)  # 0.25..0.75
    # koruyucu kurallar
    w_model = np.where((out[pred_ma_col] <= 0) & (out[pred_df_col] > 0), 1.0, w_model)
    w_model = np.where((out[pred_df_col] <= 0) & (out[pred_ma_col] > 0) & (out[last28_col] > 0), 0.0, w_model)
    out["w_model"] = w_model.astype(float)

    out["pred_14_blended"] = out["w_model"] * out[pred_df_col] + (1.0 - out["w_model"]) * out[pred_ma_col]

    out["pred_LT"] = out["pred_14_blended"] * (lead_time_days / 14.0)

    out["OUL_14"] = out[ss_col] + out["pred_14_blended"]
    expected_onhand_at_arrival = out[onhand_col] - out["pred_LT"]
    if in_transit_col:
        expected_onhand_at_arrival = expected_onhand_at_arrival + out[in_transit_col]
    out["ihtiyac_adet"] = np.ceil(np.maximum(0.0, out["OUL_14"] - expected_onhand_at_arrival)).astype(int)

    return out
result = need_with_14day_coverage(base_calc_table_4, lead_time_days=2)

# Filtreleme: ihtiyac_adet > 0 olanları kaydet
print("Filtering replenishment results for ihtiyac_adet > 0...")
original_count = len(result)

if 'ihtiyac_adet' in result.columns:
    filtered_result = result[result['ihtiyac_adet'] > 0].copy()
    filtered_count = len(filtered_result)
    print(f"Filtered replenishment results: {original_count} -> {filtered_count} rows (ihtiyac_adet > 0)")
else:
    print("'ihtiyac_adet' column not found, saving all results")
    filtered_result = result

# CSV olarak kaydet
filtered_result.to_csv(
    "output/replenishment_results.csv",
    index=False,
    sep=";",             
    encoding="utf-8-sig",
    float_format="%.6f"  
)

# Parquet olarak da kaydet (DAG için)
filtered_result.to_parquet(
    "output/replenishment_results.parquet",
    index=False,
    compression="snappy"
)

print("✅ Replenishment results saved (CSV + Parquet)")

# Warehouse distribution hesaplama
print("\n🔄 Starting warehouse distribution calculation...")

def distribute_warehouse_stock(
    replenishment_needs: pd.DataFrame,
    warehouse_stock: pd.DataFrame,
    store_col="store_code",
    product_col="product_id",
    need_col="ihtiyac_adet",
    warehouse_col="alt_yer",
    warehouse_qty_col="miktar"
) -> pd.DataFrame:
    """
    Warehouse stock dağıtım fonksiyonu - Tüm ürün-mağaza kombinasyonları için
    """
    print("Starting comprehensive warehouse stock distribution...")
    
    # Data preprocessing
    needs = replenishment_needs.copy()
    warehouse = warehouse_stock.copy()
    
    # Ensure consistent data types
    needs[store_col] = needs[store_col].astype(str)
    needs[product_col] = needs[product_col].astype(str)
    warehouse[warehouse_col] = warehouse[warehouse_col].astype(str)
    warehouse[product_col] = warehouse[product_col].astype(str)
    
    # Filter only products with positive needs
    needs = needs[needs[need_col] > 0].copy()
    
    if needs.empty:
        print("No positive needs found for distribution")
        return pd.DataFrame(columns=[
            "gonderen_depo", "alan_magaza", "urun", "baslangic_depo_stok", 
            "ihtiyac_adet", "karsilanan_adet", "karsilanmayan_adet", "kalan_ihtiyac", "kalan_depo_stok"
        ])
    
    print(f"Total needs: {len(needs)} rows")
    print(f"Total unique products with needs: {needs[product_col].nunique()}")
    
    # Get all unique products that have needs
    products_with_needs = needs[product_col].unique()
    
    # Get warehouse stock for each product
    warehouse_by_product = {}
    for product in products_with_needs:
        product_warehouse = warehouse[
            (warehouse[product_col] == product) & 
            (warehouse[warehouse_qty_col] > 0)
        ].copy()
        warehouse_by_product[product] = product_warehouse
    
    print(f"Products with warehouse stock: {len([p for p in products_with_needs if not warehouse_by_product[p].empty])}")
    print(f"Products without warehouse stock: {len([p for p in products_with_needs if warehouse_by_product[p].empty])}")
    
    distribution_results = []
    
    for product in products_with_needs:
        # Get needs for this product
        product_needs = needs[needs[product_col] == product].copy()
        product_needs = product_needs.sort_values(need_col, ascending=False)  # Büyük ihtiyaçtan küçüğe
        
        # Get warehouse stock for this product
        product_warehouse = warehouse_by_product[product]
        
        if product_warehouse.empty:
            # No warehouse stock - all needs are unmet
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
        
        # Store initial warehouse stock for each warehouse
        initial_warehouse_stocks = {}
        for _, warehouse_row in product_warehouse.iterrows():
            warehouse_code = warehouse_row[warehouse_col]
            warehouse_qty = warehouse_row[warehouse_qty_col]
            initial_warehouse_stocks[warehouse_code] = warehouse_qty
        
        # Initialize remaining needs and warehouse stock
        remaining_needs = product_needs.copy()
        remaining_warehouse = product_warehouse.copy()
        
        # Track distribution for each store-warehouse combination
        store_warehouse_distributions = {}
        
        # Round-robin distribution: 1 unit at a time
        while not remaining_needs.empty and not remaining_warehouse.empty:
            # Get current need and warehouse
            current_need = remaining_needs.iloc[0]
            current_warehouse = remaining_warehouse.iloc[0]
            
            need_store = current_need[store_col]
            need_qty = current_need[need_col]
            warehouse_code = current_warehouse[warehouse_col]
            warehouse_qty = current_warehouse[warehouse_qty_col]
            
            # Initialize tracking for this store-warehouse combination
            key = (need_store, warehouse_code)
            if key not in store_warehouse_distributions:
                store_warehouse_distributions[key] = {
                    'initial_need': need_qty,
                    'distributed': 0,
                    'initial_warehouse_stock': initial_warehouse_stocks[warehouse_code]
                }
            
            # Distribute 1 unit
            distributed_qty = 1
            remaining_qty = need_qty - distributed_qty
            warehouse_remaining = warehouse_qty - distributed_qty
            
            # Update tracking
            store_warehouse_distributions[key]['distributed'] += distributed_qty
            
            # Update remaining needs
            if remaining_qty <= 0:
                # Need is fully satisfied, remove from list
                remaining_needs = remaining_needs.iloc[1:].reset_index(drop=True)
            else:
                # Update need quantity and move to end for round-robin
                remaining_needs.iloc[0, remaining_needs.columns.get_loc(need_col)] = remaining_qty
                # Move this need to the end for round-robin distribution
                temp_need = remaining_needs.iloc[0:1].copy()
                remaining_needs = pd.concat([
                    remaining_needs.iloc[1:], 
                    temp_need
                ], ignore_index=True)
            
            # Update warehouse stock - Depo stoku 0'a kadar düşebilir
            if warehouse_remaining <= 0:
                # Warehouse is empty, remove from list
                remaining_warehouse = remaining_warehouse.iloc[1:].reset_index(drop=True)
            else:
                # Update warehouse quantity (0 dahil)
                remaining_warehouse.iloc[0, remaining_warehouse.columns.get_loc(warehouse_qty_col)] = warehouse_remaining
        
        # Calculate total shipments per warehouse for this product
        warehouse_total_shipments = {}
        for (store, warehouse_code), dist_info in store_warehouse_distributions.items():
            if warehouse_code not in warehouse_total_shipments:
                warehouse_total_shipments[warehouse_code] = 0
            warehouse_total_shipments[warehouse_code] += dist_info['distributed']
        
        # Record detailed distribution results for this product
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
    
    # Create final result DataFrame
    if distribution_results:
        result_df = pd.DataFrame(distribution_results)
        print(f"Distribution completed. Total distributions: {len(result_df)}")
        
        # Summary statistics
        total_needs = result_df['ihtiyac_adet'].sum()
        total_distributed = result_df['karsilanan_adet'].sum()
        total_unmet = result_df['kalan_ihtiyac'].sum()
        
        print(f"Summary:")
        print(f"  Total needs: {total_needs}")
        print(f"  Total distributed: {total_distributed}")
        print(f"  Total unmet: {total_unmet}")
        print(f"  Distribution rate: {total_distributed/total_needs*100:.1f}%")
        
    else:
        result_df = pd.DataFrame(columns=[
            "gonderen_depo", "alan_magaza", "urun", "baslangic_depo_stok", 
            "ihtiyac_adet", "karsilanan_adet", "kalan_ihtiyac", "depo_sevk_toplam", "depo_kalan_nihai"
        ])
        print("No distributions made")
    
    return result_df

# Warehouse distribution hesapla
distribution_result = distribute_warehouse_stock(
    replenishment_needs=filtered_result,
    warehouse_stock=warehouse_stock,
    store_col="store_code",
    product_col="product_id",
    need_col="ihtiyac_adet",
    warehouse_col="alt_yer",
    warehouse_qty_col="miktar"
)

# Warehouse distribution sonuçlarını kaydet (tüm sonuçlar)
print(f"\nSaving all warehouse distribution results: {len(distribution_result)} rows")

if not distribution_result.empty:
    distribution_result.to_csv(
        "output/warehouse_distribution.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig",
        float_format="%.6f"
    )
    
    distribution_result.to_parquet(
        "output/warehouse_distribution.parquet",
        index=False,
        compression="snappy"
    )
    
    print("✅ Warehouse distribution results saved (CSV + Parquet)")
    
    # Özet istatistikler
    total_needs = distribution_result['ihtiyac_adet'].sum()
    total_distributed = distribution_result['karsilanan_adet'].sum()
    total_unmet = distribution_result['kalan_ihtiyac'].sum()
    
    print(f"\n📊 Final Summary:")
    print(f"  Total ihtiyaç adet: {total_needs}")
    print(f"  Total karşılanan adet: {total_distributed}")
    print(f"  Total karşılanmayan adet: {total_unmet}")
    print(f"  Karşılama oranı: {total_distributed/total_needs*100:.1f}%")
    
else:
    print("⚠️ No warehouse distribution data to save")

print("\n🎉 All calculations completed successfully!")

