import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import norm

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# =======================
# 0) INPUTS + HAFİF TEMİZLİK
# =======================
product         = pd.read_csv('data/dim_product.csv', index_col=None)
store_stock_raw = pd.read_csv('data/store_stock.csv', index_col=None)
warehouse_stock = pd.read_csv('data/warehouse_stock.csv', index_col=None)  # (ileride kullanacaksan dursun)
predictions     = pd.read_parquet('daily_store_pred.parquet')
sales_raw       = pd.read_csv('data/history_sales.csv', index_col=None)
orders_raw      = pd.read_csv('data/open_order.csv', index_col=None)

# Sipariş kolonu ve isim eşitleme
orders = orders_raw.rename(columns={'depo_kodu': 'store_code'}).copy()
orders['acik_siparis'] = pd.to_numeric(
    orders['toplam_yoldaki_miktar'], errors='coerce'
).fillna(0) + pd.to_numeric(
    orders['toplanmayan_urun'], errors='coerce'
).fillna(0)

# Prediction kolon adı
predictions = predictions.rename(columns={'prediction_store_daily': 'store_predicted_quantity'})
predictions['date'] = pd.to_datetime(predictions['date']).dt.normalize()
predictions['store_code'] = predictions['store_code'].astype(str)
predictions['stok_kodu']  = predictions['stok_kodu'].astype(str)
predictions['store_predicted_quantity'] = pd.to_numeric(predictions['store_predicted_quantity'], errors='coerce').fillna(0)

# =======================
# 1) SALES: ÜRÜN SEGMENT FİLTRESİ + TİP DÜZENİ
# =======================
sales = sales_raw.copy()
sales['date'] = pd.to_datetime(sales['date']).dt.normalize()
for c in ['store_code', 'stok_kodu', 'product_id']:
    sales[c] = sales[c].astype(str)
sales['net_quantity'] = pd.to_numeric(sales['net_quantity'], errors='coerce').fillna(0)

# product’tan sadece gereken kolonları çek
product_segments_cols = ['product_id', 'product_att_01', 'product_att_02', 'product_att_03']
prod_seg = product[product_segments_cols].copy()
prod_seg['product_id'] = prod_seg['product_id'].astype(str)

# Merge + filtre (tek blok, ezilmeyi önler)
sales = (sales.merge(prod_seg, on='product_id', how='left'))
# product_att_02 içindeki D/N → 9 (veya 999) yerine:
sales['product_att_02'] = (
    sales['product_att_02']
        .astype(str)
        .str.replace('[DN]', '9', regex=True)
        .replace({'nan': np.nan})
)
sales['product_att_01'] = pd.to_numeric(sales['product_att_01'], errors='coerce')
sales['product_att_02'] = pd.to_numeric(sales['product_att_02'], errors='coerce')

# 9 veya 999 olanları çıkar
mask_valid = (~sales['product_att_01'].isin([9.0, 999])) & (~sales['product_att_02'].isin([9.0, 999]))
sales = sales[mask_valid & (sales['net_quantity'] >= 0)].reset_index(drop=True)

# =======================
# 2) STORE_STOCK: AGG + TİP
# =======================
# product_id türet (string cat daha hızlı)
ss = store_stock_raw.copy()
for c in ['stok_kodu', 'color_code', 'size', 'magaza']:
    ss[c] = ss[c].astype(str)
ss['product_id'] = ss['stok_kodu'].str.cat(ss['color_code']).str.cat(ss['size'])
ss['stok_tarihi'] = pd.to_datetime(ss['stok_tarihi']).dt.normalize()
ss['toplam_miktar'] = pd.to_numeric(ss['toplam_miktar'], errors='coerce').fillna(0)

store_stock_agg = (ss.groupby(['stok_tarihi','magaza','stok_kodu','product_id'], as_index=False)['toplam_miktar']
                     .sum()
                     .rename(columns={'toplam_miktar':'stok_miktar'}))
store_stock_agg['stok_miktar'] = store_stock_agg['stok_miktar'].astype(int)

# =======================
# 3) 7/14/28 GÜN SATIŞLARI — TEK GROUPBY
# =======================
max_date = sales['date'].max()
start_28 = max_date - pd.Timedelta(days=27)
start_14 = max_date - pd.Timedelta(days=13)
start_7  = max_date - pd.Timedelta(days=6)

# Sadece son 28 günü al, içeride 7/14’ü işaretle → tek groupby
s28 = sales[sales['date'] >= start_28].copy()
s28['w28'] = s28['net_quantity']
s28['w14'] = np.where(s28['date'] >= start_14, s28['net_quantity'], 0)
s28['w7']  = np.where(s28['date'] >= start_7,  s28['net_quantity'], 0)

sales_win = (s28.groupby(['product_id','store_code'], as_index=False)[['w7','w14','w28']].sum()
               .rename(columns={'w7':'net_quantity_7','w14':'net_quantity_14','w28':'net_quantity_28'}))

# =======================
# 4) STORE×SKU PAIRS (STOK ∪ PREDICTION)
# =======================
def get_store_sku_pairs_union(store_stock_agg: pd.DataFrame,
                              prediction: pd.DataFrame | None = None,
                              lookback_stock_days=30,
                              min_pred=0.0) -> pd.DataFrame:
    ss = store_stock_agg[['stok_tarihi','magaza','stok_kodu','stok_miktar']].copy()
    ss['stok_tarihi'] = pd.to_datetime(ss['stok_tarihi']).dt.normalize()
    ss['magaza'] = ss['magaza'].astype(str)
    ss['stok_kodu'] = ss['stok_kodu'].astype(str)
    ss['stok_miktar'] = pd.to_numeric(ss['stok_miktar'], errors='coerce').fillna(0)

    max_dates = [ss['stok_tarihi'].max()]
    if prediction is not None and not prediction.empty:
        max_dates.append(prediction['date'].max())
    today = max([d for d in max_dates if pd.notna(d)])

    start_stock = today - pd.Timedelta(days=lookback_stock_days-1)
    stock_pairs = (ss[(ss['stok_tarihi'] >= start_stock) & (ss['stok_tarihi'] <= today) & (ss['stok_miktar'] > 0)]
                     [['magaza','stok_kodu']].drop_duplicates())

    if prediction is not None and not prediction.empty:
        pred_pairs = (prediction[prediction['store_predicted_quantity'] > min_pred]
                        [['store_code','stok_kodu']].drop_duplicates()
                        .rename(columns={'store_code':'magaza'}))
        pairs = pd.concat([stock_pairs, pred_pairs], ignore_index=True).drop_duplicates()
    else:
        pairs = stock_pairs

    return pairs.sort_values(['magaza','stok_kodu'], ignore_index=True)

store_sku_pairs = get_store_sku_pairs_union(store_stock_agg, predictions)
store_sku_pairs['magaza']    = store_sku_pairs['magaza'].astype(str)
store_sku_pairs['stok_kodu'] = store_sku_pairs['stok_kodu'].astype(str)

# =======================
# 5) STORE×PRODUCT PAIRS (aktif ürünler)
# =======================
pm_active = product.copy()
pm_active['product_code'] = pm_active['product_code'].astype(str)
pm_active['product_id']   = pm_active['product_id'].astype(str)
if 'is_blocked' in pm_active.columns:
    pm_active = pm_active.loc[pm_active['is_blocked'].fillna(0).astype(int) == 0]

store_product_pairs = (store_sku_pairs
    .merge(pm_active[['product_code','product_id']], left_on='stok_kodu', right_on='product_code', how='left')
    .drop(columns=['product_code'])
    .drop_duplicates()
    .rename(columns={'magaza':'store_code'})
    .reset_index(drop=True))

# =======================
# 6) SAFETY STOCK (store+product)
# =======================
def seasonal_safety_stock(sales: pd.DataFrame,
                          date_col='date',
                          store_col='store_code',
                          product_col='product_id',
                          qty_col='net_quantity',
                          lead_time_days=2,
                          service_level=0.90,
                          month_window=1) -> pd.DataFrame:
    df = sales[[date_col, store_col, product_col, qty_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df[store_col] = df[store_col].astype(str)
    df[product_col] = df[product_col].astype(str)
    df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)

    today = df[date_col].max()
    months_to_keep = [(today.month + i - 1) % 12 + 1 for i in range(-month_window, month_window + 1)]
    df = df[df[date_col].dt.month.isin(months_to_keep)]

    daily = (df.groupby([store_col, product_col, date_col], as_index=False)[qty_col].sum())
    stats = (daily.groupby([store_col, product_col])
                  .agg(daily_avg=(qty_col,'mean'), daily_std=(qty_col,'std'))
                  .fillna(0)
                  .reset_index())
    z = norm.ppf(service_level)
    stats['safety_stock'] = (z * stats['daily_std'] * np.sqrt(lead_time_days)).round(2)
    return stats

ss_df = seasonal_safety_stock(sales, store_col='store_code', month_window=1)

# =======================
# 7) AĞIRLIK HARİTASI (sales→stock→equal)
#     *sadece prediction’ı olan store×sku’lar için*
# =======================
LOOKBACK_SALES = 90
LOOKBACK_STOCK = 60
end_date = sales['date'].max()

# Pairs’i prediction varlığına göre sınırla (gereksiz kombinasyonlardan kaçın)
pred_pairs = predictions[predictions['store_predicted_quantity'] > 0][['store_code','stok_kodu']].drop_duplicates()
pairs = (store_product_pairs[['store_code','stok_kodu','product_id']]
         .merge(pred_pairs, on=['store_code','stok_kodu'], how='inner'))

# SALES pencere
start_sales = end_date - pd.Timedelta(days=LOOKBACK_SALES - 1)
s_win = sales[(sales['date'] >= start_sales) & (sales['date'] <= end_date) & (sales['net_quantity'] > 0)]
s_win = s_win.merge(pairs, on=['store_code','stok_kodu','product_id'], how='inner')

sales_agg = (s_win.groupby(['store_code','stok_kodu','product_id'], as_index=False)['net_quantity']
               .sum().rename(columns={'net_quantity':'sales_90d'}))
sales_agg['weight_sales'] = sales_agg['sales_90d'] / sales_agg.groupby(['store_code','stok_kodu'])['sales_90d'].transform('sum')

# STOCK pencere
start_stock = end_date - pd.Timedelta(days=LOOKBACK_STOCK - 1)
st = ss.rename(columns={'magaza':'store_code'})[['store_code','stok_kodu','product_id','stok_tarihi','toplam_miktar']].copy()
st_win = st[(st['stok_tarihi'] >= start_stock) & (st['stok_tarihi'] <= end_date) & (st['toplam_miktar'] > 0)]
st_win = st_win.merge(pairs, on=['store_code','stok_kodu','product_id'], how='inner')

stock_agg = (st_win.groupby(['store_code','stok_kodu','product_id'], as_index=False)['toplam_miktar']
              .sum().rename(columns={'toplam_miktar':'onhand_30d'}))
stock_agg['weight_onhand'] = stock_agg['onhand_30d'] / stock_agg.groupby(['store_code','stok_kodu'])['onhand_30d'].transform('sum')

# Children = pairs (predictionı olan universe)
weights = (pairs
    .merge(sales_agg[['store_code','stok_kodu','product_id','weight_sales']], on=['store_code','stok_kodu','product_id'], how='left')
    .merge(stock_agg[['store_code','stok_kodu','product_id','weight_onhand']], on=['store_code','stok_kodu','product_id'], how='left'))

# Öncelik: sales → stock → equal
weights['weight'] = weights['weight_sales'].where(weights['weight_sales'].notna(), weights['weight_onhand'])
weights['n_option'] = weights.groupby(['store_code','stok_kodu'])['product_id'].transform('count')
grp_has_any = weights.groupby(['store_code','stok_kodu'])['weight'].transform(lambda s: s.notna().any())
weights.loc[~grp_has_any, 'weight'] = 1.0 / weights['n_option']

sum_w = weights.groupby(['store_code','stok_kodu'])['weight'].transform('sum')
weights.loc[(sum_w == 0) | (sum_w.isna()), 'weight'] = 1.0 / weights['n_option']
weights['weight'] = weights['weight'] / weights.groupby(['store_code','stok_kodu'])['weight'].transform('sum')

weight_map = weights[['store_code','stok_kodu','product_id','weight']].copy()

# =======================
# 8) PREDICTION’I product_id’YE KIR (14 gün toplamı)
# =======================
pred = predictions.copy()
pred['store_predicted_quantity'] = pd.to_numeric(pred['store_predicted_quantity'], errors='coerce').fillna(0)

# 14 gün toplamı için pencere (max_date referans alındı)
end14 = max_date + timedelta(days=14)
pred_window = pred[(pred['date'] > max_date) & (pred['date'] <= end14)]
pred_window = pred_window.groupby(['store_code','stok_kodu'], as_index=False)['store_predicted_quantity'].sum()

pred_pid = (pred_window
    .merge(weight_map, on=['store_code','stok_kodu'], how='inner')
    .assign(prediction_df=lambda d: d['store_predicted_quantity'] * d['weight'])
    [['store_code','stok_kodu','product_id','prediction_df']])

# =======================
# 9) BAZ TABLO (SS + OnHand + Satış pencereleri + Açık sipariş)
# =======================
# safety stock merge
base = (store_product_pairs[['store_code','product_id']]
        .merge(ss_df[['store_code','product_id','safety_stock']], on=['store_code','product_id'], how='left'))
base['safety_stock'] = pd.to_numeric(base['safety_stock'], errors='coerce').fillna(0)

# prediction (14g toplam) merge
base = base.merge(pred_pid, on=['store_code','product_id'], how='left')
base['prediction_df'] = base['prediction_df'].fillna(0.0).astype(float)

# onhand merge
store_stock_onhand = (store_stock_agg.rename(columns={'magaza': 'store_code'})
                      [['store_code','product_id','stok_miktar']])
base = base.merge(store_stock_onhand, on=['store_code','product_id'], how='left')
base['stok_miktar'] = base['stok_miktar'].fillna(0).astype(int)

# satış 7/14/28 merge (tek DF’den)
base = (base.merge(sales_win[['store_code','product_id','net_quantity_7']],  on=['store_code','product_id'], how='left')
            .merge(sales_win[['store_code','product_id','net_quantity_14']], on=['store_code','product_id'], how='left')
            .merge(sales_win[['store_code','product_id','net_quantity_28']], on=['store_code','product_id'], how='left'))
for c in ['net_quantity_7','net_quantity_14','net_quantity_28']:
    base[c] = base[c].fillna(0).astype(int)

# prediction_ma (14g eşleniği)
base['prediction_ma'] = (
    (base['net_quantity_7']/7)*0.5 +
    (base['net_quantity_14']/14)*0.3 +
    (base['net_quantity_28']/28)*0.2
) * 14

# açık sipariş
orders_min = orders[['store_code','product_id','acik_siparis']].copy()
orders_min['acik_siparis'] = pd.to_numeric(orders_min['acik_siparis'], errors='coerce').fillna(0)
base = base.merge(orders_min, on=['store_code','product_id'], how='left')
base['acik_siparis'] = base['acik_siparis'].fillna(0).astype(float)

# =======================
# 10) İHTİYAÇ HESABI (14 gün kapsama)
# =======================
def need_with_14day_coverage(
    df: pd.DataFrame,
    pred_df_col='prediction_df',
    pred_ma_col='prediction_ma',
    ss_col='safety_stock',
    onhand_col='stok_miktar',
    last14_col='net_quantity_14',
    last28_col='net_quantity_28',
    lead_time_days=2,
    in_transit_col='acik_siparis',
    w_min=0.25, w_max=0.75
) -> pd.DataFrame:
    out = df.copy()
    for c in [pred_df_col, pred_ma_col, ss_col, onhand_col, last14_col, last28_col, in_transit_col]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0.0)
        else:
            out[c] = 0.0

    # momentum: 14g / (28g/2)
    avg14_from_28 = out[last28_col] / 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        momentum = out[last14_col] / avg14_from_28.replace(0, np.nan)
    momentum = pd.Series(momentum).fillna(1.0).clip(0.0, 2.0)

    w_model = (0.5 + 0.25 * (momentum - 1.0)).clip(w_min, w_max)
    w_model = np.where((out[pred_ma_col] <= 0) & (out[pred_df_col] > 0), 1.0, w_model)
    w_model = np.where((out[pred_df_col] <= 0) & (out[pred_ma_col] > 0) & (out[last28_col] > 0), 0.0, w_model)
    out['w_model'] = w_model.astype(float)

    out['pred_14_blended'] = out['w_model'] * out[pred_df_col] + (1.0 - out['w_model']) * out[pred_ma_col]
    out['pred_LT'] = out['pred_14_blended'] * (lead_time_days / 14.0)

    out['OUL_14'] = out[ss_col] + out['pred_14_blended']
    expected_onhand_at_arrival = out[onhand_col] - out['pred_LT'] + out[in_transit_col]
    out['ihtiyac_adet'] = np.ceil(np.maximum(0.0, out['OUL_14'] - expected_onhand_at_arrival)).astype(int)
    return out

result = need_with_14day_coverage(base, lead_time_days=2)

# =======================
# 11) OUTPUT
# =======================
result.to_csv(
    "output/replenishment_results.csv",
    index=False,
    sep=";",
    encoding="utf-8-sig",
    float_format="%.6f"
)
