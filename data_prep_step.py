import pandas as pd
import numpy as np

product = pd.read_csv('data/dim_product.csv', index_col=None)
calendar = pd.read_csv('data/dim_calendar_pivot.csv', index_col=None)
sales = pd.read_csv('data/history_sales.csv', index_col=None)

# Tarih dönüşümleri
sales['date'] = pd.to_datetime(sales['date'])
calendar['date'] = pd.to_datetime(calendar['date'])
calendar['week_start_date'] = calendar['date'].dt.to_period('W').apply(lambda r: r.start_time)

# Hafta bilgisi ekle
sales = sales.merge(calendar[['date', 'week_start_date']], on='date', how='left')

# Ürün bilgilerini birleştir
product_segments_cols = [
    'product_id', 'marka_aciklama',
    'product_att_01', 'product_att_02', 'product_att_03',
    'product_att_04', 'product_att_05', 'product_att_06',
    'product_att_01_desc', 'product_att_02_desc', 'product_att_03_desc',
    'product_att_04_desc', 'product_att_05_desc', 'product_att_06_desc'
]
sales_with_segment = sales.merge(product[product_segments_cols], on='product_id', how='left')

sales_with_segment['product_att_02'] = sales_with_segment['product_att_02'].apply(
    lambda x: int(str(x).replace('D', '9').replace('N', '9')) if isinstance(x, str) and ('D' in x or 'N' in x) else x
)

sales_with_segment['product_att_02'] = pd.to_numeric(sales_with_segment['product_att_02'], errors='coerce')




# Store clustering
cluster = pd.read_csv('k_means_store.csv')
sales_with_segment = sales_with_segment.merge(cluster, on='store_code', how='left')

# Segment tanımı
granularity = [
    'product_att_01', 'product_att_02', 'product_att_05',
    'product_att_01_desc', 'product_att_02_desc', 'product_att_05_desc',
    'store_cluster'
]

# Filtrele: 9.0'lar hariç
filtered_sales = sales_with_segment[
    (~(sales_with_segment['product_att_01'] == 9.0)) &
    (~(sales_with_segment['product_att_02'] == 9.0)) 

]

fs = filtered_sales.copy()

# Adet bağımsız indirim oranı (0-0.9 kırp)
denom = fs['discount_amount'].fillna(0) + fs['net_amount_wovat'].fillna(0)
fs['discount_frac'] = np.where(denom > 0, fs['discount_amount'].fillna(0) / denom, 0.0)
fs['discount_frac'] = fs['discount_frac'].clip(0, 0.9)

# Temel price sinyali (satır bazında)
fs['unit_price'] = pd.to_numeric(fs['unit_price'], errors='coerce')
fs['unit_price'] = fs['unit_price'].clip(lower=0)

# (Opsiyonel) Ağırlık için indirim öncesi ciro
fs['rev_pre_disc'] = denom

# Haftalık + segment agregasyon anahtarı
agg_keys = granularity + ['week_start_date']

# Ağırlıklı ortalama fonksiyonu (ciro ile)
def wavg(x, w):
    wsum = w.sum()
    return (x.mul(w).sum() / wsum) if wsum > 0 else x.mean()

# Haftalık indirim/price özetleri
price_agg = (
    fs.groupby(agg_keys)
      .apply(lambda g: pd.Series({
          'unit_price_mean': g['unit_price'].mean(),
          'unit_price_median': g['unit_price'].median(),
          'discount_frac_mean': g['discount_frac'].mean(),
          'discount_frac_wavg': wavg(g['discount_frac'], g['rev_pre_disc'])
      }))
      .reset_index()
)

# segment_sales ile birleştir (toplam adet + indirim & price özetleri aynı tabloda)
segment_sales = (
    fs.groupby(agg_keys)['net_quantity']
      .sum().reset_index()
      .rename(columns={'net_quantity': 'total_quantity'})
      .merge(price_agg, on=agg_keys, how='left')
      .sort_values(by='total_quantity', ascending=False)
)

segment_sales['total_quantity'] = segment_sales['total_quantity'].astype(int)


# Kaydet (artık weekly_sales_with_cluster dosyanda indirim/price alanları da var)
segment_sales.to_csv('output/weekly_sales_with_cluster[1,2,5].csv', index=False)