import pandas as pd
import numpy as np

# Veri setlerini oku
df = pd.read_csv("output/weekly_sales_with_cluster[1,2,5].csv")
calendar = pd.read_csv("data/dim_calendar_pivot.csv")

# Tarih dönüşümleri
calendar['date'] = pd.to_datetime(calendar['date'])
calendar['week_start_date'] = calendar['date'].dt.to_period('W').apply(lambda r: r.start_time)
df['week_start_date'] = pd.to_datetime(df['week_start_date'])

df['week_of_year'] = df['week_start_date'].dt.isocalendar().week

# Negatif satışları sıfırla
df.loc[df['total_quantity'] < 0, 'total_quantity'] = 0

# Segmentasyon kolonları
segment_cols = ['product_att_01', 'product_att_02', 'product_att_05','store_cluster'] # 

# Tarihe göre sırala
df = df.sort_values(segment_cols + ['week_start_date'])

for col in ['discount_frac_wavg', 'discount_frac_mean', 'unit_price_mean', 'unit_price_median']:
    if col in df.columns:
        df[f'{col}_l1'] = df.groupby(segment_cols)[col].shift(1)
        df[f'{col}_ma4_l1'] = (
            df.groupby(segment_cols)[col]
              .shift(1).rolling(4, min_periods=2).mean()
        )
#df['store_product_share'] = df['total_quantity'] / df.groupby(['store_cluster', 'week_start_date'])['total_quantity'].transform('sum')


# Geçen yılın aynı haftasının satış miktarı
df['total_quantity_last_year'] = df.groupby(segment_cols)['total_quantity'].shift(53)




# Özellik mühendisliği fonksiyonu
def add_shifted_rolled_features(data: pd.DataFrame,
                                date_col: str,
                                granularity_cols: list,
                                target_col: str,
                                shifts: list,
                                rolls: dict,
                                compute_diffs: list = None):
    data = data.sort_values(granularity_cols + [date_col]).reset_index(drop=True)

    # LAG / shifted
    for shift in shifts:
        data[f"{target_col}_shifted_{shift}"] = (
            data.groupby(granularity_cols)[target_col].shift(shift)
        )

    # DIFF (shifted_X - shifted_{X+1})
    if compute_diffs:
        for diff in compute_diffs:
            col_1 = f"{target_col}_shifted_{diff}"
            col_2 = f"{target_col}_shifted_{diff + 1}"

            if col_1 not in data.columns:
                data[col_1] = data.groupby(granularity_cols)[target_col].shift(diff)
            if col_2 not in data.columns:
                data[col_2] = data.groupby(granularity_cols)[target_col].shift(diff + 1)

            #data[f"diff_{diff}"] = data[col_1] - data[col_2]

            data[f"pct_change_{diff}"] = (
                        (data[col_1] - data[col_2]) / data[col_2].replace(0, np.nan)
                    )

    # ROLLING
    for shift, windows in rolls.items():
        shifted = data.groupby(granularity_cols)[target_col].shift(shift)
        for window in windows:
            data[f"min_{target_col}_roll{window}_shift{shift}"] = shifted.rolling(window).min()
            data[f"mean_{target_col}_roll{window}_shift{shift}"] = shifted.rolling(window).mean()
            data[f"max_{target_col}_roll{window}_shift{shift}"] = shifted.rolling(window).max()

    return data

# Özellikleri uygula
df = add_shifted_rolled_features(
    data=df,
    date_col='week_start_date',
    granularity_cols=segment_cols,
    target_col='total_quantity',
    shifts=[1, 2, 3, 4, 5, 6, 8, 12, 24, 48, 49, 50, 51, 52],
    rolls={
        1: [4, 8],
        2: [4, 8],
        3: [4, 8],
        8: [4],
        12: [4]
    },
    compute_diffs=[1, 2, 3, 4, 8, 12]
)


# Ek hesaplamalar
df['segment_mean'] = df.groupby(segment_cols)['total_quantity'].transform('mean')
#df['yoy_growth'] = (df['total_quantity_shifted_1'] - df['total_quantity_last_year']) / (df['total_quantity_last_year'].replace(0, 1e-5))

# Takvim verisini haftalık özetle
calendar_weekly = (
    calendar.groupby('week_start_date')
    .agg({
        'month': 'first',
        'special_day_tag': lambda x: x.dropna().iloc[0] if x.notna().any() else 'YOK',
        'ramazan_bayrami': 'max',
        'kurban_bayrami': 'max',
        'kara_cuma': 'max'
    })
    .reset_index()
)

calendar_weekly['is_special_day'] = (calendar_weekly['special_day_tag'] != 'YOK').astype(int)
calendar_weekly.drop('special_day_tag', axis=1, inplace=True)

# Takvim ile birleştir
df = df.merge(calendar_weekly, on='week_start_date', how='left')


# Kategorik değişkenleri dönüştür
cat_cols = [
    'product_att_01_desc', 'product_att_02_desc',  'product_att_05_desc',
    'month'
]

for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Sonucu CSV'ye kaydet
df.to_csv('output/weekly_sales_with_features.csv', index=False)
#df.to_clipboard()
