import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

# === Parametreler ===
FUTURE_WEEKS = 12
INPUT_PATH  = "output/weekly_sales_with_features.csv"
OUTPUT_PATH = "output/future_12_weeks_hierarchical_forecast.csv"

# === Veri ===
df = pd.read_csv(INPUT_PATH)
df['week_start_date'] = pd.to_datetime(df['week_start_date'])

target = "total_quantity"
hier_cols = ['product_att_01', 'product_att_02', 'product_att_05', 'store_cluster'] #

drop_cols = [
    'week_start_date', 'total_quantity',
    'product_att_01_desc', 'product_att_02_desc', 'product_att_05_desc'
]
feature_cols = [c for c in df.columns if c not in drop_cols]

# Kategorikler
for c in feature_cols:
    if df[c].dtype == 'object':
        df[c] = df[c].astype('category')
cat_features = [c for c in feature_cols if str(df[c].dtype) == 'category']

# === Hafta + hiyerarşi seviyesinde aggregate ===
agg_map = {target: 'sum'}
for c in feature_cols:
    if c not in hier_cols:
        agg_map[c] = 'last'

df_agg = (
    df.groupby(hier_cols + ['week_start_date'], as_index=False)
      .agg(agg_map)
)

# === Modeli TÜM veriyle eğit ===
X_train = df_agg[feature_cols]
y_train = df_agg[target]

model = LGBMRegressor(
    objective='regression',
    learning_rate=0.05,
    max_depth=3,
    num_leaves=31,
    n_estimators=300,
    random_state=42
)
model.fit(
    X_train, y_train,
    categorical_feature=cat_features if len(cat_features) > 0 else None
)

# === Gelecek 12 hafta satırları ===
last_date = pd.to_datetime('2025-04-28')#df_agg['week_start_date'].max()
future_dates = [last_date + pd.Timedelta(weeks=i+1) for i in range(FUTURE_WEEKS)]

base_keys = df_agg[hier_cols].drop_duplicates().reset_index(drop=True)
future_df = (base_keys
             .assign(key=1)
             .merge(pd.DataFrame({'week_start_date': future_dates, 'key':1}), on='key')
             .drop(columns='key'))

# === Basit takvim feature'ları (varsa, üzerine yazar) ===
if 'month' in feature_cols:
    future_df['month'] = future_df['week_start_date'].dt.month
if 'week_of_year' in feature_cols:
    future_df['week_of_year'] = future_df['week_start_date'].dt.isocalendar().week.astype(int)

# Özel gün kolonlarını 0’a çek (varsa)
special_flags = [c for c in feature_cols if c.lower() in
                 ['ramazan_bayrami','kurban_bayrami','kara_cuma','is_special_day']]
for c in special_flags:
    future_df[c] = 0

# === Diğer feature'ları son gözlemden doldur ===
for col in feature_cols:
    if col in hier_cols: 
        continue
    if col in ['month','week_of_year'] + special_flags:
        continue
    mapping = df_agg.groupby(hier_cols)[col].last()
    future_df[col] = future_df.set_index(hier_cols).index.map(lambda idx: mapping.get(idx, np.nan))

# === Tahmin ===
future_df['predicted_quantity'] = model.predict(future_df[feature_cols]).clip(min=0)
future_df["predicted_quantity"] = future_df["predicted_quantity"].astype(float)

# === Çıktı ===
out_cols = hier_cols + ['week_start_date', 'predicted_quantity']

future_df[out_cols].to_csv(OUTPUT_PATH, index=False,  decimal=",", sep=";")



