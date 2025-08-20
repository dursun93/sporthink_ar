# Airflow DAG Konfigürasyonu

## DAG Yapısı

### 1. df_data_prep_dag
- **Çalışma Zamanı**: Her Pazar sabah 10:00 TRT (07:00 UTC)
- **Cron Expression**: `0 7 * * 0`
- **Görevler**:
  - `run_data_prep`: Veri hazırlama işlemi
  - `trigger_feature_eng`: Feature engineering DAG'ını tetikleme

### 2. df_feature_eng_dag
- **Çalışma Zamanı**: Sadece data_prep DAG'ı tarafından tetiklenir
- **Schedule**: `None` (manuel tetikleme)
- **Görevler**:
  - `run_feature_eng`: Feature engineering işlemi

## Dosya Yolları

### Giriş Dosyaları (input klasörü öncelikli)
- `demand_forecasting/input/` (öncelikli)
- `demand_forecasting/input_parquet/` (yedek)

### Çıkış Dosyaları
- `demand_forecasting/output/weekly_sales_with_cluster_125.parquet`
- `demand_forecasting/output/weekly_sales_with_features.parquet`

## Gerekli Dosyalar

### Data Prep için:
- `dim_product.parquet`
- `history_sales.parquet`
- `k_means_store.parquet`

### Feature Engineering için:
- `dim_calendar_pivot.parquet`
- Data prep çıktısı

## Sorun Giderme

### Feature Engineering DAG "Running" Durumunda Kalırsa:

1. **Logları Kontrol Et**: Airflow UI'da task loglarını incele
2. **Timeout Ayarları**: 
   - DAG timeout: 3 saat
   - Task timeout: 2 saat
3. **Orijinal Feature Sayısı Korundu**: 
   - Shifts: 14 (orijinal)
   - Rolls: 8 (orijinal)
   - Diffs: 6 (orijinal)

### Olası Sorunlar:
- GCS bağlantı sorunu
- Büyük veri seti işleme
- Bellek yetersizliği
- Parquet engine sorunu
- Trigger mekanizması sorunu

## Test Etme

```bash
# DAG'ları test et
python test_dags.py

# Feature engineering'i debug et
python debug_feature_eng.py
```

## Güncellemeler

- ✅ Input klasörü öncelikli hale getirildi
- ✅ Orijinal feature engineering parametreleri korundu
- ✅ Timeout ayarları artırıldı
- ✅ Hata yönetimi iyileştirildi
- ✅ Logging geliştirildi
