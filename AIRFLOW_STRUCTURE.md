# Airflow Yapısı - Tek DAG Yaklaşımı

## Dosya Yapısı

```
airflow/
├── df_sporthink_dag.py          # Ana DAG dosyası
├── data_prep_step.py            # Data preparation step
└── feature_eng_step.py          # Feature engineering step
```

## GCS Yapısı

```
europe-west1-airflow-a054c263-bucket/
└── demand_forecasting/
    ├── df_steps/                # Step dosyaları burada
    │   ├── data_prep_step.py
    │   └── feature_eng_step.py
    ├── input/                   # Giriş dosyaları (öncelikli)
    ├── input_parquet/           # Yedek giriş klasörü
    └── output/                  # Çıkış dosyaları
```

## DAG Konfigürasyonu

### demand_forecast_dag
- **Çalışma Zamanı**: Her Pazar sabah 10:00 TRT (07:00 UTC)
- **Cron Expression**: `0 7 * * 0`
- **Görevler**:
  1. `data_prep` - Veri hazırlama
  2. `feature_eng` - Feature engineering

### Task Bağımlılığı
```
data_prep >> feature_eng
```

## Avantajlar

1. **Tek DAG**: İki ayrı DAG yerine tek DAG
2. **Basit Tetikleme**: Trigger mekanizması yok, sıralı çalışma
3. **Kolay Debug**: Tek DAG'da tüm işlemler
4. **Daha Az Karmaşıklık**: Trigger timeout sorunları yok

## Gerekli Dosyalar

### Input Klasöründe:
- `dim_product.parquet`
- `history_sales.parquet`
- `k_means_store.parquet`
- `dim_calendar_pivot.parquet`

### Output Klasöründe:
- `weekly_sales_with_cluster_125.parquet`
- `weekly_sales_with_features.parquet`

## Deployment

1. Step dosyalarını GCS'ye yükle:
   ```
   gs://europe-west1-airflow-a054c263-bucket/demand_forecasting/df_steps/
   ```

2. DAG dosyasını Airflow'a yükle:
   ```
   airflow/df_sporthink_dag.py
   ```

## Sorun Giderme

- **Tek DAG**: İki task da aynı DAG'da olduğu için takılma sorunu olmaz
- **Sıralı Çalışma**: data_prep bitmeden feature_eng başlamaz
- **Basit Logging**: Tek DAG'da tüm logları görebilirsiniz
