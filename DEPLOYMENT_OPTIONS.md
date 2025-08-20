# Deployment Seçenekleri

## Seçenek 1: Basit Deployment (Önerilen)

### Dosya Yapısı
```
airflow/
├── df_sporthink_dag.py          # Ana DAG
├── data_prep_step.py            # Step dosyası
└── feature_eng_step.py          # Step dosyası
```

### Avantajlar
- ✅ Basit ve güvenilir
- ✅ Hızlı deployment
- ✅ Import sorunu yok
- ✅ Kolay debug

### Deployment
1. Tüm dosyaları Airflow DAG klasörüne kopyalayın
2. Airflow otomatik olarak DAG'ı tanıyacak

---

## Seçenek 2: GCS'den Dinamik Import

### Dosya Yapısı
```
GCS: europe-west1-airflow-a054c263-bucket/demand_forecasting/df_steps/
├── data_prep_step.py
└── feature_eng_step.py

Airflow: airflow/df_sporthink_dag_gcs.py
```

### Avantajlar
- ✅ Step dosyaları GCS'de merkezi
- ✅ Versiyon kontrolü
- ✅ Birden fazla DAG aynı step'leri kullanabilir

### Dezavantajlar
- ❌ Daha karmaşık
- ❌ GCS bağımlılığı
- ❌ Import hataları olabilir

### Deployment
1. Step dosyalarını GCS'ye yükleyin
2. `df_sporthink_dag_gcs.py` dosyasını Airflow'a yükleyin

---

## Önerilen Çözüm

**Seçenek 1'i öneriyorum** çünkü:
- Daha basit ve güvenilir
- Import sorunu yok
- Hızlı deployment
- Kolay debug

Hangi seçeneği tercih ediyorsunuz?
