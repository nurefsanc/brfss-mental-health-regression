# Büyük Veri Analitiği ile Ruh Sağlığı ve Yaşam Tarzı Korelasyonu
**Kocaeli Üniversitesi - Makine Öğrenmesinin İlkeleri Dersi Dönem Sonu Projesi**

Bu proje, CDC tarafından yayınlanan 2024 yılı Behavioral Risk Factor Surveillance System (BRFSS) veri seti üzerinde yürütülmüş bir makine öğrenmesi çalışmasıdır. Projenin temel amacı; fiziksel egzersiz, sigara kullanımı ve alkol tüketimi gibi yaşam tarzı faktörlerinin, bireylerin ruh sağlığı üzerindeki doğrusal etkilerini modellemektir.

## Kullanılan Teknolojiler ve Yöntemler
* **Veri Ön İşleme:** Pandas, NumPy
* **Eksik Veri Yönetimi (Imputation):** K-En Yakın Komşuluk (KNN) Algoritması (Scikit-Learn)
* **Keşifsel Veri Analizi (EDA):** Seaborn, Matplotlib (Korelasyon Isı Haritaları)
* **Makine Öğrenmesi Modeli:** Çoklu Lineer Regresyon (Scikit-Learn)

## Model Performansı
* **Ortalama Kare Hata (MSE):** 67.59
* **R-Kare (Açıklanan Varyans):** %1.8 
*(Not: Model performansı ve insan psikolojisinin non-lineer yapısı `docs/` klasöründeki akademik raporda detaylıca tartışılmıştır.)*

## Klasör Mimarisi
* `src/`: Veri temizleme, KNN imputasyonu ve model eğitimini içeren ana Jupyter Notebook dosyası.
* `docs/`: Projenin APA 7 standartlarında yazılmış resmi akademik makalesi ve Sorumlu Yapay Zeka etik beyanı.
* `data/`: Kullanılan veri setinin Kaggle erişim bağlantısı.
