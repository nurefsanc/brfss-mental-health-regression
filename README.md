# Büyük Veri Analitiği ile Ruh Sağlığı, Yaşam Tarzı ve Sosyal Destek Korelasyonu
**Kocaeli Üniversitesi - Makine Öğrenmesinin İlkeleri Dersi Dönem Sonu Projesi**

Bu proje, Centers for Disease Control and Prevention (CDC) tarafından yayınlanan 2024 yılı Behavioral Risk Factor Surveillance System (BRFSS) veri seti üzerinde yürütülmüş bir makine öğrenmesi çalışmasıdır. Projenin temel amacı; bireylerin yaşam tarzı faktörlerinin (fiziksel egzersiz, sigara kullanımı, alkol tüketimi) ve sosyal destek mekanizmalarının (duygusal destek algısı, medeni durum), ruh sağlığı üzerindeki doğrusal etkilerini modellemektir.

## 📌 Kullanılan Değişkenler (Öznitelikler)
Model, bağımlı değişken olarak `MENTHLTH` (Kötü ruh sağlığı gün sayısı) hedefini tahmin etmek üzere aşağıdaki 5 bağımsız değişken ile eğitilmiştir:
1. `EXERANY2` (Fiziksel Aktivite / Egzersiz Durumu)
2. `SMOKE100` (Kalıcı Sigara Kullanımı)
3. `ALCDAY4` (Aylık Alkol Tüketim Sıklığı)
4. `EMTSUPRT` (Duygusal ve Sosyal Destek Algısı)
5. `MARITAL` (Medeni Durum)

## 🛠️ Kullanılan Teknolojiler ve Yöntemler
* **Veri Ön İşleme:** Pandas, NumPy
* **Eksik Veri Yönetimi (Imputation):** Tüm evren (yaklaşık 450.000 satır) üzerinde K-En Yakın Komşuluk (KNN) Algoritması (Scikit-Learn)
* **Keşifsel Veri Analizi (EDA):** Seaborn, Matplotlib (Korelasyon Isı Haritaları)
* **Makine Öğrenmesi Modeli:** Çoklu Lineer Regresyon (Scikit-Learn)

## 📊 Model Performansı
Model %80 Eğitim ve %20 Test seti olarak ayrılmış olup, test seti üzerindeki genelleme başarısı aşağıdaki gibidir:
* **Ortalama Kare Hata (MSE):** 61.5351
* **R-Kare (Açıklanan Varyans):** 0.1079 


## 📂 Klasör Mimarisi
* `src/`: Veri temizleme, tüm veri seti üzerinde KNN imputasyonu ve model eğitimini içeren ana Jupyter Notebook (`.ipynb`) dosyası.
* `docs/`: Projenin APA 7 standartlarında yazılmış resmi akademik makalesi ve Sorumlu Yapay Zeka etik beyanını içeren PDF dosyası.
* `data/`: Kullanılan devasa boyutlu orijinal BRFSS 2024 veri setinin Kaggle erişim bağlantısı.
