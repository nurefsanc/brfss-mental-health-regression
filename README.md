# CDC BRFSS 2024 Veri Seti Üzerinde Ruh Sağlığı ve Yaşam Tarzı Korelasyonunun Makine Öğrenmesi Yaklaşımlarıyla Analizi

Bu proje, Kocaeli Üniversitesi "Makine Öğrenmesinin İlkeleri" dersi kapsamında geliştirilmiştir. Çalışmanın temel amacı, Centers for Disease Control and Prevention (CDC) tarafından yayımlanan Behavioral Risk Factor Surveillance System (BRFSS) 2024 veri seti kullanılarak bireylerin ruh sağlığı durumu ile yaşam tarzı faktörleri arasındaki doğrusal ilişkiyi modellemektir. 

Model, bireylerin "kötü ruh sağlığı yaşadığı gün sayısını" tahmin edebilen, hiperparametreleri optimize edilmiş düzenlileştirilmiş bir Ridge Regresyon (L2) algoritmasına dayanmaktadır.

## 👥 Geliştiriciler
* **Nurefşan Çapoğlu** (Öğrenci No: 250121017)
* **Amine Andıç** (Öğrenci No: 250121027)
* **Kurum:** Kocaeli Üniversitesi, Yapay Zeka ve Makine Öğrenmesi

---

## 📊 Veri Seti ve Değişkenler
Projede 457.670 gözlem içeren geniş ölçekli BRFSS 2024 veri seti kullanılmıştır. Odak hedefler doğrultusunda analiz, aşağıdaki 6 temel değişken izole edilerek gerçekleştirilmiştir:

* **MENTHLTH (Bağımlı Değişken - Hedef):** Son 30 gün içinde ruh sağlığının kötü olduğu gün sayısı (0-30 arası sürekli değişken).
* **EXERANY2:** Fiziksel aktivite / egzersiz yapma durumu.
* **SMOKE100:** Hayat boyunca en az 100 adet sigara içme durumu.
* **ALCDAY4:** Son 30 gün içindeki alkol tüketim sıklığı.
* **EMTSUPRT:** İhtiyaç duyulan sosyal ve duygusal desteği alma sıklığı (Likert Tipi).
* **MARITAL:** Yasal medeni durum.

---

## 🛠 Metodoloji ve Veri Ön İşleme (Pipeline)
Makine öğrenmesi süreçleri, algoritmik yanlılığı ve veri sızıntısını engellemek adına katı bir protokol ile yürütülmüştür.

1. **Veri Temizleme:** Anket metodolojisindeki geçersiz yanıtlar (7, 9, 77, vb.) standart `NaN` formatına dönüştürülmüştür. 
2. **Bağımlı Değişken Eliminasyonu:** Algoritmik uydurmayı (data distortion) önlemek için bağımlı değişken olan `MENTHLTH` sütununda eksik veri barındıran tüm satırlar (8.156 gözlem) elimine edilmiştir.
3. **Veri Sızıntısının (Data Leakage) Önlenmesi:** Veri seti, eksik değerler doldurulmadan önce `%80 Eğitim` ve `%20 Test` olarak ikiye ayrılmıştır.
4. **K-En Yakın Komşuluk (KNN) İmputasyonu:** Özellikle `%56.16` oranında eksik veri barındıran `EMTSUPRT` değişkeni için k=5 komşuluk değeriyle mesafe tabanlı imputasyon yapılmıştır. Bu işlem sadece eğitim setinden öğrenilip test setine uygulanmıştır.
5. **Öznitelik Ölçeklendirme:** Değişken büyüklüklerinin model ağırlıklarını yapay olarak domine etmesini engellemek için tüm özelliklere z-skoru tabanlı `StandardScaler` uygulanmıştır.

---

## ⚙️ Modellemeler: Ridge Regresyonu
Aşırı uyumu (overfitting) engellemek ve katsayıları küçültmek amacıyla doğrusal bir model olan **Ridge Regresyonu** (L2 Regularization) tercih edilmiştir. Model, varyans-yanlılık dengesini bulmak için 5 Katlı Çapraz Doğrulama (GridSearchCV) ile test edilmiştir. 

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \alpha \sum_{j=1}^{n} \theta_j^2$$

* **Optimize Edilen Alpha Değeri:** 100
* **Ortalama Kare Hata (MSE):** 65.5498
* **Açıklanan Varyans Oranı (R²):** 0.0566

---

## 📈 Temel Bulgular ve XAI Yorumu
Açıklanabilir Yapay Zeka (Explainable AI - XAI) prensiplerine uygun olarak model katsayıları şeffaf bir şekilde analiz edilmiştir:

* **Duygusal Destek (EMTSUPRT - 1.2143):** Bireylerin kötü ruh sağlığı gün sayısını en güçlü şekilde artıran (risk faktörü) değişken, sosyal ve duygusal destekten yoksunluktur.
* **Fiziksel Egzersiz (EXERANY2 - 0.7211):** Fiziksel aktivite eksikliği ikinci en büyük risk faktörüdür.
* **Alkol Kullanımı (ALCDAY4 - 0.0401):** Bu model kapsamında ruh sağlığı üzerinde en düşük istatistiksel etkiye sahip değişken olarak gözlemlenmiştir.

---

## ⚖️ Sorumlu Yapay Zeka (Responsible AI) Bildirimi
Bu projede etik algoritma tasarımı ilkeleri gözetilmiştir:
* **Duyarlı Özellik İzolasyonu:** Yaş, Irk ve Cinsiyet gibi demografik etiketler (sensitive features) kasıtlı olarak model dışında bırakılmış, model kararları bireylerin dinamik yaşam tarzlarına bağlanmıştır.
* **Toplumsal Damgalama:** Model bulguları (örneğin bekar olmanın negatif etkisi), sigorta veya işveren kararları için ayrımcılık aracı olarak kullanılmamalıdır. Korelasyon, nedensellik (causation) anlamına gelmez.
* **Azınlık Temsili:** KNN Imputer kullanımı ile alt sosyodemografik grupların veri içinde asimile olması engellenmiştir.

---

## 🚀 Kurulum ve Kullanım

### Gereksinimler
Projeyi yerel makinenizde çalıştırmak için aşağıdaki kütüphanelere ihtiyacınız vardır:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib

Projeyi Çalıştırma
Bu depoyu klonlayın:

Bash
git clone [https://github.com/nurefsanc/brfss-mental-health-regression.git](https://github.com/nurefsanc/brfss-mental-health-regression.git)
cd brfss-mental-health-regression
CDC veri setini klasör içerisine yerleştirin.

Python analiz dosyanız üzerinden hücreleri sırasıyla çalıştırarak tüm eğitim ve değerlendirme sürecini yeniden üretebilirsiniz.
