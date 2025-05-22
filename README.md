---

# 🥗 Yemek Siparişi Kişiselleştirme Projesi

Bu projede, müşterilerin yemek siparişlerindeki "Items in order" açıklamalarını doğal dil işleme (NLP) yöntemleri kullanarak analiz ederek kişiselleştirilmiş öneriler sunmak amaçlanmıştır. Bu sayede müşterilere daha iyi bir deneyim sunulması ve sipariş süreçlerinin optimize edilmesi hedeflenmektedir.

---

## 1. Hafta — Veri Hazırlama ve Önişleme
### 1.1. Veri Toplama
İlk olarak, restoran ve yemek sipariş platformlarından müşteri sipariş verileri toplanmıştır. Bu verilerden "Items in order" sütunu ayrılarak sipariş açıklamaları elde edilmiştir.  
Ayrılan "Items in order" sütunları birleştirilerek yeni bir CSV dosyası (`birlesik_siparisler.csv`) oluşturulmuştur.  
Bu işlem, `siparis_birlestirme.ipynb` adlı Jupyter Notebook dosyasında gerçekleştirilmiştir.

### 1.2. Veri Ön İşleme
"Items in order" verileri üzerinde aşağıdaki işlemler uygulanmıştır:
- Küçük harfe çevirme
- Noktalama işaretlerinin kaldırılması
- İngilizce stopword (gereksiz kelimeler) temizliği
- Tokenizasyon (metni kelimelere ayırma)
- Lemmatizasyon (kelimelerin köklerine indirgenmesi)
- Stemming (kelime köklerini bulma)

---

## 🔍 Proje Özeti
CSV dosyasından alınan "Items in order" verileri şu adımlardan geçirilmiştir:
- Verinin `pandas` ile yüklenmesi ve genel incelemesi
- Eksik verilerin kontrolü
- Cümle ve kelime seviyesinde ayrıştırma (`tokenization`)
- İngilizce stopwords (`nltk`) ile filtreleme
- Lemmatizasyon ve stemleme işlemleriyle kelimelerin kök formlarının çıkarılması
- Sipariş listesi oluşturularak yapısal analiz yapılması
- Veri ön işleme adımları, `nltk`, `pandas` ve `re` kütüphaneleri kullanılarak Python'da uygulanmıştır.

---

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler
- Python 3.x
- Jupyter Notebook
- [NLTK - Natural Language Toolkit](https://www.nltk.org/)
- spaCy (isteğe bağlı)
- pandas
- numpy
- `gensim` (Word2Vec gibi kelime vektörü modelleri için)
- `from gensim.models import Word2Vec` (Word2Vec modeli için)
- `import pandas as pd` (Veri çerçeveleri ve CSV dosyaları için)
- `import nltk` (NLP görevleri için)
- `from nltk.tokenize import word_tokenize, sent_tokenize` (Metni kelimelere ve cümlelere ayırmak için)
- `from nltk.corpus import stopwords` (Stop kelimelerini filtrelemek için)
- `from nltk.stem import WordNetLemmatizer, PorterStemmer` (Kelime köklerini bulmak için)
- `from collections import Counter` (Kelime sıklıklarını saymak için)

---

## 2. Hafta: TF-IDF Vektörleştirme ve Word2Vec Modelleri

Bu hafta, ön işlenmiş "Items in order" verileri hem TF-IDF yöntemiyle vektörleştirilecek hem de Word2Vec modeli kullanılarak kelime vektörleri elde edilecektir.

### 2.1. TF-IDF Vektörleştirme
TF-IDF (Term Frequency-Inverse Document Frequency), sipariş açıklamalarındaki kelimelerin önemini ölçmek için kullanılan bir tekniktir. Bu adımda, her bir sipariş verisi, terim frekansları (TF) ve ters belge frekansı (IDF) kullanılarak bir vektöre dönüştürülür.  
`sklearn.feature_extraction.text` kütüphanesindeki `TfidfVectorizer` sınıfı bu dönüşümü gerçekleştirmek için kullanılır.  
Bu işlem, kod klasöründeki `TF-IDF.ipynb` dosyasında gerçekleştirilmiştir. Elde edilen bulgular dosya içinde bulunmaktadır.

### 2.2. Cosine Similarity (Kosinüs Benzerliği) Hesaplaması
TF-IDF vektörleri elde edildikten sonra, siparişler arasındaki benzerliği ölçmek için Cosine Similarity yöntemi kullanılır. Bu yöntem, iki vektör arasındaki açının kosinüsünü hesaplayarak siparişlerin ne kadar benzer olduğunu belirler.  
`sklearn.metrics.pairwise` kütüphanesindeki `cosine_similarity` fonksiyonu bu hesaplamayı yapmak için kullanılır.  
Bu işlem, kod klasöründeki `TF-IDF.ipynb` dosyasında gerçekleştirilmiştir. Elde edilen bulgular dosya içinde bulunmaktadır.

### 2.3. İlk Sipariş için En Yüksek TF-IDF Skorlu Kelimeler
TF-IDF vektörleştirme işleminden sonra, her siparişteki en önemli kelimeler belirlenir. Bu, her sipariş için en yüksek TF-IDF skoruna sahip kelimelerin bulunmasıyla yapılır.  
Bu analiz, veri setindeki siparişlerin anahtar temalarını (örneğin, sık sipariş edilen yemek türleri) anlamaya yardımcı olur.

### 2.4. Cosine Similarity Matrisi Oluşturma
Tüm siparişler arasındaki Cosine Similarity skorları bir matris içinde düzenlenir. Bu matris, hangi siparişlerin birbirine daha çok benzediğini görselleştirmeyi ve analiz etmeyi kolaylaştırır.  
Bu matris, kişiselleştirilmiş öneri sistemleri veya benzer sipariş gruplarını bulma gibi uygulamalar için temel oluşturabilir.

### 2.5. Word2Vec Modelleri Eğitimi
Word2Vec modeli, sipariş açıklamalarındaki kelimelerin anlamlarını vektörler aracılığıyla temsil etmeyi amaçlar. Bu adımda, "Items in order" verilerinden kelime vektörleri elde edilir.  
Model eğitimi için farklı parametre kombinasyonları kullanılır:
- **Model tipi**: CBOW (Continuous Bag of Words) veya Skip-gram
- **Pencere boyutu**: Bir kelimenin bağlamını oluşturan kelime sayısı (ör. 2 veya 4)
- **Vektör boyutu**: Kelimelerin temsil edileceği vektörlerin boyutu (ör. 100 veya 300)  
Model eğitimi, kod klasöründeki `word2vec.ipynb` dosyasında gerçekleştirilmiştir.  
Eğitilen modeller, kullanılan parametreleri içerecek şekilde adlandırılmış ve `model` klasörüne kaydedilmiştir (örneğin, `lemmatized_model_cbow_window2_dim100.model`).

### 2.6. Model Değerlendirmesi ve Kullanımı
Eğitilen Word2Vec modelleri, kelime benzerliği (örneğin, "pizza" ile "burger" arasındaki ilişki) ve kelime analojisi gibi görevlerde değerlendirilebilir.  
Modelin performansı ve elde edilen vektörlerin kalitesi analiz edilerek en iyi performansı gösteren modeller seçilebilir.

---

## Word2Vec Modeli

Bu proje, yemek siparişi açıklamaları üzerinde **Word2Vec** modellerini eğitmek ve kelime benzerliklerini analiz ederek kişiselleştirilmiş öneriler sunmak için tasarlanmıştır.

### Adımlar

#### 1. Gerekli Kütüphanelerin Kurulumu
- **Kullanılan Araçlar**: `gensim` (Word2Vec modeli için), `pandas` (veri işleme), `nltk` (metin işleme)
- **NLTK Paketleri**: Tokenizasyon, stopwords ve lemmatization için gerekli paketler indirilir.

#### 2. Veri Setinin Hazırlanması
- **Veri Kaynakları**:
  - `lemmatized_sentences.csv`: Kelimelerin kök hallerini içeren sipariş açıklamaları
  - `stemmed_sentences.csv`: Kelime köklerini içeren sipariş açıklamaları
- **Temizlik İşlemleri**:
  - NaN ve boş değerler temizlenir
  - Metinler özel karakterlerden arındırılır ve küçük harfe çevrilir
  - Stopwords ve tek karakterli kelimeler filtrelenir

#### 3. Veri Analizi ve Vektörleştirme
- **Model Parametreleri**:
  - **Model Türü**: CBOW veya Skip-gram
  - **Pencere Boyutu**: 2 veya 4
  - **Vektör Boyutu**: 100 veya 300
- **Eğitim**:
  - Her parametre kombinasyonu için ayrı modeller eğitilir
  - Modeller `.model` uzantısıyla kaydedilir
- **Analiz**:
  - "angara" kelimesine en benzer 3 kelime ve skorları çıkarılır
  - Veri setindeki en sık kullanılan 20 kelime listelenir (ör. "pizza", "burger", "salad")

---

### Sonuçlar
- **Kaydedilen Modeller**: `lemmatized_model_cbow_vs100_w2.model`, `stemmed_model_skipgram_vs300_w4.model` gibi isimlerle kaydedilir.
- **Örnek Çıktılar**:
  - Kelime benzerlikleri yüksek skorlarla raporlanır (örneğin, "pizza" ↔ "burger": 0.9876)
  - En sık kullanılan kelimeler "pizza", "salad", "drink" gibi tematik terimlerdir

---

### Nasıl Çalıştırılır?
1. **Veri Yollarını Güncelleyin**: CSV dosyalarının doğru konumunu belirtin.
2. **Jupyter Not Defterini Başlatın**: Tüm hücreleri sırayla çalıştırın.
3. **Sonuçları Görüntüleyin**: Modeller ve analiz çıktıları otomatik olarak oluşturulur.

---

Bu uyarlama, yemek siparişi verilerini kişiselleştirme hedefiyle orijinal projenin yapısını korurken, içeriği yemek siparişleri bağlamına uygun hale getirilmiştir. "Items in order" sütunu, sipariş açıklamalarını temsil eder ve analizler bu verilere odaklanmıştır.
