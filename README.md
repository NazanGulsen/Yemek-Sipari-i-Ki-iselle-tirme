Kişiselleştirilmiş Yemek Siparişi Öneri Sistemi Projesi

Bu projede, kullanıcıların yemek sipariş geçmişlerini analiz ederek kişiselleştirilmiş yemek önerileri sunmayı amaçlayan bir doğal dil işleme (NLP) ve veri madenciliği tabanlı sistem geliştirilmiştir. Bu sistem, kullanıcıların ürün tercihleri, sipariş saatleri, harcama alışkanlıkları gibi veriler üzerinden çalışmaktadır.

-Veri Hazırlama ve Önişleme

1.1. Veri Toplama

Zomato gibi yemek sipariş platformlarından alınan sipariş geçmişi verileri kullanılmıştır. Bu verilerde aşağıdaki sütunlara odaklanılmıştır:

Customer ID

Items in order

Order Placed At

Restaurant name

Rating

Review

Veri dosyası birleştirilerek siparis_verisi.csv adında yeni bir CSV dosyası oluşturulmuştur. Bu dosyada, her satır bir sipariş ve içeriğini temsil etmektedir. Bu aşamadaki işlemler "veri_birlestirme.ipynb" dosyasında gerçekleştirilmiştir.

1.2. Veri Ön İşleme

Sipariş içerikleri ve kullanıcı yorumları üzerinde aşağıdaki ön işleme adımları uygulanmıştır:

Küçük harfe çevirme

Noktalama işaretlerinin kaldırılması

İngilizce stopword (gereksiz kelimeler) temizliği

Tokenizasyon (metni kelimelere ayırma)

Lemmatizasyon

Stemming

Bu işlemler sonucunda elde edilen temiz veri, analiz için uygun hale getirilmiştir.

🔍 Proje Özeti

CSV dosyasından alınan sipariş verileri şu adımlardan geçirilmiştir:

pandas ile veri yükleme ve genel inceleme

Eksik verilerin tespiti ve temizlenmesi

Sipariş içeriklerinin kelime seviyesinde ayrıştırılması

Review alanlarının NLP analizine hazırlanması

Order Placed At sütunundan saat ve gün gibi zaman bilgileri çıkarılmıştır

Items in order sütunundan ürün türleri ayrıştırılarak kullanıcı zevkleri analiz edilmiştir

🛠️ Kullanılan Teknolojiler ve Kütüphaneler

Python 3.x

Jupyter Notebook

pandas

numpy

nltk

gensim

sklearn

Kullanılan özel NLP fonksiyonları:

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer, PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity


— TF-IDF, Benzerlik Skorları ve Öneri Sistemi

2.1. TF-IDF Vektörleştirme

TF-IDF yöntemiyle, her kullanıcı yorumu ve sipariş içeriği vektörleştirilmiştir. Bu vektörler daha sonra kullanıcı zevklerini temsil edecek şekilde analiz edilmiştir.

2.2. Cosine Similarity Kullanımı

Sipariş içerikleri ve yorumlar arasındaki benzerlik cosine_similarity fonksiyonu ile hesaplanmıştır. Böylece benzer ürünleri seven kullanıcılar gruplanmış ve işbirlikçi filtreleme algoritması için temel oluşturulmuştur.

2.3. En Yüksek TF-IDF Skorlu Ürünler

Her kullanıcı için en çok tekrar eden ve en yüksek ağırlığa sahip ürünler çıkarılmıştır. Bu ürünler, kişiye özel öneri listelerinde öne çıkarılmıştır.

2.4. Kullanıcı Segmentasyonu

Kullanıcılar; sipariş sıklığı, saat aralıkları, restoran tercihleri ve harcama düzeylerine göre gruplandırılmıştır. K-Means gibi algoritmalarla segment oluşturulmuştur.

2.5. Word2Vec Modeli Eğitimi

Kullanıcı yorumları ve ürün içerikleri Word2Vec ile eğitilerek semantik olarak benzer ürünler bulunmuştur. Böylece, "burger" tercih eden bir kullanıcıya "krispy sandwich" gibi benzer öneriler yapılabilmiştir.

Eğitim sonucunda oluşturulan modeller:

w2v_model_skipgram_100dim.model

w2v_model_cbow_300dim.model

Sonuçlar ve Kullanım

Her kullanıcı için en uygun 5 ürün önerisi listelenmiştir.

En sık sipariş edilen ürünler analiz edilmiştir.

Yorumlardaki olumlu/olumsuz kelime dağılımları çıkarılmıştır.

Zaman ve ürün türüne göre tavsiyeler optimize edilmiştir.

Nasıl Çalıştırılır?

siparis_verisi.csv dosyasını uygun dizine yerleştirin

siparis_analizi.ipynb dosyasını açın ve hücreleri çalıştırın

oneri_olustur.ipynb ile öneri sisteminin çıktısını alın

