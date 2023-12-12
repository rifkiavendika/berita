import streamlit as st
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim import corpora
from gensim.models import LdaModel
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/rifkiavendika/PPW/main/all_kategori.csv'
data_berita = pd.read_csv(url)

import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Fungsi untuk melakukan Preprocessing dengan Porter Stemmer
def preprocess(df):
    if 'Isi Berita' in df.columns:
        # Character Cleansing
        df['CleansedText'] = df['Isi Berita'].str.replace('[^\w\s]', '', regex=True)
        
        # Stopword Removal
        stop_words = set(stopwords.words('indonesian')) # Ganti dengan bahasa sesuai kebutuhan
        df['StopWord'] = df['CleansedText'].apply(lambda x: [word for word in word_tokenize(str(x)) if word.lower() not in stop_words])
        
        # Stemming with Porter Stemmer
        porter = PorterStemmer()
        df['Stemmed'] = df['StopWord'].apply(lambda x: [porter.stem(word) for word in x])
        
        return df
    else:
        st.write("Kolom 'Isi Berita' tidak ditemukan dalam DataFrame.")
        return None

# Fungsi untuk Word2Vec dan vektor rata-rata dokumen
def get_document_vectors(df):
    if 'Stemmed' in df.columns:
        model = Word2Vec(df['Stemmed'], vector_size=100, window=5, min_count=1, sg=1)
        document_vectors = []
        for words in df['Stemmed']:
            word_vectors = [model.wv[word] for word in words if word in model.wv]
            if len(word_vectors) > 0:
                document_vectors.append(sum(word_vectors) / len(word_vectors))
            else:
                document_vectors.append([0] * model.vector_size)
        
        return pd.DataFrame(document_vectors)
    else:
        st.write("Kolom 'Stemmed' tidak ditemukan dalam DataFrame.")
        return None

# Fungsi untuk Count Vectorizer
def get_count_vectorizer(df):
    if 'Stemmed' in df.columns:
        corpus = df['Stemmed'].apply(lambda x: ' '.join(x))
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus)
        return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    else:
        st.write("Kolom 'Stemmed' tidak ditemukan dalam DataFrame.")
        return None

# Fungsi untuk TF-IDF Vectorizer
def get_tfidf_vectorizer(df):
    if 'Stemmed' in df.columns:
        corpus = df['Stemmed'].apply(lambda x: ' '.join(x))
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    else:
        st.write("Kolom 'Stemmed' tidak ditemukan dalam DataFrame.")
        return None
    
#fungsi LDA
def lda_topic_modeling(df):
    if 'Stemmed' in df.columns:
        dictionary = corpora.Dictionary(df['Stemmed'])
        corpus = [dictionary.doc2bow(text) for text in df['Stemmed']]
        lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
        return lda_model
    else:
        st.write("Kolom 'Stemmed' tidak ditemukan dalam DataFrame.")
        return None

# Fungsi untuk menampilkan hasil LDA dalam bentuk dataframe
def display_lda_results(lda_model):
    data = {'Topik': [], 'Kata-kunci': []}
    for index, topic in lda_model.show_topics(formatted=False):
        words = [word for word, _ in topic]
        data['Topik'].append(index)
        data['Kata-kunci'].append(', '.join(words))

    df_lda_results = pd.DataFrame(data)
    return df_lda_results

# Fungsi untuk menambahkan ringkasan pada kolom Isi Berita
def add_summary(df):
    if 'Isi Berita' in df.columns:
        df['Ringkasan'] = df['Isi Berita'].apply(lambda x: ' '.join(word_tokenize(str(x))[:50]))  # Misalnya, diambil 50 token pertama sebagai ringkasan
        return df
    else:
        st.write("Kolom 'Isi Berita' tidak ditemukan dalam DataFrame.")
        return None

from sklearn.metrics.pairwise import cosine_similarity

# Fungsi untuk menghitung cosine similarity
def compute_cosine_similarity(df):
    if 'Stemmed' in df.columns:
        model = Word2Vec(df['Stemmed'], vector_size=100, window=5, min_count=1, sg=1)
        document_vectors = []
        for words in df['Stemmed']:
            word_vectors = [model.wv[word] for word in words if word in model.wv]
            if len(word_vectors) > 0:
                document_vectors.append(sum(word_vectors) / len(word_vectors))
            else:
                document_vectors.append([0] * model.vector_size)
        
        similarity_matrix = cosine_similarity(document_vectors)
        return similarity_matrix
    else:
        st.write("Kolom 'Stemmed' tidak ditemukan dalam DataFrame.")
        return None
    
import networkx as nx

# Fungsi untuk menghitung closeness centrality
def compute_closeness_centrality(similarity_matrix):
    if similarity_matrix is not None:
        G = nx.from_numpy_array(similarity_matrix)
        closeness_cent = nx.closeness_centrality(G)
        return closeness_cent
    else:
        st.write("Matrix kesamaan tidak tersedia.")
        return None

# Fungsi untuk menampilkan nilai PageRank
def display_pagerank(G):
    if G is not None:  # Periksa apakah G telah didefinisikan
        pagerank = nx.pagerank(G)
        pagerank_df = pd.DataFrame(list(pagerank.items()), columns=['Node', 'PageRank'])
        return pagerank_df
    else:
        st.write("Grafik belum dibuat. Silakan pilih kategori dan proses data terlebih dahulu di tab 'Data Preprocessing' dan 'Grafik'.")


st.title("DATA BERITA DETIK")
st.write("RIFKI AVENDIKA | 170411100030")


# Membuat tab
tabs = st.tabs(["Data Preprocessing", "Word2vec", "Count Vectorizer", "TF-IDF", "LDA", "Ringkasan Dokumen", "Cosine Similarity", "Closeness Centrality", "Graph", "PageRank"])

# Membuat tab pertama: Data Preprocessing
with tabs[0]:
    st.title('Pilih Kategori')
    kategori_pilihan = st.selectbox('Pilih Kategori Berita', data_berita['Kategori'].unique())
    
    if st.button('Proses'):
        df = data_berita[data_berita['Kategori'] == kategori_pilihan].head(100)
        df_preprocessed = preprocess(df)
        if df_preprocessed is not None:
            st.subheader(f"Preprocessing untuk Kategori: {kategori_pilihan}")
            st.dataframe(df_preprocessed)

# Membuat tab kedua: Word2Vec
with tabs[1]:
    st.title('Word2Vec')
    if 'df_preprocessed' in locals():
        document_vectors = get_document_vectors(df_preprocessed)
        if document_vectors is not None:
            st.subheader("Vektor Rata-rata Dokumen")
            st.dataframe(document_vectors)
    else:
        st.write("Data belum diproses. Silakan pilih kategori dan proses data terlebih dahulu di tab 'Data Preprocessing'.")

# Membuat tab ketiga: Count Vectorizer
with tabs[2]:
    st.title('Count Vectorizer')
    if 'df_preprocessed' in locals():
        count_vectorizer = get_count_vectorizer(df_preprocessed)
        if count_vectorizer is not None:
            st.subheader("Hasil Count Vectorizer")
            st.dataframe(count_vectorizer)
    else:
        st.write("Data belum diproses. Silakan pilih kategori dan proses data terlebih dahulu di tab 'Data Preprocessing'.")

# Membuat tab keempat: TF-IDF
with tabs[3]:
    st.title('TF-IDF')
    if 'df_preprocessed' in locals():
        tfidf_vectorizer = get_tfidf_vectorizer(df_preprocessed)
        if tfidf_vectorizer is not None:
            st.subheader("Hasil TF-IDF Vectorizer")
            st.dataframe(tfidf_vectorizer)
    else:
        st.write("Data belum diproses. Silakan pilih kategori dan proses data terlebih dahulu di tab 'Data Preprocessing'.")

# Membuat tab keenam: LDA
with tabs[4]:
    st.title('Pemodelan Topik dengan LDA')
    if 'df_preprocessed' in locals():
        lda_model = lda_topic_modeling(df_preprocessed)  # Memanggil fungsi untuk membuat model LDA
        df_lda = display_lda_results(lda_model)
        st.subheader("Hasil Pemodelan Topik dengan LDA")
        st.dataframe(df_lda)
    else:
        st.write("Pemodelan LDA belum dilakukan. Silakan proses data terlebih dahulu di tab 'Preprocessing' dan 'Word2Vec'.")

# Membuat tab keenam: Ringkasan Dokumen
with tabs[5]:
    st.title('Ringkasan Dokumen')
    if 'df_preprocessed' in locals():
        df_with_summary = add_summary(df_preprocessed)
        if df_with_summary is not None:
            st.subheader("Ringkasan dari Dokumen")
            st.dataframe(df_with_summary[['Isi Berita', 'Ringkasan']])
    else:
        st.write("Data belum diproses. Silakan pilih kategori dan proses data terlebih dahulu di tab 'Data Preprocessing'.")

# Membuat tab baru untuk menampilkan cosine similarity
with tabs[6]:
    st.title('Cosine Similarity')
    if 'df_preprocessed' in locals():
        similarity_matrix = compute_cosine_similarity(df_preprocessed)
        if similarity_matrix is not None:
            st.subheader("Matrix Similarity antar Dokumen")
            st.dataframe(pd.DataFrame(similarity_matrix))
    else:
        st.write("Data belum diproses. Silakan pilih kategori dan proses data terlebih dahulu di tab 'Data Preprocessing'.")

# Membuat tab baru untuk menampilkan closeness centrality
with tabs[7]:
    st.title('Closeness Centrality')
    if 'similarity_matrix' in locals():
        closeness_centrality = compute_closeness_centrality(similarity_matrix)
        if closeness_centrality is not None:
            st.subheader("Closeness Centrality")
            st.dataframe(pd.Series(closeness_centrality, name='Closeness Centrality'))
    else:
        st.write("Matrix kesamaan tidak tersedia. Silakan hitung similarity matrix terlebih dahulu di tab 'Cosine Similarity'.")

# ... (Kode sebelumnya) ...

# Fungsi display_graph diperbarui agar dapat digunakan dalam tab yang berbeda
def display_graph(df):
    if 'Stemmed' in df.columns:
        G = nx.Graph()
        # Tambahkan node
        for index, row in df.iterrows():
            G.add_node(index)
        
        # Tambahkan edge
        # Misalnya, Anda ingin menghubungkan node berdasarkan kesamaan kata kunci
        for i in range(len(df)):
            for j in range(i+1, len(df)):
                common_words = len(set(df.iloc[i]['Stemmed']).intersection(set(df.iloc[j]['Stemmed'])))
                if common_words > 0:
                    G.add_edge(i, j, weight=common_words)
        
        return G
    else:
        st.write("Data belum diproses. Silakan pilih kategori dan proses data terlebih dahulu di tab 'Data Preprocessing'.")

# Membuat tab baru untuk menampilkan grafik
with tabs[8]:
    st.title('Grafik')
    if 'df_preprocessed' in locals():
        G = display_graph(df_preprocessed)
        if G is not None:
            # Tampilkan grafik
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, font_size=8, alpha=0.8)
            
            # Simpan grafik sebagai file gambar
            temp_file_path = "temp_graph.png"
            plt.savefig(temp_file_path)
            st.image(temp_file_path)
    else:
        st.write("Data belum diproses. Silakan pilih kategori dan proses data terlebih dahulu di tab 'Data Preprocessing'.")


# Membuat tab kedelapan: PageRank
with tabs[9]:
    st.title('PageRank')
    if 'G' in locals():
        pagerank_df = display_pagerank(G)
        if pagerank_df is not None:
            st.subheader("Nilai PageRank untuk Setiap Dokumen")
            st.dataframe(pagerank_df)
    else:
        st.write("Grafik belum dibuat. Silakan pilih kategori dan proses data terlebih dahulu di tab 'Data Preprocessing' dan 'Grafik'.")