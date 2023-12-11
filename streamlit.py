import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from streamlit_option_menu import option_menu

# with st.sidebar:
#     selected = option_menu(
#         menu_title="Main Menu",
#         options=["Crawling Data", "Normalisasi Data", "LDA"],
#         default_index=0
#     )
selected = st.selectbox("Select an option", [
                        "Crawling Data", "Normalisasi Data", "LDA", "KMeans"])

countvectn = pd.core.frame.DataFrame
if selected == "Crawling Data":
    if st.button('Mulai crawling'):
        def crawl_pta(url):
            result = []
            req = requests.get(url)
            soup = BeautifulSoup(req.text, "lxml")

            news_links = soup.find_all("li", {'data-id': 'id-1'})
            # #looping through article link
            for idx, news in enumerate(news_links):
                penulis = ""
                dosen1 = ''
                dosen2 = ''
                news_dict = {}

                # #find urll news
                url_news = news.find('a', {'class': 'gray button'}).get('href')

                # find news content in url
                req_news = requests.get(url_news)
                soup_news = BeautifulSoup(req_news.text, "lxml")

                # Judul Jurnal
                judul = soup_news.find('a', {'class': 'title'}).text
                # print("title:", title)

                # Penulis, dospem I, dospem II
                dt = soup_news.find_all(
                    'div', {'style': 'padding:2px 2px 2px 2px;'})
                penulis = dt[0].text.split(':')[1]
                dosen1 = dt[1].text.split(':')[1]
                dosen2 = dt[2].text.split(':')[1]

                # Abstrak
                abstrak = soup_news.find('p', {'align': 'justify'}).text

                # wrap in dictionary
                news_dict['Judul'] = judul
                news_dict['Penulis'] = penulis
                news_dict['Dosen Pembimbing I'] = dosen1
                news_dict['Dosen Pembimbing II'] = dosen2
                news_dict['Abstrak'] = abstrak
                result.append(news_dict)

            return result

        # Set judul halaman
        st.title('Crawling PTA Trunojoyo')

        # URL target untuk di-crawl
        url = 'https://pta.trunojoyo.ac.id/c_search/byprod/10/'

        # Mendapatkan data dari web scraping
        data = []
        for i in range(1, 170):  # Ubah batas range sesuai kebutuhan
            cr = crawl_pta(f'{url}/{i}')
            data += cr

        # Menampilkan hasil dalam DataFrame
        df = pd.DataFrame(data)

        # Menampilkan data frame
        st.write('Data Hasil Crawling:')
        st.dataframe(df)

        # Simpan dataframe ke dalam file CSV
        st.write('Simpan Data ke dalam File CSV:')

        df.to_csv('crawling_pta.csv', index=False)
        st.success('Data berhasil disimpan ke dalam file CSV.')
    # uploaded_file = st.file_uploader("Unggah file dataset (CSV)", type=["csv"])

if selected == "Normalisasi Data":
    import csv
    import string
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')

    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    import re
    file_name = 'crawling_pta.csv'

    def open_csv(file_name):
        result = []
        # Membuka file CSV untuk dibaca
        with open(file_name, mode='r', newline='') as file:

            # Membaca file CSV menggunakan modul csv.reader
            csv_reader = csv.reader(file)
            # Menggunakan perulangan untuk membaca baris-baris dalam file CSV
            for row in csv_reader:
                if row[0] == '':
                    continue
                else:
                    result.append(row)
        return result

    def normalize_text(data):
        result = []
        datahasil = {}
        for i in range(0, len(data)-1):
            text = str(data[i][5])
            text = text.lower()

            punctuation_set = set(string.punctuation)

            # Menghapus tanda baca dari teks
            text_without_punctuation = ''.join(
                char for char in text if char not in punctuation_set)
            # Membuat tabel translasi untuk menghapus angka
            # translation_table = str.maketrans("", "", string.digits)

            # # Menggunakan translate() untuk menghapus angka
            # result_string = text_without_punctuation.translate(translation_table)
            # clenasing mention
            clean_tag = re.sub("@[A-Za-z0-9_]+", "", text_without_punctuation)
            clean_hashtag = re.sub(
                "#[A-Za-z0-9_]+", "", clean_tag)  # clenasing hashtag
            # cleansing url link
            clean_https = re.sub(r'http\S+', '', clean_hashtag)
            # cleansing character
            clean_symbols = re.sub("[^a-zA-Zï ]+", " ", clean_https)

            # stopwords
            stop_words = set(stopwords.words('indonesian'))
            words = clean_symbols.split()
            filtered_words = [
                word for word in words if word.lower() not in stop_words]
            filtered_text = ' '.join(filtered_words)

            # factory = StemmerFactory()
            # stemmer = factory.create_stemmer()

            # steamwords = filtered_text.split()
            # for i in range (len(steamwords)):
            #   stem = stemmer.stem(steamwords[i])
            #   steaming = ''.join(stem)

            # tokenize
            tokens = word_tokenize(filtered_text)
            # print(tokens)
            result.append(tokens)
        return text, result

    data_file = open_csv(file_name)
    text, result = normalize_text(data_file)
    print(text, result)
    st.write('''Hasil normalisasi''')
    df = pd.DataFrame(result)
    df

    # ----- Join kata abstrak ------
    gabung = []
    for i in range(len(result)):
        joinkata = ' '.join(result[i])
        gabung.append(joinkata)

    hasil = pd.DataFrame(gabung, columns=['Abstrak'])

    hasil.to_csv('normalisasi.csv')

    # ------ countvector dan tfidf
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    import pandas as pd

    countvectorizer = CountVectorizer(analyzer='word', stop_words='english')
    tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
    count_wm = countvectorizer.fit_transform(gabung)
    tfidf_wm = tfidfvectorizer.fit_transform(gabung)

    count_tokens = countvectorizer.get_feature_names_out()
    tfidf_tokens = tfidfvectorizer.get_feature_names_out()
    df_countvect = pd.DataFrame(data=count_wm.toarray(), columns=count_tokens)
    # ambil data crawling
    judul = pd.read_csv("crawling_pta.csv")
    # ambil kolom judul
    ab = judul["Judul"]
    # inster ke countvect
    df_countvect.insert(0, 'Judul', ab)
    df_countvect.to_csv('countvect.csv')
    df_tfidfvect = pd.DataFrame(data=tfidf_wm.toarray(), columns=tfidf_tokens)
    df_tfidfvect.insert(0, 'Judul', ab)
    df_tfidfvect.to_csv('tfidfvect.csv')
    print("Count Vectorizer\n")
    st.write('''TFIDF Vectorizer''')
    df_tfidfvect

    countv = pd.read_csv("countvect.csv")
    st.write('''Count Vectorizer''')
    showcountv = countv.iloc[:, 1:]
    showcountv


if selected == "LDA":
    import numpy as np
    import sklearn
    from sklearn.decomposition import LatentDirichletAllocation
    import pandas as pd
    import os
    termFreq = pd.read_csv("countvect.csv")
    judul = termFreq["judul"]
    # mengambil kolom selain kolom pertama untuk mendapatkan kata dari abstrak
    termFreq_proses = termFreq.iloc[:, 2:]
    termFreq_proses

    k = 3
    alpha = 0.3
    beta = 0.3

    lda = LatentDirichletAllocation(
        n_components=k, doc_topic_prior=alpha, topic_word_prior=beta)
    lda.fit(termFreq_proses)
    proporsi_topik_dokumen = lda.transform(termFreq_proses)
    proporsi_topik_dokumen_df = pd.DataFrame(proporsi_topik_dokumen, columns=[
                                             'Topik 1', 'Topik 2', 'Topik 3'])
    # membaca abstrak dan insert abstrak pada kolom pertama
    # abstrakread = pd.read_csv("normalisasi.csv")
    # getabstrak = abstrakread["Abstrak"]
    proporsi_topik_dokumen_df.insert(0, 'Judul', judul)
    proporsi_topik_dokumen_df.to_csv('proporsi_topik_dokumen.csv')
    proporsi_topik_dokumen_df
    st.success('Data berhasil disimpan ke dalam file CSV.')

    # Membaca dataset frekuensi kata-kata
    termFreq = pd.read_csv("countvect.csv")
    termFreq_proses = termFreq.iloc[:, 2:]

    # Inisialisasi parameter LDA
    k = 3
    alpha = 0.3
    beta = 0.3

    # Membuat dan melatih model LDA
    lda = LatentDirichletAllocation(
        n_components=k, doc_topic_prior=alpha, topic_word_prior=beta)
    lda.fit(termFreq_proses)

    # Mendapatkan distribusi kata dalam setiap topik
    distribusi_kata_topik = lda.components_ / \
        lda.components_.sum(axis=1)[:, np.newaxis]

    # Menampilkan distribusi kata dalam setiap topik
    distribusi_kata_topik_df = pd.DataFrame(
        distribusi_kata_topik, columns=termFreq_proses.columns)
    distribusi_kata_topik_df.to_csv("distribusi_kata_topik.csv")
    distribusi_kata_topik_df
if selected == 'KMeans':
    from sklearn.cluster import KMeans
    from sklearn.cluster import KMeans
    from sklearn import datasets
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    termFreq = pd.read_csv("proporsi_topik_dokumen.csv")
    # iris = datasets.load_iris()
    X = termFreq.iloc[:, 2:]
    n_clusters = 3  # Ubah sesuai kebutuhan Anda
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    X['cluster'] = kmeans.labels_
