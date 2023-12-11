import streamlit as st

import pandas as pd
# from sklearn.model_selection import GridSearchCV
from streamlit_option_menu import option_menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Hasil Crawling Data", "Hasil Normalisasi Data",
                 "Countvector dan TFIDF", "Hasil LDA", 'KMeans'],
        default_index=0
    )
if selected == "Hasil Crawling Data":
    st.write('Data Hasil Crawling:')
    read_data = pd.read_csv("crawling_pta.csv")
    show_data = read_data.iloc[:, 1:]
    show_data
if selected == "Hasil Normalisasi Data":
    st.write('Data Setelah Normalisasi:')
    read_normalisasi = pd.read_csv("normalisasi.csv")
    show_normalisasi = read_normalisasi.iloc[:, 1:]
    show_normalisasi


if selected == "Countvector dan TFIDF":
    st.write('Countvector:')
    read_countvectorizer = pd.read_csv("countvect.csv")
    show_con = read_countvectorizer.iloc[:, 1:]
    show_con
    st.write('TFIDF:')
    read_tfidf = pd.read_csv("tfidfvect.csv")
    show_tfidf = read_tfidf.iloc[:, 1:]
    show_tfidf
if selected == "Hasil LDA":
    st.write('Proporsi Topik Dalam Dokumen:')
    read_proporsi_topik_dokumen = pd.read_csv("proporsi_topik_dokumen.csv")
    show_proporsi_topik_dokumen = read_proporsi_topik_dokumen.iloc[:, 1:]
    show_proporsi_topik_dokumen
    st.write('Proporsi Kata Dalam Topik:')
    read_distribusi_kata_topik = pd.read_csv("distribusi_kata_topik.csv")
    show_distribusi_kata_topik = read_distribusi_kata_topik.iloc[:, 1:]
    show_distribusi_kata_topik
if selected == 'KMeans':
    from sklearn.cluster import KMeans
    from sklearn.cluster import KMeans
    import pandas as pd
    import os
    termFreq = pd.read_csv("proporsi_topik_dokumen.csv")
    # iris = datasets.load_iris()
    X = termFreq.iloc[:, 2:]
    n_clusters = 3  # Ubah sesuai kebutuhan Anda
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    X['cluster'] = kmeans.labels_
    X
