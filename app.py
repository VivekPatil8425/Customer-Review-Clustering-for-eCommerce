import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
import matplotlib.pyplot as plt

# ---- Streamlit Page Config ----
st.set_page_config(page_title="Customer Review Clustering", layout="wide")

st.title("🛒 Customer Review Clustering for eCommerce")
st.write("Upload a file of customer reviews (CSV/Excel/TXT) to see clustering & sentiment analysis.")

# ---- File Upload ----
uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx", "txt"])

if uploaded_file:
    # ---- Read File ----
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, sep="\n", header=None, names=["review"])
    
    st.write("### 📄 Data Preview")
    st.write(df.head())

    # ---- Extract Reviews ----
    reviews = df[df.columns[0]].dropna().astype(str)

    # ---- Text Vectorization ----
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(reviews)

    # ---- KMeans Clustering ----
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)

    # ---- Sentiment Analysis ----
    sentiments = []
    for review in reviews:
        score = TextBlob(review).sentiment.polarity
        if score > 0:
            sentiments.append("Positive")
        elif score < 0:
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")
    df['Sentiment'] = sentiments

    # ---- Show Results ----
    st.write("### 📊 Clustered Reviews")
    st.write(df.head(20))

    # ---- Sentiment Distribution ----
    st.write("### 😊 Sentiment Distribution")
    sentiment_counts = df['Sentiment'].value_counts()

    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
    st.pyplot(fig)

    # ---- Cluster Distribution ----
    st.write("### 🔎 Cluster Distribution")
    cluster_counts = df['Cluster'].value_counts()

    fig, ax = plt.subplots()
    cluster_counts.plot(kind='bar', ax=ax)
    st.pyplot(fig)
