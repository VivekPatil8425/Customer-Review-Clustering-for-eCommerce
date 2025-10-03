# app.py
import io
import re
import math
import json
import string
import itertools
import numpy as np
import pandas as pd
import streamlit as st

from typing import List, Tuple, Optional, Dict
from collections import Counter, defaultdict

# Optional imports handled safely
SPACY_OK = True
try:
    import spacy
except Exception:
    SPACY_OK = False

DOCX_OK = True
try:
    import docx  # python-docx
except Exception:
    DOCX_OK = False

SKLEARN_OK = True
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
except Exception:
    SKLEARN_OK = False

VADER_OK = True
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except Exception:
    VADER_OK = False

# -----------------------------
# UI: Sidebar configuration
# -----------------------------
st.set_page_config(page_title="Customer Review Clustering (Aspect-Based)", layout="wide")

st.title("Customer Review Clustering for eCommerce")
st.caption("Drag & drop TXT, CSV, Excel, or Word (.docx), then cluster reviews with optional aspect extraction and sentiment analysis.")

with st.sidebar:
    st.header("Upload files")
    uploaded_files = st.file_uploader(
        "Drop TXT / CSV / XLSX / DOCX here",
        type=["txt", "csv", "xlsx", "xls", "docx"],
        accept_multiple_files=True
    )

    st.header("Text column (for CSV/Excel)")
    default_text_cols = ["review", "text", "comment", "feedback", "reviews"]
    text_col_hint = st.text_input(
        "Preferred text column name (optional)",
        value="review",
        help="If present in CSV/Excel, this column will be used. Otherwise the first text-like column will be auto-detected."
    )

    st.header("Representation")
    use_aspects = st.checkbox(
        "Extract aspects (noun chunks) with spaCy",
        value=True,
        help="Turns reviews into aspect phrases like 'fast delivery', 'customer service'. If spaCy isn't available, the app will fall back."
    )
    ngrams = st.selectbox("N-grams for TF-IDF", options=["Unigrams (1)", "Uni+Bigrams (1,2)"], index=1)

    st.header("Clustering")
    algo = st.selectbox("Algorithm", options=["KMeans", "DBSCAN"], index=0)
    if algo == "KMeans":
        auto_k = st.checkbox("Auto-pick K (silhouette)", value=True)
        k_value = st.slider("K (clusters)", min_value=2, max_value=12, value=5)
    else:
        eps = st.slider("DBSCAN eps", min_value=0.1, max_value=2.0, step=0.1, value=0.8)
        min_samples = st.slider("DBSCAN min_samples", min_value=3, max_value=20, value=5)

    st.header("Sentiment")
    do_sentiment = st.checkbox("Compute sentiment (VADER)", value=True)

    st.header("Run")
    run_btn = st.button("Run Clustering")

# -----------------------------
# Helpers
# -----------------------------
def clean_text(t: str) -> str:
    if not isinstance(t, str):
        t = str(t)
    t = t.replace("\n", " ").replace("\r", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def load_docx(file) -> str:
    if not DOCX_OK:
        return ""
    try:
        d = docx.Document(file)
        paras = [p.text.strip() for p in d.paragraphs if p.text and p.text.strip()]
        return "\n".join(paras)
    except Exception:
        return ""

def load_text_from_any(file, preferred_col: Optional[str]) -> List[str]:
    name = file.name.lower()
    if name.endswith(".txt"):
        try:
            content = file.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="ignore")
            return [clean_text(content)]
        finally:
            file.seek(0)

    if name.endswith(".csv"):
        try:
            df = pd.read_csv(file)
        finally:
            file.seek(0)
        return extract_text_from_df(df, preferred_col)

    if name.endswith(".xlsx") or name.endswith(".xls"):
        try:
            df = pd.read_excel(file)
        finally:
            file.seek(0)
        return extract_text_from_df(df, preferred_col)

    if name.endswith(".docx"):
        txt = load_docx(file)
        return [clean_text(txt)] if txt else []

    # Fallback: try read as text
    try:
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
        return [clean_text(content)]
    except Exception:
        return []
    finally:
        file.seek(0)

def extract_text_from_df(df: pd.DataFrame, preferred_col: Optional[str]) -> List[str]:
    # Preferred column if present
    if preferred_col and preferred_col in df.columns:
        return [clean_text(x) for x in df[preferred_col].dropna().tolist() if str(x).strip()]

    # Try common text columns
    for c in ["review", "text", "comment", "feedback", "reviews"]:
        if c in df.columns:
            return [clean_text(x) for x in df[c].dropna().tolist() if str(x).strip()]

    # Otherwise, pick first object/string-like column
    for c in df.columns:
        if df[c].dtype == object:
            return [clean_text(x) for x in df[c].dropna().tolist() if str(x).strip()]

    # Last resort: flatten all columns to strings
    texts = []
    for _, row in df.iterrows():
        parts = []
        for v in row.tolist():
            s = str(v).strip()
            if s and s.lower() != "nan":
                parts.append(s)
        if parts:
            texts.append(clean_text(" ".join(parts)))
    return texts

@st.cache_resource(show_spinner=False)
def load_spacy():
    if not SPACY_OK:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None

def extract_aspects_spacy(texts: List[str]) -> List[str]:
    nlp = load_spacy()
    if not nlp:
        # Fallback: return original texts
        return texts
    aspects = []
    for doc in nlp.pipe(texts, batch_size=64, disable=["ner", "lemmatizer"]):
        noun_chunks = [chunk.text.lower().strip() for chunk in doc.noun_chunks if chunk.text.strip()]
        aspects.append(" ".join(noun_chunks) if noun_chunks else doc.text.lower())
    return aspects

def vectorize_text(corpus: List[str], ngram_mode: str):
    if not SKLEARN_OK:
        return None, None
    if ngram_mode.startswith("Uni+Bigrams"):
        ngram_range = (1, 2)
    else:
        ngram_range = (1, 1)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=ngram_range,
        min_df=2
    )
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X

def auto_pick_k(X, k_min=2, k_max=10, random_state=42):
    best_k, best_score = None, -1.0
    for k in range(k_min, k_max + 1):
        try:
            km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
            labels = km.fit_predict(X)
            # Skip degenerate cases
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels, metric="cosine")
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue
    return best_k if best_k else 5

def run_kmeans(X, k, random_state=42):
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)
    return km, labels

def run_dbscan(X, eps=0.8, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = db.fit_predict(X)
    return db, labels

def top_terms_per_cluster(vectorizer, X, labels, topn=10):
    if not SKLEARN_OK or vectorizer is None:
        return {}
    terms = np.array(vectorizer.get_feature_names_out())
    cluster_terms = {}
    unique_labels = sorted(list(set(labels)))
    for cl in unique_labels:
        if cl == -1:
            # Noise for DBSCAN
            idx = np.where(labels == cl)[0]
            if len(idx) == 0:
                continue
            # Mean TF-IDF for noise cluster
            mean_vec = X[idx].mean(axis=0).A1
            top_idx = mean_vec.argsort()[::-1][:topn]
            cluster_terms[cl] = terms[top_idx].tolist()
        else:
            idx = np.where(labels == cl)[0]
            if len(idx) == 0:
                continue
            mean_vec = X[idx].mean(axis=0).A1
            top_idx = mean_vec.argsort()[::-1][:topn]
            cluster_terms[cl] = terms[top_idx].tolist()
    return cluster_terms

def compute_sentiment(texts: List[str]) -> List[float]:
    if not VADER_OK:
        return [0.0] * len(texts)
    sia = SentimentIntensityAnalyzer()
    return [sia.polarity_scores(t)["compound"] for t in texts]

def pca_2d(X):
    if not SKLEARN_OK:
        return None
    try:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X.toarray() if hasattr(X, "toarray") else X)
        return coords
    except Exception:
        return None

# -----------------------------
# Pipeline
# -----------------------------
def build_corpus(files, preferred_col) -> Tuple[List[str], List[int], List[str]]:
    texts = []
    src_index = []
    src_name = []
    for i, f in enumerate(files):
        parts = load_text_from_any(f, preferred_col)
        for p in parts:
            if p and p.strip():
                texts.append(p.strip())
                src_index.append(i)
                src_name.append(f.name)
    return texts, src_index, src_name

def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    aggs = df.groupby("cluster", dropna=False).agg(
        count=("text", "count"),
        avg_sentiment=("sentiment", "mean")
    ).reset_index()
    aggs["avg_sentiment"] = aggs["avg_sentiment"].round(3)
    return aggs.sort_values(by="count", ascending=False)

# -----------------------------
# Main run
# -----------------------------
if run_btn:
    if not uploaded_files:
        st.warning("Please upload at least one file to proceed.")
        st.stop()

    if not SKLEARN_OK:
        st.error("scikit-learn is required for clustering. Add scikit-learn to your environment.")
        st.stop()

    texts, src_i, src_name = build_corpus(uploaded_files, text_col_hint.strip() if text_col_hint else None)
    st.write(f"Loaded {len(texts)} review texts from {len(uploaded_files)} file(s).")

    if len(texts) < 3:
        st.warning("Need at least 3 texts to form meaningful clusters.")
        st.stop()

    # Aspect extraction (optional)
    if use_aspects:
        processed = extract_aspects_spacy(texts)
        used_mode = "Aspects (noun chunks)" if SPACY_OK and load_spacy() else "Original text (spaCy unavailable)"
    else:
        processed = texts
        used_mode = "Original text"

    # Vectorize
    vectorizer, X = vectorize_text(processed, ngrams)
    if X is None or X.shape[0] == 0 or X.shape[1] == 0:
        st.error("Vectorization produced an empty matrix. Try different n-gram settings or disable aspects.")
        st.stop()

    # Cluster
    if algo == "KMeans":
        k_use = auto_pick_k(X) if auto_k else k_value
        model, labels = run_kmeans(X, k_use)
        st.success(f"KMeans finished with K={len(set(labels))} cluster(s).")
    else:
        model, labels = run_dbscan(X, eps=eps, min_samples=min_samples)
        n_clusters = len([c for c in set(labels) if c != -1])
        st.success(f"DBSCAN finished with {n_clusters} cluster(s) and {(labels == -1).sum()} noise point(s).")

    # Sentiment
    sentiments = compute_sentiment(texts) if do_sentiment else [0.0] * len(texts)

    # Results DataFrame
    df = pd.DataFrame({
        "source_file": src_name,
        "text": texts,
        "used_for_clustering": processed,
        "cluster": labels,
        "sentiment": sentiments
    })

    # Top terms per cluster
    terms = top_terms_per_cluster(vectorizer, X, labels, topn=10)

    # Layout Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Clusters", "Top Terms", "2D Plot"])

    with tab1:
        st.subheader("Settings")
        st.json({
            "representation": used_mode,
            "ngrams": ngrams,
            "algorithm": algo,
            "k": int(len(set(labels))) if algo == "KMeans" else None,
            "dbscan_eps": float(eps) if algo == "DBSCAN" else None,
            "dbscan_min_samples": int(min_samples) if algo == "DBSCAN" else None,
            "sentiment": "VADER" if do_sentiment and VADER_OK else "Off"
        })

        st.subheader("Cluster Summary")
        st.dataframe(summarize_clusters(df), use_container_width=True)

        st.subheader("Download Results")
        out_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", out_csv, file_name="clustered_reviews.csv", mime="text/csv")

    with tab2:
        st.subheader("Clustered Reviews")
        sel = st.selectbox("Filter cluster", options=sorted(list(set(labels))))
        st.dataframe(df[df["cluster"] == sel][["source_file", "text", "sentiment"]], use_container_width=True)

    with tab3:
        st.subheader("Top Terms per Cluster (TF‑IDF)")
        if terms:
            for cl, words in sorted(terms.items(), key=lambda kv: kv[0]):
                st.write(f"Cluster {cl}: {', '.join(words)}")
        else:
            st.info("No terms available. Try KMeans or adjust n‑gram/min_df.")

    with tab4:
        st.subheader("PCA Scatter (2D)")
        coords = pca_2d(X)
        if coords is None:
            st.info("2D projection unavailable. Try smaller data or different settings.")
        else:
            plot_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "cluster": labels})
            st.scatter_chart(plot_df, x="x", y="y", color="cluster", height=500)

else:
    st.info("Upload files and click Run Clustering from the sidebar to begin.")

