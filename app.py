import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import silhouette_score # Kept for optional K-selection logic

# --- Configuration & Setup ---
st.set_page_config(page_title="Simple Review Clustering", layout="wide")
st.title("eCommerce Review Insight Hub (Simplified)")
st.caption("Core Pipeline: CSV Upload -> TF-IDF -> K-Means Clustering -> VADER Sentiment")

# --- Global Components ---

# Sidebar: File Upload
with st.sidebar:
    st.header("1. Upload Review Data")
    uploaded_file = st.file_uploader(
        "Upload a CSV file only",
        type=["csv"],
        help="The CSV must contain one column with review text."
    )

    st.header("2. Text Column Configuration")
    text_col = st.text_input(
        "Name of the Review Column",
        value="review",
        help="e.g., 'text', 'comment', or 'review'."
    )
    
    st.header("3. Clustering Parameters")
    # Simplified to K-Means only
    k_value = st.slider("Number of Clusters (K)", min_value=3, max_value=10, value=5)
    
    st.header("4. Run Analysis")
    run_btn = st.button("Run Clustering and Sentiment", type="primary")

# --- Helper Functions (Core Logic) ---

def clean_text(t: str) -> str:
    """Basic text cleaning."""
    if not isinstance(t, str):
        t = str(t)
    t = t.lower()
    t = re.sub(r"[^a-z\s]", "", t) # Remove punctuation/numbers, keep only letters
    t = re.sub(r"\s+", " ", t).strip()
    return t

@st.cache_data
def load_and_preprocess_data(file, col_name: str) -> Optional[pd.DataFrame]:
    """Loads CSV, extracts review column, and cleans text."""
    try:
        content = file.getvalue().decode("utf-8")
        df = pd.read_csv(io.StringIO(content))
    except Exception:
        st.error("Could not read the CSV file.")
        return None

    if col_name not in df.columns:
        st.error(f"Column '{col_name}' not found in the uploaded file.")
        return None

    # Filter out missing reviews and apply cleaning
    reviews = df[col_name].dropna().apply(clean_text)
    
    if reviews.empty or len(reviews) < 3:
        st.error("Not enough valid review texts found.")
        return None
        
    df_cleaned = pd.DataFrame({'review': reviews.tolist()})
    return df_cleaned

@st.cache_resource(show_spinner="Running K-Means...")
def run_clustering_pipeline(reviews: List[str], k: int):
    """Vectorizes, runs K-Means, and performs sentiment analysis."""
    
    # 1. Vectorization (TF-IDF)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2), # Use unigrams and bigrams
        min_df=2 # Filter out terms that appear less than twice
    )
    X = vectorizer.fit_transform(reviews)

    # 2. K-Means Clustering
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    
    # 3. Sentiment Analysis (VADER)
    sia = SentimentIntensityAnalyzer()
    sentiments = [sia.polarity_scores(t)["compound"] for t in reviews]

    # 4. Top Terms Extraction
    terms = np.array(vectorizer.get_feature_names_out())
    cluster_terms = {}
    for cl in sorted(set(labels)):
        idx = np.where(labels == cl)[0]
        if len(idx) == 0: continue
        
        # Calculate the mean TF-IDF vector for this cluster
        mean_vec = X[idx].mean(axis=0).A1
        # Get the indices of the top 5 terms
        top_idx = mean_vec.argsort()[::-1][:5] 
        cluster_terms[cl] = terms[top_idx].tolist()
        
    return labels, sentiments, cluster_terms

def summarize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates summary stats for display."""
    # Classify sentiment based on VADER compound score
    df['sentiment_class'] = np.select(
        [df['sentiment'] > 0.05, df['sentiment'] < -0.05],
        ['Positive', 'Negative'],
        default='Neutral'
    )
    
    aggs = df.groupby("cluster").agg(
        Count=("review", "count"),
        Avg_Sentiment=("sentiment", "mean"),
        Positive=('sentiment_class', lambda x: (x == 'Positive').sum()),
        Negative=('sentiment_class', lambda x: (x == 'Negative').sum())
    ).reset_index()
    aggs["Avg_Sentiment"] = aggs["Avg_Sentiment"].round(3)
    
    return aggs.sort_values(by="Count", ascending=False)


# --- Visualization Functions (Plotly) ---

def plot_sentiment_pie(df: pd.DataFrame):
    """Generates an overall sentiment pie chart."""
    sentiment_counts = df['sentiment_class'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    color_map = {'Positive': 'rgb(26, 179, 148)', 'Negative': 'rgb(220, 53, 69)', 'Neutral': 'rgb(255, 193, 7)'}
    
    fig = px.pie(
        sentiment_counts,
        values='Count',
        names='Sentiment',
        title='1. Overall Review Sentiment Distribution',
        color='Sentiment',
        color_discrete_map=color_map,
        height=350
    )
    fig.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#000000', width=1)))
    return fig

# --- Main App Execution ---

if run_btn and uploaded_file:
    # 1. Load and Preprocess Data
    data = load_and_preprocess_data(uploaded_file, text_col.strip())
    
    if data is None:
        st.stop()
        
    reviews = data['review'].tolist()

    # 2. Run ML Pipeline
    labels, sentiments, top_terms = run_clustering_pipeline(reviews, k_value)

    # 3. Compile Final DataFrame
    final_df = data.copy()
    final_df['cluster'] = labels
    final_df['sentiment'] = sentiments
    summary_df = summarize_data(final_df)

    st.success(f"Analysis Complete! {len(reviews)} reviews clustered into {k_value} groups.")
    
    # --- Results Layout ---
    
    st.header("Analysis Results")
    
    col_vis, col_summary = st.columns([1, 1])

    with col_vis:
        # A. Overall Sentiment Pie Chart
        st.subheader("Sentiment Insights")
        st.plotly_chart(plot_sentiment_pie(final_df), use_container_width=True)
        
    with col_summary:
        # B. Cluster Summary Table
        st.subheader("2. Cluster Summary Table")
        # Rename cluster column to be human-readable before display
        summary_display = summary_df.rename(columns={'cluster': 'Cluster ID', 'Avg_Sentiment': 'Avg Sentiment Score'})
        summary_display['Cluster ID'] = summary_display['Cluster ID'].apply(lambda x: f"Cluster {x}")
        
        st.dataframe(summary_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("3. Detailed Cluster Breakdown (Aspects)")

    # C. Detailed Cluster Breakdown
    cluster_labels = summary_df['cluster'].tolist()
    
    for cl in cluster_labels:
        subset = final_df[final_df["cluster"] == cl].copy()
        
        # Build expander title using top terms (our "aspects")
        if cl in top_terms:
            top_term_str = ", ".join(top_terms[cl])
            title = f"Cluster {cl}: **{top_term_str}** ({len(subset)} reviews)"
        else:
            title = f"Cluster {cl} ({len(subset)} reviews)"
            
        with st.expander(title, expanded=False):
            
            # Display sentiment data for this cluster
            avg_s = subset['sentiment'].mean()
            s_color = 'green' if avg_s > 0.05 else ('red' if avg_s < -0.05 else 'orange')
            st.markdown(f"**Average Sentiment Score:** <span style='color:{s_color}; font-weight:bold;'>{avg_s:.3f}</span>", unsafe_allow_html=True)
            
            # Display sample reviews
            st.markdown(f"**Sample Reviews (Top 5):**")
            for i, row in subset.head(5).iterrows():
                st.text(f"  - [{row['sentiment_class']}] {row['review'][:100]}...") # Show truncated review

    st.markdown("---")
    
    # D. Download Results
    st.subheader("Download Full Results")
    out_csv = final_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Clustered Data as CSV", out_csv, file_name="clustered_reviews_results_simple.csv", mime="text/csv")

else:
    st.info("Upload your CSV file in the sidebar, set the parameters, and click 'Run Clustering and Sentiment' to analyze your customer reviews.")
