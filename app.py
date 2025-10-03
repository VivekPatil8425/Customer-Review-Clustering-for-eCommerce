import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from typing import List, Optional

# --- Imports for Clustering (DBSCAN) ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN # Swapped from AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
# --- Existing Imports (Sentiment) ---
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# --- Configuration & Setup ---
st.set_page_config(page_title="DBSCAN Review Clustering", layout="wide")
st.title("eCommerce Review Insight Hub (DBSCAN Clustering) 🔬")
st.caption("Core Pipeline: CSV Upload -> TF-IDF -> DBSCAN (Density-Based) -> VADER Sentiment. Identifies core groups and critical outliers.")

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
    
    st.header("3. DBSCAN Parameters")
    st.markdown("DBSCAN finds clusters based on density, so you tune **distance** and **min points**.")
    
    # DBSCAN Parameter 1: Epsilon (Max distance between two samples)
    eps_value = st.slider(
        "Epsilon (Eps - Max distance for similarity)", 
        min_value=0.1, max_value=1.5, step=0.05, value=0.8,
        help="A smaller Eps means only very similar reviews form a group. Cosine distance is used."
    )
    
    # DBSCAN Parameter 2: Minimum Samples (Min points required to form a cluster)
    min_samples = st.slider(
        "Min Samples (Min points for a cluster)", 
        min_value=3, max_value=20, value=5,
        help="The minimum number of reviews required to form a dense area (a cluster)."
    )
    
    st.header("4. Run Analysis")
    run_btn = st.button("Run DBSCAN Clustering and Sentiment", type="primary")

# --- Helper Functions (Core Logic) ---

def clean_text(t: str) -> str:
    """Basic text cleaning."""
    if not isinstance(t, str):
        t = str(t)
    t = t.lower()
    t = re.sub(r"[^a-z\s]", "", t) 
    t = re.sub(r"\s+", " ", t).strip()
    return t

@st.cache_data
def load_and_preprocess_data(file, col_name: str) -> Optional[pd.DataFrame]:
    """Loads CSV and cleans text."""
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
    
    # Calculate sentiment on original reviews
    sia = SentimentIntensityAnalyzer()
    df_cleaned['sentiment'] = df_cleaned['review'].apply(lambda t: sia.polarity_scores(t)["compound"])

    return df_cleaned

@st.cache_resource(show_spinner="Running DBSCAN Clustering...")
def run_dbscan_pipeline(reviews: List[str], eps_value: float, min_samples: int):
    """Vectorizes, runs DBSCAN, and extracts top terms."""
    
    # 1. Vectorization (TF-IDF)
    vectorizer = TfidfVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS),
        ngram_range=(1, 2),
        min_df=2 
    )
    X = vectorizer.fit_transform(reviews)

    # 2. DBSCAN Clustering (Density-Based)
    # Using cosine metric which is appropriate for TF-IDF vectors
    db = DBSCAN(
        eps=eps_value, 
        min_samples=min_samples, 
        metric='cosine' 
    )
    # DBSCAN fit_predict requires a dense array
    labels = db.fit_predict(X.toarray())

    # 3. Top Terms Extraction (Ignore Noise cluster -1 for keywords)
    terms = np.array(vectorizer.get_feature_names_out())
    cluster_terms = {}
    
    # Iterate over unique clusters, excluding the Noise cluster (-1)
    unique_labels = sorted([l for l in set(labels) if l != -1])
    
    for cl in unique_labels:
        idx = np.where(labels == cl)[0]
        if len(idx) == 0: continue
        
        # Calculate the mean vector for the cluster
        mean_vec = X[idx].mean(axis=0).A1 
        # Get the indices of the top 5 largest values (most important terms)
        top_idx = mean_vec.argsort()[::-1][:5] 
        cluster_terms[cl] = terms[top_idx].tolist()
        
    return labels, cluster_terms, len(unique_labels)

def summarize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates summary stats for display."""
    df['sentiment_class'] = np.select(
        [df['sentiment'] > 0.05, df['sentiment'] < -0.05],
        ['Positive', 'Negative'],
        default='Neutral'
    ).astype(str)

    # Rename cluster -1 to 'Noise/Outlier' for presentation clarity
    df['cluster_display'] = df['cluster_id'].apply(lambda x: 'Noise/Outlier' if x == -1 else f'Cluster {x}')

    aggs = df.groupby("cluster_display").agg(
        Count=('review', 'count'),
        Avg_Sentiment=('sentiment', 'mean'),
        Positive=('sentiment_class', lambda x: (x == 'Positive').sum()),
        Negative=('sentiment_class', lambda x: (x == 'Negative').sum()),
        Neutral=('sentiment_class', lambda x: (x == 'Neutral').sum())
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

def plot_cluster_sentiment(subset: pd.DataFrame, cluster_name: str):
    """Generates a sentiment bar chart for a single cluster/noise group."""
    cluster_sentiment_counts = subset['sentiment_class'].value_counts().reset_index()
    cluster_sentiment_counts.columns = ['Sentiment', 'Count']
    
    color_map = {'Positive': 'rgb(26, 179, 148)', 'Negative': 'rgb(220, 53, 69)', 'Neutral': 'rgb(255, 193, 7)'}
    
    fig_bar = px.bar(
        cluster_sentiment_counts,
        x='Sentiment',
        y='Count',
        title=f'Sentiment for {cluster_name}',
        color='Sentiment',
        color_discrete_map=color_map,
        category_orders={"Sentiment": ["Positive", "Neutral", "Negative"]},
        height=300
    )
    fig_bar.update_layout(showlegend=False, margin=dict(t=30, l=10, r=10, b=10))
    st.plotly_chart(fig_bar, use_container_width=True)


# --- Main App Execution ---

if run_btn and uploaded_file:
    # 1. Load and Preprocess Data
    data = load_and_preprocess_data(uploaded_file, text_col.strip())
    
    if data is None:
        st.stop()
        
    reviews = data['review'].tolist()

    # 2. Run DBSCAN Pipeline
    labels, cluster_terms, n_clusters_found = run_dbscan_pipeline(reviews, eps_value, min_samples)

    # 3. Compile Final DataFrame
    final_df = data.copy()
    final_df['cluster_id'] = labels
    
    if final_df.empty:
        st.error("Clustering failed to assign reviews. Check your parameters or data size.")
        st.stop()
        
    summary_df = summarize_data(final_df)
    noise_count = (labels == -1).sum()
    
    st.success(f"DBSCAN Complete! Found {n_clusters_found} core cluster(s) and {noise_count} noise/outlier review(s).")
    
    # --- Results Layout ---
    
    st.header("DBSCAN Clustering Results")
    
    # Removed the HAC Validation tab, simplified to two main tabs
    tab1, tab2 = st.tabs(["Overview & Summary", "Detailed Cluster Breakdown"])

    with tab1:
        col_vis, col_summary = st.columns([1, 1])

        with col_vis:
            st.subheader("Sentiment Insights")
            st.plotly_chart(plot_sentiment_pie(final_df), use_container_width=True)
            
        with col_summary:
            st.subheader("2. Cluster Summary Table")
            summary_display = summary_df.rename(columns={'cluster_display': 'Cluster Name', 'Avg_Sentiment': 'Avg Sentiment Score'})
            
            st.dataframe(summary_display, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("3. Cluster Definitions (Aspects)")
        
        # Display core clusters only
        core_clusters = sorted([c for c in summary_df['cluster_display'].tolist() if c.startswith('Cluster')])
        
        for cluster_name in core_clusters:
            cluster_id = int(cluster_name.split(' ')[1])
            if cluster_id in cluster_terms:
                 st.write(f"**{cluster_name} Keywords:** {', '.join(cluster_terms[cluster_id])}")
             
    with tab2:
        st.subheader("Detailed Breakdown (Actionable Insights)")

        cluster_labels = summary_df['cluster_display'].tolist()
        
        # Iterate over all cluster names (including Noise)
        for cluster_name in cluster_labels:
            is_noise = cluster_name.startswith('Noise')
            cluster_id = -1 if is_noise else int(cluster_name.split(' ')[1])
            
            subset = final_df[final_df["cluster_id"] == cluster_id].copy()
            
            if subset.empty: continue
            
            # Define title and top terms
            if is_noise:
                title = f"Noise/Outlier Reviews: **Unique & Specific** ({len(subset)} reviews)"
                top_term_str = "No defining keywords (High-value outliers)"
            else:
                top_term_str = ", ".join(cluster_terms[cluster_id])
                title = f"{cluster_name}: **{top_term_str}** ({len(subset)} reviews)"

            with st.expander(title, expanded=False):
                col_chart, col_reviews = st.columns([1, 2])
                
                with col_chart:
                    plot_cluster_sentiment(subset, cluster_name)
                
                with col_reviews:
                    avg_s = subset['sentiment'].mean()
                    s_color = 'green' if avg_s > 0.05 else ('red' if avg_s < -0.05 else 'orange')
                    st.markdown(f"**Average Sentiment Score:** <span style='color:{s_color}; font-weight:bold;'>{avg_s:.3f}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Sample Reviews (Top 5):**")
                    
                    for i, row in subset.head(5).iterrows():
                        st.text(f"  - [{row['sentiment_class']}] {row['review'][:100]}...")
                        
        st.markdown("---")
        
        st.subheader("Download Full Results")
        out_csv = final_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Clustered Data as CSV", out_csv, file_name="dbscan_reviews_results.csv", mime="text/csv")
        

else:
    st.info("Upload your CSV file in the sidebar, set the DBSCAN parameters, and click 'Run DBSCAN Clustering and Sentiment' to analyze your customer reviews.")
