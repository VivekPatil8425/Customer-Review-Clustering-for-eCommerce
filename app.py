import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from typing import List, Optional

# --- New Imports for Clustering (HAC) ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage # Used for distance metrics
# --- Existing Imports (Sentiment) ---
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# --- Configuration & Setup ---
st.set_page_config(page_title="HAC Review Clustering", layout="wide")
st.title("eCommerce Review Insight Hub (Hierarchical Clustering) 🌳")
st.caption("Core Pipeline: CSV Upload -> TF-IDF -> Agglomerative Clustering (HAC) -> VADER Sentiment. Groups reviews based on hierarchy.")

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
    
    st.header("3. HAC Parameters")
    # HAC uses 'n_clusters' just like K-Means' K
    n_clusters = st.slider("Number of Clusters", min_value=3, max_value=10, value=5)
    
    # Crucial HAC Parameter to demonstrate understanding
    linkage_type = st.selectbox(
        "Linkage Method", 
        options=['ward', 'average', 'complete'], 
        index=0,
        help="How distance between clusters is calculated. 'Ward' minimizes variance."
    )
    
    st.header("4. Run Analysis")
    run_btn = st.button("Run HAC Clustering and Sentiment", type="primary")

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

@st.cache_resource(show_spinner="Running HAC Clustering...")
def run_hac_pipeline(reviews: List[str], n_clusters: int, linkage_type: str):
    """Vectorizes, runs HAC, and extracts top terms."""
    
    # 1. Vectorization (TF-IDF)
    vectorizer = TfidfVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS),
        ngram_range=(1, 2),
        min_df=2 
    )
    X = vectorizer.fit_transform(reviews)

    # 2. HAC Clustering (Agglomerative Clustering)
    hac = AgglomerativeClustering(
        n_clusters=n_clusters, 
        linkage=linkage_type, 
        metric='cosine' # Use cosine similarity for text vectors
    )
    labels = hac.fit_predict(X.toarray()) # HAC often requires dense matrix

    # 3. Top Terms Extraction (Same as K-Means for consistency)
    terms = np.array(vectorizer.get_feature_names_out())
    cluster_terms = {}
    
    for cl in sorted(set(labels)):
        idx = np.where(labels == cl)[0]
        if len(idx) == 0: continue
        
        # Calculate the mean vector for the cluster
        mean_vec = X[idx].mean(axis=0).A1 
        # Get the indices of the top 5 largest values (most important terms)
        top_idx = mean_vec.argsort()[::-1][:5] 
        cluster_terms[cl] = terms[top_idx].tolist()
        
    # 4. Linkage Matrix for Validation (The HAC value-add)
    # This matrix is the basis of the Dendrogram and shows the merge distances.
    Z = linkage(X.toarray(), method=linkage_type, metric='cosine')
    
    return labels, cluster_terms, Z

def summarize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates summary stats for display."""
    df['sentiment_class'] = np.select(
        [df['sentiment'] > 0.05, df['sentiment'] < -0.05],
        ['Positive', 'Negative'],
        default='Neutral'
    ).astype(str)

    aggs = df.groupby("cluster_id").agg(
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
    # (Same as LDA/K-Means for consistent look)
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

def plot_cluster_sentiment(subset: pd.DataFrame, cluster_id: int):
    """Generates a sentiment bar chart for a single cluster."""
    # (Same as LDA/K-Means for consistent look)
    cluster_sentiment_counts = subset['sentiment_class'].value_counts().reset_index()
    cluster_sentiment_counts.columns = ['Sentiment', 'Count']
    
    color_map = {'Positive': 'rgb(26, 179, 148)', 'Negative': 'rgb(220, 53, 69)', 'Neutral': 'rgb(255, 193, 7)'}
    
    fig_bar = px.bar(
        cluster_sentiment_counts,
        x='Sentiment',
        y='Count',
        title=f'Sentiment for Cluster {cluster_id}',
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

    # 2. Run HAC Pipeline
    labels, cluster_terms, Z_matrix = run_hac_pipeline(reviews, n_clusters, linkage_type)

    # 3. Compile Final DataFrame
    final_df = data.copy()
    final_df['cluster_id'] = labels
    
    if final_df.empty:
        st.error("Clustering failed to assign reviews. Check your parameters or data size.")
        st.stop()
        
    summary_df = summarize_data(final_df)

    st.success(f"HAC Complete! {len(final_df)} reviews grouped into {n_clusters} Clusters using '{linkage_type}' linkage.")
    
    # --- Results Layout ---
    
    st.header("Hierarchical Clustering Results")
    
    tab1, tab2, tab3 = st.tabs(["Overview & Summary", "Detailed Cluster Breakdown", "HAC Validation (Distances)"])

    with tab1:
        col_vis, col_summary = st.columns([1, 1])

        with col_vis:
            st.subheader("Sentiment Insights")
            st.plotly_chart(plot_sentiment_pie(final_df), use_container_width=True)
            
        with col_summary:
            st.subheader("2. Cluster Summary Table")
            summary_display = summary_df.rename(columns={'cluster_id': 'Cluster ID', 'Avg_Sentiment': 'Avg Sentiment Score'})
            summary_display['Cluster ID'] = summary_display['Cluster ID'].apply(lambda x: f"Cluster {x}")
            
            st.dataframe(summary_display, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("3. Cluster Definitions (Aspects)")
        
        for cluster_id in summary_df['cluster_id'].tolist():
             if cluster_id in cluster_terms:
                 st.write(f"**Cluster {cluster_id} Keywords:** {', '.join(cluster_terms[cluster_id])}")
             
    with tab2:
        st.subheader("Detailed Cluster Breakdown (Actionable Insights)")

        cluster_labels = summary_df['cluster_id'].tolist()
        
        for cluster_id in cluster_labels:
            subset = final_df[final_df["cluster_id"] == cluster_id].copy()
            
            if subset.empty: continue
            
            # Build expander title using top terms (our "aspects")
            top_term_str = ", ".join(cluster_terms[cluster_id])
            title = f"Cluster {cluster_id}: **{top_term_str}** ({len(subset)} reviews)"

            with st.expander(title, expanded=False):
                col_chart, col_reviews = st.columns([1, 2])
                
                with col_chart:
                    plot_cluster_sentiment(subset, cluster_id)
                
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
        st.download_button("Download Clustered Data as CSV", out_csv, file_name="hac_reviews_results.csv", mime="text/csv")
        
    with tab3:
        st.subheader("HAC Linkage Matrix (Validation)")
        st.markdown(
            """
            This matrix is the foundation of the Dendrogram and is used to **justify the number of clusters (K)**. 
            Each row represents a merger. The **Distance** column shows how far apart the two merging clusters were. 
            Large jumps in distance suggest a good place to **stop merging** (i.e., your chosen K is a natural separation).
            """
        )
        
        # Create a DataFrame for the Z matrix for easy viewing
        Z_df = pd.DataFrame(Z_matrix, columns=['Cluster 1 Index', 'Cluster 2 Index', 'Distance', 'New Cluster Size'])
        Z_df.index = Z_df.index.rename('Merge Step')
        Z_df['Distance'] = Z_df['Distance'].round(4)
        
        st.dataframe(Z_df, use_container_width=True)

else:
    st.info("Upload your CSV file in the sidebar, set the number of Clusters, and click 'Run HAC Clustering and Sentiment' to analyze your customer reviews.")
