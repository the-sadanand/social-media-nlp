# app.py — Streamlit Interactive Dashboard

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components


OUTPUT_DIR = Path("output")
DATA_DIR = Path("data")
RAW_DATA_PATH = DATA_DIR / "Tweets.csv"


st.set_page_config(
    page_title="Social Media NLP Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH)


@st.cache_data
def load_preprocessed_data() -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / "preprocessed_data.csv")


@st.cache_data
def load_metrics() -> dict:
    with open(OUTPUT_DIR / "sentiment_metrics.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / "sentiment_predictions.csv")


@st.cache_data
def load_topics() -> dict:
    with open(OUTPUT_DIR / "topics.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_lda_html() -> str | None:
    path = OUTPUT_DIR / "lda_visualization.html"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a section:",
    ["Dataset Overview", "Sentiment Analysis", "Topic Modeling"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Built with** Streamlit, scikit-learn, pyLDAvis, and Plotly"
)


def check_artifacts() -> bool:
    required = [
        "preprocessed_data.csv",
        "tfidf_vectorizer.pkl",
        "sentiment_model.pkl",
        "sentiment_metrics.json",
        "sentiment_predictions.csv",
        "lda_model.pkl",
        "count_vectorizer.pkl",
        "topics.json",
        "lda_visualization.html",
    ]

    missing = [
        filename
        for filename in required
        if not (OUTPUT_DIR / filename).exists()
    ]

    if missing:
        st.error(
            "Missing model artifacts. Run the pipeline first. "
            "Command: python run_pipeline.py. "
            f"Missing files: {', '.join(missing)}"
        )
        return False

    return True


def page_overview() -> None:
    st.title("Dataset Overview")
    st.markdown("Exploring the **Twitter US Airline Sentiment** dataset.")

    df = load_raw_data()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Tweets", f"{len(df):,}")
    col2.metric("Airlines", df["airline"].nunique())
    col3.metric("Avg Length", f"{df['text'].str.len().mean():.0f} chars")
    col4.metric("Unique Timestamps", f"{df['tweet_created'].nunique():,}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Sentiment Distribution")
        fig_pie = px.pie(
            df,
            names="airline_sentiment",
            color="airline_sentiment",
            color_discrete_map={
                "negative": "#EF553B",
                "neutral": "#636EFA",
                "positive": "#00CC96",
            },
            hole=0.4,
        )
        fig_pie.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("Tweets per Airline")
        counts = df["airline"].value_counts().reset_index()
        counts.columns = ["airline", "count"]

        fig_bar = px.bar(
            counts,
            x="airline",
            y="count",
            color="airline",
        )
        fig_bar.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20),
            xaxis_title="",
            yaxis_title="Number of Tweets",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Sentiment Breakdown by Airline")
    fig_stacked = px.histogram(
        df,
        x="airline",
        color="airline_sentiment",
        barmode="group",
        color_discrete_map={
            "negative": "#EF553B",
            "neutral": "#636EFA",
            "positive": "#00CC96",
        },
    )
    fig_stacked.update_layout(
        xaxis_title="",
        yaxis_title="Count",
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_stacked, use_container_width=True)

    st.subheader("Sample Tweets")
    st.dataframe(
        df[["tweet_id", "airline", "airline_sentiment", "text"]].head(20),
        use_container_width=True,
        hide_index=True,
    )


def page_sentiment() -> None:
    st.title("🎯 Sentiment Analysis Results")
    st.markdown(
        "Model: **TF-IDF + Logistic Regression** "
        "(balanced class weights, unigram + bigram features)"
    )

    metrics = load_metrics()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("Precision (macro)", f"{metrics['precision_macro']:.2%}")
    col3.metric("Recall (macro)", f"{metrics['recall_macro']:.2%}")
    col4.metric("F1 Score (macro)", f"{metrics['f1_score_macro']:.2%}")

    st.markdown("---")

    preds = load_predictions()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Predicted Sentiment Distribution (Test Set)")
        fig = px.pie(
            preds,
            names="predicted_sentiment",
            color="predicted_sentiment",
            color_discrete_map={
                "negative": "#EF553B",
                "neutral": "#636EFA",
                "positive": "#00CC96",
            },
            hole=0.4,
        )
        fig.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Prediction Counts")
        counts = preds["predicted_sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]

        fig_bar = px.bar(
            counts,
            x="Sentiment",
            y="Count",
            color="Sentiment",
            color_discrete_map={
                "negative": "#EF553B",
                "neutral": "#636EFA",
                "positive": "#00CC96",
            },
        )
        fig_bar.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Sample Predictions")

    try:
        raw = load_raw_data()[["tweet_id", "text", "airline_sentiment"]]
        display_df = preds.merge(raw, on="tweet_id", how="left")
        display_df = display_df.rename(
            columns={"airline_sentiment": "actual_sentiment"}
        )

        st.dataframe(
            display_df[
                [
                    "tweet_id",
                    "text",
                    "actual_sentiment",
                    "predicted_sentiment",
                ]
            ].head(30),
            use_container_width=True,
            hide_index=True,
        )
    except Exception as exc:
        st.warning(f"Could not merge sample tweets: {exc}")
        st.dataframe(preds.head(30), use_container_width=True, hide_index=True)


def page_topics() -> None:
    st.title("📚 Topic Modeling Results")
    st.markdown("Model: **Latent Dirichlet Allocation (LDA)** with 5 topics")

    topics = load_topics()

    st.subheader("Discovered Topics")

    for topic_name, words in topics.items():
        topic_num = topic_name.replace("topic_", "Topic ")
        st.markdown(
            f"**{topic_num}**: " + ", ".join(words)
        )

    st.markdown("---")

    st.subheader("Interactive Topic Visualisation (pyLDAvis)")
    st.markdown(
        "_Use the visualisation below to explore topic distances and "
        "word distributions. Click on a topic bubble to see its top terms._"
    )

    lda_html = load_lda_html()

    if lda_html:
        components.html(lda_html, height=800, scrolling=True)
    else:
        st.warning(
            "pyLDAvis visualisation file not found at "
            f"{OUTPUT_DIR / 'lda_visualization.html'}"
        )


def main() -> None:
    if not RAW_DATA_PATH.exists():
        st.error(
            f"Dataset not found: {RAW_DATA_PATH}. "
            "Place Tweets.csv inside the data/ directory."
        )
        st.stop()

    if not check_artifacts():
        st.stop()

    # Match the sidebar values exactly.
    if page == "Dataset Overview":
        page_overview()
    elif page == "Sentiment Analysis":
        page_sentiment()
    elif page == "Topic Modeling":
        page_topics()


if __name__ == "__main__":
    main()
