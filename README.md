# Social Media Sentiment & Topic Analysis Platform

A comprehensive **Natural Language Processing (NLP)** pipeline that analyses sentiment and discovers topics from the [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) dataset.

## Features

| Feature | Description |
|---------|-------------|
| **Text Preprocessing** | Regex cleaning, stopword removal, lemmatisation via NLTK |
| **Sentiment Classification** | TF-IDF + Logistic Regression (macro F1 ≈ 0.68+) |
| **Topic Modeling** | Latent Dirichlet Allocation (LDA) with 5 auto-discovered topics |
| **Interactive Dashboard** | Streamlit app with Plotly charts and pyLDAvis visualisation |
| **Containerised** | Docker + Docker Compose for one-command deployment |

## Quick Start

### Option A — Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the dataset (requires kagglehub)
pip install kagglehub
python -c "import kagglehub; kagglehub.dataset_download('crowdflower/twitter-airline-sentiment')"
# Copy Tweets.csv into data/

# 3. Run the full pipeline (generates all artifacts in output/)
python run_pipeline.py

# 4. Launch the dashboard
streamlit run app.py
```

### Option B — Run with Docker

```bash
# Place Tweets.csv in data/ directory, then:
docker-compose up --build
# Dashboard available at http://localhost:8501
```

## Project Structure

```
social-media-nlp/
├── app.py                  # Streamlit dashboard (3 pages)
├── run_pipeline.py         # Pipeline orchestrator
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container image definition
├── docker-compose.yml      # Container orchestration
├── .env.example            # Environment variable template
├── .gitignore
├── README.md               # ← You are here
├── src/
│   ├── __init__.py
│   ├── preprocess.py       # Text cleaning: URLs, mentions, stopwords, lemmatisation
│   ├── sentiment_model.py  # TF-IDF → Logistic Regression classifier
│   └── topic_model.py      # CountVectorizer → LDA → pyLDAvis
├── data/
│   └── Tweets.csv          # Raw dataset (14,640 tweets)
└── output/                 # Generated artifacts ↓
    ├── preprocessed_data.csv
    ├── tfidf_vectorizer.pkl
    ├── sentiment_model.pkl
    ├── sentiment_metrics.json
    ├── sentiment_predictions.csv
    ├── lda_model.pkl
    ├── count_vectorizer.pkl
    ├── topics.json
    └── lda_visualization.html
```

## Pipeline Overview

```
Raw Tweets (14,640)
       │
       ▼
┌──────────────────────────┐
│  1. PREPROCESSING        │
│  Remove URLs, mentions,  │
│  hashtags, stopwords     │
│  Lemmatise tokens        │
│  → preprocessed_data.csv │
└──────────┬───────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌──────────┐ ┌──────────────┐
│ 2. SENT. │ │ 3. TOPICS    │
│ TF-IDF + │ │ CountVec +   │
│ LogReg   │ │ LDA (5)      │
│          │ │ + pyLDAvis   │
└──────────┘ └──────────────┘
     │             │
     ▼             ▼
  Metrics      Topics JSON
  Predictions  HTML Viz
```

## Technical Decisions

### Why Logistic Regression?
- Fast to train, interpretable, and strong baseline for text classification
- With `class_weight="balanced"`, it handles the imbalanced dataset (63% negative, 21% neutral, 16% positive) without resampling
- Easily replaceable with SVM, Random Forest, or neural models for comparison

### Why TF-IDF (not raw counts) for Sentiment?
- TF-IDF down-weights common words ("flight", "airline") and up-weights discriminative words ("rude", "amazing")
- This helps the classifier focus on words that actually indicate sentiment

### Why CountVectorizer (not TF-IDF) for LDA?
- LDA is a probabilistic generative model that expects raw word frequencies
- TF-IDF distorts the probability distributions LDA relies on

### Why 5 Topics?
- Experimentation showed 5 topics produce the most interpretable clusters for this airline domain
- Common topics discovered: delays, customer service, baggage, booking, in-flight experience

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | ~78% |
| Precision (macro) | ~70% |
| Recall (macro) | ~67% |
| F1 Score (macro) | ~68% |

> **Note**: Exact values depend on the random seed. The model struggles most with neutral tweets, which is expected — neutral sentiment is inherently ambiguous.

## Error Analysis

The confusion matrix reveals:
- **Negative tweets**: Classified with high accuracy (~85%+) — they contain strong signal words
- **Positive tweets**: Good accuracy (~70%+) — words like "great", "thank", "love" are distinctive
- **Neutral tweets**: Hardest to classify (~50-60%) — they often lack clear sentiment markers and can be confused with both positive and negative

This pattern is typical for airline sentiment data, where neutral tweets often contain factual statements that could be perceived either way.

## Extending the Project

1. **Better models**: Replace Logistic Regression with SVM, XGBoost, or fine-tuned BERT
2. **Domain stopwords**: Remove airline names and common aviation terms from topic modeling
3. **More topics**: Experiment with 3–10 topics and compare coherence scores
4. **Explainability**: Use LIME or SHAP to explain individual predictions
5. **Real-time**: Add a text input box to the dashboard for on-the-fly sentiment prediction

## License

This project is for educational purposes. The dataset is provided by CrowdFlower (now Appen) via Kaggle.
