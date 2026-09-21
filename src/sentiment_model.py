# Sentiment classification Pipeline

"""

Architecture : 
raw text -> tfidfVectorizer -> LogisticRegression -> sentiment lable

"""

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import(
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import train_test_split

def trian_and_evaluate(
    preprocessed_path : str = "output/preprocessed_data.csv",
    raw_data_path : str = "data/Tweets.csv",
    output_dir : str = "output",
    test_size : float = 0.2,
    random_state : int = 42
) -> tuple:
    # tain sentiment classifier and persist all artifacts
    
    # returns : tuple of(classifier,vectorizer , metrics_dict)
    
    print("\n" + "=" * 60)
    print("step 2 - Sentiment Classification")
    print("=" * 60)
    
    out = Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    
    # 1 . Load & merge : cleaned text + sentiment labels
    
    preprocessed = pd.read_csv(preprocessed_path).drop_duplicates(subset="tweet_id")
    raw = pd.read_csv(raw_data_path)[["tweet_id","airline_sentiment"]].drop_duplicates(subset="tweet_id")
    df = preprocessed.merge(row , on="tweet_id" , how= "inner")
    print(f"  Merged dataset: {len(df):,} rows  |  "
          f"Labels: {df['airline_sentiment'].value_counts().to_dict()}")
    X = df["cleaned_text"]
    y = df["airline_sentiment"]
    
    # 2.  Train / test split  (stratified to keep class proportions)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    
    test_tweet_ids = df.loc[X_test.index, "tweet_id"]

    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    # 3.  TF-IDF vectorisation