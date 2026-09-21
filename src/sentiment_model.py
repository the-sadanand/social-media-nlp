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
    tfidf = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),
        min_df=2,
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    
    print(f"  TF-IDF vocabulary size: {len(tfidf.vocabulary_):,}")
    # 4.  Train Logistic Regression
    
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        class_weight="balanced",
        random_state=random_state,
    )
    clf.fit(X_train_vec, y_train)
    

    # 5.  Evaluate
    
    y_pred = clf.predict(X_test_vec)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_test, y_pred, average="macro")), 4),
        "recall_macro": round(float(recall_score(y_test, y_pred, average="macro")), 4),
        "f1_score_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
    }

    print(f"\n  {'Metric':<20} {'Value':>8}")
    print(f"  {'-'*28}")
    for k, v in metrics.items():
        print(f"  {k:<20} {v:>8.4f}")

    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, digits=4))


    # 6.  Save artifacts

    with open(out / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    with open(out / "sentiment_model.pkl", "wb") as f:
        pickle.dump(clf, f)

    with open(out / "sentiment_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        "tweet_id": test_tweet_ids.values,
        "predicted_sentiment": y_pred,
    }).to_csv(out / "sentiment_predictions.csv", index=False)

    print(f"  [OK] Saved: tfidf_vectorizer.pkl, sentiment_model.pkl, "
          f"sentiment_metrics.json, sentiment_predictions.csv")

    return clf, tfidf, metrics





if __name__ == "__main__":
    train_and_evaluate()