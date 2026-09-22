# Sentiment classification Pipeline

"""
Architecture:
raw text -> TF-IDF Vectorizer -> Logistic Regression / Linear SVM -> sentiment label

The project keeps Logistic Regression as the existing saved model and
runs Linear SVM as a side-by-side experiment. The two models use the
same TF-IDF features and the same train/test split for a fair comparison.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def calculate_metrics(y_true, y_pred):
    """Calculate the metrics used to compare sentiment models."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(
            float(precision_score(y_true, y_pred, average="macro"))
        ),
        "recall_macro": round(
            float(recall_score(y_true, y_pred, average="macro"))
        ),
        "f1_score_macro": round(
            float(f1_score(y_true, y_pred, average="macro"))
        ),
    }


def train_and_evaluate(
    preprocessed_path: str = "output/preprocessed_data.csv",
    raw_data_path: str = "data/Tweets.csv",
    output_dir: str = "output",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    # Train Logistic Regression and Linear SVM using the same data split
    # and TF-IDF representation, then save the existing Logistic Regression
    # artifacts plus the SVM experiment and comparison results.

    print("\n" + "=" * 60)
    print("step 2 - Sentiment Classification")
    print("=" * 60)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load & merge cleaned text with sentiment labels
    preprocessed = pd.read_csv(preprocessed_path).drop_duplicates(
        subset="tweet_id"
    )
    raw = pd.read_csv(raw_data_path)[
        ["tweet_id", "airline_sentiment"]
    ].drop_duplicates(subset="tweet_id")

    df = preprocessed.merge(raw, on="tweet_id", how="inner")

    print(
        f"  Merged dataset: {len(df):,} rows  |  "
        f"Labels: {df['airline_sentiment'].value_counts().to_dict()}"
    )

    X = df["cleaned_text"]
    y = df["airline_sentiment"]

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    test_tweet_ids = df.loc[X_test.index, "tweet_id"]

    print(
        f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}"
    )

    # 3. TF-IDF vectorisation
    # Both models receive exactly the same features.
    tfidf = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),
        min_df=2,
    )

    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    print(
        f"  TF-IDF vocabulary size: {len(tfidf.vocabulary_):,}"
    )

    # 4. Model 1 - Logistic Regression (existing baseline)
    logistic_model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        class_weight="balanced",
        random_state=random_state,
    )

    logistic_model.fit(X_train_vec, y_train)
    logistic_pred = logistic_model.predict(X_test_vec)

    logistic_metrics = calculate_metrics(y_test, logistic_pred)

    # 5. Model 2 - Linear SVM (experiment)
    svm_model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=random_state,
    )

    svm_model.fit(X_train_vec, y_train)
    svm_pred = svm_model.predict(X_test_vec)

    svm_metrics = calculate_metrics(y_test, svm_pred)

    # 6. Print model comparison
    print("\n  Model Comparison")
    print("  " + "-" * 58)
    print(
        f"  {'Model':<24}"
        f"{'Accuracy':>12}"
        f"{'Macro F1':>12}"
    )
    print("  " + "-" * 58)
    print(
        f"  {'Logistic Regression':<24}"
        f"{logistic_metrics['accuracy']:>12.4f}"
        f"{logistic_metrics['f1_score_macro']:>12.4f}"
    )
    print(
        f"  {'Linear SVM':<24}"
        f"{svm_metrics['accuracy']:>12.4f}"
        f"{svm_metrics['f1_score_macro']:>12.4f}"
    )

    print("\n  Logistic Regression Classification Report:\n")
    print(classification_report(y_test, logistic_pred, digits=4))

    print("\n  Linear SVM Classification Report:\n")
    print(classification_report(y_test, svm_pred, digits=4))

    # 7. Save the existing Logistic Regression artifacts
    # This keeps the current Streamlit app compatible.
    with open(out / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    with open(out / "sentiment_model.pkl", "wb") as f:
        pickle.dump(logistic_model, f)

    with open(out / "sentiment_metrics.json", "w") as f:
        json.dump(logistic_metrics, f, indent=2)

    pd.DataFrame(
        {
            "tweet_id": test_tweet_ids.values,
            "predicted_sentiment": logistic_pred,
        }
    ).to_csv(
        out / "sentiment_predictions.csv",
        index=False,
    )

    # 8. Save the Linear SVM experiment separately
    with open(out / "sentiment_model_svm.pkl", "wb") as f:
        pickle.dump(svm_model, f)

    # 9. Save a side-by-side comparison
    comparison = pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                **logistic_metrics,
            },
            {
                "model": "Linear SVM",
                **svm_metrics,
            },
        ]
    )

    comparison.to_csv(
        out / "sentiment_model_comparison.csv",
        index=False,
    )

    with open(out / "sentiment_model_comparison.json", "w") as f:
        json.dump(
            {
                "logistic_regression": logistic_metrics,
                "linear_svm": svm_metrics,
            },
            f,
            indent=2,
        )

    pd.DataFrame(
        {
            "tweet_id": test_tweet_ids.values,
            "logistic_regression": logistic_pred,
            "linear_svm": svm_pred,
        }
    ).to_csv(
        out / "sentiment_predictions_comparison.csv",
        index=False,
    )

    print(
        "\n  [OK] Saved: tfidf_vectorizer.pkl, sentiment_model.pkl, "
        "sentiment_metrics.json, sentiment_predictions.csv"
    )
    print(
        "  [OK] Saved SVM experiment: sentiment_model_svm.pkl"
    )
    print(
        "  [OK] Saved comparison: sentiment_model_comparison.csv"
    )

    # Keep the existing return contract: the Logistic Regression model
    # remains the current application model until the experiment is reviewed.
    return logistic_model, tfidf, logistic_metrics


if __name__ == "__main__":
    train_and_evaluate()
