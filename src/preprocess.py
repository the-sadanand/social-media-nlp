"""
preprocess.py -> Text Preprocessing pipeline
"""

import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)

_STOP_WORDS: set[str] = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Remove URLs, mentions, hashtags, HTML entities and punctuation."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"&\w+;", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_and_lemmatize(text: str) -> str:
    """Tokenize, remove stopwords, and lemmatize cleaned text."""
    tokens = word_tokenize(text)
    tokens = [
        _LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in _STOP_WORDS and len(tok) > 2
    ]

    return " ".join(tokens)


def preprocess_data(
    input_path: str = "data/Tweets.csv",
    output_path: str = "output/preprocessed_data.csv",
) -> pd.DataFrame:
    """Run the complete text preprocessing pipeline and save the result."""

    print("=" * 60)
    print("Step 1: Text Preprocessing")
    print("=" * 60)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} tweets from {input_path}")

    print("  Cleaning text (URLs, mentions, hashtags, punctuation)...")
    df["cleaned_text"] = df["text"].apply(clean_text)

    print("  Tokenizing, removing stopwords, lemmatizing...")
    df["cleaned_text"] = df["cleaned_text"].apply(tokenize_and_lemmatize)

    before = len(df)
    df = df[df["cleaned_text"].str.strip().astype(bool)].copy()
    dropped = before - len(df)

    if dropped:
        print(f"Dropped {dropped} empty rows after cleaning")

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df[["tweet_id", "cleaned_text"]].to_csv(output_path, index=False)
    print(f"  [OK] Saved {len(df):,} preprocessed tweets -> {output_path}")

    return df


if __name__ == "__main__":
    preprocess_data()
