"""
preprocess.py -> Text Preprocessing pipline
"""
import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

# Download nltk resouces for once

for resource in ["punkt","punkt_tab","stopwords","wordnet","omw-1.4"]:
    nltk.download(resource,quiet=True)
# pre initializing object to avoid repeated constructionl overhead
_STOP_WORDS:set[str] = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

def clean_text(text:str) -> str:
    
    if not isinstance(text,str):
        return ""
    
    text = re.sub(r"http\S+|www\.\S+", "", text) #urls
    text = re.sub(r"@\w+","",text) # mentions
    text = re.sub(r"#\w+","",text) # hastags
    text = re.sub(r"&\w+;","",text) # html tags
    text = re.sub(r"[^a-zA-Z\s]","",text) #keep only letters and spaces
    text = text.lower() #lower
    text = re.sub(r"\s+","",text) # collaspe white spaces
    
    return text 

def tokenize_and_lemmatize(text:str) -> str:
    # Tokenize, remove stopwords, and lemmatize a cleaned text string.
    
    # returns a single joined spaced string cuz downstream TF-IDF/CountVectorizer except stirng input
    
    tokens = word_tokenize(text)
    tokens = [_LEMMATIZER.lemmatize(tok) for tok in tokens if tok not in _STOP_WORDS and len(tok)>2]
    
    return " ".join(tokens)

def preprocess_data(input_path:"data/Tweets.csv",
    output_path:"output/preprocessed_data.csv") -> pd.DataFrame:
    
    print("="*60)
    print("Step 1 : text preprocessing")
    print("="*60)
    df = pd.read_csv(input_path)
    print(f"Loaded{len(df):,} tweets from {input_path}")
    
    # apply cleaning pipeline
    print("  Cleaning text (URLs, mentions, hashtags, punctuation)...")
    df["cleaned_text"] = df["cleaned_text"].apply(clean_text)
    print("  Tokenizing, removing stopwords, lemmatizing...")
    df["cleaned_text"] = df["cleaned_text"].apply(tokenize_and_lemmatize)
    # drop rows where cleaning produced a empty string
    
    before = len(df)
    df = df[df["cleaned_text"].str.strip().astype(bool)].copy()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} empty rows after cleaning")
        
    # save only the colum the spec requires 
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True ,exist_ok=True)
    df[["tweet_id", "cleaned_text"]].to_csv(output_path ,index=False)
    print(f"  [OK] Saved {len(df):,} preprocessed tweets -> {output_path}")
    
    return df

if __name__ == "__main__":
    preprocess_data()
    