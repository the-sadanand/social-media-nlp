"""
preprocess.py -> Text Preprocessing pipline
"""
import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
form nltk.stem import WordNetLemmatizer

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


    
    