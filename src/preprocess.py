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