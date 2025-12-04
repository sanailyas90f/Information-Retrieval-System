import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

StopWords = set(stopwords.words("english"))
Stemmer = PorterStemmer()

# basic cleaning

def normalize_text(text: str) -> str:
    """
    Lowercase + remove punctuation/numbers + strip whitespace.
    """
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Replace multiple spaces with one space
    text = re.sub(r"\s+", " ", text).strip()

    return text


# tokenization

def tokenize(text: str, apply_stemming=False):
    """
    Convert text into tokens:
        - normalized
        - split into words
        - stopword removal
        - optional stemming
    """
    text = normalize_text(text)
    tokens = text.split()

    # stop words removal
    tokens = [t for t in tokens if t not in StopWords]

    # optional stemming 
    if apply_stemming:
        tokens = [Stemmer.stem(t) for t in tokens] # optional for TF–IDF and good for BM25

    return tokens


# clean text for TF-IDF which returns a string

def PrepareDataForTFIDF(text: str) -> str:
    """
    Produce a clean normalized string for TF–IDF vectorizer.
    No stemming is recommended for TF–IDF.
    """
    tokens = tokenize(text, apply_stemming=False)
    return " ".join(tokens)


# clean text for BM25 which returns a string

def PrepareDataForBM25(text: str):
    """
    Produce token list for BM25.
    Stemming is recommended for BM25.
    """
    return tokenize(text, apply_stemming=True)
