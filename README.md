
# News Articles Information Retrieval System

A fully local, reproducible Information Retrieval (IR) system implemented in Python using TF–IDF, BM25, and Hybrid retrieval strategies. The system indexes a dataset of 2692 news articles and supports fast, accurate querying through a command-line interface.

This IR system follows the following pipeline: 

data ingestion → preprocessing → indexing → retrieval → evaluation

--------------------------------------------------------------------------------------------------------------------------------

# Features

Fully local IR system, no cloud vector databases
TF–IDF ranking using cosine similarity
BM25 ranking
Hybrid ranking α-weighted fusion using
CLI search tool
Evaluation metrics: Precision@k, Recall@k, nDCG@k
Reproducible run instructions
Works on any OS (Windows/Mac/Linux)

--------------------------------------------------------------------------------------------------------------------------------

# IR System Structure & Architecture

IR-News-Articles/
│
├── data/
│   ├── Articles.csv               # Kaggle Dataset (https://www.kaggle.com/datasets/asad1m9a9h6mood/news-articles)
│   └── clean_data.csv             # Cleaned and standardized dataset used by indexer
│
├── artifacts/
│   ├── tfidf_vectorizer.pkl       # Saved TF-IDF vectorizer
│   ├── tfidf_matrix.pkl           # TF-IDF sparse matrix
│   ├── bm25_model.pkl             # Trained BM25 Okapi model
│   └── bm25_cleanData_tokens.pkl  # Tokenized CleanData used for BM25 scoring
│
├── notebooks/
│   └── exploration.ipynb          # Jupyter notebook for exploratory data analysis (EDA)
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # IR System setting with paths configuration
│   ├── data_loader.py             # Loads raw and cleaned datasets
│   ├── preprocess.py              # Normalization, tokenization, cleaning
│   ├── indexer.py                 # Builds TF-IDF and BM25 indexes and saves artifacts
│   ├── retriever.py               # Loads indexes & performs TF-IDF, BM25, Hybrid search
│   ├── evaluator.py               # Computes Precision@K, Recall@K, NDCG@K, latency
│   ├── cli.py                     # Command-line IR search tool
│   ├── test_preprocess.py         # Unit tests for preprocessing module
│   ├── test_retriever.py          # Unit tests for retrieval module
│   └── test_evaluator.py          # Unit tests for evaluator module
│
├── IR System Architecture.png     # IR System Architecture
├── README.md                      # Full project documentation
└── requirements.txt               # Python dependencies

--------------------------------------------------------------------------------------------------------------------------------

# Installation

# 1. Clone the repository

```
git clone <your-repo-url>
cd IR-News-Articles
```

# 2. Create a virtual environment (not necessary)

```
python -m venv .venv
.\.venv\Scripts\activate      # Windows
source .venv/bin/activate    # Linux/Mac
```

# 3. Install dependencies 

```
pip install -r requirements.txt
```

# 4. Place dataset in `data/`

Rename the CSV to:

```
data/Articles.csv
```

--------------------------------------------------------------------------------------------------------------------------------

# Data Ingestion & Cleaning

Run:

```
python -m src.data_loader
```

It detects dataset encoding, loads the dataset and creates a clean version at `data/clean_data.csv`

--------------------------------------------------------------------------------------------------------------------------------

# Build Indexes (TF–IDF & BM25)

Run:

```
python -m src.indexer
```

It generates:

```
artifacts/
 ├── tfidf_vectorizer.pkl
 ├── tfidf_matrix.pkl
 ├── bm25_model.pkl
 └── bm25_cleanData_tokens.pkl
```

--------------------------------------------------------------------------------------------------------------------------------

# Run Search Using CLI

# Hybrid search:

```
python -m src.cli "oil prices"
```

# BM25 only:

```
python -m src.cli "pakistan politics" --method bm25
```

# TF–IDF only:

```
python -m src.cli "stock market" --method tfidf
```

### Change top-k:

```
python -m src.cli "inflation economy" --top_k 15
```

--------------------------------------------------------------------------------------------------------------------------------

# Evaluation

Edit `src/test_evaluation.py` to include:

* Queries
* Relevant doc_ids

Then run:

```
python -m src.test_evaluation
```

Outputs metrics:

* Precision@k
* Recall@k
* nDCG@k
* Query latency

--------------------------------------------------------------------------------------------------------------------------------

# System Overview

# Preprocessing:

* Lowercasing
* Punctuation removal
* Stopword removal
* Tokenization
* Stemming (BM25 only)

# Indexing:

* TF–IDF vectorizer (1–2 grams, 50k features)
* BM25 Okapi scoring model

# Retrieval:

* TF–IDF cosine similarity
* BM25 token scoring
* Hybrid = α·BM25 + (1−α)·TF–IDF (default α = 0.6)

# Evaluation metrics:

* Precision@10
* Recall@10
* nDCG@10
* Query latency (ms)

--------------------------------------------------------------------------------------------------------------------------------

#  IR System designed by:

Sana Ilyas
MSDS24058
Information Retrieval Assignment

--------------------------------------------------------------------------------------------------------------------------------

# Sequence to run the code:
python -m pip install -r requirements.txt  
python -m src.config
python -m src.dataloader
python -m src.test_preprocess
python -m src.indexer
python -m src.test_retriever
python -m src.cli
python -m src.test_evaluator

--------------------------------------------------------------------------------------------------------------------------------

