import numpy as np
import pandas as pd
import joblib

from sklearn.metrics.pairwise import cosine_similarity

from .config import datasetDirectory, artifactsDirectory
from .preprocess import PrepareDataForTFIDF, PrepareDataForBM25


# Loading indexes and clean data

def loadIndexes():
    print("Loading TF-IDF and BM25 indexes.")

    tfidfVectorizer = joblib.load(artifactsDirectory / "tfidf_vectorizer.pkl")
    tfidfMatrix = joblib.load(artifactsDirectory / "tfidf_matrix.pkl")

    bm25Model = joblib.load(artifactsDirectory / "bm25_model.pkl")
    tokenizedCleanData = joblib.load(artifactsDirectory / "bm25_cleanData_tokens.pkl")

    cleanDataFrame = pd.read_csv(datasetDirectory / "clean_data.csv")

    print("Indexes loaded.")

    return tfidfVectorizer, tfidfMatrix, bm25Model, tokenizedCleanData, cleanDataFrame



# TF–IDF retrieval

def retrieveTfIdf(query, tfidfVectorizer, tfidfMatrix, topK=10):
    queryClean = PrepareDataForTFIDF(query)
    queryVector = tfidfVectorizer.transform([queryClean])

    similarityScores = cosine_similarity(queryVector, tfidfMatrix)[0]

    topIndices = np.argsort(similarityScores)[::-1][:topK]

    return topIndices, similarityScores[topIndices]


# BM25 Retrieval

def retrieveBM25(query, bm25Model, topK=10):
    queryTokens = PrepareDataForBM25(query)

    scores = bm25Model.get_scores(queryTokens)

    topIndices = np.argsort(scores)[::-1][:topK]

    return topIndices, scores[topIndices]


# Hybrid (TF–IDF + BM25) retrieval

def retrieveHybrid(query, tfidfVectorizer, tfidfMatrix, bm25Model, alpha=0.5, topK=10):
    fullRange = tfidfMatrix.shape[0]
    tfidfIndices, tfidfScoresSubset = retrieveTfIdf(
        query, tfidfVectorizer, tfidfMatrix, topK=fullRange
    ) 

    tfidfScores = np.zeros(fullRange)
    tfidfScores[tfidfIndices] = tfidfScoresSubset 

    bm25Scores = bm25Model.get_scores(PrepareDataForBM25(query)) 

    tfidfNormalized = (tfidfScores - tfidfScores.min()) / (tfidfScores.max() - tfidfScores.min() + 1e-9)
    bm25Normalized = (bm25Scores - bm25Scores.min()) / (bm25Scores.max() - bm25Scores.min() + 1e-9)

    hybridScores = alpha * bm25Normalized + (1 - alpha) * tfidfNormalized

    topIndices = np.argsort(hybridScores)[::-1][:topK]

    return topIndices, hybridScores[topIndices]


def formatResults(indices, scores, cleanDataFrame):
    results = []

    for index, score in zip(indices, scores):
        row = cleanDataFrame.iloc[index]
        snippet = row["text"][:200].replace("\n", " ")

        results.append({
            "documentId": int(row["documentId"]),
            "title": row["title"],
            "score": float(score),
            "snippet": snippet,
            "fullText": row["text"],
            "date": row["date"],
            "category": row["category"]
        })

    return results
