import pandas as pd
import joblib
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

from .config import datasetDirectory, artifactsDirectory
from .preprocess import PrepareDataForTFIDF, PrepareDataForBM25


# Clean data loading

def loadCleanData():
    cleanDataPath = datasetDirectory / "clean_data.csv"
    print(f"Loading cleaned dataset from {cleanDataPath}!")

    cleanDataFrame = pd.read_csv(cleanDataPath)

    print(f"Loaded {len(cleanDataFrame)} documents.")
    return cleanDataFrame


# Build TF-IDF Index

def buildTfIdfIndex(cleanDataFrame):
    print("\nBuilding TF-IDF index!")

    cleanedTextList = [
        PrepareDataForTFIDF(text)
        for text in tqdm(cleanDataFrame["text"])
    ]

    tfidfVectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        lowercase=False, # since the data is already clean
    )

    tfidfMatrix = tfidfVectorizer.fit_transform(cleanedTextList)

    print("TF-IDF index built.")

    return tfidfVectorizer, tfidfMatrix


# Build BM25 Index

def buildBM25Index(cleanDataFrame):
    print("\nBuilding BM25 index...")

    tokenizedDocuments = [
        PrepareDataForBM25(text)
        for text in tqdm(cleanDataFrame["text"])
    ]

    bm25Model = BM25Okapi(tokenizedDocuments)

    print("BM25 index built.")

    return bm25Model, tokenizedDocuments


# Save indexed artifacts

def saveIndexArtifacts(tfidfVectorizer, tfidfMatrix, bm25Model, tokenizedDocuments):
    print("\nSaving index artifacts.")

    joblib.dump(tfidfVectorizer, artifactsDirectory / "tfidf_vectorizer.pkl")
    joblib.dump(tfidfMatrix, artifactsDirectory / "tfidf_matrix.pkl")

    joblib.dump(bm25Model, artifactsDirectory / "bm25_model.pkl")
    joblib.dump(tokenizedDocuments, artifactsDirectory / "bm25_cleanData_tokens.pkl")

    print(f"Indexes saved to: {artifactsDirectory}")


# Main function

def buildAllIndexes():
    cleanDataFrame = loadCleanData()

    tfidfVectorizer, tfidfMatrix = buildTfIdfIndex(cleanDataFrame)
    bm25Model, tokenizedDocuments = buildBM25Index(cleanDataFrame)

    saveIndexArtifacts(tfidfVectorizer, tfidfMatrix, bm25Model, tokenizedDocuments)

    print("\nAll indexes built.")

if __name__ == "__main__":
    buildAllIndexes()
