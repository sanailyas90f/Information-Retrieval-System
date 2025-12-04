from src.retriever import (
    loadIndexes,
    retrieveTfIdf,
    retrieveBM25,
    formatResults
)

# load indexes and cleaned dataset
tfidfVectorizer, tfidfMatrix, bm25Model, tokenizedCleanData, cleanDataFrame = loadIndexes()

query = "oil prices pakistan economy"

print("\nTF-IDF Results:")
tfidfIndices, tfidfScores = retrieveTfIdf(query, tfidfVectorizer, tfidfMatrix, topK=5)

for result in formatResults(tfidfIndices, tfidfScores, cleanDataFrame):
    print(result["title"], "| score =", result["score"])


print("\nBM25 Results:")
bm25Indices, bm25Scores = retrieveBM25(query, bm25Model, topK=5)

for result in formatResults(bm25Indices, bm25Scores, cleanDataFrame):
    print(result["title"], "| score =", result["score"])
