import time
import numpy as np
import pandas as pd

from .retriever import (
    loadIndexes,
    retrieveTfIdf,
    retrieveBM25,
    retrieveHybrid
)

# Evaluation metrics

def precisionAtK(relevantDocs, retrievedDocs, k):
    retrievedAtK = retrievedDocs[:k]
    hits = sum(doc in relevantDocs for doc in retrievedAtK)
    return hits / k


def recallAtK(relevantDocs, retrievedDocs, k):
    retrievedAtK = retrievedDocs[:k]
    hits = sum(doc in relevantDocs for doc in retrievedAtK)
    return hits / len(relevantDocs) if relevantDocs else 0


def ndcgAtK(relevantDocs, retrievedDocs, k):
    dcg = 0
    for i, docId in enumerate(retrievedDocs[:k]):
        if docId in relevantDocs:
            dcg += 1 / np.log2(i + 2)

    idealDcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevantDocs))))
    return dcg / idealDcg if idealDcg > 0 else 0


# Evaluation pipeline

def evaluateSystem(queries, relevanceDict, method="hybrid", k=10, alpha=0.6):
    """
    queries : list of query strings
    relevanceDict : {query_index: [list of relevant documentIds]}
    """

    tfidfVectorizer, tfidfMatrix, bm25Model, _, cleanDataFrame = loadIndexes()

    evaluationResults = []

    for i, query in enumerate(queries):
        print(f"\nEvaluating Query {i+1}: {query}")

        startTime = time.time()

        if method == "tfidf":
            indices, _ = retrieveTfIdf(query, tfidfVectorizer, tfidfMatrix, topK=k)

        elif method == "bm25":
            indices, _ = retrieveBM25(query, bm25Model, topK=k)

        else:  
            indices, _ = retrieveHybrid(
                query, tfidfVectorizer, tfidfMatrix, bm25Model, alpha=alpha, topK=k
            )

        latencyMs = (time.time() - startTime) * 1000
        retrievedDocs = list(indices)

        relevantDocs = relevanceDict.get(i, [])

        pAtK = precisionAtK(relevantDocs, retrievedDocs, k)
        rAtK = recallAtK(relevantDocs, retrievedDocs, k)
        ndcg = ndcgAtK(relevantDocs, retrievedDocs, k)

        evaluationResults.append({
            "query": query,
            "precision@k": pAtK,
            "recall@k": rAtK,
            "ndcg@k": ndcg,
            "latency_ms": latencyMs
        })

    return pd.DataFrame(evaluationResults)
