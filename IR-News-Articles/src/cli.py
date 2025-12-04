import argparse

from .retriever import (
    loadIndexes,
    retrieveTfIdf,
    retrieveBM25,
    retrieveHybrid,
    formatResults
)

def main():
    parser = argparse.ArgumentParser(description="IR System CLI Search Tool")
    
    parser.add_argument("query", type=str, help="Search query text")

    parser.add_argument(
        "--method",
        type=str,
        default="hybrid",
        choices=["tfidf", "bm25", "hybrid"],
        help="Retrieval method to use"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results to return"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Hybrid weight (higher = more BM25 influence)"
    )

    args = parser.parse_args()

    # Load indexes and cleaned dataset
    tfidfVectorizer, tfidfMatrix, bm25Model, tokenizedCleanData, cleanDataFrame = loadIndexes()

    query = args.query
    topK = args.top_k

    print(f"\nSearching for: '{query}'")
    print(f"Retrieval method: {args.method}\n")

    if args.method == "tfidf":
        indices, scores = retrieveTfIdf(query, tfidfVectorizer, tfidfMatrix, topK=topK)

    elif args.method == "bm25":
        indices, scores = retrieveBM25(query, bm25Model, topK=topK)

    else:  
        indices, scores = retrieveHybrid(
            query,
            tfidfVectorizer,
            tfidfMatrix,
            bm25Model,
            alpha=args.alpha,
            topK=topK
        )

    results = formatResults(indices, scores, cleanDataFrame)

    for i, result in enumerate(results):
        print(f"--- Result {i+1} ---")
        print("Document ID:", result["documentId"])
        print("Title:", result["title"])
        print("Score:", round(result["score"], 4))
        print("Snippet:", result["snippet"])
        print("Category:", result["category"], "| Date:", result["date"])
        print()


if __name__ == "__main__":
    main()

# & "C:/Users/sana.ilyas/AppData/Local/Programs/Python/Python312/python.exe" -m src.cli "stock market" --method hybrid --alpha 0.8   
# & "C:/Users/sana.ilyas/AppData/Local/Programs/Python/Python312/python.exe" -m src.cli "oil prices" --method hybrid --top_k 10        