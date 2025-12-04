from src.preprocess import prepare_for_tfidf, prepare_for_bm25

sample = "Breaking News: Oil prices increase to $85 amid global concerns!"

print("TF-IDF text:", prepare_for_tfidf(sample))
print("BM25 tokens:", prepare_for_bm25(sample))
