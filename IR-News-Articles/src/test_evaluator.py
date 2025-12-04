from src.evaluator import evaluateSystem

queries = [
    "oil prices",
    "pakistan politics",
    "stock market crash",
]

relevance = {
    0: [4, 14, 12, 9, 7, 10, 23, 50, 5, 30],
    1: [54, 315, 2471, 174, 97],
    2: [6, 354, 224, 798, 1],
}

df = evaluateSystem(
    queries,
    relevance,
    method="hybrid",
    k=10,
    alpha=0.6
)

print("\nEvaluation Results:")
print(df)
