import json, sys
from app.rag import QdrantMotorsportsRetriever, chain, format_docs

K = int(sys.argv[1]) if len(sys.argv) > 1 else 5

with open("golden_set.json", encoding="utf-8") as f:
    golden = json.load(f)

retriever = QdrantMotorsportsRetriever(top_k=K)
results = []

for item in golden:
    docs = retriever.invoke(item["question"])

    rank = None
    for i, d in enumerate(docs):
        if any(exp in d.metadata["source_url"] for exp in item["expected_sources"]):
            rank = i + 1
            break

    context = format_docs(docs)
    answer = chain.invoke({"context": context, "question": item["question"]})

    results.append({
        "question": item["question"],
        "expected_answer": item["expected_answer"],
        "answer": answer,
        "sources": [d.metadata["source_url"] for d in docs],
        "contexts": [d.page_content for d in docs],
        "rank": rank
    })


hit_rate = sum(r["rank"] is not None for r in results) / len(results)
mrr = sum(1 / r["rank"] for r in results if r["rank"]) / len(results)
print(f"k={K} Hit Rate={hit_rate:.3f} MRR={mrr:.3f}")


for r in results:
    if r["rank"] is None:
        print("MISS:", r["question"])

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

with open(f"results_k{K}.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)