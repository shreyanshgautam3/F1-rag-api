import json, sys
from app.rag import QdrantMotorsportsRetriever, chain, format_docs

args = [a for a in sys.argv[1:] if not a.startswith("--")]

K = int(args[0] if args else 5)

RETRIEVAL_ONLY = "--retrieval-only" in sys.argv
MIN_HIT_RATE = 0.96

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

    answer = ""
    if not RETRIEVAL_ONLY:
        answer = chain.invoke({"context": format_docs(docs), "question": item["question"]})

    results.append({
        "question": item["question"],
        "expected_answer": item["expected_answer"],
        "answer": answer,
        "contexts": [d.page_content for d in docs],
        "sources": [d.metadata["source_url"] for d in docs],
        "rank": rank,
    })

hit_rate = sum(r["rank"] is not None for r in results) / len(results)
mrr = sum(1 / r["rank"] for r in results if r["rank"]) / len(results)
print(f"k={K}   Hit Rate={hit_rate:.3f}   MRR={mrr:.3f}")

for r in results:
    if r["rank"] is None:
        print("MISS:", r["question"])

if RETRIEVAL_ONLY:
    if hit_rate < MIN_HIT_RATE:
        print(f"Failed: hit rate {hit_rate:.3f} is below {MIN_HIT_RATE}")
        sys.exit(1)

else:
    with open(f"results_k{K}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)