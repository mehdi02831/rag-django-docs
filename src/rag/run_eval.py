import json
from pathlib import Path
from search import dense_search

def load_questions():
    lines = Path("eval_set/questions.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]

def evaluate(retriever, ks=(1, 5, 10)):
    qs = [q for q in load_questions() if q["sources"]]
    hits = {k: 0 for k in ks}
    rr = 0.0
    for item in qs:
        results = retriever(item["q"], max(ks))
        rank = next((i + 1 for i, r in enumerate(results) if r["source"] in item["sources"]), None)
        if rank:
            rr += 1 / rank
            for k in ks:
                if rank <= k:
                    hits[k] += 1
        else:
            print("MISS:", item["q"])
    n = len(qs)
    for k in ks:
        print(f"recall@{k}: {hits[k]/n:.2f}")
    print(f"MRR: {rr/n:.3f}  (n={n})")

if __name__ == "__main__":
    import time
    from search import dense_search, bm25_search, hybrid_search, rerank_search
    qs = [q["q"] for q in load_questions() if q["sources"]]
    for name, fn in [("dense", dense_search), ("bm25", bm25_search),
                     ("hybrid", hybrid_search), ("hybrid+rerank", rerank_search)]:
        print(f"\n=== {name} ===")
        evaluate(fn)
        t = time.perf_counter()
        for q in qs:
            fn(q, 10)
        print(f"latency: {(time.perf_counter() - t) / len(qs) * 1000:.0f} ms/query")