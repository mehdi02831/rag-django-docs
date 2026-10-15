from run_eval import load_questions
from search import hybrid_search, rerank_search

def rank(res, item):
    return next((i + 1 for i, x in enumerate(res) if x["source"] in item["sources"]), None)

for item in [q for q in load_questions() if q["sources"]]:
    h, r = hybrid_search(item["q"], 50), rerank_search(item["q"], 50)
    rh, rr = rank(h, item), rank(r, item)
    if (rr or 99) > (rh or 99):
        print(f"\n{item['q']}\n  hybrid rank {rh} -> rerank rank {rr}")
        print("  expected:", item["sources"])
        print("  rerank top3:", [x["source"] for x in r[:3]])