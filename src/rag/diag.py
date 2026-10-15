from run_eval import load_questions
from search import dense_search, bm25_search, hybrid_search

fns = {"dense": dense_search, "bm25": bm25_search, "hybrid": hybrid_search}

def rank_of(fn, item):
    res = fn(item["q"], 50)
    return next((i + 1 for i, r in enumerate(res) if r["source"] in item["sources"]), None)

print(f"{'question':60} dense bm25 hybrid")
for item in [q for q in load_questions() if q["sources"]]:
    print(f"{item['q'][:58]:60}", *[f"{str(rank_of(f, item)):>5}" for f in fns.values()])

print("\nTop 5 dense for the signals question:")
for r in dense_search("How do I run code automatically after a model instance is saved?", 5):
    print(r["source"], "|", r["text"][:90].replace("\n", " "))