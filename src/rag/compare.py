import json, time
from pathlib import Path
from generate import answer
from lc_pipeline import lc_answer
from li_pipeline import li_answer

qs = [json.loads(l) for l in Path("eval_set/questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
pipes = {"manual": lambda q: answer(q), "langchain": lc_answer, "llamaindex": li_answer}
answer("warm up")
for name, fn in pipes.items():
    fr, fa, bc, t0 = [], [], [], time.perf_counter()
    for item in qs:
        out = fn(item["q"])
        refused = "I don't know based on the documentation" in out["answer"]
        if item["sources"]:
            if refused: fr.append(item["q"])
            elif not set(out["sources"]) & set(item["sources"]): bc.append(item["q"])
        elif not refused:
            fa.append(item["q"])
    n_ans = sum(1 for q in qs if q["sources"])
    print(f"\n=== {name} ({(time.perf_counter()-t0)/len(qs):.1f} s/question) ===")
    print(f"false refusals: {len(fr)}/{n_ans}  false answers: {len(fa)}/{len(qs)-n_ans}  cited none of expected: {len(bc)}")
    for q in fr + fa: print("   -", q)