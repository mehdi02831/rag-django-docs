import json
from pathlib import Path
from generate import answer

REFUSAL = "I don't know based on the documentation."
qs = [json.loads(l) for l in Path("eval_set/questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

false_ref, false_ans, bad_cite = [], [], []
for item in qs:
    out = answer(item["q"])
    refused = out["answer"].startswith("I don't know")
    if item["sources"]:
        if refused:
            false_ref.append(item["q"])
        elif not set(out["sources"]) & set(item["sources"]):
            bad_cite.append(item["q"])
    elif not refused:
        false_ans.append(item["q"])

ans_n = sum(1 for q in qs if q["sources"])
un_n = len(qs) - ans_n
print(f"\nFalse refusals (answerable but refused): {len(false_ref)}/{ans_n}")
for q in false_ref: print("  -", q)
print(f"False answers (unanswerable but answered): {len(false_ans)}/{un_n}")
for q in false_ans: print("  -", q)
print(f"Answered but cited none of the expected files: {len(bad_cite)}")
for q in bad_cite: print("  -", q)