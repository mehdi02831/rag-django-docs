import json
from pathlib import Path

DOCS = Path("data/django/docs")
bad = 0
for line in Path("eval_set/questions.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    item = json.loads(line)
    for s in item["sources"]:
        if not (DOCS / s).exists():
            print("MISSING:", s, "<-", item["q"])
            bad += 1
print("all paths exist" if bad == 0 else f"{bad} bad paths")