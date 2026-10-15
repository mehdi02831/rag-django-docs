import json, re
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

DOCS = Path("data/django/docs")
OUT = Path("data/index")
OUT.mkdir(parents=True, exist_ok=True)

def chunk_file(text, max_chars=1000):
    paras = re.split(r"\n\s*\n", text)
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > max_chars:
            chunks.append(cur.strip())
            cur = ""
        cur += p + "\n\n"
    if cur.strip():
        chunks.append(cur.strip())
    return chunks

records = []
for f in DOCS.rglob("*.txt"):
    rel = f.relative_to(DOCS).as_posix()
    if rel.startswith("releases/"):
        continue
    for i, c in enumerate(chunk_file(f.read_text(encoding="utf-8", errors="ignore"))):
        if len(c) > 80:
            records.append({"source": rel, "chunk_id": i, "text": c})

print(len(records), "chunks")
model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode([r["text"] for r in records], batch_size=64,
                   normalize_embeddings=True, show_progress_bar=True)
np.save(OUT / "emb.npy", emb)
(OUT / "chunks.jsonl").write_text(
    "\n".join(json.dumps(r) for r in records), encoding="utf-8")