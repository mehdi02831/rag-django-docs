import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


IDX = Path(__file__).resolve().parents[2] / "data" / "index"
_emb = np.load(IDX / "emb.npy")
_chunks = [json.loads(l) for l in (IDX / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
_model = SentenceTransformer("all-MiniLM-L6-v2")

def dense_search(query: str, k: int = 10):
    q = _model.encode([query], normalize_embeddings=True)[0]
    scores = _emb @ q          # cosine similarity (vectors are normalized)
    top = np.argsort(-scores)[:k]
    return [{**_chunks[i], "score": float(scores[i])} for i in top]



_STOP = {"how","do","i","a","an","the","to","of","in","is","it","and","or","for",
         "my","what","does","can","with","on","from","are","be","by","this","that","between"}

def _tok(s: str):
    return [t for t in re.findall(r"[a-z0-9_]+", s.lower()) if t not in _STOP]

_bm25 = BM25Okapi([_tok(c["text"]) for c in _chunks])

def bm25_search(query: str, k: int = 10):
    scores = _bm25.get_scores(_tok(query))
    top = np.argsort(-scores)[:k]
    return [{**_chunks[i], "score": float(scores[i])} for i in top]

def hybrid_search(query: str, k: int = 10, pool: int = 50, c: int = 60):
    # Reciprocal Rank Fusion: each list gives 1/(c+rank) to every chunk it returns
    fused = {}
    for results in (dense_search(query, pool), bm25_search(query, pool)):
        for rank, r in enumerate(results, start=1):
            key = (r["source"], r["chunk_id"])
            if key not in fused:
                fused[key] = {**r, "rrf": 0.0}
            fused[key]["rrf"] += 1 / (c + rank)
    return sorted(fused.values(), key=lambda r: -r["rrf"])[:k]

_ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_search(query: str, k: int = 10, pool: int = 50):
    cands = hybrid_search(query, k=pool, pool=pool)
    scores = _ce.predict([(query, c["text"]) for c in cands])
    order = np.argsort(-scores)[:k]
    return [{**cands[i], "score": float(scores[i])} for i in order]