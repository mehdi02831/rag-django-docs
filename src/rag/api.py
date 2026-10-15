from fastapi import FastAPI
from pydantic import BaseModel
from generate import answer
from search import hybrid_search

app = FastAPI(title="Django Docs RAG")

class Ask(BaseModel):
    question: str
    k: int = 5

class Answer(BaseModel):
    answer: str
    sources: list[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=Answer)
def ask(body: Ask):
    out = answer(body.question, body.k)
    return {"answer": out["answer"], "sources": out["sources"]}

@app.get("/search")
def search(q: str, k: int = 5):
    return [{"source": r["source"], "text": r["text"][:300]} for r in hybrid_search(q, k)]