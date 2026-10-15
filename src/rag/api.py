from fastapi import FastAPI
from pydantic import BaseModel
from generate import answer
from search import hybrid_search
from pathlib import Path
from fastapi.responses import FileResponse

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Django Docs RAG")
@app.get("/", include_in_schema=False)

def home():
    return FileResponse(STATIC / "index.html")

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