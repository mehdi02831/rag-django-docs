# Django Docs RAG: local, measured

A retrieval-augmented question-answering system over the official Django
documentation. Everything runs locally: no paid APIs, no data leaves the machine.

## Architecture

    Django docs (.txt) -> paragraph chunking (~1000 chars, release notes excluded)
      -> embeddings (all-MiniLM-L6-v2) + BM25 index
      -> hybrid retrieval (Reciprocal Rank Fusion)
      -> local LLM (Ollama, qwen3.5:4b) with a strict grounded prompt
      -> FastAPI (/ask, /search, /health)  [Docker]
      -> n8n workflow: webhook -> /ask -> escalate refusals

## Retrieval results (20 answerable questions)

| Variant | recall@1 | recall@5 | recall@10 | MRR | latency/query |
|---|---|---|---|---|---|
| Dense only | 0.60 | 0.95 | 0.95 | 0.746 | 8 ms |
| BM25 only | 0.50 | 0.85 | 1.00 | 0.658 | 6 ms |
| Hybrid (RRF) | 0.65 | 0.95 | 0.95 | 0.777 | 17 ms |
| Hybrid + cross-encoder rerank | 0.45 | 0.90 | 0.95 | 0.636 | 1687 ms |

Hybrid is best, but its margin over dense is one question, which is within noise at n=20.
The off-the-shelf reranker made results worse and about 100x slower, so it is optional and off.

## Generation results (20 answerable + 3 unanswerable)

| Pipeline | false refusals | false answers | s/question |
|---|---|---|---|
| Manual | 0/20 | 0/3 | 5.1 |
| LangChain | 0/20 | 0/3 | 1.8 |
| LlamaIndex | 1/20 | 0/3 | 12.7 |

Same retrieval in all three: frameworks changed convenience and latency, not quality.
Speed differences are unexplained and unverified.

## ReAct agent

A hand-written agent loop (tool: `search_docs`) rewrote one query and fixed the single
retrieval miss ("run code after a model is saved" -> `post_save`). That is one example,
not a measured improvement. Not yet benchmarked on the full set.

## Failure analysis

- Vocabulary gap: dense and BM25 both miss questions that avoid the docs' own terms.
- A "verify with the LLM" step was tried and dropped: the 4B model gave false NOs.
- A strict prompt stopped an early hallucinated Stripe answer; tuned on that case,
  so the 0/3 false-answer result is partly circular.
- Answers sometimes include claims without citations (model drifting from the prompt).
- Labels are file-level, so a correct answer from an unlisted file can count as a miss.

## Limitations

- Only 20 answerable and 3 unanswerable questions; one question = 5 points.
- Outputs vary between runs (sampling).
- Citation correctness is not verified automatically.
- n8n flow tested by hand only.

## RGPD / privacy

- All processing is local: embeddings, LLM (Ollama) and n8n run on the machine.
- No external API calls at query time. Models are downloaded once from Hugging Face.
- The corpus is public documentation; no personal data is ingested.
- Questions are not stored by the API. n8n keeps execution data locally by default,
  so set a retention policy before using real user questions.
- n8n's optional AI Assistant sends data to an external provider and is not used.

## Run

    ollama pull qwen3.5:4b
    python src/rag/build_index.py
    docker compose up --build
    # n8n: docker run -it --rm -p 5678:5678 -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n

The Django docs are not committed. Clone them into `data/django` (BSD-licensed project).
Check n8n's licence terms before any commercial use.