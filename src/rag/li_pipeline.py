from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.ollama import Ollama
from search import hybrid_search
from generate import SYSTEM, MODEL

class HybridRetriever(BaseRetriever):
    def _retrieve(self, query_bundle: QueryBundle):
        return [NodeWithScore(node=TextNode(text=r["text"], metadata={"source": r["source"]}),
                              score=r.get("rrf", 0.0))
                for r in hybrid_search(query_bundle.query_str, 5)]

llm = Ollama(model=MODEL, request_timeout=300, thinking=False, temperature=0.2)

qa_prompt = PromptTemplate(SYSTEM + "\n\nContext:\n{context_str}\n\nQuestion: {query_str}\nAnswer:")
synth = get_response_synthesizer(llm=llm, text_qa_template=qa_prompt, response_mode="compact")
engine = RetrieverQueryEngine(retriever=HybridRetriever(), response_synthesizer=synth)

def li_answer(question: str):
    r = engine.query(question)
    return {"answer": str(r), "sources": sorted({n.node.metadata["source"] for n in r.source_nodes})}

if __name__ == "__main__":
    for q in ["How do I write a custom middleware class?",
              "How do I integrate Stripe payments in Django?"]:
        print("\nQ:", q, "\nA:", li_answer(q)["answer"][:400])