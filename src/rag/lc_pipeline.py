from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from search import hybrid_search
from generate import SYSTEM, MODEL

class HybridRetriever(BaseRetriever):
    k: int = 5
    def _get_relevant_documents(self, query, *, run_manager=None):
        return [Document(page_content=r["text"], metadata={"source": r["source"]})
                for r in hybrid_search(query, self.k)]

llm = ChatOllama(model=MODEL, reasoning=False, temperature=0.2)
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user", "Context:\n{context}\n\nQuestion: {question}"),
])
chain = prompt | llm | StrOutputParser()
retriever = HybridRetriever()

def lc_answer(question: str):
    docs = retriever.invoke(question)
    context = "\n\n".join(f"[{i+1}] ({d.metadata['source']})\n{d.page_content}" for i, d in enumerate(docs))
    text = chain.invoke({"context": context, "question": question})
    return {"answer": text, "sources": sorted({d.metadata["source"] for d in docs})}

if __name__ == "__main__":
    for q in ["How do I write a custom middleware class?",
              "How do I integrate Stripe payments in Django?"]:
        print("\nQ:", q, "\nA:", lc_answer(q)["answer"][:400])