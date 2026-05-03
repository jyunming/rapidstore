"""End-to-end RAG with TurboQuantRetriever — embed, store, retrieve, ground.

Demonstrates the LangChain-style retriever wrapper: pass an embedding function,
add documents, run text-in / documents-out retrieval. Skips the LLM call so the
example runs without an API key — swap in any chat model at the marked TODO.

Run:
    pip install tqdb sentence-transformers
    python examples/rag.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from tqdb.rag import TurboQuantRetriever

DB_PATH = Path("./rag_db")

CORPUS = [
    ("ts-2024-rust",   "Rust 1.78 stabilized async fn in traits, removing the need for the async-trait crate in many cases."),
    ("ts-2024-python", "Python 3.13 introduced an experimental free-threaded build that disables the GIL."),
    ("ts-2024-llm",    "LLM context windows grew from 8k tokens in early 2023 to 1M+ tokens by mid-2024."),
    ("ts-2024-ann",    "DiskANN paper introduced the first SSD-resident graph index that scales to 1B vectors per node."),
    ("ts-2024-quant",  "TurboQuant achieves near-optimal vector compression at 2-4 bits per dim with zero training time."),
    ("ts-2024-rag",    "Retrieval-augmented generation reduces hallucination by grounding LLM output in retrieved passages."),
    ("ts-2024-bm25",   "BM25 remains a strong sparse baseline; hybrid sparse-dense fusion via RRF often beats dense alone."),
]


def main() -> None:
    if DB_PATH.exists():
        shutil.rmtree(DB_PATH)

    print("Loading sentence-transformers model ...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    dim = model.get_sentence_embedding_dimension()

    def embed(texts: list[str]) -> np.ndarray:
        return model.encode(texts, normalize_embeddings=True).astype(np.float32)

    retriever = TurboQuantRetriever(
        db_path=str(DB_PATH),
        dimension=dim,
        bits=4,
        rerank_precision="f16",
        embedding_function=embed,
    )

    ids = [doc_id for doc_id, _ in CORPUS]
    texts = [text for _, text in CORPUS]
    metadatas = [{"year": 2024} for _ in CORPUS]
    retriever.add_texts(texts=texts, ids=ids, metadatas=metadatas)
    print(f"Indexed {len(ids)} passages.")

    question = "How big are LLM context windows now?"
    docs = retriever.similarity_search(query=question, k=3)

    print(f"\nQ: {question}\n")
    print("Top-3 retrieved passages:")
    for i, d in enumerate(docs, 1):
        print(f"  {i}. {d.page_content}")

    context = "\n".join(f"- {d.page_content}" for d in docs)
    prompt = f"Use the context to answer.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    print("\n--- Prompt ready for an LLM (skipped in this example) ---")
    print(prompt[:300] + " ...")
    # TODO: feed `prompt` into your chat model of choice (OpenAI, Anthropic,
    # local Llama, etc.) — TQDB's role ends at retrieval.


if __name__ == "__main__":
    main()
