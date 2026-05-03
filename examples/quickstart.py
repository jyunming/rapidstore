"""End-to-end quickstart: embed real text, store, search.

Uses sentence-transformers/all-MiniLM-L6-v2 (d=384, ~80 MB download) so the
search results actually mean something — unlike np.random.randn snippets.

Run:
    pip install tqdb sentence-transformers
    python examples/quickstart.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from tqdb import Database

DB_PATH = Path("./quickstart_db")
DOCS = [
    ("rust",    "Rust uses an ownership model with borrowing for memory safety without a garbage collector."),
    ("python",  "Python is a dynamically typed language that prioritizes readability and rapid prototyping."),
    ("ml",      "Gradient descent is the workhorse optimizer for training deep neural networks."),
    ("vector",  "A vector database stores high-dimensional embeddings and supports nearest-neighbour search."),
    ("rag",     "Retrieval-augmented generation grounds language model output in retrieved documents."),
    ("ann",     "HNSW is a graph-based approximate nearest neighbour index with logarithmic search complexity."),
    ("hadoop",  "Hadoop popularised distributed batch processing on commodity hardware."),
    ("kafka",   "Apache Kafka is a distributed log used for high-throughput event streaming."),
]


def main() -> None:
    if DB_PATH.exists():
        shutil.rmtree(DB_PATH)

    print("Loading sentence-transformers model (first run downloads ~80 MB) ...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    dim = model.get_sentence_embedding_dimension()
    print(f"  embedding dimension = {dim}")

    db = Database.open(str(DB_PATH), dimension=dim, bits=4, metric="ip", rerank=True)

    ids = [doc_id for doc_id, _ in DOCS]
    texts = [text for _, text in DOCS]
    vectors = model.encode(texts, normalize_embeddings=True).astype(np.float32)
    db.insert_batch(ids, vectors, documents=texts)
    print(f"Inserted {len(ids)} documents.")

    queries = [
        "How do I avoid memory leaks?",
        "What is a vector store for LLMs?",
        "Tell me about distributed streaming.",
    ]
    for q in queries:
        q_vec = model.encode(q, normalize_embeddings=True).astype(np.float32)
        results = db.search(q_vec, top_k=3)
        print(f"\nQ: {q}")
        for r in results:
            print(f"  [{r['score']:.3f}] {r['id']:7s} — {r['document']}")


if __name__ == "__main__":
    main()
