"""
Real-world RAG quality test.

Ingests a folder of PDFs and Markdown files into TurboQuantDB,
then runs a set of domain-relevant queries and evaluates retrieval quality.

Usage:
    py -3.13 benchmarks/real_world_rag_test.py --folder "C:/dev/studio_brain_open/Qualification/papers"
"""

import argparse
import os
import re
import sys
import time
import shutil
from tempfile import mkdtemp

import numpy as np
import pymupdf          # PDF extraction
import turboquantdb
from fastembed import TextEmbedding

sys.stdout.reconfigure(encoding="utf-8")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE   = 400   # tokens (approx words)
CHUNK_OVERLAP = 80

# Queries relevant to the lithography/EUV paper collection
QUERIES = [
    "What is the effect of shot noise in EUV lithography?",
    "How does acid diffusion affect critical dimension in chemically amplified resists?",
    "What are the stochastic limits to EUV scaling?",
    "What is the role of mask topography in EUV imaging?",
    "How does high-NA EUV improve resolution?",
    "What are the sources of line edge roughness in photoresist?",
    "How is optical proximity correction used in lithography simulation?",
    "What are the metrics for stochastic defects in EUV?",
    "How does crosslinking mechanism affect MOR photoresist performance?",
    "What are the fundamental limits of optical lithography?",
]


# ── Text extraction ────────────────────────────────────────────────────────────

def extract_pdf(path):
    doc = pymupdf.open(path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n".join(pages)


def extract_md(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def extract_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_pdf(path)
    elif ext in (".md", ".txt"):
        return extract_md(path)
    return ""


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--bits",   type=int, default=8)
    parser.add_argument("--top_k",  type=int, default=5)
    args = parser.parse_args()

    folder = args.folder

    # ── Step 1: Load files ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Real-World RAG Test  |  model={EMBED_MODEL}  bits={args.bits}")
    print(f"  Folder: {folder}")
    print(f"{'='*70}\n")

    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith((".pdf", ".md", ".txt")))
    print(f"  Found {len(files)} files:")
    for f in files:
        print(f"    {f}")

    # ── Step 2: Extract + chunk ───────────────────────────────────────────────
    print(f"\n  Extracting and chunking...")
    all_chunks = []   # list of (id, text, source_file)
    for fname in files:
        path = os.path.join(folder, fname)
        text = extract_file(path)
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append((f"{fname}::{i}", chunk, fname))
        print(f"    {fname:<55} {len(chunks):>4} chunks")

    print(f"\n  Total chunks: {len(all_chunks)}")

    # ── Step 3: Embed ─────────────────────────────────────────────────────────
    print(f"\n  Embedding with {EMBED_MODEL}...")
    t0 = time.perf_counter()
    embedder = TextEmbedding(EMBED_MODEL)
    texts = [c[1] for c in all_chunks]
    embeddings = np.array(list(embedder.embed(texts)), dtype=np.float32)
    # Normalize for inner product similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    embeddings /= norms
    embed_time = time.perf_counter() - t0
    print(f"  Embedded {len(embeddings)} chunks in {embed_time:.1f}s  "
          f"({len(embeddings)/embed_time:.0f} chunks/s)  dim={embeddings.shape[1]}")

    # ── Step 4: Ingest into TurboQuantDB ──────────────────────────────────────
    print(f"\n  Ingesting into TurboQuantDB (bits={args.bits}, rerank=True)...")
    tmp = mkdtemp(prefix="tqdb_rag_")
    db_path = os.path.join(tmp, "rag.tqdb")
    db = turboquantdb.TurboQuantDB.open(
        db_path, dimension=embeddings.shape[1],
        bits=args.bits, metric="ip", rerank=True)

    ids       = [c[0] for c in all_chunks]
    metadatas = [{"source": c[2], "chunk_idx": int(c[0].split("::")[-1])} for c in all_chunks]
    documents = [c[1] for c in all_chunks]

    t0 = time.perf_counter()
    db.insert_batch(ids, embeddings, metadatas=metadatas, documents=documents)
    ingest_time = time.perf_counter() - t0
    print(f"  Inserted {len(ids)} vectors in {ingest_time:.2f}s")

    print(f"  Building HNSW index (m=32, ef_construction=200)...")
    t0 = time.perf_counter()
    db.create_index(max_degree=32, ef_construction=200, search_list_size=128)
    index_time = time.perf_counter() - t0
    print(f"  Index built in {index_time:.2f}s")

    stats = db.stats()
    print(f"  DB stats: {stats}")

    # ── Step 5: Query ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  RETRIEVAL RESULTS  (top_k={args.top_k})")
    print(f"{'='*70}")

    query_vecs = np.array(list(embedder.embed(QUERIES)), dtype=np.float32)
    query_vecs /= np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-9

    latencies = []
    for i, (query, qvec) in enumerate(zip(QUERIES, query_vecs)):
        print(f"\n  Q{i+1}: {query}")
        print(f"  {'-'*66}")
        t0 = time.perf_counter()
        results = db.search(qvec, top_k=args.top_k, ann_search_list_size=128)
        latencies.append((time.perf_counter() - t0) * 1000)

        for rank, r in enumerate(results, 1):
            source = r.get("metadata", {}).get("source", "?")
            score  = r.get("score", 0)
            doc    = r.get("document") or ""
            snippet = " ".join(doc.split()[:25])
            print(f"  [{rank}] {source:<50} score={score:.4f}")
            print(f"       \"{snippet}...\"")

    # ── Step 6: Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Total chunks ingested : {len(all_chunks)}")
    print(f"  Embedding time        : {embed_time:.1f}s")
    print(f"  Ingest time           : {ingest_time:.2f}s")
    print(f"  Index build time      : {index_time:.2f}s")
    print(f"  Query p50 latency     : {float(np.median(latencies)):.2f}ms")
    print(f"  Query p95 latency     : {float(np.percentile(latencies, 95)):.2f}ms")
    print()

    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
