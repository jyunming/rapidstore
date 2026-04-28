# Multi-vector documents (ColBERT-style late interaction)

`MultiVectorStore` lets each document hold N token vectors and scores queries
via **MaxSim**:

> `score(q, d) = Σ_i max_j <q_i, d_j>`

Each query token finds its best-matching document token; the per-token best
matches are summed to produce the document score. This is the late-interaction
retrieval that ColBERT / ColBERTv2 use, and it tends to recover keyword and
syntactic signal that single-vector dense retrieval misses.

## Status

This is a **Python-layer wrapper** in v0.8: token vectors are stored as
regular slots in the underlying `Database`; a JSON sidecar tracks the
`doc_id → [token_id]` mapping; raw float32 token vectors are kept in a
NumPy `.npz` sidecar for exact MaxSim scoring.

A future v0.9 release will move multi-vector into the engine (native slot
grouping in `IdPool`, on-disk MaxSim kernel) for tighter storage and native
filter pushdown. **The public Python API is designed to stay stable across
that move** — your code shouldn't need to change.

## Quick start

```python
import numpy as np
from tqdb import MultiVectorStore

store = MultiVectorStore.open("./mvstore", dimension=96, bits=4, metric="cosine")

# Insert a doc with 8 token vectors and the source text.
tokens = np.random.randn(8, 96).astype(np.float32)
tokens /= np.linalg.norm(tokens, axis=1, keepdims=True)
store.insert("paper-2504.19874", tokens, document="full text here",
             metadata={"year": 2025})

# Query with K query-token vectors → MaxSim ranking.
query_tokens = np.random.randn(5, 96).astype(np.float32)
query_tokens /= np.linalg.norm(query_tokens, axis=1, keepdims=True)
hits = store.search(query_tokens, top_k=10)
for h in hits:
    print(h["doc_id"], h["score"], h["document"])
```

## API

```python
store = MultiVectorStore.open(path, dimension, bits=4, metric="cosine")
# or wrap an existing Database:
store = MultiVectorStore(db, directory="./sidecars")

store.insert(doc_id, vectors, document=None, metadata=None) -> [token_id]
store.insert_many([(doc_id, vectors, document, metadata), ...])
store.delete(doc_id) -> bool
store.search(query_vectors, top_k=10, oversample=4, candidate_filter=None) -> [hit]
store.get(doc_id) -> {"doc_id", "n_tokens", "document", "metadata"} | None
store.doc_ids() -> [str]
len(store)
"doc_id" in store
```

`hit` is `{"doc_id": str, "score": float, "document": str | None, "metadata": dict}`.

## Knobs

- `oversample` (default 4): each query token asks the engine for
  `oversample × top_k` token-level hits during candidate generation. Higher
  values improve recall on long-tail queries at the cost of more candidate-doc
  scoring work.
- `candidate_filter`: a metadata filter forwarded to the engine to restrict
  the candidate token universe before MaxSim. Same syntax as
  `Database.search(filter=...)`.

## Recommended config

For ColBERTv2-style late interaction:

| Knob | Value | Why |
|------|-------|-----|
| `dimension` | 96 (ColBERTv2) or 128 (ColBERT) | Per the model |
| `bits` | 4 | Good recall at modest disk |
| `metric` | `cosine` | ColBERT vectors are unit-normalised |
| `oversample` | 4 | Default; raise to 8-16 for long-tail recall |

## Limitations (v0.8 — temporary)

- Inserting `N` token vectors does `N` engine row-inserts. A single insert
  with N=8 is ~5× slower than a single-vector insert.
- Search wall-clock scales with the candidate doc count after union, not
  with `top_k`. Set a tight `candidate_filter` for narrow queries.
- The raw float32 sidecar (`_mv_tokens.npz`) is read-back-rewrite for every
  modification — fine at 10k docs / 80k tokens, slow at 100k+ docs.

All three are addressed by the v0.9 native engine integration.
