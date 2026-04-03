# Python API Reference

Complete reference for the `turboquantdb` Python package.

---

## Installation

```bash
pip install tqdb
```

Requires Python 3.10+. Pre-built wheels available for Linux, Windows, and macOS.

---

## Opening a Database

```python
from turboquantdb import Database

db = Database.open(
    path,                    # str — base directory path, created if it doesn't exist
    dimension,               # int — vector dimension, must match on every reopen
    bits=4,                  # int — quantization bits: 4 (4.2× compression) or 8 (2.47×, higher recall)
    seed=42,                 # int — RNG seed for quantizer, must stay the same across sessions
    metric="ip",             # str — "ip" (inner product), "cosine", or "l2"
    rerank=True,             # bool — enable reranking of ANN candidates; precision via rerank_precision
    fast_mode=False,         # bool — skip QJL stage: ~30% faster ingest, ~5pp recall loss
    rerank_precision=None,   # str|None — None = dequant reranking (no extra storage)
                             #            "f16" = float16 exact reranking (+n×d×2 bytes)
                             #            "f32" = float32 exact reranking (+n×d×4 bytes)
    collection=None,         # str|None — subdirectory name for the collection; if given,
                             #            the DB is stored at path/collection/ instead of path/
)
```

`path` must use the same `dimension`, `bits`, `seed`, and `metric` every time it is opened — these are baked into the quantizer and cannot be changed after creation.

### Multi-collection pattern

Use the `collection` parameter to store multiple isolated namespaces under one base directory:

```python
articles = Database.open("./mydb", dimension=1536, collection="articles")
images   = Database.open("./mydb", dimension=512,  collection="images")
# Stored at ./mydb/articles/ and ./mydb/images/ respectively
```

---

## Recommended Presets

### High Quality — recall matters most

```python
db = Database.open(path, dimension=DIM, bits=8, rerank=True)
db.create_index(max_degree=32, ef_construction=200, n_refinements=8)
results = db.search(query, top_k=10, ann_search_list_size=200)
# ~97% Recall@10 at 50k×1536  |  119 MB disk  |  12ms p50
```

### Balanced — default recommendation

```python
db = Database.open(path, dimension=DIM, bits=4, rerank=True)
db.create_index(max_degree=32, ef_construction=200, n_refinements=5)
results = db.search(query, top_k=10, ann_search_list_size=128)
# ~89% Recall@10 at 50k×1536  |  70 MB disk  |  10ms p50
```

### Fast Build — ingest speed is priority

```python
db = Database.open(path, dimension=DIM, bits=4, fast_mode=True, rerank=False)
db.create_index(max_degree=32, ef_construction=200, n_refinements=5)
results = db.search(query, top_k=10, ann_search_list_size=128)
# ~83% Recall@10 at 50k×1536  |  70 MB disk  |  5ms p50
```

*Benchmarked at 50,000 vectors, dim=1536, top_k=10, brute-force ground truth.*

---

## Insert

```python
import numpy as np

# Single insert
db.insert(
    id,              # str
    vector,          # np.ndarray (float32) or list[float]
    metadata=None,   # dict | None
    document=None    # str | None
)

# Batch insert — recommended for loading large datasets
db.insert_batch(
    ids,             # list[str]
    vectors,         # np.ndarray shape (N, D) or list[list[float]]
    metadatas=None,  # list[dict] | None
    documents=None,  # list[str | None] | None
    mode="insert"    # "insert" | "upsert" | "update"
)
# mode semantics:
#   "insert"  — raises RuntimeError if any ID already exists (atomic: fails at first duplicate)
#   "upsert"  — insert new IDs or replace existing ones (always succeeds)
#   "update"  — raises RuntimeError if any ID does not exist (atomic: fails at first missing ID)

# Single upsert / update
db.upsert(id, vector, metadata=None, document=None)  # insert or replace (always succeeds)
db.update(id, vector, metadata=None, document=None)  # raises RuntimeError if id not found
```

---

## Delete & Retrieve

```python
db.delete(id)                    # bool — True if id existed
db.delete_batch(ids)             # deletes multiple ids at once

db.get(id)                       # dict | None — {id, metadata, document}
db.get_many(ids)                 # list[dict | None]
db.list_all()                    # list[str] — all active ids
db.list_ids(                     # paginated id list with optional filter
    where_filter=None,           # dict | None — same filter syntax as search
    limit=None,                  # int | None — max results (None = all)
    offset=0,                    # int — skip this many results
)
db.count(filter=None)            # int — number of matching vectors

# Update metadata without re-uploading the vector
db.update_metadata(
    id,                          # str — must exist; raises RuntimeError otherwise
    metadata=None,               # dict | None — replaces metadata; None = keep existing
    document=None,               # str | None — replaces document; None = keep existing
)

db.stats()                       # dict — vector_count, disk_bytes, has_index, …

# Python container protocol
len(db)                          # int — number of active vectors
"my-id" in db                    # bool — True if id exists
```

---

## Search

```python
results = db.search(
    query,                       # np.ndarray (float32) or list[float]
    top_k=10,                    # int
    filter=None,                 # dict | None  (see Metadata Filtering below)
    _use_ann=True,               # bool — use HNSW index if available
    ann_search_list_size=None,   # int | None — HNSW ef_search (default: max_degree × 2)
    include=None,                # list[str] | None — fields to return; default all
                                 #   valid values: "id", "score", "metadata", "document"
)
# Returns list of dicts: {"id": str, "score": float, "metadata": dict, "document": str | None}
```

`ann_search_list_size` trades recall for latency — higher values find better results but take longer. Values between 64 and 256 cover the practical range.

### Batch query

Search with multiple vectors in one call:

```python
all_results = db.query(
    query_embeddings,            # np.ndarray shape (N, D), float32 or float64
    n_results=10,                # int — results per query
    where_filter=None,           # dict | None
    _use_ann=True,
    ann_search_list_size=None,
)
# Returns list[list[dict]] — one inner list per query vector
```

---

## Index (HNSW)

Build the index **after** loading your data. Rebuild after large batches of inserts — it is not updated incrementally.

```python
db.create_index(
    max_degree=32,        # int — max neighbors per node; higher = better recall, larger graph (default 32)
    ef_construction=200,  # int — beam size during build; higher = better quality, slower build (default 200)
    n_refinements=5,      # int — refinement passes; higher = better graph, slower build (default 5)
    search_list_size=128, # int — alias for ef_construction (default 128)
    alpha=1.2,            # float — pruning aggressiveness (default 1.2)
)
```

All parameters are optional. `None` uses the listed defaults.

| Parameter | Recall impact | Build time impact |
|-----------|--------------|------------------|
| `max_degree` 16 → 32 | +5–8pp | +50% |
| `ef_construction` 64 → 200 | +3–5pp | +2× |
| `n_refinements` 0 → 8 | +2–4pp | +30% |

---

## Metadata Filtering

Filters are evaluated in-process during search, applied before scoring.

```python
# Simple equality
db.search(query, top_k=5, filter={"topic": "ml"})

# Multiple fields (implicit $and)
db.search(query, top_k=5, filter={"topic": "ml", "year": 2024})

# Comparison operators: $eq  $ne  $gt  $gte  $lt  $lte
db.search(query, top_k=5, filter={"year": {"$gte": 2023}})

# Set operators: $in  $nin
db.search(query, top_k=5, filter={"status": {"$in": ["published", "featured"]}})
db.search(query, top_k=5, filter={"status": {"$nin": ["draft", "archived"]}})

# Field presence: $exists
db.search(query, top_k=5, filter={"tags": {"$exists": True}})   # field must be present
db.search(query, top_k=5, filter={"tags": {"$exists": False}})  # field must be absent

# Substring match: $contains (strings only)
db.search(query, top_k=5, filter={"title": {"$contains": "neural"}})

# Logical: $and  $or
db.search(query, top_k=5, filter={
    "$and": [
        {"topic": "ml"},
        {"year": {"$gte": 2023}}
    ]
})

# Nested field paths (dot notation)
db.search(query, top_k=5, filter={"profile.region": "eu"})
```

Filter semantics:
- `$ne` and `$nin` match documents where the field is missing
- `$eq`, range operators, and `$in` do not match missing fields
- `$exists: true` matches documents that have the field; `$exists: false` matches those that don't
- `$contains` does substring matching on string fields only
- No implicit type coercion — `{"year": "2023"}` will not match `{"year": 2023}`

---

## RAG Integration

`TurboQuantRetriever` is a lightweight LangChain-style wrapper around `Database`.

```python
from turboquantdb.rag import TurboQuantRetriever

retriever = TurboQuantRetriever(
    db_path,                # str — directory path for the database
    dimension=1536,         # int — vector dimension
    bits=4,                 # int — quantization bits (4 or 8)
    seed=42,                # int — RNG seed
    metric="ip",            # str — "ip", "cosine", or "l2"
    rerank_precision=None,  # str|None — None (dequant), "f16", or "f32"
)
```

### `add_texts(texts, embeddings, metadatas=None)`

Batch-insert documents with their embeddings.

```python
retriever.add_texts(
    texts=["Document one.", "Document two."],
    embeddings=[vec1, vec2],       # np.ndarray or list[list[float]]
    metadatas=[{"src": "a"}, {"src": "b"}]  # optional
)
```

IDs are auto-assigned as `doc_0`, `doc_1`, … continuing from the current count.

### `similarity_search(query_embedding, k=4)`

Search for the `k` most similar documents.

```python
results = retriever.similarity_search(query_embedding=query_vec, k=5)
for r in results:
    print(r["score"], r["text"], r["metadata"])
```

Returns a `list[dict]` with keys: `"text"`, `"metadata"`, `"score"`.

---

## On-disk Layout

```
./my_db/
├── manifest.json        — DB config (dimension, bits, seed, metric)
├── quantizer.bin        — Serialized quantizer state
├── live_codes.bin       — Memory-mapped quantized vectors (hot path)
├── live_vectors.bin     — Raw vectors for exact reranking (only if rerank_precision="f16" or "f32")
├── wal.log              — Write-ahead log (crash recovery)
├── metadata.bin         — Per-vector metadata and documents
├── live_ids.bin         — ID → slot index
├── graph.bin            — HNSW adjacency list (if index built)
└── seg-XXXXXXXX.bin     — Immutable flushed segment files
```
