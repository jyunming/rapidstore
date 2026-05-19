# Python API Reference

Complete reference for the `tqdb` Python package.

---

## Installation

```bash
pip install tqdb
```

Requires Python 3.10+. Pre-built wheels available for Linux, Windows, and macOS.

---

## Opening a Database

```python
from tqdb import Database

db = Database.open(
    path,                    # str — base directory path, created if it doesn't exist
    dimension=None,          # int|None — vector dimension. Required for new databases.
                             #            Omit to reopen an existing database — the
                             #            dimension and fixed params are loaded from manifest.json.
    bits=4,                  # int — quantization bits, 1..8. bits=1 is allowed only
                             #        with fast_mode=True; common values are 2, 4, 8.
    seed=42,                 # int — RNG seed for quantizer, must stay the same across sessions
    metric="ip",             # str — "ip" (inner product), "cosine", or "l2"
    rerank=False,            # bool — store raw vectors for exact second-pass rescoring.
                             #        False (default) = MSE codes only, minimum disk.
                             #        True = stores INT8 raw vectors; +5–25 pp R@1 depending on d.
    fast_mode=True,          # bool — True (default) = MSE-only (paper Figure 5 allocation,
                             #        fastest ingest, minimum disk).
                             #        False = also stores 1-bit QJL residual; adds +5–10 pp R@1
                             #        at d ≥ 1536 when rerank=False; no benefit when rerank=True.
    rerank_precision=None,   # str|None — None → "int8" (default when rerank=True)
                             #            "int8"/"i8" = INT8 per-vector-scaled (+n×(d+4) bytes)
                             #            "residual_int4"/"ri4" = residual INT4 rerank
                             #            "int4"/"i4" = deprecated raw INT4 rerank
                             #            "f16" = float16 exact reranking (+n×d×2 bytes)
                             #            "f32" = float32 exact reranking (+n×d×4 bytes)
                             #            "disabled"/"dequant" = no extra storage (legacy)
    collection=None,         # str|None — subdirectory name for the collection; if given,
                             #            the DB is stored at path/collection/ instead of path/
    normalize=False,         # bool — L2-normalize every inserted vector and every query at
                             #        write time; makes inner-product scoring equivalent to
                             #        cosine similarity without changing the metric parameter
    quantizer_type=None,     # str|None — None = auto: "dense" if d<1024, "srht" if d>=1024.
                             #            "dense" = Haar-uniform QR (O(d²) ingest, n=d).
                             #                     Rotation matrix stored as bf16 on disk.
                             #            "exact" is a legacy alias for "dense".
                             #            "srht" = structured Walsh-Hadamard (O(d log d) ingest,
                             #                    n=next_power_of_two(d), faster ingest+search at
                             #                    d>=1024).
)

# Parameterless reopen — reads all parameters from manifest.json:
db = Database.open("./mydb")
```

When reopening an existing database, `dimension` may be omitted; `bits`, `seed`, and `metric` are loaded from the stored `manifest.json` automatically.  `dimension` is required only when creating a new database (no manifest present).

Pass an `s3://bucket/prefix` URI (requires the `cloud` Cargo feature) to store segments in Amazon S3 with a local write-through cache (see [On-disk Layout](#on-disk-layout)).

### Multi-collection pattern

Use the `collection` parameter to store multiple isolated namespaces under one base directory:

```python
articles = Database.open("./mydb", dimension=1536, collection="articles")
images   = Database.open("./mydb", dimension=512,  collection="images")
# Stored at ./mydb/articles/ and ./mydb/images/ respectively
```

### Quantizer modes

TurboQuantDB exposes the same two-stage MSE + residual-QJL layout through two quantizer families:

- **`None` (default)** — auto-pick based on dimension. `"dense"` for `d < 1024`, `"srht"` for `d >= 1024`. The crossover reflects empirical wins for SRHT once the rotation dominates ingest/latency cost.
- **`"dense"`** — Haar-uniform QR rotation + dense i.i.d. Gaussian QJL with `n = d`. O(d²) ingest cost; rotation matrix stored as bf16 on disk (~50% rotation-tax savings vs f32). `"exact"` is a legacy alias.
- **`"srht"`** — structured Walsh-Hadamard + random-sign transforms, `n = next_power_of_two(d)`, O(d log d) ingest. At `d >= 1024` this is now the default — typically 2–3× faster ingest, 1.5–3× lower p50 latency, with recall equal to or slightly better than dense in our public-bench numbers.

See [`docs/QUANTIZER_MODES.md`](QUANTIZER_MODES.md) for the full per-dim trade-off table including disk and RAM.

### `fast_mode` and `rerank` interaction

These two parameters work together and must be understood as a pair:

| `fast_mode` | `rerank` | Storage | Recall impact |
|-------------|----------|---------|---------------|
| `True` (default) | `False` (default) | MSE codes only | Minimum disk; fastest ingest |
| `True` | `True` | MSE codes + INT8 raw vectors | +5–25 pp R@1 vs default; ~2–4× more disk |
| `False` | `False` | MSE+QJL codes | d ≥ 1536: +5–10 pp R@1; d < 512: hurts recall |
| `False` | `True` | MSE+QJL codes + INT8 raw vectors | Maximum recall at d ≥ 1536 |

**`rerank=True`** stores per-vector-scaled INT8 vectors by default for exact second-pass rescoring. The quantized pass pre-selects `rerank_factor × top_k` candidates; exact dot products on the dequantized INT8 vectors then re-rank to the final `top_k`. This consistently improves R@1 by +5–25 pp depending on dimension and bits, at ~2× lower disk cost than F16.

**`fast_mode=False`** adds 1-bit QJL residual codes on top of MSE codes. At d < 512, the projections are too noisy and reduce recall. At d ≥ 1536, enough bits accumulate to add meaningful signal — gains +5–10 pp R@1 when `rerank=False`. When `rerank=True`, the QJL residuals provide secondary benefit because raw-vector re-scoring dominates.

Validation edge cases are intentionally Python-visible: `dimension=0`, `bits=0`, `bits>8`, `bits=1` with `fast_mode=False`, unknown `quantizer_type`, non-finite vectors, vector dimension mismatches, `top_k<=0`, `n_results<=0`, negative `limit`, and negative `offset` raise `ValueError`. `insert_batch(..., metadatas=[None, ...])` treats each `None` row as empty metadata, matching Chroma collections that have no metadata.

---

## Recommended Presets

See [`docs/CONFIGURATION.md`](CONFIGURATION.md) for a full decision guide with storage estimates and scenario presets.

### Default — minimum disk, fastest ingest

```python
db = Database.open(path, dimension=DIM, bits=4)
# rerank=False, fast_mode=True (library defaults) — MSE codes only; minimum disk
results = db.search(query, top_k=10)
# GloVe-200 (d=200):     R@1 ≈ 0.82  |  ~22 MB disk
# arXiv-768 (d=768):     R@1 ≈ 0.70  |  ~48 MB disk
# DBpedia-1536 (d=1536): R@1 ≈ 0.92  |  ~108 MB disk
```

### Best recall — enable INT8 rerank

```python
db = Database.open(path, dimension=DIM, bits=4, rerank=True)
# INT8 raw-vector reranking (default precision); +5–25 pp R@1 vs no-rerank
results = db.search(query, top_k=10)
# GloVe-200 (d=200):     R@1 ≈ 1.00  |  ~30 MB disk
# arXiv-768 (d=768):     R@1 ≈ 0.98  |  ~116 MB disk
# DBpedia-1536 (d=1536): R@1 ≈ 0.95  |  ~231 MB disk
```

### Best recall, high-d (d ≥ 1536)

```python
db = Database.open(path, dimension=1536, bits=4, rerank=True, fast_mode=False)
# QJL residuals provide +3–8 pp R@1 vs fast_mode=True at d≥1536
results = db.search(query, top_k=10)
```

### Minimum disk — compressed codes only

```python
db = Database.open(path, dimension=DIM, bits=4)   # rerank=False is the default
# No raw vectors stored; same as Default preset above
```

### Low latency at scale — HNSW index

```python
db = Database.open(path, dimension=DIM, bits=4)
db.create_index(max_degree=32, ef_construction=200, n_refinements=5)
results = db.search(query, top_k=10, _use_ann=True, rerank_factor=20)
# p50 < 10ms at N=100k; R@1 ~2–5 pp below brute-force
```

### Fast ingest, high-dimensional

```python
db = Database.open(path, dimension=1536, bits=4, rerank=False, quantizer_type="srht")
# 4,000–7,000 vps vs 2,000 vps for dense; O(d log d) rotation
```

*Full benchmark data for all 32 config × 4 dataset combinations: run `python benchmarks/full_config_bench.py --full` — results appear in `benchmarks/_full_config_report.md`.*

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
db.delete_batch(                 # → int — count of ids that were deleted
    ids=[],                      # list[str] — explicit IDs to delete (may be empty)
    where_filter=None,           # dict | None — delete all vectors matching this filter
)                                #   Examples:
                                 #     db.delete_batch(["id1", "id2"])
                                 #     db.delete_batch(where_filter={"year": {"$lt": 2020}})
                                 #     db.delete_batch(["id1"], where_filter={"tag": "old"})

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

db.stats()                       # dict — see Stats Keys below
db.flush()                       # flush WAL to a segment file immediately
db.close()                       # flush and release all file handles

# Python container protocol
len(db)                          # int — number of active vectors
"my-id" in db                    # bool — True if id exists
```

### Crash Recovery Playbook (WAL)

`wal.log` is replayed automatically on reopen. Use this playbook after an unclean shutdown:

1. Stop writes to the database path.
2. Copy the DB directory as a safety backup.
3. Reopen:
   ```python
   db = Database.open("./my_db")
   ```
4. Validate:
   - `db.stats()["vector_count"]`
   - representative `db.get(...)` and `db.search(...)`
5. Finalize recovered state:
   ```python
   db.checkpoint()   # flush WAL + compact segments
   db.close()
   ```

If recovery cannot complete due to file corruption, restore from snapshot/backup (server mode: `POST .../snapshot` and `POST .../restore`; see `docs/SERVER_API.md`).

**Stats keys** returned by `db.stats()`:

| Key | Type | Description |
|-----|------|-------------|
| `vector_count` | `int` | Total active (non-deleted) vectors |
| `segment_count` | `int` | Number of immutable segment files |
| `buffered_vectors` | `int` | Vectors in the WAL (not yet flushed) |
| `dimension` | `int` | Vector dimension |
| `bits` | `int` | Quantization bits |
| `total_disk_bytes` | `int` | Total on-disk footprint in bytes |
| `has_index` | `bool` | Whether a HNSW index has been built |
| `index_nodes` | `int` | Number of nodes in the HNSW graph |
| `delta_size` | `int` | Vectors inserted after last `create_index()` (delta overlay); when large, consider rebuilding |
| `live_codes_bytes` | `int` | Size of the in-memory quantized codes buffer |
| `live_slot_count` | `int` | Allocated slots in the live slab |
| `ram_estimate_bytes` | `int` | Estimated total in-memory footprint |

---

## Search

```python
results = db.search(
    query,                       # np.ndarray (float32) or list[float]
    top_k=10,                    # int
    filter=None,                 # dict | None  (see Metadata Filtering below)
    _use_ann=None,               # None=auto, True=force ANN, False=force brute-force
    ann_search_list_size=None,   # int | None — HNSW ef_search (default: max_degree × 2)
    include=None,                # list[str] | None — fields to return; default all
                                 #   valid values: "id", "score", "metadata", "document"
    rerank_factor=None,          # int | None — rerank oversampling multiplier (requires rerank=True)
                                 #   default: 10 (brute-force), 20 (ANN)
                                 #   top (rerank_factor × top_k) candidates are re-scored exactly
    nprobe=None,                 # int | None — IVF clusters to probe (requires create_coarse_index())
    hybrid=None,                 # dict | None — sparse+dense fusion via RRF (see Hybrid Search below)
)
# Returns list of dicts: {"id": str, "score": float, "metadata": dict, "document": str | None}
```

With `_use_ann=None` (default), the planner auto-selects: ANN if an index exists, N ≥ 10k, and delta < 20% of N; otherwise brute-force. Pass `_use_ann=True` to force ANN (requires `create_index()` first) or `_use_ann=False` to force brute-force.

`ann_search_list_size` trades recall for latency — higher values find better results but take longer. Values between 64 and 256 cover the practical range.

`rerank_factor` follows the industry convention (Qdrant `oversampling`, LanceDB `refine_factor`). Higher values improve recall at the cost of exact re-score latency. Useful when `top_k` is small (1–10) and precision is critical.

### Hybrid search (sparse + dense)

When documents were inserted via `db.insert(..., document=...)` or `db.insert_batch(..., documents=[...])`, an in-memory BM25 sparse index is maintained automatically. Pass `hybrid={...}` to combine it with the dense vector search at query time using Reciprocal Rank Fusion (RRF):

```python
results = db.search(
    query_vec,
    top_k=10,
    hybrid={
        "text": "exact arXiv-2504.19874 paper",  # required — query string for the BM25 leg
        "weight": 0.3,                            # optional — BM25 contribution in [0, 1]; default 0.5
        "rrf_k": 60,                              # optional — RRF smoothing constant; default 60
        "oversample": 4,                          # optional — per-list candidate multiplier; default 4
    },
)
```

How to read the parameters:

- `weight=0.0` collapses to pure dense (the BM25 leg contributes nothing). `weight=1.0` collapses to pure BM25 (with the dense leg's metadata still used to enrich results that BM25 surfaces). `weight=0.5` (the default) gives sparse and dense equal voice.
- `rrf_k=60` is the textbook value from Cormack et al. 2009. Larger `rrf_k` flattens the rank-decay curve so consensus picks across both lists are rewarded more than rank-1 winners on either alone.
- `oversample=4` asks each leg for `4 × top_k` candidates so RRF has room to find shared hits that don't lead either list. Lower it (down to 1) for slightly faster queries; raise it for slightly higher recall on long-tail queries.
- An empty `text` (or text with no alphanumeric tokens) collapses to the dense-only fast path — no overhead.
- `hybrid` is mutually exclusive with `nprobe`: if both are passed, `hybrid` wins.

The BM25 index itself is built from the `document` field only. It is persisted in `bm25.idx` and rebuilt automatically from the doc store on cold start if the sidecar is missing.

### Batch query

Search with multiple vectors in one call:

```python
all_results = db.query(
    query_embeddings,            # np.ndarray shape (N, D), float32 or float64
    n_results=10,                # int — results per query
    where_filter=None,           # dict | None
    _use_ann=None,               # None=auto, True=force ANN, False=force brute-force (same semantics as search())
    ann_search_list_size=None,
    rerank_factor=None,          # int | None — same semantics as search(); default 10 (brute) / 20 (ANN)
    hybrid=None,                 # dict | None — same shape as search(); accepts "texts": [str] for per-row text
                                 # or "text": str to broadcast a single query to all rows
)
# Returns list[list[dict]] — one inner list per query vector
```

---

## Index (HNSW)

Build the index **after** loading your data. Rebuild after large batches of inserts — it is not updated incrementally.

```python
db.create_index(
    max_degree=32,        # int — max neighbors per node; higher = better recall, larger graph (default 32)
    ef_construction=200,  # int — beam size during graph build; higher = better quality, slower build (default 200)
    n_refinements=5,      # int — refinement passes after build; higher = better graph, slower (default 5)
    search_list_size=128, # int — default query-time HNSW search breadth (ef_search); higher = better recall, higher latency (default 128)
    alpha=1.2,            # float — edge pruning aggressiveness (default 1.2)
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

### Enumerating metadata values

Use `list_metadata_values` to discover all distinct values for a field — useful for building filter UIs or faceted search:

```python
counts = db.list_metadata_values("topic")
# {"finance": 120, "ml": 84, "sports": 31, ...}
```

Returns a `dict[str, int]` mapping each distinct non-null value to its occurrence count across active vectors. Supports dotted paths (e.g. `"profile.region"`). Non-string values are stringified via their JSON representation.

---

## Hybrid search (delta index)

Vectors inserted **after** `create_index()` are tracked in an in-memory **delta overlay** (`delta_ids.json`). ANN search automatically queries both the HNSW graph and the delta overlay (brute-force), merging results before returning top-k. There is no extra configuration required.

```python
stats = db.stats()
print(stats["delta_size"])   # vectors in the delta overlay

# When delta grows large, rebuild to incorporate them into the graph:
if stats["delta_size"] > 10_000:
    db.create_index()        # merges delta into HNSW graph, clears delta_size to 0
```

The delta overlay is persisted to `delta_ids.json` and survives restarts. Deletes still invalidate the index (the deleted vector would otherwise remain navigable in the HNSW graph).

---

## Async API

`AsyncDatabase` is an asyncio-friendly wrapper around `Database`. Every long-running method is awaitable; the underlying call dispatches to a thread-pool executor, so concurrent awaits fan out through the pool and the event loop stays responsive. Rust engine calls release the GIL while they run; actual throughput still depends on executor size, CPU cores, and storage.

```python
import asyncio
from tqdb.aio import AsyncDatabase

async def main():
    async with await AsyncDatabase.open("./mydb", dimension=1536, bits=4) as db:
        await db.insert("doc1", vec, document="...")
        results = await db.search(query, top_k=5)
        # Concurrent searches fan out through the executor:
        all_results = await asyncio.gather(
            *(db.search(q, top_k=5) for q in queries)
        )

asyncio.run(main())
```

Every sync `Database` method has an async counterpart with the same signature and return shape: `insert`, `insert_batch`, `upsert`, `update`, `update_metadata`, `delete`, `delete_batch`, `get`, `get_many`, `list_all`, `list_ids`, `count`, `search`, `query`, `create_index`, `create_coarse_index`, `checkpoint`, `stats`, `close`.

Constructor arguments:

- `executor=None` — share an existing `ThreadPoolExecutor` across multiple `AsyncDatabase` instances or with other code. When `None`, `AsyncDatabase` creates its own pool sized to `min(32, cpu_count + 4)` and shuts it down on `close()`.
- `max_workers=None` — override the auto-created pool size.
- All other kwargs forward to `Database.open` (`dimension`, `bits`, `metric`, …).

Use `db.sync` (a property) to access the underlying synchronous `Database` for cheap O(1) operations like `len(db.sync)` or `"id" in db.sync` that don't need an executor round-trip.

---

## RAG Integration

TQDB has five Python-side surfaces for RAG pipelines:

| Use case | Class | Module |
|---|---|---|
| LangChain v2 `VectorStore` ABC | `TurboQuantVectorStore` | `tqdb.vectorstore` |
| LlamaIndex `BasePydanticVectorStore` | `TurboQuantVectorStore` | `tqdb.llama_index` |
| Multi-vector / ColBERT (MaxSim) | `MultiVectorStore` | `tqdb.multivector` |
| Asyncio facade for any of the above | `AsyncDatabase` | `tqdb.aio` |
| Legacy LangChain-style retriever | `TurboQuantRetriever` | `tqdb.rag` |

Dedicated guides:

- **LangChain** → [`docs/integrations/langchain.md`](integrations/langchain.md)
- **LlamaIndex** → [`docs/integrations/llama_index.md`](integrations/llama_index.md)
- **Multi-vector / ColBERT** → [`docs/MULTI_VECTOR.md`](MULTI_VECTOR.md)
- **Async API** → see the [Async API section above](#async-api)
- **Migration from Chroma / LanceDB** → [`docs/MIGRATION.md`](MIGRATION.md)

The legacy `TurboQuantRetriever` is preserved for back-compat:

```python
from tqdb.rag import TurboQuantRetriever

retriever = TurboQuantRetriever(
    db_path,                # str — directory path for the database
    dimension=1536,         # int — vector dimension
    bits=4,                 # int — quantization bits (any int >= 2; common: 2, 4, 8)
    seed=42,                # int — RNG seed
    metric="ip",            # str — "ip", "cosine", or "l2"
    rerank_precision=None,  # str|None — None (→ "int8"), "int8", "int4", "f16", "f32", "disabled"
)
```

For new code prefer `TurboQuantVectorStore` (it's the full LangChain v2 ABC and supports `as_retriever()` for the same call sites). See `docs/integrations/langchain.md` for the migration recipe.

### Legacy `add_texts(texts, embeddings, metadatas=None)`

Batch-insert documents with their embeddings.

```python
retriever.add_texts(
    texts=["Document one.", "Document two."],
    embeddings=[vec1, vec2],       # np.ndarray or list[list[float]]
    metadatas=[{"src": "a"}, {"src": "b"}]  # optional
)
```

IDs are auto-assigned as `doc_0`, `doc_1`, … continuing from the current count.

### Legacy `similarity_search(query_embedding, k=4)`

Search for the `k` most similar documents.

```python
results = retriever.similarity_search(query_embedding=query_vec, k=5)
for r in results:
    print(r["score"], r["text"], r["metadata"])
```

Returns a `list[dict]` with keys: `"text"`, `"metadata"`, `"score"`.

---

## ChromaDB Compatibility Shim

`tqdb.chroma_compat` provides a drop-in `PersistentClient` that mirrors the chromadb ≥ 1.5 API. Each collection is stored as a `tqdb.Database` under `{path}/{collection_name}/`.

```python
from tqdb.chroma_compat import PersistentClient

client = PersistentClient(path="/data/chroma")
col = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})

col.add(ids=["a", "b"], embeddings=[[0.1, ...], [0.2, ...]], metadatas=[{"src": "web"}, {}])
col.upsert(ids=["a"], embeddings=[[0.3, ...]], metadatas=[{"src": "updated"}])

results = col.query(query_embeddings=[[0.1, ...]], n_results=5)
# {"ids": [[...]], "distances": [[...]], "metadatas": [[...]], "documents": [[...]]}

col.delete(ids=["b"])
col.delete(where={"src": {"$eq": "web"}})

print(col.count())          # int
print(col.peek(limit=3))    # dict same shape as query result
```

**Metric mapping:** `metadata={"hnsw:space": "cosine"}` → `metric="cosine"`, `"ip"` → inner product (default), `"l2"` → L2. The metric is fixed at collection creation and cannot be changed.

**Empty behavior:** `get(ids=[])` returns an empty Chroma-shaped response, not a full-table scan. `query(...)` on an empty collection returns one empty result row per query embedding. `add(ids=[], embeddings=[])` raises `ValueError` because Chroma-compatible adds require at least one embedding to validate or establish dimension.

**Not implemented:** `HttpClient`, `Settings`, server/cloud mode, `chromadb.Client()` (ephemeral), `where_document` filtering, automatic text embedding (pass pre-computed `embeddings`; or provide an `embedding_function` callable at collection creation time).

**Where-filter operators supported:** `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$and`, `$or`, `$exists`, `$contains`.

---

## LanceDB Compatibility Shim

`tqdb.lancedb_compat` provides a `connect()` factory mirroring the LanceDB v0/v1 Python API. Each table is stored as a `tqdb.Database` under `{uri}/{table_name}/`.

```python
from tqdb.lancedb_compat import connect
import pyarrow as pa

db = connect("/data/lancedb")

# Create from PyArrow Table or list[dict] with "id" + "vector" columns
tbl = db.create_table("docs", data=pa_table)
tbl = db.create_table("docs", data=pa_table, mode="overwrite")  # wipe and recreate

# Fluent query builder
results = (
    tbl.search(query_vec)
       .metric("dot")          # "dot"/"ip" → ip, "cosine" → cosine, "l2"/"euclidean" → l2
       .limit(10)
       .where("id IN ('a', 'b', 'c')")
       .to_list()               # list[dict] with all fields + "_distance"
)

tbl.delete("id IN ('a', 'b')")
tbl.delete("status = 'archived'")

print(tbl.count_rows())
tbl.optimize()   # no-op; tqdb handles compaction automatically
```

**SQL WHERE parser:** supports `field = 'value'`, `field != 'value'`, `field IN ('a', 'b', ...)` (including `id IN (...)`), and numeric comparisons (`field > 10`, `field >= 10`, `field < 10`, `field <= 10`). More complex predicates raise `NotImplementedError`.

**Empty behavior:** a table without a manifest reports `count_rows()==0`; `head()`, `to_arrow()`, `to_pandas()`, and searches with no matches return empty tables/lists rather than raising. `add([])` and `add(empty_pyarrow_table)` are always no-ops; they do not create a manifest or establish a brand-new table's schema.

**Implemented compatibility subset:** `update(where, values)` and `merge_insert(on).when_matched_update_all().when_not_matched_insert_all().execute(data)` are supported for basic upsert/update workflows. Other merge clauses are accepted as no-ops.

**Not implemented:** `create_fts_index`, `create_scalar_index`, `drop_database`, remote/cloud URIs.

---

## On-disk Layout

```
./my_db/
├── manifest.json        — DB config (dimension, bits, seed, metric)
├── quantizer.bin        — Serialized quantizer state
├── live_codes.bin       — Memory-mapped quantized vectors (hot path)
├── live_vectors.bin     — Raw vectors for exact reranking (only if rerank=True; precision set by rerank_precision)
├── wal.log              — Write-ahead log (crash recovery)
├── metadata.bin         — Per-vector metadata properties
├── metadata.docs.zst    — Document sidecar (adaptive codec container; backward-compatible loader)
├── live_ids.bin         — ID → slot index (dense/sparse compact encodings; optional zstd for sparse)
├── graph.bin            — HNSW adjacency list (if index built)
└── seg-XXXXXXXX.bin     — Immutable flushed segment files
```

Notes:
- `live_ids.bin` is persisted in a compact format (`TQID2D1` dense, `TQID2S1` sparse, `TQID2Z1` sparse+zstd). Legacy format is still readable.
- `metadata.docs.zst` uses a small container header (`M2DZ`) with adaptive codec selection and remains backward-compatible with legacy `M2D1` zstd payloads.
