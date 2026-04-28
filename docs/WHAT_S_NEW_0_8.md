# What's new in v0.8

> Released 2026-04-28. See the full entry in [`CHANGELOG.md`](../CHANGELOG.md).

## TL;DR

Five new Python-side features and **no breaking changes**:

1. **Multi-vector / ColBERT** retrieval with MaxSim scoring.
2. **LangChain v2** native `VectorStore` ABC.
3. **LlamaIndex** native `BasePydanticVectorStore` integration.
4. **`AsyncDatabase`** — asyncio facade with real concurrency (PyO3 releases the GIL).
5. **`tqdb.migrate`** — one-command import from existing Chroma / LanceDB collections.

Existing v0.7 code keeps working. Upgrade is `pip install -U tqdb`.

## Feature highlights

### Multi-vector / ColBERT-style late interaction

```python
from tqdb import MultiVectorStore

store = MultiVectorStore.open("./mvstore", dimension=96, bits=4, metric="cosine")
store.insert("paper-2504.19874", token_vectors_8x96, document="full text")
hits = store.search(query_token_vectors_5x96, top_k=10)
# MaxSim: score(q, d) = Σ_i max_j <q_i, d_j>
```

Each document holds N token vectors; queries are tokenized to K vectors and scored
via MaxSim. Built as a Python-layer wrapper over `tqdb.Database` so it ships now;
the v0.9 sprint will move it into the engine for tighter storage and native filter
pushdown — **the public Python API stays the same**.

Full guide: [`docs/MULTI_VECTOR.md`](MULTI_VECTOR.md).

### LangChain v2 `VectorStore`

```python
from tqdb.vectorstore import TurboQuantVectorStore
store = TurboQuantVectorStore.from_texts(texts, embedding, path="./mydb")
docs = store.similarity_search("query", k=4)
```

Implements the full LangChain v2 ABC: `add_texts`, `add_documents`, all
`similarity_search*` variants, `delete`, `from_texts` / `from_documents`,
`as_retriever`, `_select_relevance_score_fn`, `embeddings`. Lazy-imports
LangChain so installing plain `tqdb` doesn't pull the dep tree.

Full guide: [`docs/integrations/langchain.md`](integrations/langchain.md).

### LlamaIndex `BasePydanticVectorStore`

```python
from tqdb.llama_index import TurboQuantVectorStore
vstore = TurboQuantVectorStore.open("./tqdb_llama", dimension=1536)
storage = StorageContext.from_defaults(vector_store=vstore)
index = VectorStoreIndex.from_documents(docs, storage_context=storage)
```

LlamaIndex's `MetadataFilters` / `MetadataFilter` / `FilterOperator` /
`FilterCondition` are translated to TQDB's MongoDB-style filter dialect.
Optional dep `tqdb[llamaindex]`.

Full guide: [`docs/integrations/llama_index.md`](integrations/llama_index.md).

### Async API — `AsyncDatabase`

```python
from tqdb.aio import AsyncDatabase

async with await AsyncDatabase.open("./mydb", dimension=1536) as db:
    await db.insert("doc1", vec)
    # 50 concurrent searches — actually run in parallel:
    all_results = await asyncio.gather(*(db.search(q, top_k=5) for q in queries))
```

Thread-pool-backed wrapper over the sync `Database`. Because every PyO3 method
already releases the GIL via `py.allow_threads`, concurrent `await` calls
genuinely parallelise — the existing test suite includes a 50-task proof.

Full guide: [`docs/PYTHON_API.md` — Async API](PYTHON_API.md#async-api).

### Migration toolkit — `tqdb.migrate`

```bash
pip install 'tqdb[migrate]'
python -m tqdb.migrate chroma /old/chroma_db /new/tqdb_db --collection my_docs
python -m tqdb.migrate lancedb /old/lance_dataset /new/tqdb_db --table my_docs
```

Reads each source library's native on-disk format via the source library itself
(no schema parsing) and bulk-inserts into a fresh TQDB. Preserves IDs, vectors,
metadata, document text. CLI plus programmatic `migrate_chroma()` /
`migrate_lancedb()` functions.

Full guide: [`docs/MIGRATION.md`](MIGRATION.md).

## Migration from v0.7

**Nothing breaks.** All v0.7 APIs continue to work in v0.8. The only behaviour change
in core code is a previously-undocumented Rust-side fix: `Database.insert_batch`'s
`metadatas=` parameter now correctly accepts `[None, None, …]` per-row entries (the
public `.pyi` type stub already documented this; the Rust implementation hadn't
followed). Code passing `[None, ...]` previously raised `TypeError` and now succeeds.

### Optional: move from `TurboQuantRetriever` to `TurboQuantVectorStore`

The pre-v0.8 `tqdb.rag.TurboQuantRetriever` is preserved for back-compat — your
existing imports keep working. New code should prefer the LangChain-v2-native
`TurboQuantVectorStore`:

```python
# Before — still works
from tqdb.rag import TurboQuantRetriever
ret = TurboQuantRetriever(db_path, dimension=1536, embedding_function=embed)

# After — recommended for new code
from tqdb.vectorstore import TurboQuantVectorStore
store = TurboQuantVectorStore.open(db_path, embedding=embed)
ret = store.as_retriever()
```

The new class implements the full LangChain `VectorStore` ABC, so calls like
`store.add_documents(...)`, `store.similarity_search_with_score(...)`, and
`store.from_texts(...)` work as documented in the LangChain reference.

## What's coming in v0.9

The v0.9-v1.0 hardening sprint targets:

- **Native multi-vector engine integration** — replaces the v0.8 Python-layer
  wrapper with `IdPool` doc-id grouping, an on-disk MaxSim kernel, and native
  filter pushdown. Same public Python API, much tighter storage and faster search.
- Server mode v0.8 feature parity (LangChain / LlamaIndex / multi-vector exposed
  over HTTP) is a candidate but not yet committed.
- Pre-1.0 hardening: `≥80%` test coverage, `.unwrap()` audit pass 2, API-key
  authentication baseline, complete `0.x → 1.0` migration guide.

See the full [v0.9-v1.0 sprint board](https://github.com/users/jyunming/projects/3)
for the running list.
