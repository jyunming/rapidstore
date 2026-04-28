# LangChain integration

`TurboQuantVectorStore` is a complete implementation of LangChain v2's
[`VectorStore`](https://api.python.langchain.com/en/latest/vectorstores/langchain_core.vectorstores.base.VectorStore.html)
ABC backed by `tqdb.Database`.

## Install

```bash
pip install 'tqdb[langchain]'
```

The `[langchain]` extra pins the supported `langchain-core` version range.
LangChain itself is **lazy-loaded** — TQDB imports it only when you actually
touch `TurboQuantVectorStore`, so users without LangChain installed don't pay
its import cost.

## Quickstart

```python
from langchain_openai import OpenAIEmbeddings  # or any langchain_core.embeddings.Embeddings
from tqdb.vectorstore import TurboQuantVectorStore

embed = OpenAIEmbeddings()

# Build a fresh store from raw texts.
store = TurboQuantVectorStore.from_texts(
    texts=["The quick brown fox", "Jumped over the lazy dog"],
    embedding=embed,
    metadatas=[{"source": "fable"}, {"source": "fable"}],
    path="./mydb",
    bits=4,             # forwarded to Database.open
    metric="cosine",
)

# Query.
docs = store.similarity_search("a fast fox", k=5)
for d in docs:
    print(d.id, d.metadata, d.page_content)

# Or with scores:
for doc, score in store.similarity_search_with_score("query", k=5):
    print(score, doc.id)
```

## Wrap an existing database

```python
from tqdb import Database
from tqdb.vectorstore import TurboQuantVectorStore

db = Database.open("./mydb")  # parameterless reopen via manifest.json
store = TurboQuantVectorStore(db, embedding=embed)
```

`embedding=None` is allowed if you only call `similarity_search_by_vector`
or `add_texts(_vectors=...)` (i.e. you've pre-computed embeddings).

## Methods

`TurboQuantVectorStore` implements the full LangChain v2 `VectorStore` ABC:

| Method | Notes |
|--------|-------|
| `add_texts(texts, metadatas=None, ids=None)` | Returns the assigned IDs (UUIDs unless caller supplies them). |
| `add_documents(documents, ids=None)` | Same, taking `Document` objects. |
| `similarity_search(query, k=4, filter=None)` | Embeds `query` then queries TQDB. Returns `List[Document]`. |
| `similarity_search_with_score(query, k=4, filter=None)` | Same but returns `List[Tuple[Document, float]]`. |
| `similarity_search_by_vector(embedding, k=4, filter=None)` | Skips the embedder; useful when you've pre-computed query vectors. |
| `delete(ids)` | Returns `True` if any IDs were deleted. |
| `get_by_ids(ids)` | Returns `List[Document]` (only IDs that exist). |
| `as_retriever(search_kwargs=...)` | Standard LangChain retriever bridge. |
| `_select_relevance_score_fn()` | Maps TQDB's IP/cosine score to LangChain's `[0, 1]` relevance space; for L2 uses `1 / (1 + distance)`. |
| `from_texts(...)` (classmethod) | Bootstrap helper — builds the database, embeds the texts, populates. |
| `from_documents(...)` (classmethod) | Same with `Document` inputs. |

## Filters

Pass a TQDB-style filter dict as `filter=` on any search method. TQDB
operators (`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`,
`$exists`, `$contains`, `$and`, `$or`) all work:

```python
store.similarity_search(
    "query",
    k=5,
    filter={"source": {"$in": ["fable", "myth"]}, "year": {"$gte": 2020}},
)
```

LangChain itself doesn't standardise filter shapes; we pass dicts through
unchanged. Vendor-specific filter wrappers (e.g.
`StructuredFilter` Pydantic models) will raise `ValueError` — convert them
to plain dicts on your side.

## Hybrid search

If you've inserted documents (i.e. the `document` field is set), TQDB's BM25
hybrid path is available — pass `hybrid={"text": "...", "weight": 0.3}` as a
search kwarg:

```python
docs = store.similarity_search(
    "exact-keyword query",
    k=10,
    hybrid={"text": "exact-keyword query", "weight": 0.3},
)
```

See [`docs/PYTHON_API.md` — Hybrid search](../PYTHON_API.md#hybrid-search-sparse--dense)
for details.

## Migrating from `tqdb.rag.TurboQuantRetriever`

The pre-v0.8 `TurboQuantRetriever` (in `python/tqdb/rag.py`) still works
for back-compat. New code should prefer `TurboQuantVectorStore`. The two
interfaces are not identical:

| Capability | `TurboQuantRetriever` | `TurboQuantVectorStore` |
|------------|-----------------------|-------------------------|
| LangChain `BaseRetriever` `Runnable` | yes (`invoke`, `get_relevant_documents`) | use `store.as_retriever()` |
| Full `VectorStore` ABC | partial | yes |
| `from_texts`/`from_documents` factories | yes | yes |
| Hybrid kwarg passthrough | yes | yes |
| Async API | no | use `tqdb.AsyncDatabase` for an async DB; vectorstore async is a future addition |

Move recipe:

```python
# Before
from tqdb.rag import TurboQuantRetriever
ret = TurboQuantRetriever(db_path, dimension=1536, embedding_function=embed)

# After
from tqdb.vectorstore import TurboQuantVectorStore
store = TurboQuantVectorStore.open(db_path, embedding=embed)
ret = store.as_retriever()
```

## Supported versions

The `[langchain]` extra pins `langchain-core>=0.3`. Earlier `langchain<0.3`
versions are not tested.
