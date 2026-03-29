# Python API Migration: `Database` to Collection-First `Client`

This guide shows how to migrate existing code that opens a single `Database` directly to the newer collection-first API.

## Summary

- Old style: open one `Database` and store all vectors there.
- New style: open a `Client`, then create/open named collections and operate per collection.

## Quick Mapping

| Legacy (`Database`) | Collection-first (`Client` + `Database`) |
|---|---|
| `Database.open(uri, ...)` | `Client(uri, ...).get_or_create_collection(name)` |
| Single namespace | Multiple isolated collections |
| `insert` / `upsert` / `update` / `delete` | Same methods on collection object |
| `search` | Same + `query` for batched queries |
| N/A | `list_collections`, `snapshot_collection`, `restore_collection` |

## 1) Open Database

Legacy:

```python
import turboquantdb as tq

db = tq.Database.open(
    uri="data/tqdb",
    dimension=1536,
    bits=4,
    seed=42,
    metric="ip",
)
```

Collection-first:

```python
import turboquantdb as tq

client = tq.Client(
    uri="data/tqdb",
    dimension=1536,
    bits=4,
    seed=42,
    metric="ip",
)
collection = client.get_or_create_collection("docs")
```

## 2) CRUD Operations

Legacy:

```python
db.upsert("id-1", embedding, {"source": "faq"}, "document text")
got = db.get("id-1")
db.delete("id-1")
```

Collection-first:

```python
collection.upsert("id-1", embedding, {"source": "faq"}, "document text")
got = collection.get("id-1")
collection.delete("id-1")
```

## 3) Batch Write + Partial Failure Report

```python
report = collection.upsert_many_report(
    ids=["id-1", "id-2"],
    vectors=[vec1, vec2],
    metadatas=[{"tenant": "a"}, {"tenant": "a"}],
    documents=["doc1", "doc2"],
)
print(report["applied"])
print(report["failed"])  # list of {index, id, error}
```

## 4) Search and Query

Single query:

```python
hits = collection.search(
    query=query_vec,
    top_k=10,
    where_filter={"tenant": {"$eq": "a"}},
    include_document=True,
)
```

Batched query with include/pagination:

```python
out = collection.query(
    query_embeddings=[q1, q2],
    n_results=5,
    where_filter={"tenant": {"$eq": "a"}},
    include=["ids", "scores", "metadatas", "documents"],
    offset=0,
)
```

## 5) Collection Management

```python
client.create_collection("images")
print(client.list_collections())
client.snapshot_collection("docs", "snapshots/docs-2026-03-28")
client.restore_collection("docs", "snapshots/docs-2026-03-28")
client.delete_collection("images")
```

## Migration Checklist

- Replace `Database.open(...)` calls with `Client(...).get_or_create_collection(name)`.
- Choose explicit collection names (for isolation by app/domain/tenant).
- Keep CRUD/search calls mostly unchanged (same methods on collection object).
- Update call sites to use `query(...)` where batched retrieval is needed.
- Add include/pagination controls where response size matters.
