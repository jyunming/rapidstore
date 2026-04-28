# LlamaIndex integration

`tqdb.llama_index.TurboQuantVectorStore` is a
[`BasePydanticVectorStore`](https://docs.llamaindex.ai/en/stable/api_reference/storage/vector_store/)
implementation backed by `tqdb.Database`.

## Install

```bash
pip install 'tqdb[llamaindex]'
```

The `[llamaindex]` extra pins `llama-index-core>=0.10`. The integration
module imports LlamaIndex **lazily** — TQDB doesn't pull the heavy
LlamaIndex dep tree on import; you only pay for it when you actually use
the class.

## Quickstart with `VectorStoreIndex`

```python
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

from tqdb.llama_index import TurboQuantVectorStore

Settings.embed_model = OpenAIEmbedding()  # any LlamaIndex embedder

# Open or create the underlying TQDB. dimension must match the embedder.
vstore = TurboQuantVectorStore.open(
    "./tqdb_llama",
    dimension=1536,
    bits=4,
    metric="cosine",
)
storage = StorageContext.from_defaults(vector_store=vstore)

# Build an index — LlamaIndex computes embeddings and calls vstore.add(nodes).
docs = [Document(text="The quick brown fox"), Document(text="Jumped over the lazy dog")]
index = VectorStoreIndex.from_documents(docs, storage_context=storage)

# Query.
qe = index.as_query_engine(similarity_top_k=5)
print(qe.query("a fast fox"))
```

## Direct `add` / `query`

For users who want to manage embeddings themselves (e.g., custom batched
embedding pipelines):

```python
import numpy as np
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery

# Insert pre-embedded nodes.
nodes = [
    TextNode(id_="a", text="alpha", metadata={"k": 1}),
    TextNode(id_="b", text="bravo", metadata={"k": 2}),
]
nodes[0].embedding = my_embedder.embed("alpha")  # list[float]
nodes[1].embedding = my_embedder.embed("bravo")
vstore.add(nodes)

# Query.
result = vstore.query(VectorStoreQuery(
    query_embedding=my_embedder.embed("alpha"),
    similarity_top_k=5,
))
print(result.nodes, result.similarities, result.ids)
```

## Filters

`MetadataFilters` / `MetadataFilter` / `FilterOperator` / `FilterCondition`
are translated to TQDB's MongoDB-style filter dialect automatically:

```python
from llama_index.core.vector_stores import (
    FilterCondition, FilterOperator, MetadataFilter, MetadataFilters,
)

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="year", value=2024, operator=FilterOperator.GTE),
        MetadataFilter(key="lang", value="en", operator=FilterOperator.EQ),
    ],
    condition=FilterCondition.AND,
)
result = vstore.query(VectorStoreQuery(
    query_embedding=qvec,
    similarity_top_k=10,
    filters=filters,
))
```

Supported operators (LlamaIndex → TQDB):

| LlamaIndex | TQDB |
|------------|------|
| `==` (`EQ`) | `$eq` |
| `!=` (`NE`) | `$ne` |
| `>` / `>=` (`GT` / `GTE`) | `$gt` / `$gte` |
| `<` / `<=` (`LT` / `LTE`) | `$lt` / `$lte` |
| `in` (`IN`) | `$in` |
| `nin` / `not in` (`NIN`) | `$nin` |
| `contains` (`CONTAINS`) | `$contains` |
| `condition: and` | `$and` |
| `condition: or` | `$or` |

## Methods

| Method | Notes |
|--------|-------|
| `add(nodes)` | Each node must have `embedding` set. Auto-generates `node_id` if absent. Stores text + metadata. |
| `query(VectorStoreQuery)` | Requires `query_embedding`. Returns `VectorStoreQueryResult` with nodes / similarities / ids. |
| `delete(ref_doc_id)` | LlamaIndex `delete` semantics — single ref_doc_id at a time. |
| `delete_nodes(node_ids=None, filters=None)` | Bulk delete by IDs or by metadata filter. |
| `clear()` | Drop every node. |
| `persist()` | Calls `db.checkpoint()` (TQDB writes through, so this is a no-op-style flush). |
| `client` | Property exposing the underlying `Database` for advanced use. |

Class flags:

- `stores_text = True` — TQDB stores the document text alongside the
  vector, so retrieval round-trips full nodes (LlamaIndex won't fetch text
  from a separate doc store).
- `flat_metadata = True` — TQDB indexes only scalar metadata fields.
  Flatten nested metadata before insert.

## Filter ↔ TQDB mapping internals

The translator lives at [`python/tqdb/_filter_translator.py`](https://github.com/jyunming/TurboQuantDB/blob/main/python/tqdb/_filter_translator.py).
If you hit an unsupported filter shape it raises `ValueError` with the
operator/key it couldn't translate; that's the place to report bugs.

## Supported versions

`[llamaindex]` pins `llama-index-core>=0.10`. The class uses Pydantic v2
(matching LlamaIndex 0.10+'s `BasePydanticVectorStore`); older versions
are not tested.
