# TurboQuantDB Compatibility Matrix

This matrix tracks practical API/behavior parity against common embedded RAG vector-store expectations.

## Scope

- Baselines compared: ChromaDB and LanceDB-style collection workflows.
- Focus: collection CRUD, batch writes, query/get filters, pagination/include, and service controls.

## Feature Matrix

| Area | TurboQuantDB | Notes |
|---|---|---|
| Collection create/get/list/delete | Yes | Embedded + service (`tenant/database/collection` in service mode). |
| Add / Upsert / Update / Delete | Yes | Batch paths and report-mode partial failures are available. |
| Delete by IDs and/or filter | Yes | Service supports `ids`, `filter`, `where_filter`. |
| Get by IDs and/or filter | Yes | Service supports `ids`, `filter`, `where_filter` plus include/pagination. |
| Query with filters | Yes | Supports `filter` and alias `where_filter`. |
| Query result count aliases | Yes | `top_k` and alias `n_results`. |
| Include controls | Yes | `ids`, `scores`, `metadatas`, `documents` by endpoint. |
| Pagination controls | Yes | `offset`/`limit` (where applicable). |
| Distance metrics | Yes | `ip`, `cosine`, `l2` (collection-configured). |
| ANN build + tuning | Yes | Graph index build/search controls and persisted index state. |
| Snapshot / Restore | Yes | Engine + service job-based snapshot and restore workflow via background jobs. |
| Multi-tenant service auth/RBAC | Yes | Persisted API keys, principals, role bindings. |
| Quota controls | Yes | Collections, vectors, disk, concurrent jobs. |

## Integration Version Matrix

Versions pinned in [`pyproject.toml`](../pyproject.toml). Install the matching extra to pull in the third-party library and enable the corresponding TQDB module.

| Integration | Min version | TQDB extra | Module |
|---|---|---|---|
| Python | 3.10 | (built into wheel) | — |
| LangChain | `langchain-core>=0.3` | `tqdb[langchain]` | `tqdb.vectorstore.TurboQuantVectorStore` |
| LlamaIndex | `llama-index-core>=0.10` | `tqdb[llamaindex]` | `tqdb.llama_index.TurboQuantVectorStore` |
| Chroma (migrator) | `chromadb>=0.5` | `tqdb[migrate]` / `tqdb[migrate-chroma]` | `tqdb.migrate.migrate_chroma` |
| LanceDB (migrator) | `lancedb>=0.10` | `tqdb[migrate]` / `tqdb[migrate-lancedb]` | `tqdb.migrate.migrate_lancedb` |

The integration modules import their third-party deps **lazily** — TQDB never pulls
them in at top-level import time. A user who installs plain `pip install tqdb` and
never touches `tqdb.vectorstore` / `tqdb.llama_index` / `tqdb.migrate` doesn't pay
their import cost.

## Known Gaps / Follow-ups

- Additional protocol-level compatibility shims may still be needed for drop-in client replacement.
- Full workload validation is pending comprehensive user-side test plan.
- LangChain / LlamaIndex compatibility is tested against the pinned versions above; older versions are not tested. No async / streaming integration wrappers yet for either framework — `AsyncDatabase` is a separate facade that doesn't yet have LangChain/LlamaIndex async parity.

