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

## Known Gaps / Follow-ups

- Additional protocol-level compatibility shims may still be needed for drop-in client replacement.
- Full workload validation is pending comprehensive user-side test plan.

