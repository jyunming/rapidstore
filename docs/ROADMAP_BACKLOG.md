# TurboQuantDB GitHub Roadmap Backlog

This backlog is ordered by urgency for making TurboQuantDB comparable to embedded RAG vector stores as an embedded RAG vector store.

## Milestones

- `M1-Embedded-Core-Parity` (2-3 weeks)
- `M2-Retrieval-Parity` (3-4 weeks)
- `M3-Storage-Operability` (2-3 weeks)
- `M4-Python-API-Ergonomics` (1-2 weeks)
- `M5-MultiTenant-Service` (later)

## Labels

- `epic`
- `high-priority`
- `medium-priority`
- `api`
- `storage`
- `indexing`
- `retrieval`
- `python-sdk`
- `tests`
- `benchmark`
- `docs`

## Epic 1: Embedded Core Parity

### Issue: Add Collection Namespace Model
- Labels: `epic`, `high-priority`, `storage`, `api`
- Milestone: `M1-Embedded-Core-Parity`
- Status: `completed` (engine collection namespace model + Python collection APIs + lifecycle guard tests)
- Scope:
  - Collection manifest and storage layout.
  - Isolated vector/metadata/index per collection.
- Acceptance:
  - [x] `create/get/list/delete` collection APIs.
  - [x] Cross-collection query isolation tests.
  - [x] Lifecycle guards: cannot open missing collection or delete open collection.

### Issue: Persist Documents in Rust Metadata Layer
- Labels: `high-priority`, `storage`, `api`
- Milestone: `M1-Embedded-Core-Parity`
- Status: `completed` (documents persisted in metadata layer and returned by get/search/query with include controls)
- Scope:
  - Store optional `document` payload with metadata.
  - Return documents from `get/query`.
- Acceptance:
  - [x] Restart persistence test for documents.
  - [x] Python API can include/exclude documents.

### Issue: Batch CRUD APIs
- Labels: `high-priority`, `api`, `python-sdk`, `tests`
- Milestone: `M1-Embedded-Core-Parity`
- Status: `completed` (batch CRUD + report-mode partial-failure aggregation in Rust/Python)
- Acceptance:
  - [x] Batch insert/upsert/update/delete APIs.
  - [x] Report mode for partial failures (`insert_many_report` / `upsert_many_report` / `update_many_report`).
  - [x] Integration tests for continue-on-error behavior.

### Issue: Filter Grammar v1
- Labels: `high-priority`, `retrieval`, `tests`
- Milestone: `M1-Embedded-Core-Parity`
- Status: `completed` (`$and/$or/$eq/$ne/$gt/$gte/$lt/$lte/$in`, nested paths, strict type policy, missing-field semantics, and negative-path tests implemented)

## Epic 2: Retrieval Parity

### Issue: Distance Metric Selection
- Labels: `high-priority`, `retrieval`, `indexing`
- Milestone: `M2-Retrieval-Parity`
- Status: `completed` (`ip/cosine/l2` selection wired with deterministic tie-break and manifest mismatch enforcement)
- Scope:
  - Per-collection metric: `cosine`, `ip`, `l2`.
  - Deterministic tie-break behavior.
- Acceptance:
  - [x] Metric correctness against brute-force reference.
  - [x] Persisted-manifest metric mismatch policy (enforce).

### Issue: ANN Quality/Tuning Surface
- Labels: `high-priority`, `indexing`, `benchmark`
- Milestone: `M2-Retrieval-Parity`
- Status: `completed` (scorer-driven graph build, build/search knobs, persisted index config, benchmark artifact emitters + CI upload workflow)
- Scope:
  - Expose build/search knobs.
  - Persist index config and status.
- Acceptance:
  - [x] Build knobs (`max_degree`, `alpha`) and search-time override (`ann_search_list_size`).
  - [x] Persisted index config/state and surfaced stats (`index_search_list_size`, `index_alpha`).
  - [x] Recall/latency benchmark artifacts emitted and uploaded via CI workflow.

### Issue: Dense+Sparse Hybrid Search
- Labels: `high-priority`, `retrieval`
- Milestone: `M2-Retrieval-Parity`
- Status: `completed` (hybrid dense+sparse scoring + filter support + tests)
- Scope:
  - BM25/FTS sparse path.
  - Weighted fusion with dense scores.
- Acceptance:
  - [x] Hybrid query API and relevance tests.

### Issue: Reranker Hook
- Labels: `medium-priority`, `retrieval`, `python-sdk`
- Milestone: `M2-Retrieval-Parity`
- Status: `completed` (reranker callback path implemented in Python/Rust with integration coverage)
- Scope:
  - Re-rank top-N via pluggable callback/model.
- Acceptance:
  - [x] End-to-end rerank integration test.

## Epic 3: Storage & Operability

### Issue: Crash-Safe Compaction with Recovery
- Labels: `high-priority`, `storage`, `tests`
- Milestone: `M3-Storage-Operability`
- Status: `completed` (two-phase marker + startup recovery path + tests)
- Scope:
  - Two-phase compaction commit marker.
  - Recovery for interrupted compaction.
- Acceptance:
  - [x] Fault-injection style recovery tests pass.

### Issue: Snapshot/Restore
- Labels: `high-priority`, `storage`, `api`
- Milestone: `M3-Storage-Operability`
- Status: `completed` (engine and collection snapshot/restore with roundtrip verification tests)
- Scope:
  - Collection-level snapshot and restore.
- Acceptance:
  - [x] Bitwise and logical restore verification tests.

### Issue: Rich Engine/Index Stats
- Labels: `medium-priority`, `storage`, `docs`
- Milestone: `M3-Storage-Operability`
- Status: `completed` (live/deleted/segment/disk/index/compaction stats surfaced in Rust + Python)
- Scope:
  - live/deleted counts, segment skew, compaction counters.
- Acceptance:
  - [x] `stats()` includes actionable metrics.

## Epic 4: Python API Ergonomics

### Issue: Collection-First Python API
- Labels: `high-priority`, `python-sdk`, `api`
- Milestone: `M4-Python-API-Ergonomics`
- Status: `completed` (collection-first API + migration examples documented in README and README.md)
- Scope:
  - `client.get_or_create_collection(...)`
  - `collection.add/upsert/update/delete/get/query`
- Acceptance:
  - [x] Migration examples from current `Database` API.

### Issue: Pagination and Include Controls
- Labels: `medium-priority`, `python-sdk`, `api`
- Milestone: `M4-Python-API-Ergonomics`
- Status: `completed` (`offset/limit` and `include=[...]` in get/query/search paths with deterministic behavior tests)
- Scope:
  - `offset/limit` or cursor.
  - `include=["ids","scores","metadatas","documents"]`.
- Acceptance:
  - [x] Deterministic pagination tests.

## Epic 5: Multi-Tenant / Service Mode

### Issue: Tenant/Database Hierarchy
- Labels: `epic`, `medium-priority`, `storage`, `api`
- Milestone: `M5-MultiTenant-Service`
- Status: `in progress` (scoped storage path resolver/catalog helpers + service CRUD wiring completed; auth/RBAC persistence done and quota baseline added)
- Scope:
  - tenant->database->collection isolation.

### Issue: Auth/RBAC and Quotas
- Labels: `medium-priority`, `api`
- Milestone: `M5-MultiTenant-Service`
- Status: `in progress` (persisted API key + RBAC bindings enforced for collection CRUD; quota admission baseline added for collection creation)

### Issue: Service Mode and Background Workers
- Labels: `medium-priority`, `storage`, `indexing`
- Milestone: `M5-MultiTenant-Service`
- Status: `in progress` (persisted job store + async job lifecycle + restart-safe recovery/retry budget + real compact/index/snapshot worker execution + job status endpoints + cancel/retry + concurrency cap + data-plane add/upsert/delete/get/query APIs)

## Open Work (Priority Order)

1. [x] Publish migration examples for Collection-first Python API in README/docs.
2. [x] Add benchmark quality gates (recall/latency regression thresholds) in CI.
3. [x] Start M5 design doc: tenant/database hierarchy and auth/RBAC boundaries. See `docs/SERVER_DESIGN.md`.
4. [x] Scaffold server crate with routing, auth middleware, and request context.
5. [x] Implement tenant/database path resolver and catalog helpers in storage layer.

6. [x] Wire service collection CRUD endpoints to scoped storage APIs (create/list/delete in tenant/database scope).
7. [x] Persist auth/RBAC bindings and enforce policy from store (M5.2 baseline for collection CRUD endpoints).
8. [x] Add quotas + admission control hooks in service write paths (M5.3 baseline: collection-create quota limits).

9. [x] Add service job queue + job status endpoints with persisted job store scaffolding (M5.4 baseline).
10. [x] Add restart-safe job recovery semantics (`running -> queued` on startup) with retry budget tracking (M5.4 hardening).
11. [x] Execute compact/index/snapshot jobs via real engine operations and startup queue dispatch (M5.5 baseline).
12. [x] Add job cancel/retry endpoints and worker concurrency cap (`TQ_JOB_WORKERS`) (M5.6 baseline).
13. [x] Add service data-plane endpoints (add/upsert/delete/get/query) in tenant/database collection scope (M5.7 baseline).
14. [x] Add service include/pagination parity for get/query (include fields, query offset, get offset/limit).
15. [x] Add service-side report-mode partial-failure semantics for batch add/upsert to match embedded APIs.
16. [x] Add service e2e tests for data-plane include/pagination/report-mode behavior.
17. [x] Harden quota admission for data-plane/job enqueue (max_vectors on add/upsert + max_concurrent_jobs on maintenance jobs) with service e2e tests.
18. [x] Enforce disk quota admission (`max_disk_bytes`) for add/upsert (estimated WAL growth) and snapshot enqueue with service e2e tests.

19. [x] Add filter-based delete selectors (`ids` + `filter`/`where_filter`) for service parity.
20. [x] Add query request alias parity (`n_results`, `where_filter`) for service compatibility.
21. [x] Add get selector parity (`ids` + `filter`/`where_filter`) and validation semantics.
22. [x] Add service update endpoint parity (`/update`) with report-mode partial-failure behavior.
23. [x] Publish compatibility matrix doc for current parity status (`docs/COMPATIBILITY_MATRIX.md`).


