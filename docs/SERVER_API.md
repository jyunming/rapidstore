# Server API Reference

## Purpose

Define the HTTP/JSON contract for multi-tenant TurboQuantDB service mode.

Base path: `/v1`

## Conventions

- All endpoints require authentication (except `/healthz`).
- IDs are ASCII strings unless stated otherwise.
- Timestamps are ISO-8601 UTC strings.
- Error responses are stable and machine-readable.

## Headers

Required:

- `Authorization: Bearer <token>` or `Authorization: ApiKey <key>`
- `X-Tenant-Id: <tenant_id>` (required when token does not bind tenant)

Optional:

- `X-Request-Id: <client-id>`

## Error Model

All non-2xx responses:

```json
{
  "error": {
    "code": "quota_exceeded",
    "message": "tenant vector quota exceeded",
    "details": {
      "dimension": "vectors",
      "current": 1000001,
      "limit": 1000000
    }
  },
  "request_id": "req_01J..."
}
```

Canonical error codes:

- `unauthenticated`
- `forbidden`
- `not_found`
- `already_exists`
- `invalid_argument`
- `conflict`
- `quota_exceeded`
- `rate_limited`
- `resource_busy`
- `internal`

## Health

### `GET /healthz`

Response `200`:

```json
{ "status": "ok" }
```

## Collections

### `POST /tenants/{tenant}/databases/{database}/collections`

Create collection.

Request:

```json
{
  "name": "docs",
  "dimension": 1536,
  "bits": 4,
  "seed": 42,
  "metric": "ip"
}
```

Response `201`:

```json
{
  "tenant": "t1",
  "database": "db1",
  "name": "docs",
  "dimension": 1536,
  "bits": 4,
  "seed": 42,
  "metric": "ip",
  "created_at": "2026-03-28T18:00:00Z"
}
```

### `GET /tenants/{tenant}/databases/{database}/collections`

List collections.

Response `200`:

```json
{
  "collections": [
    {"name": "docs"},
    {"name": "images"}
  ]
}
```

### `DELETE /tenants/{tenant}/databases/{database}/collections/{collection}`

Delete collection.

Response `200`:

```json
{ "deleted": true }
```

## Vector Write APIs

### `POST /tenants/{tenant}/databases/{database}/collections/{collection}/upsert`

Request:

```json
{
  "ids": ["id-1", "id-2"],
  "vectors": [[0.1, 0.2], [0.3, 0.4]],
  "metadatas": [{"source": "faq"}, {"source": "blog"}],
  "documents": ["doc1", "doc2"],
  "mode": "report"
}
```

- `mode` values:
  - `fail_fast` (default)
  - `report` (continue-on-error)

Response `200` (`fail_fast`):

```json
{ "applied": 2 }
```

Response `200` (`report`):

```json
{
  "applied": 1,
  "failed": [
    {"index": 1, "id": "id-2", "error": "vector dimension mismatch"}
  ]
}
```

### `POST /tenants/{tenant}/databases/{database}/collections/{collection}/delete`

Request:

```json
{ "ids": ["id-1", "id-2"] }
```

Response `200`:

```json
{ "deleted": 2 }
```

## Query/Search APIs

### `POST /tenants/{tenant}/databases/{database}/collections/{collection}/search`

Single query search.

Request:

```json
{
  "query": [0.1, 0.2, 0.3],
  "top_k": 10,
  "where_filter": {"tenant": {"$eq": "a"}},
  "include_document": true,
  "ann_search_list_size": 64
}
```

Response `200`:

```json
{
  "results": [
    {
      "id": "id-1",
      "score": 0.98,
      "metadata": {"source": "faq"},
      "document": "..."
    }
  ]
}
```

### `POST /tenants/{tenant}/databases/{database}/collections/{collection}/query`

Batched query API.

Request:

```json
{
  "query_embeddings": [[0.1, 0.2], [0.3, 0.4]],
  "n_results": 5,
  "offset": 0,
  "include": ["ids", "scores", "metadatas", "documents"],
  "where_filter": {"source": {"$in": ["faq", "blog"]}},
  "ann_search_list_size": 64
}
```

Response `200`:

```json
{
  "ids": [["id-1"], ["id-7"]],
  "scores": [[0.97], [0.91]],
  "metadatas": [[{"source": "faq"}], [{"source": "blog"}]],
  "documents": [["..."], ["..."]]
}
```

### `POST /tenants/{tenant}/databases/{database}/collections/{collection}/search_hybrid`

Request:

```json
{
  "query": [0.1, 0.2, 0.3],
  "query_text": "turboquant roadmap",
  "top_k": 10,
  "dense_weight": 0.7,
  "sparse_weight": 0.3,
  "where_filter": {"tenant": {"$eq": "a"}},
  "include_document": true
}
```

Response shape: same as `/search`.

### `POST /tenants/{tenant}/databases/{database}/collections/{collection}/search_rerank`

Request:

```json
{
  "query": [0.1, 0.2, 0.3],
  "top_k": 10,
  "rerank_top_n": 50,
  "where_filter": {"tenant": {"$eq": "a"}},
  "ann_search_list_size": 64
}
```

Response shape: same as `/search`.

Note: reranker model callback transport is implementation-defined (local plugin/model ID in service context).

## Index and Maintenance

### `POST /tenants/{tenant}/databases/{database}/collections/{collection}/index`

Request:

```json
{
  "max_degree": 32,
  "search_list_size": 100,
  "alpha": 1.2,
  "async": true
}
```

Response `202` (async):

```json
{ "job_id": "job_01J...", "status": "queued" }
```

### `POST /tenants/{tenant}/databases/{database}/collections/{collection}/compact`

Request:

```json
{ "async": true }
```

Response `202`: same job format.

### `POST /tenants/{tenant}/databases/{database}/collections/{collection}/snapshot`

Request:

```json
{
  "snapshot_name": "docs-2026-03-28",
  "async": true
}
```

Response `202`: same job format.

### `POST /tenants/{tenant}/databases/{database}/collections/{collection}/restore`

Request:

```json
{
  "snapshot_name": "docs-2026-03-28",
  "async": true
}
```

Response `202`: same job format.

## Stats

### `GET /tenants/{tenant}/databases/{database}/collections/{collection}/stats`

Response `200`:

```json
{
  "vector_count": 120000,
  "physical_record_count": 130000,
  "deleted_record_count": 10000,
  "segment_count": 8,
  "buffered_vectors": 0,
  "dimension": 1536,
  "bits": 4,
  "total_disk_bytes": 123456789,
  "segment_min_bytes": 1000,
  "segment_max_bytes": 2000,
  "segment_avg_bytes": 1500.0,
  "segment_skew_ratio": 2.0,
  "has_index": true,
  "index_nodes": 120000,
  "index_search_list_size": 100,
  "index_alpha": 1.2,
  "compaction_runs": 12,
  "compaction_recovery_runs": 1,
  "last_reclaimed_segments": 4
}
```

## Jobs

### `GET /jobs/{job_id}`

Response `200`:

```json
{
  "job_id": "job_01J...",
  "type": "index_build",
  "status": "running",
  "tenant": "t1",
  "database": "db1",
  "collection": "docs",
  "created_at": "2026-03-28T18:00:00Z",
  "started_at": "2026-03-28T18:00:02Z",
  "completed_at": null,
  "progress": {"done": 1200, "total": 10000},
  "error": null
}
```

### `GET /tenants/{tenant}/databases/{database}/collections/{collection}/jobs`

Response `200`:

```json
{
  "jobs": [
    {"job_id": "job_01J...", "type": "index_build", "status": "succeeded"}
  ]
}
```

## Security and Isolation Rules

- Request tenant must match authenticated tenant unless caller has tenant-admin cross-scope role.
- All reads/writes require resource-level authorization.
- Quota checks run before mutating operations.
- Job APIs must enforce same tenant/database/collection visibility constraints.

## Versioning

- Breaking changes require `/v2`.
- Additive fields are allowed in responses.
- Clients must ignore unknown fields.

## Implementation Notes

- Map `invalid_argument` to HTTP 400.
- Map `unauthenticated` to 401.
- Map `forbidden` to 403.
- Map `not_found` to 404.
- Map `conflict` and `resource_busy` to 409.
- Map `quota_exceeded` and `rate_limited` to 429.
- Map `internal` to 500.

