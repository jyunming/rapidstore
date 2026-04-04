# Server API Reference

## Purpose

Define the HTTP/JSON contract for multi-tenant TurboQuantDB service mode.

## Conventions

- All endpoints require authentication (except `/healthz`).
- IDs are ASCII strings unless stated otherwise.
- Timestamps in job responses are Unix epoch seconds as strings (e.g., `"1743800000"`).
- Error responses are stable and machine-readable.

## Headers

Required:

- `Authorization: ApiKey <key>`

Optional:

- `X-Request-Id: <client-id>` — echoed back in error responses

## Error Model

All non-2xx responses:

```json
{
  "error": {
    "code": "quota_exceeded",
    "message": "vector quota exceeded for tenant='t1' database='db1' ..."
  },
  "request_id": "my-req-id"
}
```

Canonical error codes:

- `unauthenticated` — missing or invalid API key (401)
- `forbidden` — valid key but insufficient permissions (403)
- `invalid_argument` — bad request body or parameters (400)
- `not_found` — resource does not exist (404)
- `conflict` — resource already exists, or job state mismatch (409)
- `quota_exceeded` — vector/disk/job quota exceeded (429)
- `internal` — unexpected server error (500)

## Health

### `GET /healthz`

No authentication required.

Response `200`:

```json
{ "status": "ok" }
```

## Collections

### `POST /v1/tenants/{tenant}/databases/{database}/collections`

Create collection. Returns `409` if the collection already exists.

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

Fields: `name` (required), `dimension` (required), `bits` (required), `seed` (optional, default `42`), `metric` (optional, default `"ip"`; values: `"ip"`, `"cosine"`, `"l2"`).

Response `201`:

```json
{
  "tenant": "t1",
  "database": "db1",
  "name": "docs",
  "dimension": 1536,
  "bits": 4,
  "seed": 42,
  "metric": "ip"
}
```

### `GET /v1/tenants/{tenant}/databases/{database}/collections`

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

### `DELETE /v1/tenants/{tenant}/databases/{database}/collections/{collection}`

Delete collection. Returns `404` if it does not exist.

Response `200`:

```json
{ "deleted": true }
```

## Vector Write APIs

### `POST /v1/tenants/{tenant}/databases/{database}/collections/{collection}/add`

Insert vectors. Fails with `409` if any ID already exists (fail-fast mode) or reports per-item failures when `report=true`.

Request:

```json
{
  "ids": ["id-1", "id-2"],
  "embeddings": [[0.1, 0.2], [0.3, 0.4]],
  "metadatas": [{"source": "faq"}, {"source": "blog"}],
  "documents": ["doc1", "doc2"],
  "report": false
}
```

- `embeddings` — required; list of float arrays matching the collection's dimension
- `metadatas`, `documents` — optional; must have same length as `ids` if provided
- `report` — optional bool (default `false`); when `true`, returns partial-failure report instead of failing on first error

Response `201` (all succeeded):

```json
{ "count": 2, "applied": 2, "failed": [] }
```

Response `200` (partial failure, `report=true`):

```json
{
  "count": 1,
  "applied": 1,
  "failed": [
    {"index": 1, "id": "id-2", "error": "embedding dimension mismatch for id 'id-2': expected 1536, got 128"}
  ]
}
```

### `POST /v1/tenants/{tenant}/databases/{database}/collections/{collection}/upsert`

Insert or replace vectors. Same request/response shape as `/add`.

Response `200` (always, even without `report`):

```json
{ "count": 2, "applied": 2, "failed": [] }
```

### `POST /v1/tenants/{tenant}/databases/{database}/collections/{collection}/delete`

Delete vectors by IDs and/or metadata filter. At least one of `ids` or `filter` must be non-empty.

Request:

```json
{
  "ids": ["id-1", "id-2"],
  "filter": {"source": {"$eq": "faq"}}
}
```

Alias `where_filter` is accepted in place of `filter`.

Response `200`:

```json
{ "deleted": 2 }
```

### `POST /v1/tenants/{tenant}/databases/{database}/collections/{collection}/get`

Fetch vectors by IDs and/or metadata filter. At least one selector is required.

Request:

```json
{
  "ids": ["id-1"],
  "filter": {"source": "faq"},
  "include": ["ids", "metadatas", "documents"],
  "offset": 0,
  "limit": 10
}
```

`include` defaults to all fields (`ids`, `metadatas`, `documents`). Alias `where_filter` accepted in place of `filter`.

Response `200`:

```json
{
  "ids": ["id-1"],
  "metadatas": [{"source": "faq"}],
  "documents": ["doc text"]
}
```

Only fields listed in `include` are present in the response.

## Query API

### `POST /v1/tenants/{tenant}/databases/{database}/collections/{collection}/query`

Vector similarity search. Accepts a batch of query vectors in a single request.

Request:

```json
{
  "query_embeddings": [[0.1, 0.2], [0.3, 0.4]],
  "top_k": 5,
  "n_results": 5,
  "filter": {"source": "faq"},
  "include": ["ids", "scores", "metadatas", "documents"],
  "offset": 0
}
```

- `query_embeddings` — required; list of query vectors
- `top_k` or `n_results` — results per query (default `10`); both are accepted, `top_k` takes precedence
- `filter` / `where_filter` — optional metadata filter
- `include` — optional; defaults to all fields (`ids`, `scores`, `metadatas`, `documents`)
- `offset` — optional; skip the first N results per query (default `0`)

Response `200` — one entry in `results` per query vector:

```json
{
  "results": [
    {
      "ids": ["id-1", "id-3"],
      "scores": [0.97, 0.91],
      "metadatas": [{"source": "faq"}, {"source": "faq"}],
      "documents": ["doc text 1", "doc text 3"]
    },
    {
      "ids": ["id-7"],
      "scores": [0.88],
      "metadatas": [{"source": "faq"}],
      "documents": ["doc text 7"]
    }
  ]
}
```

Only fields listed in `include` are present in each result row.

## Index and Maintenance

### `POST /v1/tenants/{tenant}/databases/{database}/collections/{collection}/index`

Enqueue an HNSW index build job. Index parameters (max_degree, ef_construction, etc.) are fixed at the server's defaults — use the embedded `Database.create_index()` API for custom tuning.

Request:

```json
{ "async": true }
```

Response `202`:

```json
{ "job_id": "job_0000000000000001", "status": "queued" }
```

### `POST /v1/tenants/{tenant}/databases/{database}/collections/{collection}/compact`

Enqueue a background compaction job.

Request:

```json
{ "async": true }
```

Response `202`:

```json
{ "job_id": "job_0000000000000002", "status": "queued" }
```

### `POST /v1/tenants/{tenant}/databases/{database}/collections/{collection}/snapshot`

Enqueue a snapshot job. Copies the collection directory to `<TQ_LOCAL_ROOT>/snapshots/{tenant}/{database}/{collection}/{snapshot_name}/`.

Request:

```json
{
  "snapshot_name": "docs-backup",
  "async": true
}
```

`snapshot_name` is optional; defaults to `snapshot_{created_at}_{job_id}`.

Response `202`:

```json
{ "job_id": "job_0000000000000003", "status": "queued" }
```

## Quota Usage

### `GET /v1/tenants/{tenant}/databases/{database}/quota_usage`

Returns current quota limits and usage for a database.

Response `200`:

```json
{
  "tenant": "t1",
  "database": "db1",
  "max_collections": null,
  "max_vectors": 1000000,
  "max_disk_bytes": null,
  "max_concurrent_jobs": 4,
  "current_collections": 3,
  "current_vectors": 50000,
  "current_disk_bytes": 123456789,
  "queued_jobs": 0,
  "running_jobs": 1
}
```

`null` limit means unrestricted.

## Jobs

### `GET /v1/jobs/{job_id}`

Response `200` — job record nested under `"job"`:

```json
{
  "job": {
    "job_id": "job_0000000000000001",
    "job_type": "index_build",
    "status": "running",
    "tenant": "t1",
    "database": "db1",
    "collection": "docs",
    "snapshot_name": null,
    "created_at": "1743800000",
    "started_at": "1743800002",
    "completed_at": null,
    "error": null,
    "attempts": 1,
    "max_attempts": 3
  }
}
```

`job_type` values: `"compact"`, `"index_build"`, `"snapshot"`

`status` values: `"queued"`, `"running"`, `"succeeded"`, `"failed"`, `"canceled"`

Timestamps are Unix epoch seconds as strings.

### `GET /v1/tenants/{tenant}/databases/{database}/collections/{collection}/jobs`

List all jobs for a collection.

Response `200`:

```json
{
  "jobs": [
    {
      "job_id": "job_0000000000000001",
      "job_type": "index_build",
      "status": "succeeded",
      ...
    }
  ]
}
```

### `POST /v1/jobs/{job_id}/cancel`

Cancel a queued job. Returns `409` if the job is already running or terminal.

Response `200`: same `JobStatusResponse` shape as `GET /v1/jobs/{job_id}`.

### `POST /v1/jobs/{job_id}/retry`

Retry a failed or canceled job. Returns `409` if the job is not in a retryable state or retry budget is exhausted (max 3 attempts).

Response `200`: same `JobStatusResponse` shape as `GET /v1/jobs/{job_id}`.

## Security and Isolation Rules

- All requests (except `/healthz`) require `Authorization: ApiKey <key>`.
- The authenticated principal's `tenant_id` must match the request's `{tenant}` path segment.
- All reads/writes require a matching role binding (checked against `auth_store.json`).
- Quota checks run before every mutating operation.
- Job visibility is scoped to the tenant/database/collection of the job.

## Versioning

- Breaking changes require `/v2`.
- Additive fields are allowed in responses.
- Clients must ignore unknown fields.

## HTTP Status Code Mapping

| Code | Error code |
|------|------------|
| 400 | `invalid_argument` |
| 401 | `unauthenticated` |
| 403 | `forbidden` |
| 404 | `not_found` |
| 409 | `conflict` |
| 429 | `quota_exceeded` |
| 500 | `internal` |

