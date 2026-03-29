# rapidstore-server (crate: turboquantdb-server)

Service mode currently includes:

- Collection CRUD (tenant/database scoped)
- Persisted auth store (`auth_store.json`) with API keys + RBAC bindings
- Persisted quota store (`quota_store.json`) with collection-count admission checks
- Persisted job store (`job_store.json`) with async lifecycle, restart-safe recovery, and real compact/index/snapshot worker execution

## Endpoints

- `GET /healthz`
- `GET/POST /v1/tenants/:tenant/databases/:database/collections`
- `DELETE /v1/tenants/:tenant/databases/:database/collections/:collection`
- `POST /v1/tenants/:tenant/databases/:database/collections/:collection/add`
- `POST /v1/tenants/:tenant/databases/:database/collections/:collection/upsert`
- `POST /v1/tenants/:tenant/databases/:database/collections/:collection/update`
- `POST /v1/tenants/:tenant/databases/:database/collections/:collection/delete`
- `POST /v1/tenants/:tenant/databases/:database/collections/:collection/get`
- `POST /v1/tenants/:tenant/databases/:database/collections/:collection/query`
- `POST /v1/tenants/:tenant/databases/:database/collections/:collection/compact`
- `POST /v1/tenants/:tenant/databases/:database/collections/:collection/index`
- `POST /v1/tenants/:tenant/databases/:database/collections/:collection/snapshot`
- `GET /v1/tenants/:tenant/databases/:database/collections/:collection/jobs`
- `GET /v1/jobs/:job_id`
- `POST /v1/jobs/:job_id/cancel`
- `POST /v1/jobs/:job_id/retry`

## Env Vars

- `TQ_SERVER_ADDR` (default `127.0.0.1:8080`)
- `TQ_LOCAL_ROOT` (default `./data`)
- `TQ_STORAGE_URI` (default `TQ_LOCAL_ROOT`)
- `TQ_AUTH_STORE_PATH` (default `<TQ_LOCAL_ROOT>/auth_store.json`)
- `TQ_QUOTA_STORE_PATH` (default `<TQ_LOCAL_ROOT>/quota_store.json`)
- `TQ_JOB_STORE_PATH` (default `<TQ_LOCAL_ROOT>/job_store.json`)
- `TQ_JOB_WORKERS` (default `2`)



### Data-Plane Request Notes

- `POST .../add` and `POST .../upsert` support optional `report` (when `true`, returns partial-failure report with `applied` and `failed[]` instead of fail-fast).
- `POST .../get` supports optional selectors (`ids`, `filter`, `where_filter`) plus `include`, `offset`, `limit`.
- `POST .../query` supports `top_k` (or alias `n_results`), `filter` (or alias `where_filter`), optional `include`, and `offset`.
- `include` defaults to all allowed fields for each endpoint.




