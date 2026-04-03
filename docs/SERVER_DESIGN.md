# Server Mode Design: Multi-Tenant Architecture

## Objective

Define an implementation-ready design for:

- tenant -> database -> collection isolation
- auth/RBAC and quota enforcement
- service-mode APIs with background workers

This is the target architecture for evolving TurboQuantDB from embedded-only usage to a shared service while preserving the existing embedded engine as the storage core.

## Current Baseline (as of M1-M4)

- Engine already supports per-collection isolation and collection lifecycle operations.
- Persistence model is local/remote-backed files per collection with manifest/WAL/segments/metadata.
- Python API has both direct `Database` access and collection-first `Client` APIs.
- No service process, no auth, no tenant boundary enforcement in request path.

## Scope and Non-Goals

Scope:

- Introduce explicit tenant and database namespaces above collections.
- Add authenticated request context and authorization checks.
- Add quotas and admission control.
- Add background workers for compaction/index/snapshot jobs.

Non-goals (initial M5):

- Full distributed control plane.
- Cross-region replication.
- Hard real-time SLO guarantees.

## Namespace Model

Canonical path hierarchy:

- `tenant_id`
- `database_id` (scoped to tenant)
- `collection_name` (scoped to database)

Logical identifiers:

- `resource_id = tenant_id/database_id/collection_name`

Storage layout (local example):

- `root/tenants/<tenant_id>/databases/<database_id>/collections/<collection_name>/...`

Catalog files:

- `tenants/_catalog.json`
- `tenants/<tenant_id>/databases/_catalog.json`
- `.../collections/_catalog.json`

## AuthN/AuthZ Model

### Authentication

Supported modes (ordered):

1. Static API keys (MVP)
2. JWT/OIDC bearer tokens (next)

Request context fields:

- `subject` (user/service)
- `tenant_id` (explicit or derived)
- `roles` (set)
- `scopes` (set)

### Authorization

RBAC resources:

- tenant
- database
- collection

Actions:

- `read`
- `write`
- `admin`
- `manage_snapshots`
- `manage_index`

Policy default:

- deny by default
- explicit allow required
- tenant boundary cannot be crossed by role escalation at lower levels

## Quotas and Limits

Quota levels:

- tenant-level hard limits
- database-level optional overrides

Quota dimensions:

- max collections
- max vectors
- max disk bytes
- max QPS (read/write)
- max concurrent background jobs

Enforcement points:

- pre-write admission (insert/upsert/update)
- collection creation
- snapshot creation
- index build request

Behavior on limit:

- return typed error code (`quota_exceeded`) with dimension + current/limit payload

## Service API Surface

Protocol:

- HTTP/JSON MVP
- optional gRPC later

Core endpoints:

- `POST /v1/tenants/{tenant}/databases/{db}/collections`
- `GET /v1/tenants/{tenant}/databases/{db}/collections`
- `DELETE /v1/tenants/{tenant}/databases/{db}/collections/{collection}`
- `POST /v1/.../{collection}/upsert`
- `POST /v1/.../{collection}/query`
- `POST /v1/.../{collection}/search`
- `POST /v1/.../{collection}/index`
- `POST /v1/.../{collection}/snapshot`

Job endpoints:

- `GET /v1/jobs/{job_id}`
- `GET /v1/.../{collection}/jobs`

Error format:

- stable code + message + details

## Background Worker Model

Worker queue types:

- compaction
- index build/rebuild
- snapshot/restore

Execution model:

- persisted job table/queue
- at-least-once execution
- idempotent job handlers

Job states:

- `queued`
- `running`
- `succeeded`
- `failed`
- `cancelled`

Recovery:

- on restart, resume `running` as `queued` with retry budget

## Concurrency and Isolation Guarantees

- Per-collection lock domain for write operations.
- Read operations allowed concurrently with immutable segment reads.
- Background jobs coordinate via collection job mutex to prevent conflicting maintenance tasks.
- Cross-tenant operations have no shared mutable state by design except global scheduler metadata.

## Data Model Changes (Minimum)

Add service metadata stores:

- `auth_store` (keys, principals, role bindings)
- `quota_store`
- `job_store`
- optional `audit_log`

Existing engine data remains unchanged under each collection path.

## Phased Implementation Plan

### Phase M5.1: Namespace + Service Skeleton

- Introduce tenant/database/collection path resolver.
- Add service process crate (`server`) with health + collection CRUD endpoints.
- Thread `RequestContext` through API handlers.

Exit criteria:

- End-to-end create/list/delete collection in tenant/database scope.

### Phase M5.2: Auth + RBAC

- API key auth middleware.
- RBAC policy evaluator with deny-by-default.
- Role bindings at tenant/database/collection levels.

Exit criteria:

- Positive and negative authorization tests for all core actions.

### Phase M5.3: Quotas + Admission Control

- Quota schema and enforcement hooks.
- Usage accounting updates from write/index/snapshot paths.

Exit criteria:

- Deterministic quota-exceeded behavior and tests.

### Phase M5.4: Background Jobs

- Job queue + worker loop.
- Async compaction/index/snapshot endpoints.
- Job status APIs.

Exit criteria:

- Restart-safe job recovery tests.

### Phase M5.5: Hardening + Observability

- Metrics and audit logs.
- P95 latency/error dashboards.
- Load and chaos testing.

Exit criteria:

- Production readiness checklist signed off.

## Risks and Mitigations

Risk: authorization bypass through missing checks.

- Mitigation: centralized middleware + integration tests per endpoint/action.

Risk: quota drift due to async jobs.

- Mitigation: reconcile worker + periodic accounting sweep.

Risk: background job contention with foreground writes.

- Mitigation: per-collection maintenance lock + retry semantics.

## Testing Strategy

- Unit tests: auth policy, quota evaluator, path resolution.
- Integration tests: tenant isolation, RBAC allow/deny, quota enforcement.
- Recovery tests: interrupted jobs and service restart semantics.
- Performance tests: service QPS, ANN query latency under multi-tenant load.

## Immediate Follow-up Tasks

1. [x] Create `docs/M5_API_SPEC.md` with request/response schemas and error codes.
2. [x] Scaffold `server` crate with routing, auth middleware, and request context.
3. Implement tenant/database path resolver and catalog helpers in storage layer.


Related spec: docs/M5_API_SPEC.md


