use axum::extract::{Extension, Path, Request, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::path::{Path as StdPath, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{error, info};
use turboquantdb::storage::engine::{BatchWriteItem, DistanceMetric, TurboQuantEngine};

#[derive(Clone)]
struct AppState {
    auth: Arc<AuthStore>,
    quotas: Arc<QuotaStore>,
    jobs: Arc<Mutex<JobStore>>,
    storage: StorageConfig,
    job_worker_concurrency: usize,
}

#[derive(Clone)]
struct StorageConfig {
    uri: String,
    local_root: String,
    auth_store_path: String,
    quota_store_path: String,
    job_store_path: String,
}

#[derive(Clone)]
struct AuthStore {
    api_key_subjects: HashMap<String, String>,
    principals: HashMap<String, Principal>,
    role_bindings: Vec<RoleBinding>,
}

#[derive(Clone)]
struct QuotaStore {
    tenant_quotas: HashMap<String, TenantQuota>,
    database_quotas: HashMap<String, DatabaseQuota>,
}

struct JobStore {
    jobs: HashMap<String, JobRecord>,
    next_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Principal {
    subject: String,
    tenant_id: Option<String>,
    roles: HashSet<String>,
    scopes: HashSet<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RoleBinding {
    subject: String,
    tenant: Option<String>,
    database: Option<String>,
    collection: Option<String>,
    actions: HashSet<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ApiKeyEntry {
    key: String,
    subject: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AuthStoreFile {
    api_keys: Vec<ApiKeyEntry>,
    principals: Vec<Principal>,
    role_bindings: Vec<RoleBinding>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TenantQuota {
    tenant: String,
    max_collections: Option<usize>,
    max_vectors: Option<u64>,
    max_disk_bytes: Option<u64>,
    max_concurrent_jobs: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DatabaseQuota {
    tenant: String,
    database: String,
    max_collections: Option<usize>,
    max_vectors: Option<u64>,
    max_disk_bytes: Option<u64>,
    max_concurrent_jobs: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuotaStoreFile {
    tenant_quotas: Vec<TenantQuota>,
    database_quotas: Vec<DatabaseQuota>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum JobType {
    Compact,
    IndexBuild,
    Snapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum JobStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
    Canceled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JobRecord {
    job_id: String,
    job_type: JobType,
    status: JobStatus,
    tenant: String,
    database: String,
    collection: String,
    snapshot_name: Option<String>,
    created_at: String,
    started_at: Option<String>,
    completed_at: Option<String>,
    error: Option<String>,
    #[serde(default)]
    attempts: u32,
    #[serde(default = "default_job_max_attempts")]
    max_attempts: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JobStoreFile {
    jobs: Vec<JobRecord>,
}

fn default_job_max_attempts() -> u32 { 3 }

fn build_batch_items(
    ids: &[String],
    embeddings: &[Vec<f64>],
    metadatas: Option<&Vec<HashMap<String, JsonValue>>>,
    documents: Option<&Vec<String>>,
) -> Vec<BatchWriteItem> {
    (0..ids.len())
        .map(|i| BatchWriteItem {
            id: ids[i].clone(),
            vector: Array1::from(embeddings[i].clone()),
            metadata: metadatas
                .and_then(|m| m.get(i).cloned())
                .unwrap_or_default(),
            document: documents.and_then(|d| d.get(i).cloned()),
        })
        .collect()
}
fn parse_include_set(
    include: Option<&Vec<String>>,
    defaults: &[&str],
    allowed: &[&str],
    request_id: Option<String>,
) -> Result<HashSet<String>, ApiError> {
    let allowed_set = allowed.iter().map(|s| s.to_string()).collect::<HashSet<_>>();
    let values = include
        .cloned()
        .unwrap_or_else(|| defaults.iter().map(|s| s.to_string()).collect());

    let mut out = HashSet::new();
    for raw in values {
        let key = raw.trim().to_ascii_lowercase();
        if !allowed_set.contains(&key) {
            return Err(ApiError::invalid_argument(
                format!(
                    "invalid include field '{}' (allowed: {})",
                    raw,
                    allowed.join(", ")
                ),
                request_id,
            ));
        }
        out.insert(key);
    }
    Ok(out)
}

#[derive(Clone)]
struct RequestContext {
    subject: String,
    tenant_id: Option<String>,
    scopes: HashSet<String>,
    request_id: Option<String>,
}

#[derive(Serialize)]
struct ErrorBody {
    error: ErrorInfo,
    request_id: Option<String>,
}

#[derive(Serialize)]
struct ErrorInfo {
    code: &'static str,
    message: String,
}

struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
    request_id: Option<String>,
}

impl ApiError {
    fn unauthenticated(msg: impl Into<String>, request_id: Option<String>) -> Self {
        Self { status: StatusCode::UNAUTHORIZED, code: "unauthenticated", message: msg.into(), request_id }
    }
    fn forbidden(msg: impl Into<String>, request_id: Option<String>) -> Self {
        Self { status: StatusCode::FORBIDDEN, code: "forbidden", message: msg.into(), request_id }
    }
    fn invalid_argument(msg: impl Into<String>, request_id: Option<String>) -> Self {
        Self { status: StatusCode::BAD_REQUEST, code: "invalid_argument", message: msg.into(), request_id }
    }
    fn not_found(msg: impl Into<String>, request_id: Option<String>) -> Self {
        Self { status: StatusCode::NOT_FOUND, code: "not_found", message: msg.into(), request_id }
    }
    fn conflict(msg: impl Into<String>, request_id: Option<String>) -> Self {
        Self { status: StatusCode::CONFLICT, code: "conflict", message: msg.into(), request_id }
    }
    fn quota_exceeded(msg: impl Into<String>, request_id: Option<String>) -> Self {
        Self { status: StatusCode::TOO_MANY_REQUESTS, code: "quota_exceeded", message: msg.into(), request_id }
    }
    fn internal(msg: impl Into<String>, request_id: Option<String>) -> Self {
        Self { status: StatusCode::INTERNAL_SERVER_ERROR, code: "internal", message: msg.into(), request_id }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = ErrorBody { error: ErrorInfo { code: self.code, message: self.message }, request_id: self.request_id };
        (self.status, Json(body)).into_response()
    }
}

#[derive(Serialize)]
struct HealthResponse { status: &'static str }

#[derive(Deserialize)]
struct CreateCollectionRequest {
    name: String,
    dimension: usize,
    bits: usize,
    seed: Option<u64>,
    metric: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct CollectionInfo { name: String }

#[derive(Serialize, Deserialize)]
struct ListCollectionsResponse { collections: Vec<CollectionInfo> }

#[derive(Serialize, Deserialize)]
struct CreateCollectionResponse {
    tenant: String,
    database: String,
    name: String,
    dimension: usize,
    bits: usize,
    seed: u64,
    metric: String,
}

#[derive(Serialize, Deserialize)]
struct DeleteCollectionResponse { deleted: bool }

#[derive(Deserialize)]
struct CompactRequest { r#async: Option<bool> }
#[derive(Deserialize)]
struct IndexRequest { r#async: Option<bool> }
#[derive(Deserialize)]
struct SnapshotRequest { r#async: Option<bool>, snapshot_name: Option<String> }
#[derive(Deserialize)]
struct AddVectorsRequest {
    ids: Vec<String>,
    embeddings: Vec<Vec<f64>>,
    metadatas: Option<Vec<HashMap<String, JsonValue>>>,
    documents: Option<Vec<String>>,
    report: Option<bool>,
}

#[derive(Deserialize)]
struct UpsertVectorsRequest {
    ids: Vec<String>,
    embeddings: Vec<Vec<f64>>,
    metadatas: Option<Vec<HashMap<String, JsonValue>>>,
    documents: Option<Vec<String>>,
    report: Option<bool>,
}

#[derive(Deserialize)]
struct DeleteVectorsRequest { ids: Vec<String> }

#[derive(Deserialize)]
struct GetVectorsRequest {
    ids: Vec<String>,
    include: Option<Vec<String>>,
    offset: Option<usize>,
    limit: Option<usize>,
}

#[derive(Deserialize)]
struct QueryVectorsRequest {
    query_embeddings: Vec<Vec<f64>>,
    top_k: usize,
    filter: Option<HashMap<String, JsonValue>>,
    include: Option<Vec<String>>,
    offset: Option<usize>,
}

#[derive(Serialize, Deserialize)]
struct BatchWriteFailureResponse {
    index: usize,
    id: String,
    error: String,
}

#[derive(Serialize, Deserialize)]
struct WriteCountResponse {
    count: usize,
    applied: usize,
    failed: Vec<BatchWriteFailureResponse>,
}

#[derive(Serialize, Deserialize)]
struct DeleteCountResponse { deleted: usize }

#[derive(Serialize, Deserialize)]
struct GetVectorsResponse {
    ids: Option<Vec<String>>,
    metadatas: Option<Vec<HashMap<String, JsonValue>>>,
    documents: Option<Vec<Option<String>>>,
}

#[derive(Serialize, Deserialize)]
struct QueryRow {
    ids: Option<Vec<String>>,
    scores: Option<Vec<f64>>,
    metadatas: Option<Vec<HashMap<String, JsonValue>>>,
    documents: Option<Vec<Option<String>>>,
}

#[derive(Serialize, Deserialize)]
struct QueryVectorsResponse { results: Vec<QueryRow> }

#[derive(Serialize, Deserialize)]
struct JobEnqueueResponse { job_id: String, status: JobStatus }
#[derive(Serialize, Deserialize)]
struct JobStatusResponse { job: JobRecord }
#[derive(Serialize, Deserialize)]
struct ListJobsResponse { jobs: Vec<JobRecord> }

#[derive(Serialize, Deserialize)]
struct QuotaUsageResponse {
    tenant: String,
    database: String,
    max_collections: Option<usize>,
    max_vectors: Option<u64>,
    max_disk_bytes: Option<u64>,
    max_concurrent_jobs: Option<usize>,
    current_collections: usize,
    current_vectors: u64,
    current_disk_bytes: u64,
    queued_jobs: usize,
    running_jobs: usize,
}

fn build_app(state: AppState) -> Router {
    let protected = Router::new()
        .route("/v1/tenants/:tenant/databases/:database/collections", get(list_collections).post(create_collection))
        .route("/v1/tenants/:tenant/databases/:database/collections/:collection", delete(delete_collection))
        .route("/v1/tenants/:tenant/databases/:database/collections/:collection/add", post(add_vectors))
        .route("/v1/tenants/:tenant/databases/:database/collections/:collection/upsert", post(upsert_vectors))
        .route("/v1/tenants/:tenant/databases/:database/collections/:collection/delete", post(delete_vectors))
        .route("/v1/tenants/:tenant/databases/:database/collections/:collection/get", post(get_vectors))
        .route("/v1/tenants/:tenant/databases/:database/collections/:collection/query", post(query_vectors))
        .route("/v1/tenants/:tenant/databases/:database/collections/:collection/compact", post(start_compact_job))
        .route("/v1/tenants/:tenant/databases/:database/collections/:collection/index", post(start_index_job))
        .route("/v1/tenants/:tenant/databases/:database/collections/:collection/snapshot", post(start_snapshot_job))
        .route("/v1/tenants/:tenant/databases/:database/collections/:collection/jobs", get(list_collection_jobs))
        .route("/v1/tenants/:tenant/databases/:database/quota_usage", get(get_quota_usage))
        .route("/v1/jobs/:job_id", get(get_job_status))
        .route("/v1/jobs/:job_id/cancel", post(cancel_job))
        .route("/v1/jobs/:job_id/retry", post(retry_job))
        .with_state(state.clone())
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

    Router::new().route("/healthz", get(healthz)).merge(protected).with_state(state)
}

fn default_auth_store_file() -> AuthStoreFile {
    AuthStoreFile {
        api_keys: vec![ApiKeyEntry { key: "dev-key".into(), subject: "dev-user".into() }],
        principals: vec![Principal {
            subject: "dev-user".into(),
            tenant_id: Some("dev".into()),
            roles: HashSet::from(["tenant_admin".into()]),
            scopes: HashSet::from(["read".into(), "write".into(), "admin".into()]),
        }],
        role_bindings: vec![RoleBinding {
            subject: "dev-user".into(),
            tenant: Some("dev".into()),
            database: None,
            collection: None,
            actions: HashSet::from(["read".into(), "write".into(), "admin".into()]),
        }],
    }
}

fn default_quota_store_file() -> QuotaStoreFile { QuotaStoreFile { tenant_quotas: vec![], database_quotas: vec![] } }
fn default_job_store_file() -> JobStoreFile { JobStoreFile { jobs: vec![] } }

fn load_or_init_auth_store(path: &str) -> Result<AuthStore, Box<dyn std::error::Error + Send + Sync>> {
    let p = StdPath::new(path);
    let file = if p.exists() {
        serde_json::from_str::<AuthStoreFile>(&std::fs::read_to_string(p)?)?
    } else {
        if let Some(parent) = p.parent() { std::fs::create_dir_all(parent)?; }
        let d = default_auth_store_file();
        std::fs::write(p, serde_json::to_string_pretty(&d)?)?;
        d
    };
    let mut api_key_subjects = HashMap::new();
    for e in file.api_keys { api_key_subjects.insert(e.key, e.subject); }
    let mut principals = HashMap::new();
    for pr in file.principals { principals.insert(pr.subject.clone(), pr); }
    Ok(AuthStore { api_key_subjects, principals, role_bindings: file.role_bindings })
}

fn load_or_init_quota_store(path: &str) -> Result<QuotaStore, Box<dyn std::error::Error + Send + Sync>> {
    let p = StdPath::new(path);
    let file = if p.exists() {
        serde_json::from_str::<QuotaStoreFile>(&std::fs::read_to_string(p)?)?
    } else {
        if let Some(parent) = p.parent() { std::fs::create_dir_all(parent)?; }
        let d = default_quota_store_file();
        std::fs::write(p, serde_json::to_string_pretty(&d)?)?;
        d
    };
    let mut tenant_quotas = HashMap::new();
    for q in file.tenant_quotas { tenant_quotas.insert(q.tenant.clone(), q); }
    let mut database_quotas = HashMap::new();
    for q in file.database_quotas { database_quotas.insert(format!("{}/{}", q.tenant, q.database), q); }
    Ok(QuotaStore { tenant_quotas, database_quotas })
}

fn load_or_init_job_store(path: &str) -> Result<JobStore, Box<dyn std::error::Error + Send + Sync>> {
    let p = StdPath::new(path);
    let file = if p.exists() {
        serde_json::from_str::<JobStoreFile>(&std::fs::read_to_string(p)?)?
    } else {
        if let Some(parent) = p.parent() { std::fs::create_dir_all(parent)?; }
        let d = default_job_store_file();
        std::fs::write(p, serde_json::to_string_pretty(&d)?)?;
        d
    };
    let mut jobs = HashMap::new();
    let mut next_id = 1u64;
    let mut recovered_running_jobs = false;
    for mut j in file.jobs {
        if let Some(n) = j.job_id.strip_prefix("job_").and_then(|s| s.parse::<u64>().ok()) { next_id = next_id.max(n + 1); }
        if j.max_attempts == 0 {
            j.max_attempts = default_job_max_attempts();
        }
        if matches!(j.status, JobStatus::Running) {
            recovered_running_jobs = true;
            j.attempts = j.attempts.saturating_add(1);
            j.started_at = None;
            j.completed_at = None;
            if j.attempts > j.max_attempts {
                j.status = JobStatus::Failed;
                j.error = Some(format!("retry budget exhausted (attempts={}, max_attempts={})", j.attempts, j.max_attempts));
            } else {
                j.status = JobStatus::Queued;
                j.error = Some("recovered after restart".to_string());
            }
        }
        jobs.insert(j.job_id.clone(), j);
    }
    let store = JobStore { jobs, next_id };
    if recovered_running_jobs {
        save_job_store(path, &store)?;
    }
    Ok(store)
}

fn save_job_store(path: &str, store: &JobStore) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut jobs = store.jobs.values().cloned().collect::<Vec<_>>();
    jobs.sort_by(|a, b| a.job_id.cmp(&b.job_id));
    std::fs::write(path, serde_json::to_string_pretty(&JobStoreFile { jobs })?)?;
    Ok(())
}

fn now_ts() -> String {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0).to_string()
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "turboquantdb_server=info,axum=info".to_string()))
        .init();

    let local_root = std::env::var("TQ_LOCAL_ROOT").unwrap_or_else(|_| "./data".to_string());
    let uri = std::env::var("TQ_STORAGE_URI").unwrap_or_else(|_| local_root.clone());
    let auth_store_path = std::env::var("TQ_AUTH_STORE_PATH").unwrap_or_else(|_| format!("{local_root}/auth_store.json"));
    let quota_store_path = std::env::var("TQ_QUOTA_STORE_PATH").unwrap_or_else(|_| format!("{local_root}/quota_store.json"));
    let job_store_path = std::env::var("TQ_JOB_STORE_PATH").unwrap_or_else(|_| format!("{local_root}/job_store.json"));
    let job_worker_concurrency = std::env::var("TQ_JOB_WORKERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(2);

    let state = AppState {
        auth: Arc::new(load_or_init_auth_store(&auth_store_path).expect("failed to load auth store")),
        quotas: Arc::new(load_or_init_quota_store(&quota_store_path).expect("failed to load quota store")),
        jobs: Arc::new(Mutex::new(load_or_init_job_store(&job_store_path).expect("failed to load job store"))),
        storage: StorageConfig { uri, local_root, auth_store_path, quota_store_path, job_store_path },
        job_worker_concurrency,
    };

    dispatch_queued_jobs(&state);

    let addr: SocketAddr = std::env::var("TQ_SERVER_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string()).parse().expect("invalid TQ_SERVER_ADDR");
    info!(%addr, auth_store=%state.storage.auth_store_path, quota_store=%state.storage.quota_store_path, job_store=%state.storage.job_store_path, job_workers=state.job_worker_concurrency, "starting turboquantdb server");

    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind failed");
    axum::serve(listener, build_app(state)).await.expect("server failed");
}

async fn healthz() -> Json<HealthResponse> { Json(HealthResponse { status: "ok" }) }

async fn auth_middleware(State(state): State<AppState>, mut req: Request, next: Next) -> Result<Response, ApiError> {
    let request_id = header_value(req.headers(), "x-request-id");
    let principal = authenticate(req.headers(), &state.auth)
        .ok_or_else(|| ApiError::unauthenticated("missing or invalid authorization; expected 'Authorization: ApiKey <key>'", request_id.clone()))?;
    req.extensions_mut().insert(RequestContext { subject: principal.subject, tenant_id: principal.tenant_id, scopes: principal.scopes, request_id });
    Ok(next.run(req).await)
}

fn authenticate(headers: &HeaderMap, store: &AuthStore) -> Option<Principal> {
    let auth = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    let key = auth.strip_prefix("ApiKey ")?.trim();
    let subject = store.api_key_subjects.get(key)?;
    store.principals.get(subject).cloned()
}

fn header_value(headers: &HeaderMap, name: &str) -> Option<String> { headers.get(name)?.to_str().ok().map(str::to_string) }

async fn list_collections(State(state): State<AppState>, Path((tenant, database)): Path<(String, String)>, Extension(ctx): Extension<RequestContext>) -> Result<Json<ListCollectionsResponse>, ApiError> {
    authorize(&ctx, &state.auth, "read", &tenant, &database, None)?;
    let collections = TurboQuantEngine::list_collections_scoped(&state.storage.local_root, &tenant, &database)
        .map_err(|e| map_engine_err(e, ctx.request_id.clone()))?
        .into_iter()
        .map(|name| CollectionInfo { name })
        .collect();
    Ok(Json(ListCollectionsResponse { collections }))
}

async fn create_collection(State(state): State<AppState>, Path((tenant, database)): Path<(String, String)>, Extension(ctx): Extension<RequestContext>, Json(body): Json<CreateCollectionRequest>) -> Result<(StatusCode, Json<CreateCollectionResponse>), ApiError> {
    authorize(&ctx, &state.auth, "write", &tenant, &database, None)?;
    if body.dimension == 0 || body.bits == 0 {
        return Err(ApiError::invalid_argument("dimension and bits must be greater than 0", ctx.request_id.clone()));
    }

    if TurboQuantEngine::get_collection_scoped_with_uri(&state.storage.uri, &state.storage.local_root, &tenant, &database, &body.name)
        .map_err(|e| map_engine_err(e, ctx.request_id.clone()))?
        .is_some()
    {
        return Err(ApiError::conflict(format!("collection '{}' already exists in tenant '{}' database '{}'", body.name, tenant, database), ctx.request_id.clone()));
    }

    enforce_create_collection_quota(&state.quotas, &state.storage.local_root, &tenant, &database, ctx.request_id.clone())?;

    TurboQuantEngine::create_collection_scoped_with_uri(&state.storage.uri, &state.storage.local_root, &tenant, &database, &body.name)
        .map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;

    let metric = parse_metric(body.metric.as_deref(), ctx.request_id.clone())?;
    let seed = body.seed.unwrap_or(42);
    let mut engine = TurboQuantEngine::open_collection_scoped(&state.storage.uri, &state.storage.local_root, &tenant, &database, &body.name, body.dimension, body.bits, seed, metric.clone())
        .map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;
    engine.close().map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;

    Ok((StatusCode::CREATED, Json(CreateCollectionResponse { tenant, database, name: body.name, dimension: body.dimension, bits: body.bits, seed, metric: metric_to_str(&metric).to_string() })))
}

async fn delete_collection(State(state): State<AppState>, Path((tenant, database, collection)): Path<(String, String, String)>, Extension(ctx): Extension<RequestContext>) -> Result<Json<DeleteCollectionResponse>, ApiError> {
    authorize(&ctx, &state.auth, "write", &tenant, &database, Some(&collection))?;
    let deleted = TurboQuantEngine::delete_collection_scoped_with_uri(&state.storage.uri, &state.storage.local_root, &tenant, &database, &collection)
        .map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;
    if !deleted {
        return Err(ApiError::not_found(format!("collection '{}' does not exist in tenant '{}' database '{}'", collection, tenant, database), ctx.request_id.clone()));
    }
    Ok(Json(DeleteCollectionResponse { deleted }))
}

async fn add_vectors(State(state): State<AppState>, Path((tenant, database, collection)): Path<(String, String, String)>, Extension(ctx): Extension<RequestContext>, Json(body): Json<AddVectorsRequest>) -> Result<(StatusCode, Json<WriteCountResponse>), ApiError> {
    authorize(&ctx, &state.auth, "write", &tenant, &database, Some(&collection))?;
    if body.ids.is_empty() {
        return Err(ApiError::invalid_argument("ids cannot be empty", ctx.request_id.clone()));
    }
    if body.ids.len() != body.embeddings.len() {
        return Err(ApiError::invalid_argument("ids and embeddings length must match", ctx.request_id.clone()));
    }
    if let Some(m) = &body.metadatas {
        if m.len() != body.ids.len() {
            return Err(ApiError::invalid_argument("metadatas length must match ids length", ctx.request_id.clone()));
        }
    }
    if let Some(d) = &body.documents {
        if d.len() != body.ids.len() {
            return Err(ApiError::invalid_argument("documents length must match ids length", ctx.request_id.clone()));
        }
    }

    let report_mode = body.report.unwrap_or(false);
    let mut engine = open_scoped_engine_from_manifest(&state, &tenant, &database, &collection)
        .map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;
    enforce_vector_quota_for_ids(&state.quotas, &tenant, &database, &engine, &body.ids, ctx.request_id.clone())?;

    if report_mode {
        let mut pre_failed = Vec::new();
        let mut items = Vec::new();
        for i in 0..body.ids.len() {
            if body.embeddings[i].len() != engine.d {
                pre_failed.push(BatchWriteFailureResponse {
                    index: i,
                    id: body.ids[i].clone(),
                    error: format!("embedding dimension mismatch for id '{}': expected {}, got {}", body.ids[i], engine.d, body.embeddings[i].len()),
                });
                continue;
            }
            items.push(BatchWriteItem {
                id: body.ids[i].clone(),
                vector: Array1::from(body.embeddings[i].clone()),
                metadata: body.metadatas.as_ref().and_then(|m| m.get(i).cloned()).unwrap_or_default(),
                document: body.documents.as_ref().and_then(|d| d.get(i).cloned()),
            });
        }

        let report = engine.insert_many_report(items);
        engine.close().map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;

        let mut failed = report
            .failed
            .into_iter()
            .map(|f| BatchWriteFailureResponse { index: f.index, id: f.id, error: f.error })
            .collect::<Vec<_>>();
        failed.extend(pre_failed);
        failed.sort_by_key(|f| f.index);

        let status = if failed.is_empty() {
            StatusCode::CREATED
        } else {
            StatusCode::OK
        };
        return Ok((status, Json(WriteCountResponse { count: report.applied, applied: report.applied, failed })));
    }

    for i in 0..body.ids.len() {
        if body.embeddings[i].len() != engine.d {
            engine.close().map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;
            return Err(ApiError::invalid_argument(
                format!("embedding dimension mismatch for id '{}': expected {}, got {}", body.ids[i], engine.d, body.embeddings[i].len()),
                ctx.request_id.clone(),
            ));
        }
    }

    let items = build_batch_items(&body.ids, &body.embeddings, body.metadatas.as_ref(), body.documents.as_ref());
    let write_result = engine.insert_many(items);
    engine.close().map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;
    write_result.map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;

    Ok((
        StatusCode::CREATED,
        Json(WriteCountResponse {
            count: body.ids.len(),
            applied: body.ids.len(),
            failed: vec![],
        }),
    ))
}

async fn upsert_vectors(State(state): State<AppState>, Path((tenant, database, collection)): Path<(String, String, String)>, Extension(ctx): Extension<RequestContext>, Json(body): Json<UpsertVectorsRequest>) -> Result<Json<WriteCountResponse>, ApiError> {
    authorize(&ctx, &state.auth, "write", &tenant, &database, Some(&collection))?;
    if body.ids.is_empty() {
        return Err(ApiError::invalid_argument("ids cannot be empty", ctx.request_id.clone()));
    }
    if body.ids.len() != body.embeddings.len() {
        return Err(ApiError::invalid_argument("ids and embeddings length must match", ctx.request_id.clone()));
    }
    if let Some(m) = &body.metadatas {
        if m.len() != body.ids.len() {
            return Err(ApiError::invalid_argument("metadatas length must match ids length", ctx.request_id.clone()));
        }
    }
    if let Some(d) = &body.documents {
        if d.len() != body.ids.len() {
            return Err(ApiError::invalid_argument("documents length must match ids length", ctx.request_id.clone()));
        }
    }

    let report_mode = body.report.unwrap_or(false);
    let mut engine = open_scoped_engine_from_manifest(&state, &tenant, &database, &collection)
        .map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;
    enforce_vector_quota_for_ids(&state.quotas, &tenant, &database, &engine, &body.ids, ctx.request_id.clone())?;

    if report_mode {
        let mut pre_failed = Vec::new();
        let mut items = Vec::new();
        for i in 0..body.ids.len() {
            if body.embeddings[i].len() != engine.d {
                pre_failed.push(BatchWriteFailureResponse {
                    index: i,
                    id: body.ids[i].clone(),
                    error: format!("embedding dimension mismatch for id '{}': expected {}, got {}", body.ids[i], engine.d, body.embeddings[i].len()),
                });
                continue;
            }
            items.push(BatchWriteItem {
                id: body.ids[i].clone(),
                vector: Array1::from(body.embeddings[i].clone()),
                metadata: body.metadatas.as_ref().and_then(|m| m.get(i).cloned()).unwrap_or_default(),
                document: body.documents.as_ref().and_then(|d| d.get(i).cloned()),
            });
        }

        let report = engine.upsert_many_report(items);
        engine.close().map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;

        let mut failed = report
            .failed
            .into_iter()
            .map(|f| BatchWriteFailureResponse { index: f.index, id: f.id, error: f.error })
            .collect::<Vec<_>>();
        failed.extend(pre_failed);
        failed.sort_by_key(|f| f.index);

        return Ok(Json(WriteCountResponse { count: report.applied, applied: report.applied, failed }));
    }

    for i in 0..body.ids.len() {
        if body.embeddings[i].len() != engine.d {
            engine.close().map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;
            return Err(ApiError::invalid_argument(
                format!("embedding dimension mismatch for id '{}': expected {}, got {}", body.ids[i], engine.d, body.embeddings[i].len()),
                ctx.request_id.clone(),
            ));
        }
    }

    let items = build_batch_items(&body.ids, &body.embeddings, body.metadatas.as_ref(), body.documents.as_ref());
    let write_result = engine.upsert_many(items);
    engine.close().map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;
    write_result.map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;

    Ok(Json(WriteCountResponse {
        count: body.ids.len(),
        applied: body.ids.len(),
        failed: vec![],
    }))
}

async fn delete_vectors(State(state): State<AppState>, Path((tenant, database, collection)): Path<(String, String, String)>, Extension(ctx): Extension<RequestContext>, Json(body): Json<DeleteVectorsRequest>) -> Result<Json<DeleteCountResponse>, ApiError> {
    authorize(&ctx, &state.auth, "write", &tenant, &database, Some(&collection))?;
    let mut engine = open_scoped_engine_from_manifest(&state, &tenant, &database, &collection)
        .map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;
    let deleted = engine.delete_many(&body.ids).map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;
    engine.close().map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;
    Ok(Json(DeleteCountResponse { deleted }))
}

async fn get_vectors(State(state): State<AppState>, Path((tenant, database, collection)): Path<(String, String, String)>, Extension(ctx): Extension<RequestContext>, Json(body): Json<GetVectorsRequest>) -> Result<Json<GetVectorsResponse>, ApiError> {
    authorize(&ctx, &state.auth, "read", &tenant, &database, Some(&collection))?;
    let include_set = parse_include_set(
        body.include.as_ref(),
        &["ids", "metadatas", "documents"],
        &["ids", "metadatas", "documents"],
        ctx.request_id.clone(),
    )?;
    let offset = body.offset.unwrap_or(0);

    let mut engine = open_scoped_engine_from_manifest(&state, &tenant, &database, &collection)
        .map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;
    let mut rows = engine.get_many(&body.ids).map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;
    if offset > 0 {
        rows = rows.into_iter().skip(offset).collect();
    }
    if let Some(limit) = body.limit {
        rows.truncate(limit);
    }
    engine.close().map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;

    Ok(Json(GetVectorsResponse {
        ids: include_set.contains("ids").then(|| rows.iter().map(|r| r.id.clone()).collect()),
        metadatas: include_set
            .contains("metadatas")
            .then(|| rows.iter().map(|r| r.metadata.clone()).collect()),
        documents: include_set
            .contains("documents")
            .then(|| rows.iter().map(|r| r.document.clone()).collect()),
    }))
}

async fn query_vectors(State(state): State<AppState>, Path((tenant, database, collection)): Path<(String, String, String)>, Extension(ctx): Extension<RequestContext>, Json(body): Json<QueryVectorsRequest>) -> Result<Json<QueryVectorsResponse>, ApiError> {
    authorize(&ctx, &state.auth, "read", &tenant, &database, Some(&collection))?;
    if body.top_k == 0 {
        return Err(ApiError::invalid_argument("top_k must be greater than 0", ctx.request_id.clone()));
    }
    if body.query_embeddings.is_empty() {
        return Err(ApiError::invalid_argument("query_embeddings cannot be empty", ctx.request_id.clone()));
    }

    let include_set = parse_include_set(
        body.include.as_ref(),
        &["ids", "scores", "metadatas", "documents"],
        &["ids", "scores", "metadatas", "documents"],
        ctx.request_id.clone(),
    )?;
    let offset = body.offset.unwrap_or(0);
    let candidate_n = body.top_k.saturating_add(offset);

    let mut engine = open_scoped_engine_from_manifest(&state, &tenant, &database, &collection)
        .map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;
    let mut rows = Vec::new();
    for q in &body.query_embeddings {
        if q.len() != engine.d {
            engine.close().map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;
            return Err(ApiError::invalid_argument(format!("query vector dimension mismatch: expected {}, got {}", engine.d, q.len()), ctx.request_id.clone()));
        }
        let query = Array1::from(q.clone());
        let mut hits = engine
            .search_with_filter(&query, candidate_n, body.filter.as_ref())
            .map_err(|e| map_engine_err(e, ctx.request_id.clone()))?
            .into_iter()
            .skip(offset)
            .collect::<Vec<_>>();
        hits.truncate(body.top_k);

        rows.push(QueryRow {
            ids: include_set.contains("ids").then(|| hits.iter().map(|h| h.id.clone()).collect()),
            scores: include_set.contains("scores").then(|| hits.iter().map(|h| h.score).collect()),
            metadatas: include_set
                .contains("metadatas")
                .then(|| hits.iter().map(|h| h.metadata.clone()).collect()),
            documents: include_set
                .contains("documents")
                .then(|| hits.iter().map(|h| h.document.clone()).collect()),
        });
    }
    engine.close().map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;
    Ok(Json(QueryVectorsResponse { results: rows }))
}
async fn start_compact_job(State(state): State<AppState>, Path((tenant, database, collection)): Path<(String, String, String)>, Extension(ctx): Extension<RequestContext>, Json(body): Json<CompactRequest>) -> Result<(StatusCode, Json<JobEnqueueResponse>), ApiError> {
    authorize(&ctx, &state.auth, "write", &tenant, &database, Some(&collection))?;
    enforce_job_enqueue_quota(&state, &tenant, &database, Some(&collection), ctx.request_id.clone())?;
    let _ = body.r#async.unwrap_or(true);
    let (job_id, status) = enqueue_job(&state, JobType::Compact, tenant, database, collection, None, ctx.request_id.clone())?;
    Ok((StatusCode::ACCEPTED, Json(JobEnqueueResponse { job_id, status })))
}

async fn start_index_job(State(state): State<AppState>, Path((tenant, database, collection)): Path<(String, String, String)>, Extension(ctx): Extension<RequestContext>, Json(body): Json<IndexRequest>) -> Result<(StatusCode, Json<JobEnqueueResponse>), ApiError> {
    authorize(&ctx, &state.auth, "write", &tenant, &database, Some(&collection))?;
    enforce_job_enqueue_quota(&state, &tenant, &database, Some(&collection), ctx.request_id.clone())?;
    let _ = body.r#async.unwrap_or(true);
    let (job_id, status) = enqueue_job(&state, JobType::IndexBuild, tenant, database, collection, None, ctx.request_id.clone())?;
    Ok((StatusCode::ACCEPTED, Json(JobEnqueueResponse { job_id, status })))
}

async fn start_snapshot_job(State(state): State<AppState>, Path((tenant, database, collection)): Path<(String, String, String)>, Extension(ctx): Extension<RequestContext>, Json(body): Json<SnapshotRequest>) -> Result<(StatusCode, Json<JobEnqueueResponse>), ApiError> {
    authorize(&ctx, &state.auth, "write", &tenant, &database, Some(&collection))?;
    enforce_job_enqueue_quota(&state, &tenant, &database, Some(&collection), ctx.request_id.clone())?;
    let _ = body.r#async.unwrap_or(true);
    let (job_id, status) = enqueue_job(&state, JobType::Snapshot, tenant, database, collection, body.snapshot_name, ctx.request_id.clone())?;
    Ok((StatusCode::ACCEPTED, Json(JobEnqueueResponse { job_id, status })))
}

async fn get_job_status(State(state): State<AppState>, Path(job_id): Path<String>, Extension(ctx): Extension<RequestContext>) -> Result<Json<JobStatusResponse>, ApiError> {
    let job = state.jobs.lock().map_err(|_| ApiError::internal("job store lock poisoned", ctx.request_id.clone()))?.jobs.get(&job_id).cloned()
        .ok_or_else(|| ApiError::not_found(format!("job '{}' not found", job_id), ctx.request_id.clone()))?;
    authorize(&ctx, &state.auth, "read", &job.tenant, &job.database, Some(&job.collection))?;
    Ok(Json(JobStatusResponse { job }))
}

async fn list_collection_jobs(State(state): State<AppState>, Path((tenant, database, collection)): Path<(String, String, String)>, Extension(ctx): Extension<RequestContext>) -> Result<Json<ListJobsResponse>, ApiError> {
    authorize(&ctx, &state.auth, "read", &tenant, &database, Some(&collection))?;
    let jobs = state.jobs.lock().map_err(|_| ApiError::internal("job store lock poisoned", ctx.request_id.clone()))?
        .jobs.values().filter(|j| j.tenant == tenant && j.database == database && j.collection == collection).cloned().collect();
    Ok(Json(ListJobsResponse { jobs }))
}


async fn cancel_job(State(state): State<AppState>, Path(job_id): Path<String>, Extension(ctx): Extension<RequestContext>) -> Result<Json<JobStatusResponse>, ApiError> {
    let mut guard = state.jobs.lock().map_err(|_| ApiError::internal("job store lock poisoned", ctx.request_id.clone()))?;
    let job = guard.jobs.get_mut(&job_id)
        .ok_or_else(|| ApiError::not_found(format!("job '{}' not found", job_id), ctx.request_id.clone()))?;
    authorize(&ctx, &state.auth, "write", &job.tenant, &job.database, Some(&job.collection))?;

    if matches!(job.status, JobStatus::Queued) {
        job.status = JobStatus::Canceled;
        job.completed_at = Some(now_ts());
        job.error = Some("canceled by user".to_string());
        let out = job.clone();
        save_job_store(&state.storage.job_store_path, &guard).map_err(|e| ApiError::internal(e.to_string(), ctx.request_id.clone()))?;
        return Ok(Json(JobStatusResponse { job: out }));
    }
    if matches!(job.status, JobStatus::Running) {
        return Err(ApiError::conflict("running jobs are not cancelable in this phase".to_string(), ctx.request_id.clone()));
    }
    Err(ApiError::conflict(format!("job '{}' is already terminal", job_id), ctx.request_id.clone()))
}

async fn retry_job(State(state): State<AppState>, Path(job_id): Path<String>, Extension(ctx): Extension<RequestContext>) -> Result<Json<JobStatusResponse>, ApiError> {
    {
        let mut guard = state.jobs.lock().map_err(|_| ApiError::internal("job store lock poisoned", ctx.request_id.clone()))?;
        let job = guard.jobs.get_mut(&job_id)
            .ok_or_else(|| ApiError::not_found(format!("job '{}' not found", job_id), ctx.request_id.clone()))?;
        authorize(&ctx, &state.auth, "write", &job.tenant, &job.database, Some(&job.collection))?;

        if !(matches!(job.status, JobStatus::Failed) || matches!(job.status, JobStatus::Canceled)) {
            return Err(ApiError::conflict(format!("job '{}' is not retryable unless failed/canceled", job_id), ctx.request_id.clone()));
        }
        if job.attempts >= job.max_attempts {
            return Err(ApiError::conflict(format!("job '{}' retry budget exhausted", job_id), ctx.request_id.clone()));
        }
        job.status = JobStatus::Queued;
        job.started_at = None;
        job.completed_at = None;
        job.error = None;
        save_job_store(&state.storage.job_store_path, &guard).map_err(|e| ApiError::internal(e.to_string(), ctx.request_id.clone()))?;
    }

    dispatch_queued_jobs(&state);

    let job = state.jobs.lock().map_err(|_| ApiError::internal("job store lock poisoned", ctx.request_id.clone()))?
        .jobs.get(&job_id).cloned()
        .ok_or_else(|| ApiError::not_found(format!("job '{}' not found", job_id), ctx.request_id.clone()))?;
    Ok(Json(JobStatusResponse { job }))
}
fn dispatch_queued_jobs(state: &AppState) {
    let mut to_spawn = Vec::new();
    if let Ok(mut guard) = state.jobs.lock() {
        let running = guard.jobs.values().filter(|j| matches!(j.status, JobStatus::Running)).count();
        let slots = state.job_worker_concurrency.saturating_sub(running);
        if slots == 0 {
            return;
        }

        let mut queued_ids = guard
            .jobs
            .values()
            .filter(|j| matches!(j.status, JobStatus::Queued))
            .map(|j| j.job_id.clone())
            .collect::<Vec<_>>();
        queued_ids.sort();

        for job_id in queued_ids.into_iter().take(slots) {
            if let Some(j) = guard.jobs.get_mut(&job_id) {
                if j.attempts >= j.max_attempts {
                    j.status = JobStatus::Failed;
                    j.completed_at = Some(now_ts());
                    j.error = Some(format!("retry budget exhausted before dispatch (attempts={}, max_attempts={})", j.attempts, j.max_attempts));
                    continue;
                }
                j.status = JobStatus::Running;
                j.attempts = j.attempts.saturating_add(1);
                j.started_at = Some(now_ts());
                j.error = None;
                to_spawn.push(job_id.clone());
            }
        }
        let _ = save_job_store(&state.storage.job_store_path, &guard);
    }

    for job_id in to_spawn {
        let state_clone = state.clone();
        tokio::spawn(async move { run_job_lifecycle(state_clone, job_id); });
    }
}

fn enqueue_job(
    state: &AppState,
    job_type: JobType,
    tenant: String,
    database: String,
    collection: String,
    snapshot_name: Option<String>,
    request_id: Option<String>,
) -> Result<(String, JobStatus), ApiError> {
    let mut guard = state.jobs.lock().map_err(|_| ApiError::internal("job store lock poisoned", request_id.clone()))?;
    let job_id = format!("job_{:016}", guard.next_id);
    guard.next_id += 1;
    guard.jobs.insert(job_id.clone(), JobRecord {
        job_id: job_id.clone(),
        job_type,
        status: JobStatus::Queued,
        tenant,
        database,
        collection,
        snapshot_name,
        created_at: now_ts(),
        started_at: None,
        completed_at: None,
        error: None,
        attempts: 0,
        max_attempts: default_job_max_attempts(),
    });
    save_job_store(&state.storage.job_store_path, &guard).map_err(|e| ApiError::internal(e.to_string(), request_id.clone()))?;

    dispatch_queued_jobs(state);

    Ok((job_id, JobStatus::Queued))
}

fn run_job_lifecycle(state: AppState, job_id: String) {
    let job_for_execution = match state.jobs.lock() {
        Ok(guard) => guard.jobs.get(&job_id).cloned(),
        Err(_) => None,
    };
    let Some(job_for_execution) = job_for_execution else { return; };

    let op_result = execute_job_operation(&state, &job_for_execution);

    if let Ok(mut guard) = state.jobs.lock() {
        if let Some(j) = guard.jobs.get_mut(&job_id) {
            if matches!(j.status, JobStatus::Canceled) {
                let _ = save_job_store(&state.storage.job_store_path, &guard);
                return;
            }
            j.completed_at = Some(now_ts());
            match op_result {
                Ok(()) => {
                    j.status = JobStatus::Succeeded;
                    j.error = None;
                }
                Err(err) => {
                    j.status = JobStatus::Failed;
                    j.error = Some(err.to_string());
                }
            }
        }
        let _ = save_job_store(&state.storage.job_store_path, &guard);
    }

    dispatch_queued_jobs(&state);
}

fn execute_job_operation(
    state: &AppState,
    job: &JobRecord,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    match job.job_type {
        JobType::Compact => {
            let mut engine = open_scoped_engine_from_manifest(state, &job.tenant, &job.database, &job.collection)?;
            engine.compact()?;
            engine.close()?;
            Ok(())
        }
        JobType::IndexBuild => {
            let mut engine = open_scoped_engine_from_manifest(state, &job.tenant, &job.database, &job.collection)?;
            engine.create_index(32, 64)?;
            engine.close()?;
            Ok(())
        }
        JobType::Snapshot => {
            let collection_dir = scoped_collection_dir(&state.storage.local_root, &job.tenant, &job.database, &job.collection);
            if !collection_dir.exists() {
                return Err(format!(
                    "collection '{}' does not exist in tenant '{}' database '{}'",
                    job.collection, job.tenant, job.database
                )
                .into());
            }
            let snapshot_name = job
                .snapshot_name
                .clone()
                .unwrap_or_else(|| format!("snapshot_{}_{}", job.created_at, job.job_id));
            let snapshot_dir = PathBuf::from(&state.storage.local_root)
                .join("snapshots")
                .join(&job.tenant)
                .join(&job.database)
                .join(&job.collection)
                .join(snapshot_name);
            if let Some(parent) = snapshot_dir.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let collection_dir_s = collection_dir.to_string_lossy().to_string();
            let snapshot_dir_s = snapshot_dir.to_string_lossy().to_string();
            TurboQuantEngine::snapshot_local_dir(&collection_dir_s, &snapshot_dir_s)
        }
    }
}

fn open_scoped_engine_from_manifest(
    state: &AppState,
    tenant: &str,
    database: &str,
    collection: &str,
) -> Result<TurboQuantEngine, Box<dyn std::error::Error + Send + Sync>> {
    #[derive(Deserialize)]
    struct CollectionManifestProbe {
        d: usize,
        b: usize,
        seed: u64,
        metric: DistanceMetric,
    }

    let collection_dir = scoped_collection_dir(&state.storage.local_root, tenant, database, collection);
    let manifest_path = collection_dir.join("manifest.json");
    let manifest = serde_json::from_str::<CollectionManifestProbe>(&std::fs::read_to_string(&manifest_path)?)?;
    TurboQuantEngine::open_collection_scoped(
        &state.storage.uri,
        &state.storage.local_root,
        tenant,
        database,
        collection,
        manifest.d,
        manifest.b,
        manifest.seed,
        manifest.metric,
    )
}

fn authorize(ctx: &RequestContext, store: &AuthStore, action: &str, tenant: &str, database: &str, collection: Option<&str>) -> Result<(), ApiError> {
    if !(ctx.scopes.contains(action) || ctx.scopes.contains("admin")) {
        return Err(ApiError::forbidden(format!("subject '{}' does not have required {} scope", ctx.subject, action), ctx.request_id.clone()));
    }
    if let Some(bound) = &ctx.tenant_id {
        if bound != tenant {
            return Err(ApiError::forbidden(format!("subject '{}' is bound to tenant '{}' and cannot access tenant '{}'", ctx.subject, bound, tenant), ctx.request_id.clone()));
        }
    }
    let allowed = store.role_bindings.iter().any(|b| {
        b.subject == ctx.subject
            && (b.actions.contains(action) || b.actions.contains("admin"))
            && b.tenant.as_deref().map(|v| v == tenant).unwrap_or(true)
            && b.database.as_deref().map(|v| v == database).unwrap_or(true)
            && match (b.collection.as_deref(), collection) { (Some(x), Some(y)) => x == y, (Some(_), None) => false, (None, _) => true }
    });
    if !allowed {
        return Err(ApiError::forbidden(format!("subject '{}' has no {} role binding for tenant='{}' database='{}'", ctx.subject, action, tenant, database), ctx.request_id.clone()));
    }
    Ok(())
}

fn enforce_create_collection_quota(quotas: &QuotaStore, local_root: &str, tenant: &str, database: &str, request_id: Option<String>) -> Result<(), ApiError> {
    let key = format!("{tenant}/{database}");
    let limit = quotas.database_quotas.get(&key).and_then(|q| q.max_collections).or_else(|| quotas.tenant_quotas.get(tenant).and_then(|q| q.max_collections));
    if let Some(max_collections) = limit {
        let current = TurboQuantEngine::list_collections_scoped(local_root, tenant, database)
            .map_err(|e| map_engine_err(e, request_id.clone()))?
            .len();
        if current >= max_collections {
            return Err(ApiError::quota_exceeded(format!("collection quota exceeded for tenant='{}' database='{}' (current={}, limit={})", tenant, database, current, max_collections), request_id));
        }
    }
    Ok(())
}

fn effective_max_vectors(quotas: &QuotaStore, tenant: &str, database: &str) -> Option<u64> {
    let key = format!("{tenant}/{database}");
    quotas
        .database_quotas
        .get(&key)
        .and_then(|q| q.max_vectors)
        .or_else(|| quotas.tenant_quotas.get(tenant).and_then(|q| q.max_vectors))
}

fn effective_max_concurrent_jobs(quotas: &QuotaStore, tenant: &str, database: &str) -> Option<usize> {
    let key = format!("{tenant}/{database}");
    quotas
        .database_quotas
        .get(&key)
        .and_then(|q| q.max_concurrent_jobs)
        .or_else(|| quotas.tenant_quotas.get(tenant).and_then(|q| q.max_concurrent_jobs))
}

fn enforce_vector_quota_for_ids(
    quotas: &QuotaStore,
    tenant: &str,
    database: &str,
    engine: &TurboQuantEngine,
    ids: &[String],
    request_id: Option<String>,
) -> Result<(), ApiError> {
    let Some(limit) = effective_max_vectors(quotas, tenant, database) else {
        return Ok(());
    };

    let current = engine.vector_count();
    let mut seen = HashSet::new();
    let mut growth = 0u64;
    for id in ids {
        if !seen.insert(id.clone()) {
            continue;
        }
        if engine.get(id).map_err(|e| map_engine_err(e, request_id.clone()))?.is_none() {
            growth = growth.saturating_add(1);
        }
    }

    let projected = current.saturating_add(growth);
    if projected > limit {
        return Err(ApiError::quota_exceeded(
            format!(
                "vector quota exceeded for tenant='{}' database='{}' (current={}, projected={}, limit={})",
                tenant, database, current, projected, limit
            ),
            request_id,
        ));
    }
    Ok(())
}

async fn get_quota_usage(
    State(state): State<AppState>,
    Path((tenant, database)): Path<(String, String)>,
    Extension(ctx): Extension<RequestContext>,
) -> Result<Json<QuotaUsageResponse>, ApiError> {
    authorize(&ctx, &state.auth, "read", &tenant, &database, None)?;

    let collection_names = TurboQuantEngine::list_collections_scoped(&state.storage.local_root, &tenant, &database)
        .map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;
    let mut current_vectors = 0u64;
    let mut current_disk_bytes = 0u64;

    for collection in &collection_names {
        let mut engine = open_scoped_engine_from_manifest(&state, &tenant, &database, collection)
            .map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;
        current_vectors = current_vectors.saturating_add(engine.vector_count());
        current_disk_bytes = current_disk_bytes.saturating_add(engine.stats().total_disk_bytes);
        engine.close().map_err(|e| map_engine_err(e, ctx.request_id.clone()))?;
    }

    let (queued_jobs, running_jobs) =
        count_jobs_for_scope(&state, &tenant, &database, None, ctx.request_id.clone())?;
    let limits = effective_quota_limits(&state.quotas, &tenant, &database);

    Ok(Json(QuotaUsageResponse {
        tenant,
        database,
        max_collections: limits.max_collections,
        max_vectors: limits.max_vectors,
        max_disk_bytes: limits.max_disk_bytes,
        max_concurrent_jobs: limits.max_concurrent_jobs,
        current_collections: collection_names.len(),
        current_vectors,
        current_disk_bytes,
        queued_jobs,
        running_jobs,
    }))
}

#[derive(Clone, Copy)]
struct QuotaLimits {
    max_collections: Option<usize>,
    max_vectors: Option<u64>,
    max_disk_bytes: Option<u64>,
    max_concurrent_jobs: Option<usize>,
}

fn effective_quota_limits(quotas: &QuotaStore, tenant: &str, database: &str) -> QuotaLimits {
    let key = format!("{tenant}/{database}");
    let db_quota = quotas.database_quotas.get(&key);
    let tenant_quota = quotas.tenant_quotas.get(tenant);
    QuotaLimits {
        max_collections: db_quota
            .and_then(|q| q.max_collections)
            .or_else(|| tenant_quota.and_then(|q| q.max_collections)),
        max_vectors: db_quota
            .and_then(|q| q.max_vectors)
            .or_else(|| tenant_quota.and_then(|q| q.max_vectors)),
        max_disk_bytes: db_quota
            .and_then(|q| q.max_disk_bytes)
            .or_else(|| tenant_quota.and_then(|q| q.max_disk_bytes)),
        max_concurrent_jobs: db_quota
            .and_then(|q| q.max_concurrent_jobs)
            .or_else(|| tenant_quota.and_then(|q| q.max_concurrent_jobs)),
    }
}

fn count_jobs_for_scope(
    state: &AppState,
    tenant: &str,
    database: &str,
    collection: Option<&str>,
    request_id: Option<String>,
) -> Result<(usize, usize), ApiError> {
    let guard = state
        .jobs
        .lock()
        .map_err(|_| ApiError::internal("job store lock poisoned", request_id))?;

    let mut queued = 0usize;
    let mut running = 0usize;
    for job in guard.jobs.values() {
        if job.tenant != tenant || job.database != database {
            continue;
        }
        if let Some(c) = collection {
            if job.collection != c {
                continue;
            }
        }
        if matches!(job.status, JobStatus::Queued) {
            queued = queued.saturating_add(1);
        } else if matches!(job.status, JobStatus::Running) {
            running = running.saturating_add(1);
        }
    }
    Ok((queued, running))
}

fn enforce_job_enqueue_quota(
    state: &AppState,
    tenant: &str,
    database: &str,
    collection: Option<&str>,
    request_id: Option<String>,
) -> Result<(), ApiError> {
    let Some(limit) = effective_max_concurrent_jobs(&state.quotas, tenant, database) else {
        return Ok(());
    };

    let (queued, running) =
        count_jobs_for_scope(state, tenant, database, collection, request_id.clone())?;
    let active = queued.saturating_add(running);
    if active >= limit {
        return Err(ApiError::quota_exceeded(
            format!(
                "job quota exceeded for tenant='{}' database='{}' (active={}, limit={})",
                tenant, database, active, limit
            ),
            request_id,
        ));
    }
    Ok(())
}
fn parse_metric(metric: Option<&str>, request_id: Option<String>) -> Result<DistanceMetric, ApiError> {
    match metric.unwrap_or("ip").to_ascii_lowercase().as_str() {
        "ip" => Ok(DistanceMetric::Ip),
        "cosine" => Ok(DistanceMetric::Cosine),
        "l2" => Ok(DistanceMetric::L2),
        other => Err(ApiError::invalid_argument(format!("unsupported metric '{}'; expected one of: ip, cosine, l2", other), request_id)),
    }
}

fn metric_to_str(metric: &DistanceMetric) -> &'static str {
    match metric { DistanceMetric::Ip => "ip", DistanceMetric::Cosine => "cosine", DistanceMetric::L2 => "l2" }
}

fn map_engine_err(err: Box<dyn std::error::Error + Send + Sync>, request_id: Option<String>) -> ApiError {
    let msg = err.to_string();
    let lower = msg.to_ascii_lowercase();
    if lower.contains("does not exist") || lower.contains("not found") { return ApiError::not_found(msg, request_id); }
    if lower.contains("already exists") || lower.contains("currently open") { return ApiError::conflict(msg, request_id); }
    if lower.contains("invalid") || lower.contains("must be") || lower.contains("cannot contain") || lower.contains("schema mismatch") || lower.contains("metric mismatch") {
        return ApiError::invalid_argument(msg, request_id);
    }
    error!(error = %msg, "storage operation failed");
    ApiError::internal(msg, request_id)
}

fn scoped_collection_dir(local_root: &str, tenant: &str, database: &str, collection: &str) -> PathBuf {
    PathBuf::from(local_root).join("tenants").join(tenant).join("databases").join(database).join("collections").join(collection)
}






























































#[cfg(test)]
mod tests {
    use super::*;

    fn mk_state_with_jobs(max_concurrent_jobs: Option<usize>, jobs: Vec<JobRecord>) -> AppState {
        let mut tenant_quotas = HashMap::new();
        tenant_quotas.insert(
            "dev".to_string(),
            TenantQuota {
                tenant: "dev".to_string(),
                max_collections: None,
                max_vectors: None,
                max_disk_bytes: None,
                max_concurrent_jobs,
            },
        );

        let mut job_map = HashMap::new();
        for j in jobs {
            job_map.insert(j.job_id.clone(), j);
        }

        AppState {
            auth: Arc::new(AuthStore {
                api_key_subjects: HashMap::new(),
                principals: HashMap::new(),
                role_bindings: vec![],
            }),
            quotas: Arc::new(QuotaStore {
                tenant_quotas,
                database_quotas: HashMap::new(),
            }),
            jobs: Arc::new(Mutex::new(JobStore { jobs: job_map, next_id: 1 })),
            storage: StorageConfig {
                uri: "".to_string(),
                local_root: "".to_string(),
                auth_store_path: "".to_string(),
                quota_store_path: "".to_string(),
                job_store_path: "".to_string(),
            },
            job_worker_concurrency: 1,
        }
    }

    fn mk_job(id: &str, status: JobStatus) -> JobRecord {
        JobRecord {
            job_id: id.to_string(),
            job_type: JobType::Compact,
            status,
            tenant: "dev".to_string(),
            database: "db".to_string(),
            collection: "c1".to_string(),
            snapshot_name: None,
            created_at: "0".to_string(),
            started_at: None,
            completed_at: None,
            error: None,
            attempts: 0,
            max_attempts: default_job_max_attempts(),
        }
    }

    #[test]
    fn effective_quota_limits_database_overrides_tenant() {
        let mut tenant_quotas = HashMap::new();
        tenant_quotas.insert(
            "dev".to_string(),
            TenantQuota {
                tenant: "dev".to_string(),
                max_collections: Some(10),
                max_vectors: Some(1_000),
                max_disk_bytes: Some(10_000),
                max_concurrent_jobs: Some(4),
            },
        );

        let mut database_quotas = HashMap::new();
        database_quotas.insert(
            "dev/db".to_string(),
            DatabaseQuota {
                tenant: "dev".to_string(),
                database: "db".to_string(),
                max_collections: Some(3),
                max_vectors: None,
                max_disk_bytes: Some(7_000),
                max_concurrent_jobs: Some(2),
            },
        );

        let limits = effective_quota_limits(
            &QuotaStore {
                tenant_quotas,
                database_quotas,
            },
            "dev",
            "db",
        );

        assert_eq!(limits.max_collections, Some(3));
        assert_eq!(limits.max_vectors, Some(1_000));
        assert_eq!(limits.max_disk_bytes, Some(7_000));
        assert_eq!(limits.max_concurrent_jobs, Some(2));
    }

    #[test]
    fn count_jobs_for_scope_tracks_queued_and_running() {
        let state = mk_state_with_jobs(
            Some(10),
            vec![
                mk_job("job_1", JobStatus::Queued),
                mk_job("job_2", JobStatus::Running),
                mk_job("job_3", JobStatus::Succeeded),
            ],
        );

        let (queued, running) = match count_jobs_for_scope(&state, "dev", "db", Some("c1"), None) { Ok(v) => v, Err(_) => panic!("count_jobs_for_scope should succeed"), };
        assert_eq!(queued, 1);
        assert_eq!(running, 1);
    }

    #[test]
    fn enforce_job_enqueue_quota_blocks_when_limit_reached() {
        let state = mk_state_with_jobs(
            Some(2),
            vec![
                mk_job("job_1", JobStatus::Queued),
                mk_job("job_2", JobStatus::Running),
            ],
        );

        let err = enforce_job_enqueue_quota(&state, "dev", "db", Some("c1"), None).unwrap_err();
        assert_eq!(err.code, "quota_exceeded");
        assert_eq!(err.status, StatusCode::TOO_MANY_REQUESTS);
    }
}

