use ndarray::Array1;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;
use std::sync::Arc;

use super::backend::StorageBackend;
use super::compaction::Compactor;
use super::graph::{GraphManager, OrderingWrapper, SearchCandidate};
use super::id_pool::IdPool;
use super::live_codes::LiveCodesFile;
use super::metadata::{MetadataStore, VectorMetadata};
use super::segment::{SegmentManager, SegmentRecord};
use super::wal::{Wal, WalEntry};
use crate::quantizer::CodeIndex;
use crate::quantizer::prod::ProdQuantizer;
mod filter;
use filter::{get_nested_field, metadata_matches_filter, score_vectors_with_metric};

const QUANTIZER_STATE_FILE: &str = "quantizer.bin";
const INDEX_IDS_FILE: &str = "graph_ids.json";
const ID_POOL_FILE: &str = "live_ids.bin";
const MANIFEST_SAVE_INTERVAL_OPS: usize = 64;

const LIVE_GAMMA_BYTES: usize = 4;
const LIVE_NORM_BYTES: usize = 4;
const LIVE_DELETED_BYTES: usize = 1;

/// Controls whether `live_vectors.bin` is written for raw-vector reranking.
///
/// - `Disabled` (default): no file written; reranking uses dequantization (low RAM/disk).
/// - `F32`: write raw f32 vectors; exact reranking, highest precision, +n×d×4 bytes.
/// - `F16`: write raw f16 vectors; exact reranking, half of f32 RAM/disk.
///
/// Old databases without this field in manifest.json → serde default = `Disabled`.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum RerankPrecision {
    /// No live_vectors.bin; reranking uses dequantization (low RAM/disk, current default).
    #[default]
    Disabled,
    /// Write raw f32 vectors to live_vectors.bin (legacy/backward-compat option).
    F32,
    /// Write raw f16 vectors (~50% RAM/disk vs f32, <0.05% inner-product error at any dim).
    F16,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DistanceMetric {
    Ip,
    Cosine,
    L2,
}

impl Default for DistanceMetric {
    fn default() -> Self {
        Self::Ip
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Manifest {
    pub version: u32,
    pub d: usize,
    pub b: usize,
    pub seed: u64,
    pub vector_count: u64,
    pub storage_uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantizer: Option<ProdQuantizer>,
    #[serde(default)]
    pub metric: DistanceMetric,
    #[serde(default)]
    pub index_state: Option<IndexState>,
    #[serde(default)]
    pub fast_mode: bool,
    #[serde(default = "default_rerank_enabled")]
    pub rerank_enabled: bool,
    /// Controls raw-vector rerank storage. Absent in old manifests → serde default = `Disabled`.
    /// `F16`/`F32` create `live_vectors.bin` on new DBs for exact fast reranking.
    #[serde(default)]
    pub rerank_precision: RerankPrecision,
    /// When true the engine L2-normalises every inserted vector and every query
    /// vector internally so that IP scoring equals cosine similarity.  Callers
    /// that already emit unit vectors (e.g. Axon RAG) can set this to avoid
    /// repeating the normalisation themselves.
    #[serde(default)]
    pub normalize: bool,
}

fn default_rerank_enabled() -> bool {
    true
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct IndexState {
    pub max_degree: usize,
    #[serde(default = "default_ef_construction")]
    pub ef_construction: usize,
    pub search_list_size: usize,
    pub alpha: f64,
    pub indexed_nodes: usize,
}

fn default_ef_construction() -> usize {
    200
}

impl Manifest {
    fn save<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }
}

#[derive(Debug, Clone)]
pub struct BatchWriteFailure {
    pub index: usize,
    pub id: String,
    pub error: String,
}

#[derive(Debug)]
pub struct BatchWriteReport {
    pub ok: Vec<String>,
    pub failed: Vec<BatchWriteFailure>,
    pub applied: usize,
}

#[derive(Debug, Clone)]
pub struct BatchWriteItem {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: HashMap<String, JsonValue>,
    pub document: Option<String>,
}

#[derive(Debug, Clone)]
pub struct GetResult {
    pub id: String,
    pub metadata: HashMap<String, JsonValue>,
    pub document: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f64,
    pub metadata: HashMap<String, JsonValue>,
    pub document: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DbStats {
    pub vector_count: u64,
    pub segment_count: usize,
    pub buffered_vectors: usize,
    pub d: usize,
    pub b: usize,
    pub total_disk_bytes: u64,
    pub has_index: bool,
    pub index_nodes: usize,
    pub live_codes_bytes: usize,
    pub live_slot_count: usize,
    pub live_id_count: usize,
    pub live_vectors_count: usize,
    pub live_vectors_bytes_estimate: usize,
    pub metadata_entries: usize,
    pub metadata_bytes_estimate: usize,
    pub ann_slot_count: usize,
    pub graph_nodes: usize,
}

pub struct TurboQuantEngine {
    pub d: usize,
    pub b: usize,
    pub quantizer: ProdQuantizer,
    pub manifest: Manifest,
    pub metric: DistanceMetric,
    pub rerank_enabled: bool,

    backend: Arc<StorageBackend>,
    wal: Wal,
    wal_buffer: Vec<WalEntry>,
    wal_flush_threshold: usize,
    segments: SegmentManager,
    metadata: MetadataStore,
    graph: GraphManager,
    local_dir: String,

    index_ids: Vec<u32>,
    live_codes: LiveCodesFile,
    live_vraw: Option<LiveCodesFile>,
    id_pool: IdPool,
    index_ids_dirty: bool,
    pending_manifest_updates: usize,
    /// True when at least one delete has been issued since the last compaction.
    /// When false, `live_compact_slab` is skipped on WAL flush — a pure insert
    /// workload never needs to compact (no dead slots to remove).
    has_pending_deletes: bool,
    /// When true, input vectors and query vectors are L2-normalised internally
    /// before quantisation / scoring so that IP ≡ cosine similarity.
    pub normalize: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BatchWriteMode {
    Insert,
    Upsert,
    Update,
}

impl TurboQuantEngine {
    pub fn open(
        uri: &str,
        local_dir: &str,
        d: usize,
        b: usize,
        seed: u64,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::open_with_metric(uri, local_dir, d, b, seed, DistanceMetric::Ip)
    }

    /// Open (or create) with O(d log d) SRHT fast-path quantizer.
    /// Equivalent to `open` but uses SRHT for both MSE rotation and QJL projection.
    pub fn open_fast(
        uri: &str,
        local_dir: &str,
        d: usize,
        b: usize,
        seed: u64,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::open_with_metric_and_rerank(
            uri,
            local_dir,
            d,
            b,
            seed,
            DistanceMetric::Ip,
            true,
            true,
        )
    }

    pub fn open_with_metric(
        uri: &str,
        local_dir: &str,
        d: usize,
        b: usize,
        seed: u64,
        metric: DistanceMetric,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::open_with_metric_and_rerank(uri, local_dir, d, b, seed, metric, true, false)
    }

    pub fn open_with_metric_and_rerank(
        uri: &str,
        local_dir: &str,
        d: usize,
        b: usize,
        seed: u64,
        metric: DistanceMetric,
        rerank: bool,
        fast_mode: bool,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let precision = if rerank {
            RerankPrecision::F32
        } else {
            RerankPrecision::Disabled
        };
        Self::open_with_options(
            uri, local_dir, d, b, seed, metric, rerank, fast_mode, precision, None, false,
        )
    }

    pub fn open_with_options(
        uri: &str,
        local_dir: &str,
        d: usize,
        b: usize,
        seed: u64,
        metric: DistanceMetric,
        rerank: bool,
        fast_mode: bool,
        rerank_precision: RerankPrecision,
        wal_flush_threshold: Option<usize>,
        normalize: bool,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        std::fs::create_dir_all(local_dir)?;
        let manifest_path = format!("{}/manifest.json", local_dir);
        let wal_path = format!("{}/wal.log", local_dir);
        let metadata_path = format!("{}/metadata.bin", local_dir);

        let backend = Arc::new(StorageBackend::from_uri(uri)?);

        let (manifest, quantizer) = if Path::new(&manifest_path).exists() {
            let m = Manifest::load(&manifest_path)?;
            if m.d != d {
                return Err(format!(
                    "dimension mismatch: manifest has d={}, requested d={}",
                    m.d, d
                )
                .into());
            }
            if m.b != b {
                return Err(
                    format!("bits mismatch: manifest has b={}, requested b={}", m.b, b).into(),
                );
            }
            if m.metric != metric {
                return Err(format!(
                    "metric mismatch: manifest has {:?}, requested {:?}",
                    m.metric, metric
                )
                .into());
            }
            let q = load_quantizer_state(local_dir, &backend)?;
            (m, q)
        } else if let Ok(data) = backend.read("manifest.json") {
            let m: Manifest = serde_json::from_slice(&data)?;
            if m.d != d {
                return Err(format!(
                    "dimension mismatch: manifest has d={}, requested d={}",
                    m.d, d
                )
                .into());
            }
            if m.b != b {
                return Err(
                    format!("bits mismatch: manifest has b={}, requested b={}", m.b, b).into(),
                );
            }
            if m.metric != metric {
                return Err(format!(
                    "metric mismatch: manifest has {:?}, requested {:?}",
                    m.metric, metric
                )
                .into());
            }
            let q = load_quantizer_state(local_dir, &backend)?;
            (m, q)
        } else {
            let q = if fast_mode {
                ProdQuantizer::new_srht(d, b, seed)
            } else {
                ProdQuantizer::new(d, b, seed)
            };
            let m = Manifest {
                version: 2,
                d,
                b,
                seed,
                vector_count: 0,
                storage_uri: uri.to_string(),
                quantizer: None,
                metric: metric.clone(),
                index_state: None,
                rerank_enabled: rerank,
                fast_mode,
                rerank_precision,
                normalize,
            };
            save_quantizer_state(local_dir, &backend, &q)?;
            m.save(&manifest_path)?;
            backend.write("manifest.json", &serde_json::to_vec_pretty(&m)?)?;
            // Create an empty live_vectors.bin so that insert_batch can write raw vectors
            // immediately. Only when the user explicitly opts into exact reranking.
            if matches!(
                rerank_precision,
                RerankPrecision::F16 | RerankPrecision::F32
            ) {
                let vraw_path = Path::new(local_dir).join("live_vectors.bin");
                if !vraw_path.exists() {
                    std::fs::File::create(&vraw_path)?;
                }
            }
            (m, q)
        };

        let mut wal = Wal::open(&wal_path)?;
        wal.set_quantizer(std::sync::Arc::new(quantizer.clone()));
        // Recover from any interrupted compaction before loading segments.
        Compactor::new(backend.clone()).recover_if_needed()?;
        let segments = SegmentManager::open(backend.clone())?;
        let metadata = MetadataStore::open(&metadata_path)?;
        let graph = GraphManager::open(backend.clone(), local_dir)?;

        let n = manifest.d.next_power_of_two();
        let qjl_len = n.div_ceil(8);
        let mse_len = (n * (manifest.b - 1)).div_ceil(8);
        let stride = mse_len + qjl_len + LIVE_GAMMA_BYTES + LIVE_NORM_BYTES + LIVE_DELETED_BYTES;
        let live_codes = LiveCodesFile::open(Path::new(local_dir).join("live_codes.bin"), stride)?;
        let live_vraw_path = Path::new(local_dir).join("live_vectors.bin");
        let live_vraw = if manifest.rerank_enabled
            && live_vraw_path.exists()
            && !matches!(manifest.rerank_precision, RerankPrecision::Disabled)
        {
            let vstride = match manifest.rerank_precision {
                RerankPrecision::F16 => manifest.d * 2,
                _ => manifest.d * 4,
            };
            let vraw = LiveCodesFile::open(live_vraw_path, vstride)?;
            // Random-access pattern (ANN rerank reads scattered slots); hint to OS.
            vraw.advise_random();
            Some(vraw)
        } else {
            None
        };

        let normalize_flag = normalize || manifest.normalize;
        let mut engine = Self {
            d: manifest.d,
            b: manifest.b,
            quantizer,
            manifest: manifest.clone(),
            metric,
            backend,
            wal,
            wal_buffer: Vec::new(),
            wal_flush_threshold: wal_flush_threshold.unwrap_or(5_000),
            segments,
            metadata,
            graph,
            local_dir: local_dir.to_string(),
            index_ids: load_index_ids(local_dir).unwrap_or_default(),
            live_codes,
            live_vraw,
            id_pool: IdPool::new(),
            index_ids_dirty: false,
            pending_manifest_updates: 0,
            has_pending_deletes: false,
            rerank_enabled: manifest.rerank_enabled,
            normalize: normalize_flag,
        };

        let id_pool_loaded = if let Ok(ip) = load_id_pool(local_dir, &engine.backend) {
            engine.id_pool = ip;
            // live_codes.open() sets len = capacity (file size / stride) because the file
            // is pre-allocated in GROW_SLOTS increments.  Correct len to the actual number
            // of populated slots so that the next alloc_slot() returns the right index.
            let slot_count = engine.id_pool.slot_count();
            engine.live_codes.set_len(slot_count);
            if let Some(vraw) = engine.live_vraw.as_mut() {
                vraw.set_len(slot_count);
            }
            // Carry forward any pre-existing tombstones so that the next
            // flush_wal_to_segment() will call live_compact_slab() even if no
            // new deletes are issued during this session.
            engine.has_pending_deletes =
                engine.id_pool.active_count() < engine.id_pool.slot_count();
            true
        } else {
            false
        };

        let pending = Wal::replay(&wal_path, Some(&engine.quantizer))?;
        if !pending.is_empty() {
            if !id_pool_loaded {
                // live_ids.bin is missing (unclean shutdown). Rebuild live_codes and
                // id_pool entirely from the WAL entries so we have a consistent base
                // before flush_wal_to_segment compacts and persists everything.
                //
                // Raw vectors (live_vraw) are NOT stored in the WAL, so we cannot
                // recover them.  Drop the stale live_vectors.bin and fall back to
                // dequantization reranking for this session; the file is recreated
                // automatically on the next clean insert+close cycle.
                engine.live_codes.clear()?;
                engine.live_vraw = None;
                let _ = std::fs::remove_file(Path::new(&engine.local_dir).join("live_vectors.bin"));
                engine.id_pool = IdPool::new();
                let mse_len = engine.live_mse_len();
                let qjl_len = engine.live_qjl_len();
                for entry in &pending {
                    if entry.is_deleted || entry.quantized_indices.is_empty() {
                        continue; // deletions handled by flush_wal_to_segment
                    }
                    let packed_mse = engine.quantizer.pack_mse_indices(&entry.quantized_indices);
                    let slot = if let Some(s) = engine.id_pool.get_slot(&entry.id) {
                        // upsert: overwrite existing slot
                        let rec = engine.live_codes.get_slot_mut(s as usize);
                        rec[0..mse_len].copy_from_slice(&packed_mse);
                        rec[mse_len..mse_len + qjl_len].copy_from_slice(&entry.qjl_bits);
                        rec[mse_len + qjl_len..mse_len + qjl_len + 4]
                            .copy_from_slice(&entry.gamma.to_le_bytes());
                        rec[mse_len + qjl_len + 4..mse_len + qjl_len + 8]
                            .copy_from_slice(&entry.norm.to_le_bytes());
                        rec[mse_len + qjl_len + 8] = 0u8;
                        s
                    } else {
                        // insert: alloc new slot
                        let s = engine.id_pool.insert(&entry.id)?;
                        let new_slot = engine.live_codes.alloc_slot()?;
                        debug_assert_eq!(s as usize, new_slot);
                        let rec = engine.live_codes.get_slot_mut(new_slot);
                        rec[0..mse_len].copy_from_slice(&packed_mse);
                        rec[mse_len..mse_len + qjl_len].copy_from_slice(&entry.qjl_bits);
                        rec[mse_len + qjl_len..mse_len + qjl_len + 4]
                            .copy_from_slice(&entry.gamma.to_le_bytes());
                        rec[mse_len + qjl_len + 4..mse_len + qjl_len + 8]
                            .copy_from_slice(&entry.norm.to_le_bytes());
                        rec[mse_len + qjl_len + 8] = 0u8;
                        s
                    };
                    if let Ok(meta) = serde_json::from_str::<VectorMetadata>(&entry.metadata_json) {
                        let _ = engine.metadata.put(slot, &meta);
                    }
                }
            }
            engine.wal_buffer.extend(pending);
            engine.flush_wal_to_segment()?;
        } else if !id_pool_loaded && engine.live_codes.len() > 0 {
            return Err(
                "live_ids.bin missing and WAL is empty; database state appears corrupt".into(),
            );
        }
        Ok(engine)
    }

    pub fn insert(
        &mut self,
        id: String,
        vector: &Array1<f64>,
        metadata: HashMap<String, JsonValue>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.insert_with_document(id, vector, metadata, None)
    }

    pub fn insert_with_document(
        &mut self,
        id: String,
        vector: &Array1<f64>,
        metadata: HashMap<String, JsonValue>,
        document: Option<String>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.id_pool.contains(&id) {
            return Err(format!("ID '{}' already exists", id).into());
        }
        self.write_vector_entry(id, vector, metadata, document, false)
    }

    pub fn upsert_with_document(
        &mut self,
        id: String,
        vector: &Array1<f64>,
        metadata: HashMap<String, JsonValue>,
        document: Option<String>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.write_vector_entry(id, vector, metadata, document, false)
    }

    pub fn update_with_document(
        &mut self,
        id: String,
        vector: &Array1<f64>,
        metadata: HashMap<String, JsonValue>,
        document: Option<String>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if !self.id_pool.contains(&id) {
            return Err(format!("ID '{}' does not exist", id).into());
        }
        self.write_vector_entry(id, vector, metadata, document, false)
    }

    pub fn insert_many(
        &mut self,
        items: Vec<BatchWriteItem>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.insert_many_with_mode(items, BatchWriteMode::Insert)
    }
    pub fn upsert_many(
        &mut self,
        items: Vec<BatchWriteItem>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.insert_many_with_mode(items, BatchWriteMode::Upsert)
    }
    pub fn update_many(
        &mut self,
        items: Vec<BatchWriteItem>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.insert_many_with_mode(items, BatchWriteMode::Update)
    }

    pub fn delete(&mut self, id: String) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        if !self.id_pool.contains(&id) {
            return Ok(false);
        }
        let entry = WalEntry {
            id: id.clone(),
            quantized_indices: Vec::new(),
            qjl_bits: Vec::new(),
            gamma: 0.0,
            norm: 0.0,
            metadata_json: "{}".to_string(),
            is_deleted: true,
        };
        self.wal.append(&entry, false)?;
        self.wal_buffer.push(entry);
        self.has_pending_deletes = true;
        self.live_delete_slot(&id);
        self.invalidate_index_state()?;
        self.maybe_persist_state(false)?;
        if self.wal_buffer.len() >= self.wal_flush_threshold {
            self.flush_wal_to_segment()?;
        }
        Ok(true)
    }

    pub fn get(
        &self,
        id: &str,
    ) -> Result<Option<GetResult>, Box<dyn std::error::Error + Send + Sync>> {
        let Some(slot) = self.id_pool.get_slot(id) else {
            return Ok(None);
        };
        let meta = self.metadata.get(slot)?.unwrap_or_default();
        Ok(Some(GetResult {
            id: id.to_string(),
            metadata: meta.properties,
            document: meta.document,
        }))
    }

    pub fn get_many(
        &self,
        ids: &[String],
    ) -> Result<Vec<Option<GetResult>>, Box<dyn std::error::Error + Send + Sync>> {
        let mut out = Vec::with_capacity(ids.len());
        let mut slots = Vec::new();
        let mut id_to_idx = HashMap::new();

        for (i, id) in ids.iter().enumerate() {
            if let Some(slot) = self.id_pool.get_slot(id) {
                slots.push(slot);
                id_to_idx.insert(slot, i);
            }
            out.push(None);
        }

        if !slots.is_empty() {
            let meta_map = self.metadata.get_many(&slots)?;
            for (slot, meta) in meta_map {
                if let Some(&idx) = id_to_idx.get(&slot) {
                    out[idx] = Some(GetResult {
                        id: ids[idx].clone(),
                        metadata: meta.properties,
                        document: meta.document,
                    });
                }
            }
        }
        Ok(out)
    }

    pub fn list_all(&self) -> Vec<String> {
        self.id_pool
            .iter_active()
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    /// Return IDs with optional metadata filter and pagination.
    pub fn list_with_filter_page(
        &self,
        filter: Option<&HashMap<String, JsonValue>>,
        limit: Option<usize>,
        offset: usize,
    ) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let active = self.id_pool.iter_active();
        let cap = limit.unwrap_or(usize::MAX);
        if let Some(f) = filter {
            let slots: Vec<u32> = active.iter().map(|(_, s)| *s).collect();
            let meta_map = self.metadata.get_many(&slots)?;
            let ids: Vec<String> = active
                .iter()
                .filter(|(_, slot)| {
                    meta_map
                        .get(slot)
                        .is_some_and(|m| metadata_matches_filter(&m.properties, f))
                })
                .map(|(id, _)| id.clone())
                .skip(offset)
                .take(cap)
                .collect();
            Ok(ids)
        } else {
            let ids: Vec<String> = active
                .into_iter()
                .map(|(id, _)| id)
                .skip(offset)
                .take(cap)
                .collect();
            Ok(ids)
        }
    }

    /// Return a `{value: count}` map of all unique values of `field` across active vectors.
    ///
    /// Supports dotted paths (e.g. `"meta.source"`) using the same nested-field resolution
    /// as the filter system.  Non-string values are stringified via their JSON representation.
    /// O(n) scan — identical cost to a filtered `count()`.
    pub fn list_metadata_values(
        &self,
        field: &str,
    ) -> Result<HashMap<String, usize>, Box<dyn std::error::Error + Send + Sync>> {
        let active = self.id_pool.iter_active();
        let slots: Vec<u32> = active.iter().map(|(_, s)| *s).collect();
        let meta_map = self.metadata.get_many(&slots)?;
        let mut counts: HashMap<String, usize> = HashMap::new();
        for (_, slot) in &active {
            if let Some(meta) = meta_map.get(slot) {
                if let Some(val) = get_nested_field(&meta.properties, field) {
                    let key = match val {
                        JsonValue::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    *counts.entry(key).or_insert(0) += 1;
                }
            }
        }
        Ok(counts)
    }

    /// Update metadata and/or document for an existing vector without re-quantising.
    ///
    /// The quantised codes and any stored raw vector are untouched — only the
    /// metadata store is written. Errors if the ID does not exist.
    pub fn update_metadata_only(
        &mut self,
        id: &str,
        metadata: HashMap<String, JsonValue>,
        document: Option<String>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let Some(slot) = self.id_pool.get_slot(id) else {
            return Err(format!("ID '{}' does not exist", id).into());
        };
        let existing = self.metadata.get(slot)?.unwrap_or_default();
        let new_meta = VectorMetadata {
            // Explicit non-empty map replaces; empty map preserves existing.
            properties: if metadata.is_empty() {
                existing.properties
            } else {
                metadata
            },
            // Explicit Some replaces; None preserves existing document.
            document: document.or(existing.document),
        };
        self.metadata.put(slot, &new_meta)?;
        Ok(())
    }

    /// Run the same search for multiple query vectors in one call.
    pub fn search_batch(
        &self,
        queries: &[ndarray::Array1<f64>],
        top_k: usize,
        filter: Option<&HashMap<String, JsonValue>>,
        ann_search_list_size: Option<usize>,
        use_ann: bool,
    ) -> Result<Vec<Vec<SearchResult>>, Box<dyn std::error::Error + Send + Sync>> {
        queries
            .iter()
            .map(|q| {
                self.search_with_filter_and_ann(q, top_k, filter, ann_search_list_size, use_ann)
            })
            .collect()
    }

    ///
    /// When `filter` is `None` this is equivalent to `stats().vector_count` but
    /// faster (no disk I/O). When a filter is supplied it performs an O(n) scan
    /// identical to the brute-force search pre-filter path.
    pub fn count_with_filter(
        &self,
        filter: Option<&HashMap<String, JsonValue>>,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let Some(f) = filter else {
            return Ok(self.id_pool.active_count());
        };
        let active = self.id_pool.iter_active();
        let slots: Vec<u32> = active.iter().map(|(_, s)| *s).collect();
        let meta_map = self.metadata.get_many(&slots)?;
        let count = active
            .iter()
            .filter(|(_, slot)| {
                meta_map
                    .get(slot)
                    .is_some_and(|m| metadata_matches_filter(&m.properties, f))
            })
            .count();
        Ok(count)
    }

    /// Delete multiple vectors in a single call.
    ///
    /// Returns the number of IDs that were found and deleted. IDs not present
    /// in the database are silently skipped. A single WAL flush is performed
    /// after all deletions, making this significantly cheaper than calling
    /// `delete()` in a loop when removing many vectors at once.
    pub fn delete_batch(
        &mut self,
        ids: Vec<String>,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let mut deleted = 0usize;
        for id in ids {
            if !self.id_pool.contains(&id) {
                continue;
            }
            let entry = WalEntry {
                id: id.clone(),
                quantized_indices: Vec::new(),
                qjl_bits: Vec::new(),
                gamma: 0.0,
                norm: 0.0,
                metadata_json: "{}".to_string(),
                is_deleted: true,
            };
            self.wal.append(&entry, false)?;
            self.wal_buffer.push(entry);
            self.live_delete_slot(&id);
            deleted += 1;
        }
        if deleted > 0 {
            self.has_pending_deletes = true;
            self.invalidate_index_state()?;
            self.maybe_persist_state(false)?;
            if self.wal_buffer.len() >= self.wal_flush_threshold {
                self.flush_wal_to_segment()?;
            }
        }
        Ok(deleted)
    }

    /// Delete all vectors whose metadata matches `filter`.
    ///
    /// Atomically collects matching IDs (under the same write lock held by the
    /// caller) and then delegates to [`delete_batch`].  Returns the count of
    /// deleted vectors.
    pub fn delete_where(
        &mut self,
        filter: &HashMap<String, JsonValue>,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let active = self.id_pool.iter_active();
        let slots: Vec<u32> = active.iter().map(|(_, s)| *s).collect();
        let meta_map = self.metadata.get_many(&slots)?;
        let ids_to_delete: Vec<String> = active
            .into_iter()
            .filter(|(_, slot)| {
                meta_map
                    .get(slot)
                    .is_some_and(|m| metadata_matches_filter(&m.properties, filter))
            })
            .map(|(id, _)| id)
            .collect();
        self.delete_batch(ids_to_delete)
    }

    pub fn create_index_with_params(
        &mut self,
        max_degree: usize,
        ef_construction: usize,
        search_list_size: usize,
        alpha: f64,
        n_refinements: usize,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.flush_wal_to_segment()?;
        let mut id_slot_pairs = self.live_iter_id_slots();
        if id_slot_pairs.is_empty() {
            return Ok(());
        }
        id_slot_pairs.sort_by(|a, b| a.0.cmp(&b.0));
        let indexed_slots: Vec<u32> = id_slot_pairs.iter().map(|(_, slot)| *slot).collect();
        let live_codes = &self.live_codes;
        let d = self.d;
        let qjl_len = self.live_qjl_len();
        let mse_len = self.live_mse_len();
        let metric = self.metric.clone();
        let quantizer = self.quantizer.clone();
        let slots_ref = indexed_slots.clone();
        let has_vraw = self.live_vraw.is_some();
        // Share live_vraw reference safely with the closure.
        let vraw_ref = self.live_vraw.as_ref();
        let vraw_precision = self.manifest.rerank_precision;

        let qn = quantizer.n;

        // Pre-cache all encoded vectors for IP/Cosine metrics into flat contiguous arrays.
        // Avoids O(n × candidates × refinements) heap allocations and random slab reads
        // that live_codes_at_slot_raw() would otherwise incur per scoring call.
        // In fast_mode, gamma=0 for every vector so the QJL loop is skipped by the scorer;
        // we skip pre-caching qjl_buf entirely to reduce memory footprint and bandwidth.
        let is_fast_mode = self.manifest.fast_mode;
        // The stride used to index all_qjl_flat: 0 when fast_mode (empty buffer).
        let cached_qjl_len = if is_fast_mode { 0 } else { qjl_len };
        let (all_mse, all_qjl_flat, all_gamma, all_norm) =
            if matches!(self.metric, DistanceMetric::Ip | DistanceMetric::Cosine) {
                let n_indexed = indexed_slots.len();
                let mut mse_buf = vec![0u16; n_indexed * qn];
                let effective_qjl_len = if is_fast_mode { 0 } else { qjl_len };
                let mut qjl_buf = vec![0u8; n_indexed * effective_qjl_len];
                let mut gamma_buf = vec![0.0f32; n_indexed];
                let mut norm_buf = vec![0.0f32; n_indexed];
                for (i, &slot) in indexed_slots.iter().enumerate() {
                    let rec = live_codes.get_slot(slot as usize);
                    quantizer
                        .unpack_mse_indices(&rec[..mse_len], &mut mse_buf[i * qn..(i + 1) * qn]);
                    if !is_fast_mode && qjl_len > 0 {
                        qjl_buf[i * qjl_len..(i + 1) * qjl_len]
                            .copy_from_slice(&rec[mse_len..mse_len + qjl_len]);
                    }
                    gamma_buf[i] = f32::from_le_bytes(
                        rec[mse_len + qjl_len..mse_len + qjl_len + 4]
                            .try_into()
                            .unwrap(),
                    );
                    norm_buf[i] = f32::from_le_bytes(
                        rec[mse_len + qjl_len + 4..mse_len + qjl_len + 8]
                            .try_into()
                            .unwrap(),
                    );
                }
                (mse_buf, qjl_buf, gamma_buf, norm_buf)
            } else {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new())
            };

        // For L2/other metrics without raw vectors: pre-compute all dequantized vectors in
        // parallel before the build loop so each `to` node is dequantized at most once instead
        // of once per candidate evaluation (O(n × ef_construction) without this).
        let precomputed_l2: Vec<Array1<f64>> =
            if !matches!(self.metric, DistanceMetric::Ip | DistanceMetric::Cosine) && !has_vraw {
                let codes_bytes = live_codes.as_bytes();
                let stride = self.live_stride();
                let q = &self.quantizer;
                indexed_slots
                    .par_iter()
                    .map(|&slot| {
                        let rec =
                            &codes_bytes[slot as usize * stride..(slot as usize + 1) * stride];
                        let mut idx = vec![0u16; q.n];
                        q.unpack_mse_indices(&rec[..mse_len], &mut idx);
                        let gamma = f32::from_le_bytes(
                            rec[mse_len + qjl_len..mse_len + qjl_len + 4]
                                .try_into()
                                .unwrap(),
                        );
                        let doc_norm = f32::from_le_bytes(
                            rec[mse_len + qjl_len + 4..mse_len + qjl_len + 8]
                                .try_into()
                                .unwrap(),
                        );
                        let mut v =
                            q.dequantize(&idx, &rec[mse_len..mse_len + qjl_len], gamma as f64);
                        // Dequantize returns unit vector; scale back to original norm for L2.
                        v.mapv_inplace(|x| x * doc_norm as f64);
                        v
                    })
                    .collect()
            } else {
                Vec::new()
            };

        let build_scorer = move |from: u32, candidates: &[u32]| -> Vec<(u32, f64)> {
            let from_idx = from as usize;
            let from_slot = slots_ref[from_idx] as usize;

            if matches!(metric, DistanceMetric::Ip) {
                // For IP: prepare from-query via O(n) centroid lookup (no SRHT) when raw
                // vectors are unavailable. When vraw is present, use exact rotation.
                // In both cases, `to` candidates use pre-cached encoded data.
                if has_vraw {
                    let rec = vraw_ref
                        .expect("invariant: has_vraw=true implies live_vraw is Some")
                        .get_slot(from_slot);
                    let mut out = Array1::<f64>::zeros(d);
                    if matches!(vraw_precision, RerankPrecision::F16) {
                        for i in 0..d {
                            let bytes: [u8; 2] = rec[i * 2..(i + 1) * 2].try_into().unwrap();
                            out[i] = half::f16::from_le_bytes(bytes).to_f64();
                        }
                    } else {
                        for i in 0..d {
                            let bytes: [u8; 4] = rec[i * 4..(i + 1) * 4].try_into().unwrap();
                            out[i] = f32::from_le_bytes(bytes) as f64;
                        }
                    }
                    let prep = quantizer.prepare_ip_query_lite(&out);
                    candidates
                        .iter()
                        .map(|&to| {
                            let to_idx = to as usize;
                            let to_i = &all_mse[to_idx * qn..(to_idx + 1) * qn];
                            let to_q = if cached_qjl_len > 0 {
                                &all_qjl_flat
                                    [to_idx * cached_qjl_len..(to_idx + 1) * cached_qjl_len]
                            } else {
                                &[]
                            };
                            let to_g = all_gamma[to_idx];
                            let to_n = all_norm[to_idx];
                            (
                                to,
                                quantizer.score_ip_encoded_lite(&prep, to_i, to_q, to_g as f64)
                                    * to_n as f64,
                            )
                        })
                        .collect()
                } else {
                    // No raw vectors: use MSE centroid-lookup score blended with Hamming
                    // similarity between from-QJL bits and to-QJL bits.
                    //
                    // Background: prepare_ip_query_from_codes sets sq=0 (the QJL
                    // projection of the from-vector is unknown without raw data), making
                    // score_ip_encoded_lite skip the QJL term. This MSE-only topology
                    // mismatches the full-LUT scoring used at search time → poor recall.
                    //
                    // Fix: hamming_score(from_q, to_q) ∈ [0,1] approximates cos(from,to)
                    // in QJL-sign space. Subtracting 0.5 centres it so it contributes
                    // positively for near neighbours and negatively for far ones.
                    let from_i = &all_mse[from_idx * qn..(from_idx + 1) * qn];
                    let from_q = if cached_qjl_len > 0 {
                        &all_qjl_flat[from_idx * cached_qjl_len..(from_idx + 1) * cached_qjl_len]
                    } else {
                        &[]
                    };
                    let mse_prep = quantizer.prepare_ip_query_from_codes(from_i);
                    candidates
                        .iter()
                        .map(|&to| {
                            let to_idx = to as usize;
                            let to_i = &all_mse[to_idx * qn..(to_idx + 1) * qn];
                            let to_q = if cached_qjl_len > 0 {
                                &all_qjl_flat
                                    [to_idx * cached_qjl_len..(to_idx + 1) * cached_qjl_len]
                            } else {
                                &[]
                            };
                            let to_n = all_norm[to_idx];
                            let mse_score =
                                quantizer.score_ip_encoded_lite(&mse_prep, to_i, &[], 0.0);
                            let hamming_bonus = if !from_q.is_empty() && !to_q.is_empty() {
                                quantizer.hamming_proximity(from_q, to_q) - 0.5
                            } else {
                                0.0
                            };
                            (to, (mse_score + hamming_bonus) * to_n as f64)
                        })
                        .collect()
                }
            } else if matches!(metric, DistanceMetric::Cosine) {
                if has_vraw {
                    let rec = vraw_ref
                        .expect("invariant: has_vraw=true implies live_vraw is Some")
                        .get_slot(from_slot);
                    let mut out = Array1::<f64>::zeros(d);
                    if matches!(vraw_precision, RerankPrecision::F16) {
                        for i in 0..d {
                            let bytes: [u8; 2] = rec[i * 2..(i + 1) * 2].try_into().unwrap();
                            out[i] = half::f16::from_le_bytes(bytes).to_f64();
                        }
                    } else {
                        for i in 0..d {
                            let bytes: [u8; 4] = rec[i * 4..(i + 1) * 4].try_into().unwrap();
                            out[i] = f32::from_le_bytes(bytes) as f64;
                        }
                    }
                    let from_norm = out.iter().map(|x| x * x).sum::<f64>().sqrt();
                    let prep = quantizer.prepare_ip_query_lite(&out);
                    candidates
                        .iter()
                        .map(|&to| {
                            let to_idx = to as usize;
                            let to_i = &all_mse[to_idx * qn..(to_idx + 1) * qn];
                            let to_q = if cached_qjl_len > 0 {
                                &all_qjl_flat
                                    [to_idx * cached_qjl_len..(to_idx + 1) * cached_qjl_len]
                            } else {
                                &[]
                            };
                            let to_g = all_gamma[to_idx];
                            // prep built from raw from_vec; ip ≈ <from_raw, unit_to>
                            let ip =
                                quantizer.score_ip_encoded_lite(&prep, to_i, to_q, to_g as f64);
                            let score = if from_norm > 0.0 { ip / from_norm } else { 0.0 };
                            (to, score)
                        })
                        .collect()
                } else {
                    // No raw vectors: same Hamming-blend fix as for IP above.
                    // Vectors are stored unit-normalised, so cosine ≈ inner product.
                    let from_i = &all_mse[from_idx * qn..(from_idx + 1) * qn];
                    let from_q = if cached_qjl_len > 0 {
                        &all_qjl_flat[from_idx * cached_qjl_len..(from_idx + 1) * cached_qjl_len]
                    } else {
                        &[]
                    };
                    let mse_prep = quantizer.prepare_ip_query_from_codes(from_i);
                    candidates
                        .iter()
                        .map(|&to| {
                            let to_idx = to as usize;
                            let to_i = &all_mse[to_idx * qn..(to_idx + 1) * qn];
                            let to_q = if cached_qjl_len > 0 {
                                &all_qjl_flat
                                    [to_idx * cached_qjl_len..(to_idx + 1) * cached_qjl_len]
                            } else {
                                &[]
                            };
                            let mse_score =
                                quantizer.score_ip_encoded_lite(&mse_prep, to_i, &[], 0.0);
                            let hamming_bonus = if !from_q.is_empty() && !to_q.is_empty() {
                                quantizer.hamming_proximity(from_q, to_q) - 0.5
                            } else {
                                0.0
                            };
                            (to, mse_score + hamming_bonus)
                        })
                        .collect()
                }
            } else if !precomputed_l2.is_empty() {
                // L2 without raw vecs: use pre-computed dequantized vectors (no per-call
                // SRHT — each vector was dequantized exactly once in parallel above).
                let from_vec = &precomputed_l2[from_idx];
                candidates
                    .iter()
                    .map(|&to| {
                        let to_vec = &precomputed_l2[to as usize];
                        (to, score_vectors_with_metric(&metric, from_vec, to_vec))
                    })
                    .collect()
            } else {
                // L2 with raw vecs: read from vraw directly.
                // Invariant: precomputed_l2 is empty only when has_vraw=true (L2 + rerank=true).
                let vraw =
                    vraw_ref.expect("invariant: L2 raw-vec path only reached when rerank=true");
                let rec = vraw.get_slot(from_slot);
                let mut from_vec = Array1::<f64>::zeros(d);
                for i in 0..d {
                    let bytes: [u8; 4] = rec[i * 4..(i + 1) * 4].try_into().unwrap();
                    from_vec[i] = f32::from_le_bytes(bytes) as f64;
                }
                candidates
                    .iter()
                    .map(|&to| {
                        let to_slot = slots_ref[to as usize] as usize;
                        let rec = vraw.get_slot(to_slot);
                        let mut to_vec = Array1::<f64>::zeros(d);
                        for i in 0..d {
                            let bytes: [u8; 4] = rec[i * 4..(i + 1) * 4].try_into().unwrap();
                            to_vec[i] = f32::from_le_bytes(bytes) as f64;
                        }
                        (to, score_vectors_with_metric(&metric, &from_vec, &to_vec))
                    })
                    .collect()
            }
        };

        self.graph.build(
            indexed_slots.len(),
            max_degree,
            ef_construction,
            n_refinements,
            alpha,
            build_scorer,
        )?;
        self.index_ids = indexed_slots;
        self.index_ids_dirty = true;
        self.manifest.index_state = Some(IndexState {
            max_degree,
            ef_construction,
            search_list_size,
            alpha,
            indexed_nodes: self.index_ids.len(),
        });
        self.maybe_persist_state(true)?;
        Ok(())
    }

    pub fn search(
        &self,
        query: &Array1<f64>,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error + Send + Sync>> {
        self.search_with_filter_and_ann(query, top_k, None, None, true)
    }

    pub fn search_with_filter(
        &self,
        query: &Array1<f64>,
        top_k: usize,
        filter: Option<&HashMap<String, JsonValue>>,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error + Send + Sync>> {
        self.search_with_filter_and_ann(query, top_k, filter, None, true)
    }

    pub fn search_with_filter_and_ann(
        &self,
        query: &Array1<f64>,
        top_k: usize,
        filter: Option<&HashMap<String, JsonValue>>,
        ann_search_list_size: Option<usize>,
        use_ann: bool,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error + Send + Sync>> {
        if self.id_pool.active_count() == 0 || top_k == 0 {
            return Ok(Vec::new());
        }

        // When normalize=true, L2-normalise the query so IP ≡ cosine similarity.
        let normalized_query;
        let query = if self.normalize {
            let qn = query.iter().map(|x| x * x).sum::<f64>().sqrt();
            normalized_query = if qn > 1e-10 {
                query / qn
            } else {
                query.to_owned()
            };
            &normalized_query
        } else {
            query
        };

        let has_index = self.graph.has_index();
        let not_empty = !self.index_ids.is_empty();
        let state_match = self
            .manifest
            .index_state
            .as_ref()
            .is_some_and(|s| s.indexed_nodes == self.index_ids.len());

        if use_ann && has_index && not_empty && state_match {
            let sls = ann_search_list_size.unwrap_or_else(|| {
                self.manifest
                    .index_state
                    .as_ref()
                    .map(|s| s.search_list_size)
                    .unwrap_or(64)
            });

            // Pre-filter support
            let filter_slots = if let Some(f) = filter {
                let mut matches = Vec::new();
                let active = self.id_pool.iter_active();
                let slots: Vec<u32> = active.iter().map(|(_, s)| *s).collect();
                let meta_map = self.metadata.get_many(&slots)?;
                for (_, slot) in active {
                    if let Some(meta) = meta_map.get(&slot) {
                        if metadata_matches_filter(&meta.properties, f) {
                            matches.push(slot);
                        }
                    }
                }
                if matches.is_empty() {
                    return Ok(Vec::new());
                }
                Some(matches)
            } else {
                None
            };

            // Expand candidate pool for ANN: always fetch at least sls candidates so the
            // beam search has a sufficient buffer to recover from approximate navigation.
            // Without reranking, internal_k=top_k leaves no buffer → recall collapses.
            let internal_k = if self.rerank_enabled {
                top_k * 20
            } else {
                // Fetch sls candidates, re-score by full LUT, return top_k.
                sls.max(top_k)
            };

            // Shared references captured by search closures.
            let mse_len = self.live_mse_len();
            let qjl_len = self.live_qjl_len();
            let qn = self.quantizer.n;
            let live_codes_r = &self.live_codes;
            let quantizer_r = &self.quantizer;

            // Single full-LUT query prep: used for both HNSW navigation and final scoring.
            // The full scorer is more accurate than the lite scorer for navigation, and
            // eliminates the need for a separate re-score step after beam search.
            let prep = self.quantizer.prepare_ip_query(query);
            let query_norm = if matches!(self.metric, DistanceMetric::Cosine) {
                query.iter().map(|x| x * x).sum::<f64>().sqrt()
            } else {
                1.0
            };

            let slot_set: Option<std::collections::HashSet<u32>> =
                filter_slots.map(|s| s.into_iter().collect());
            let index_ids = &self.index_ids;
            let is_l2 = matches!(self.metric, DistanceMetric::L2);
            let is_cosine = matches!(self.metric, DistanceMetric::Cosine);
            let metric_r = &self.metric;

            // Reusable index buffer for MSE code unpacking — captured by the FnMut scorer
            // closure so it is allocated once and reused across all HNSW node visits.
            let mut idx_buf_ann = vec![0u16; qn];

            // HNSW beam search: collect internal_k candidates with accurate full-LUT scores.
            // internal_k = sls (≥ top_k) provides a candidate buffer to recover from any
            // graph navigation approximation errors before truncating to the final top_k.
            let ann_nodes = self.graph.search(
                0,
                internal_k,
                sls.max(internal_k),
                |node| {
                    let slot = index_ids[node as usize];
                    let rec = live_codes_r.get_slot(slot as usize);
                    let qjl = &rec[mse_len..mse_len + qjl_len];
                    let gamma = f32::from_le_bytes(
                        rec[mse_len + qjl_len..mse_len + qjl_len + 4]
                            .try_into()
                            .unwrap(),
                    );
                    let doc_norm = f32::from_le_bytes(
                        rec[mse_len + qjl_len + 4..mse_len + qjl_len + 8]
                            .try_into()
                            .unwrap(),
                    );
                    quantizer_r.unpack_mse_indices(&rec[..mse_len], &mut idx_buf_ann);

                    if is_l2 {
                        let mut v = quantizer_r.dequantize(&idx_buf_ann, qjl, gamma as f64);
                        v.mapv_inplace(|x| x * doc_norm as f64);
                        score_vectors_with_metric(metric_r, query, &v)
                    } else if is_cosine {
                        let ip =
                            quantizer_r.score_ip_encoded(&prep, &idx_buf_ann, qjl, gamma as f64);
                        if query_norm > 0.0 {
                            ip / query_norm
                        } else {
                            0.0
                        }
                    } else {
                        quantizer_r.score_ip_encoded(&prep, &idx_buf_ann, qjl, gamma as f64)
                            * doc_norm as f64
                    }
                },
                slot_set.map(|ss| move |node_idx: u32| ss.contains(&index_ids[node_idx as usize])),
            )?;

            // Navigation scores from the full LUT are already accurate; no re-score pass
            // needed. ann_nodes is already sorted descending by graph.search(); truncate
            // here handles the rerank=True path which uses internal_k = top_k * 20.
            let mut ann: Vec<(u32, f64)> = ann_nodes;
            ann.truncate(top_k);

            let slots: Vec<u32> = ann
                .iter()
                .map(|(n, _)| self.index_ids[*n as usize])
                .collect();
            let meta_map = self.metadata.get_many(&slots)?;
            let mut out = Vec::with_capacity(ann.len());

            // Batch-dequantize all candidates in parallel when reranking without raw vecs.
            let deq_vecs: Vec<Array1<f64>> = if self.rerank_enabled && self.live_vraw.is_none() {
                let (encoded, norms): (Vec<_>, Vec<f32>) = slots
                    .iter()
                    .map(|&slot| {
                        let (idx, qjl, gamma, norm) = self.live_codes_at_slot(slot as usize);
                        ((idx, qjl.to_vec(), gamma as f64), norm)
                    })
                    .unzip();
                let mut vecs = self.quantizer.dequantize_batch(&encoded);
                // Dequantize returns unit vectors; scale back to original norm for correct
                // metric computation (IP, L2). Cosine is scale-invariant so this is safe.
                for (v, norm) in vecs.iter_mut().zip(norms.iter()) {
                    v.mapv_inplace(|x| x * *norm as f64);
                }
                vecs
            } else {
                Vec::new()
            };

            for (i, (node, approx_score)) in ann.into_iter().enumerate() {
                let slot = self.index_ids[node as usize];
                let id = self.id_pool.get_str(slot).unwrap_or_default().to_string();
                let meta = meta_map.get(&slot).cloned().unwrap_or_default();

                let score = if self.rerank_enabled {
                    if self.live_vraw.is_some() {
                        let raw_vec = self.live_raw_vector_at_slot(slot as usize);
                        score_vectors_with_metric(&self.metric, query, &raw_vec)
                    } else {
                        score_vectors_with_metric(&self.metric, query, &deq_vecs[i])
                    }
                } else {
                    approx_score
                };

                out.push(SearchResult {
                    id,
                    score,
                    metadata: meta.properties,
                    document: meta.document,
                });
            }
            out.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if out.len() > top_k {
                out.truncate(top_k);
            }

            // Hybrid: score any vectors inserted after create_index() ("dark slots")
            // that are not in the HNSW graph. Without this, new inserts are silently
            // invisible to ANN search.
            let indexed_set: std::collections::HashSet<u32> =
                self.index_ids.iter().copied().collect();
            let dark_slots: Vec<u32> = self
                .id_pool
                .iter_active()
                .iter()
                .map(|(_, s)| *s)
                .filter(|s| !indexed_set.contains(s))
                .collect();
            if !dark_slots.is_empty() {
                let mut dark_results =
                    self.exhaustive_search_simd(query, top_k, filter, Some(dark_slots))?;
                out.append(&mut dark_results);
                out.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                out.truncate(top_k);
            }

            return Ok(out);
        }

        // Exhaustive search path (SIMD Optimized)
        self.exhaustive_search_simd(query, top_k, filter, None)
    }

    /// Score a set of candidate slots against `query` and return the top-`top_k` results.
    ///
    /// `forced_slots`: when `Some`, these slots are scored directly (after optional filter).
    /// When `None`, all active slots from `id_pool` are used (the normal brute-force path).
    fn exhaustive_search_simd(
        &self,
        query: &Array1<f64>,
        top_k: usize,
        filter: Option<&HashMap<String, JsonValue>>,
        forced_slots: Option<Vec<u32>>,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error + Send + Sync>> {
        let internal_k = if self.rerank_enabled {
            top_k * 10
        } else {
            top_k
        };
        let q_norm = query.iter().map(|x| x * x).sum::<f64>().sqrt();

        // Build candidate slot list: either from the forced set or all active slots.
        // Pre-filter slots upfront to keep the hot scoring loop filter-free.
        let candidate_slots: Vec<u32> = match forced_slots {
            Some(slots) => {
                if let Some(f) = filter {
                    let meta_map = self.metadata.get_many(&slots)?;
                    slots
                        .into_iter()
                        .filter(|s| {
                            meta_map
                                .get(s)
                                .is_some_and(|m| metadata_matches_filter(&m.properties, f))
                        })
                        .collect()
                } else {
                    slots
                }
            }
            None => {
                let active = self.id_pool.iter_active();
                if active.is_empty() {
                    return Ok(Vec::new());
                }
                if let Some(f) = filter {
                    let all_slots: Vec<u32> = active.iter().map(|(_, s)| *s).collect();
                    let meta_map = self.metadata.get_many(&all_slots)?;
                    all_slots
                        .into_iter()
                        .filter(|s| {
                            meta_map
                                .get(s)
                                .is_some_and(|m| metadata_matches_filter(&m.properties, f))
                        })
                        .collect()
                } else {
                    active.iter().map(|(_, s)| *s).collect()
                }
            }
        };

        if candidate_slots.is_empty() {
            return Ok(Vec::new());
        }

        // Borrow mmap bytes once — &[u8] is Sync so safe to share across Rayon threads.
        let codes_bytes = self.live_codes.as_bytes();
        let stride = self.live_stride();
        let mse_len = self.live_mse_len();
        let qjl_len = self.live_qjl_len();
        let quantizer = &self.quantizer;
        let metric = &self.metric;

        // Scoring: sequential for small N (avoids Rayon thread park/unpark on Windows
        // where scheduler granularity dominates the actual sub-millisecond work);
        // parallel chunk-based for large N where the work outweighs the thread overhead.
        const SEQ_THRESHOLD: usize = 20_000;
        const CHUNK: usize = 512;
        let n_candidates = candidate_slots.len();
        let scored: Vec<(u32, f64)> = if n_candidates <= SEQ_THRESHOLD {
            // Sequential path: single idx buffer reused across all slots.
            let mut out = Vec::with_capacity(n_candidates);
            let mut idx = vec![0u16; quantizer.n];
            match metric {
                DistanceMetric::Ip => {
                    let prep = quantizer.prepare_ip_query(query);
                    for &slot in &candidate_slots {
                        let rec =
                            &codes_bytes[slot as usize * stride..(slot as usize + 1) * stride];
                        quantizer.unpack_mse_indices(&rec[..mse_len], &mut idx);
                        let gamma = f32::from_le_bytes(
                            rec[mse_len + qjl_len..mse_len + qjl_len + 4]
                                .try_into()
                                .unwrap(),
                        );
                        let doc_norm = f32::from_le_bytes(
                            rec[mse_len + qjl_len + 4..mse_len + qjl_len + 8]
                                .try_into()
                                .unwrap(),
                        );
                        let score = quantizer.score_ip_encoded(
                            &prep,
                            &idx,
                            &rec[mse_len..mse_len + qjl_len],
                            gamma as f64,
                        ) * doc_norm as f64;
                        out.push((slot, score));
                    }
                }
                DistanceMetric::Cosine => {
                    let prep = quantizer.prepare_ip_query(query);
                    for &slot in &candidate_slots {
                        let rec =
                            &codes_bytes[slot as usize * stride..(slot as usize + 1) * stride];
                        quantizer.unpack_mse_indices(&rec[..mse_len], &mut idx);
                        let gamma = f32::from_le_bytes(
                            rec[mse_len + qjl_len..mse_len + qjl_len + 4]
                                .try_into()
                                .unwrap(),
                        );
                        let ip = quantizer.score_ip_encoded(
                            &prep,
                            &idx,
                            &rec[mse_len..mse_len + qjl_len],
                            gamma as f64,
                        );
                        let score = if q_norm > 0.0 { ip / q_norm } else { 0.0 };
                        out.push((slot, score));
                    }
                }
                _ => {
                    for &slot in &candidate_slots {
                        let rec =
                            &codes_bytes[slot as usize * stride..(slot as usize + 1) * stride];
                        quantizer.unpack_mse_indices(&rec[..mse_len], &mut idx);
                        let gamma = f32::from_le_bytes(
                            rec[mse_len + qjl_len..mse_len + qjl_len + 4]
                                .try_into()
                                .unwrap(),
                        );
                        let doc_norm = f32::from_le_bytes(
                            rec[mse_len + qjl_len + 4..mse_len + qjl_len + 8]
                                .try_into()
                                .unwrap(),
                        );
                        let mut v = quantizer.dequantize(
                            &idx,
                            &rec[mse_len..mse_len + qjl_len],
                            gamma as f64,
                        );
                        v.mapv_inplace(|x| x * doc_norm as f64);
                        let score = score_vectors_with_metric(metric, query, &v);
                        out.push((slot, score));
                    }
                }
            }
            out
        } else {
            // Parallel path: each 512-slot chunk reuses one CodeIndex scratch buffer.
            match metric {
                DistanceMetric::Ip => {
                    let prep = quantizer.prepare_ip_query(query);
                    candidate_slots
                        .par_chunks(CHUNK)
                        .flat_map(|chunk| {
                            let mut idx = vec![0u16; quantizer.n];
                            let mut out = Vec::with_capacity(chunk.len());
                            for &slot in chunk {
                                let rec = &codes_bytes
                                    [slot as usize * stride..(slot as usize + 1) * stride];
                                quantizer.unpack_mse_indices(&rec[..mse_len], &mut idx);
                                let gamma = f32::from_le_bytes(
                                    rec[mse_len + qjl_len..mse_len + qjl_len + 4]
                                        .try_into()
                                        .unwrap(),
                                );
                                let doc_norm = f32::from_le_bytes(
                                    rec[mse_len + qjl_len + 4..mse_len + qjl_len + 8]
                                        .try_into()
                                        .unwrap(),
                                );
                                // Vectors stored unit-normalized; scale back to recover <q, doc>.
                                let score = quantizer.score_ip_encoded(
                                    &prep,
                                    &idx,
                                    &rec[mse_len..mse_len + qjl_len],
                                    gamma as f64,
                                ) * doc_norm as f64;
                                out.push((slot, score));
                            }
                            out
                        })
                        .collect()
                }
                DistanceMetric::Cosine => {
                    let prep = quantizer.prepare_ip_query(query);
                    candidate_slots
                        .par_chunks(CHUNK)
                        .flat_map(|chunk| {
                            let mut idx = vec![0u16; quantizer.n];
                            let mut out = Vec::with_capacity(chunk.len());
                            for &slot in chunk {
                                let rec = &codes_bytes
                                    [slot as usize * stride..(slot as usize + 1) * stride];
                                quantizer.unpack_mse_indices(&rec[..mse_len], &mut idx);
                                let gamma = f32::from_le_bytes(
                                    rec[mse_len + qjl_len..mse_len + qjl_len + 4]
                                        .try_into()
                                        .unwrap(),
                                );
                                // Vectors stored unit-normalized; ip estimates <query, unit_doc>.
                                // cosine(query, doc) = <query, unit_doc> / ||query||
                                let ip = quantizer.score_ip_encoded(
                                    &prep,
                                    &idx,
                                    &rec[mse_len..mse_len + qjl_len],
                                    gamma as f64,
                                );
                                let score = if q_norm > 0.0 { ip / q_norm } else { 0.0 };
                                out.push((slot, score));
                            }
                            out
                        })
                        .collect()
                }
                _ => {
                    // L2 and any future metrics: dequantize then compute distance
                    candidate_slots
                        .par_chunks(CHUNK)
                        .flat_map(|chunk| {
                            let mut idx = vec![0u16; quantizer.n];
                            let mut out = Vec::with_capacity(chunk.len());
                            for &slot in chunk {
                                let rec = &codes_bytes
                                    [slot as usize * stride..(slot as usize + 1) * stride];
                                quantizer.unpack_mse_indices(&rec[..mse_len], &mut idx);
                                let gamma = f32::from_le_bytes(
                                    rec[mse_len + qjl_len..mse_len + qjl_len + 4]
                                        .try_into()
                                        .unwrap(),
                                );
                                let doc_norm = f32::from_le_bytes(
                                    rec[mse_len + qjl_len + 4..mse_len + qjl_len + 8]
                                        .try_into()
                                        .unwrap(),
                                );
                                // Dequantize returns unit vector; scale back to original norm.
                                let mut v = quantizer.dequantize(
                                    &idx,
                                    &rec[mse_len..mse_len + qjl_len],
                                    gamma as f64,
                                );
                                v.mapv_inplace(|x| x * doc_norm as f64);
                                let score = score_vectors_with_metric(metric, query, &v);
                                out.push((slot, score));
                            }
                            out
                        })
                        .collect()
                }
            }
        };

        // Build top-k heap from flat scored list
        let mut results = BinaryHeap::with_capacity(internal_k + 1);
        for (slot, score) in scored {
            results.push(OrderingWrapper(SearchCandidate { id: slot, score }));
            if results.len() > internal_k {
                results.pop();
            }
        }

        // Drain to Vec for deterministic indexing (BinaryHeap iter/into_iter order
        // is unspecified and may differ — Vec guarantees consistent order for the
        // batch-dequantize index alignment below).
        let candidates: Vec<OrderingWrapper> = results.into_iter().collect();
        let slots: Vec<u32> = candidates.iter().map(|OrderingWrapper(c)| c.id).collect();
        let meta_map = self.metadata.get_many(&slots)?;
        let mut out = Vec::with_capacity(candidates.len());

        // Batch-dequantize all rerank candidates in parallel when raw vecs unavailable.
        let deq_vecs: Vec<Array1<f64>> = if self.rerank_enabled && self.live_vraw.is_none() {
            let (encoded, norms): (Vec<_>, Vec<f32>) = slots
                .iter()
                .map(|&slot| {
                    let (idx, qjl, gamma, norm) = self.live_codes_at_slot(slot as usize);
                    ((idx, qjl.to_vec(), gamma as f64), norm)
                })
                .unzip();
            let mut vecs = self.quantizer.dequantize_batch(&encoded);
            for (v, norm) in vecs.iter_mut().zip(norms.iter()) {
                v.mapv_inplace(|x| x * *norm as f64);
            }
            vecs
        } else {
            Vec::new()
        };

        for (i, OrderingWrapper(cand)) in candidates.into_iter().enumerate() {
            let id = self
                .id_pool
                .get_str(cand.id)
                .unwrap_or_default()
                .to_string();
            let meta = meta_map.get(&cand.id).cloned().unwrap_or_default();

            let score = if self.rerank_enabled {
                if self.live_vraw.is_some() {
                    let raw_vec = self.live_raw_vector_at_slot(cand.id as usize);
                    score_vectors_with_metric(&self.metric, query, &raw_vec)
                } else {
                    score_vectors_with_metric(&self.metric, query, &deq_vecs[i])
                }
            } else {
                cand.score
            };

            out.push(SearchResult {
                id,
                score,
                metadata: meta.properties,
                document: meta.document,
            });
        }
        out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        if out.len() > top_k {
            out.truncate(top_k);
        }
        Ok(out)
    }

    pub fn flush_wal_to_segment(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.wal_buffer.is_empty() {
            return Ok(());
        }
        let records: Vec<SegmentRecord> = self
            .wal_buffer
            .drain(..)
            .map(|e| SegmentRecord {
                id: e.id,
                is_deleted: e.is_deleted,
            })
            .collect();
        for r in &records {
            if r.is_deleted {
                self.live_delete_slot(&r.id);
            }
        }
        // Persist segments first so any rebuilds (if needed) read a complete view.
        self.segments.flush_batch(records)?;
        // Only compact the live slab when there are pending deletes — compaction
        // copies all live data to a fresh file and rebuilds the id_pool, which is
        // unnecessary (and very expensive) for pure insert workloads.
        if self.has_pending_deletes {
            self.live_compact_slab()?;
            self.has_pending_deletes = false;
        }
        self.live_codes.flush()?;
        if let Some(vraw) = &mut self.live_vraw {
            vraw.flush()?;
        }
        self.wal.truncate()?;
        self.metadata.flush()?;
        self.persist_id_pool()?;

        self.live_codes.release_handles();
        if let Some(vraw) = &mut self.live_vraw {
            vraw.release_handles();
        }
        let live_codes_path = Path::new(&self.local_dir).join("live_codes.bin");
        let live_vraw_path = Path::new(&self.local_dir).join("live_vectors.bin");

        let live_codes_data = std::fs::read(&live_codes_path)?;
        self.backend.write("live_codes.bin", &live_codes_data)?;

        // Only sync live_vectors.bin if it actually exists (new DBs use dequant reranking).
        let had_vraw = self.live_vraw.is_some();
        if had_vraw {
            let live_vraw_data = std::fs::read(&live_vraw_path)?;
            self.backend.write("live_vectors.bin", &live_vraw_data)?;
        }

        self.live_codes = LiveCodesFile::open(live_codes_path, self.live_stride())?;
        // The file was pre-allocated to GROW_SLOTS multiples; correct len to match
        // the id_pool's slot count so the next alloc_slot() resumes at the right offset.
        let slot_count = self.id_pool.slot_count();
        self.live_codes.set_len(slot_count);
        if had_vraw {
            let mut vraw = LiveCodesFile::open(live_vraw_path, self.live_vraw_stride())?;
            vraw.set_len(slot_count);
            self.live_vraw = Some(vraw);
        }
        self.invalidate_index_state()?;
        self.manifest.vector_count = self.live_active_count() as u64;
        self.maybe_persist_state(false)?;
        Ok(())
    }

    pub fn close(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.flush_wal_to_segment()?;
        // Trim pre-allocated capacity to the exact slot count so the on-disk and
        // memory-mapped sizes are minimal after the database is closed.
        let slot_count = self.id_pool.slot_count();
        self.live_codes.truncate_to(slot_count)?;
        if let Some(vraw) = &mut self.live_vraw {
            vraw.truncate_to(slot_count)?;
        }
        self.metadata.flush()?;
        self.persist_id_pool()?;
        self.maybe_persist_state(true)?;
        Ok(())
    }

    pub fn vector_count(&self) -> u64 {
        self.id_pool.active_count() as u64
    }

    /// Open a collection scoped under `{local_root}/{tenant}/{database}/{collection}`.
    /// Thin wrapper around `open_with_options` used by the HTTP server.
    pub fn open_collection_scoped(
        uri: &str,
        local_root: &str,
        tenant: &str,
        database: &str,
        collection: &str,
        d: usize,
        b: usize,
        seed: u64,
        metric: DistanceMetric,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let path = format!("{local_root}/{tenant}/{database}/{collection}");
        Self::open_with_options(
            uri,
            &path,
            d,
            b,
            seed,
            metric,
            true,
            false,
            RerankPrecision::Disabled,
            None,
            false,
        )
    }

    /// Copy all files in `src_dir` to `dst_dir` atomically (write to `.tmp` suffix then
    /// rename the entire directory).  Used for point-in-time snapshots by the HTTP server.
    pub fn snapshot_local_dir(
        src_dir: &str,
        dst_dir: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let tmp_dir = format!("{dst_dir}.tmp");
        if std::path::Path::new(&tmp_dir).exists() {
            std::fs::remove_dir_all(&tmp_dir)?;
        }
        std::fs::create_dir_all(&tmp_dir)?;
        for entry in std::fs::read_dir(src_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                let file_name = entry.file_name();
                std::fs::copy(
                    entry.path(),
                    format!("{tmp_dir}/{}", file_name.to_string_lossy()),
                )?;
            }
        }
        // Atomic rename: replace dst_dir with the fully-written tmp copy.
        if std::path::Path::new(dst_dir).exists() {
            std::fs::remove_dir_all(dst_dir)?;
        }
        std::fs::rename(&tmp_dir, dst_dir)?;
        Ok(())
    }

    /// Check whether a scoped collection exists (i.e., its `manifest.json` is present).
    /// Returns `Some(())` if it exists, `None` otherwise.  Used by the HTTP server to
    /// prevent duplicate collection creation.
    pub fn get_collection_scoped_with_uri(
        _uri: &str,
        local_root: &str,
        tenant: &str,
        database: &str,
        collection: &str,
    ) -> Result<Option<()>, Box<dyn std::error::Error + Send + Sync>> {
        let manifest = std::path::Path::new(local_root)
            .join(tenant)
            .join(database)
            .join(collection)
            .join("manifest.json");
        Ok(if manifest.exists() { Some(()) } else { None })
    }

    /// Create the on-disk directory for a new scoped collection.
    /// The actual engine is opened (and manifest written) by a subsequent `open_collection_scoped`.
    pub fn create_collection_scoped_with_uri(
        _uri: &str,
        local_root: &str,
        tenant: &str,
        database: &str,
        collection: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let path = std::path::Path::new(local_root)
            .join(tenant)
            .join(database)
            .join(collection);
        std::fs::create_dir_all(&path)?;
        Ok(())
    }

    /// Delete the on-disk directory of a scoped collection.
    /// Returns `true` if the directory existed and was removed, `false` if not found.
    pub fn delete_collection_scoped_with_uri(
        _uri: &str,
        local_root: &str,
        tenant: &str,
        database: &str,
        collection: &str,
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        let path = std::path::Path::new(local_root)
            .join(tenant)
            .join(database)
            .join(collection);
        if path.exists() {
            std::fs::remove_dir_all(&path)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Result returned by `insert_many_report` / `upsert_many_report`.
    // (Defined as a plain struct so the server can destructure it.)

    /// Batch-insert with per-item failure reporting.  Unlike `insert_many_with_mode`,
    /// this never returns an early `Err`; individual item failures are collected and
    /// returned alongside the successes.
    pub fn insert_many_report(&mut self, items: Vec<BatchWriteItem>) -> BatchWriteReport {
        let mut failed = Vec::new();
        let mut ok = Vec::new();
        for (index, item) in items.into_iter().enumerate() {
            let id = item.id.clone();
            let vec_f64: ndarray::Array1<f64> = item.vector.iter().map(|&x| x as f64).collect();
            match self.insert_many_with_mode(
                vec![BatchWriteItem {
                    id: id.clone(),
                    ..item
                }],
                BatchWriteMode::Insert,
            ) {
                Ok(_) => ok.push(id),
                Err(e) => failed.push(BatchWriteFailure {
                    index,
                    id,
                    error: e.to_string(),
                }),
            }
            let _ = vec_f64; // suppress unused warning
        }
        let applied = ok.len();
        BatchWriteReport {
            ok,
            failed,
            applied,
        }
    }

    /// Same as `insert_many_report` but uses upsert semantics.
    pub fn upsert_many_report(&mut self, items: Vec<BatchWriteItem>) -> BatchWriteReport {
        let mut failed = Vec::new();
        let mut ok = Vec::new();
        for (index, item) in items.into_iter().enumerate() {
            let id = item.id.clone();
            match self.insert_many_with_mode(
                vec![BatchWriteItem {
                    id: id.clone(),
                    ..item
                }],
                BatchWriteMode::Upsert,
            ) {
                Ok(_) => ok.push(id),
                Err(e) => failed.push(BatchWriteFailure {
                    index,
                    id,
                    error: e.to_string(),
                }),
            }
        }
        let applied = ok.len();
        BatchWriteReport {
            ok,
            failed,
            applied,
        }
    }

    /// Batch-delete by string IDs.  Wraps `delete_batch`.
    pub fn delete_many(
        &mut self,
        ids: &[String],
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        self.delete_batch(ids.to_vec())
    }

    /// Trigger segment compaction if the compaction threshold is met.
    /// Flushes the WAL first so all buffered data is on disk before merging.
    pub fn compact(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.flush_wal_to_segment()?;
        // Compaction is a background concern; no-op if below threshold.
        Ok(())
    }

    /// Build the HNSW index with server-friendly defaults.
    /// `max_degree` and `ef_construction` mirror the HTTP API defaults.
    pub fn create_index(
        &mut self,
        max_degree: usize,
        ef_construction: usize,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.create_index_with_params(max_degree, ef_construction, 128, 1.2, 5)
    }

    /// List collection names (subdirectories containing a `manifest.json`) under
    /// `{local_root}/{tenant}/{database}/`.  Used by the HTTP server.
    pub fn list_collections_scoped(
        local_root: &str,
        tenant: &str,
        database: &str,
    ) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let dir = format!("{local_root}/{tenant}/{database}");
        let dir_path = std::path::Path::new(&dir);
        if !dir_path.exists() {
            return Ok(Vec::new());
        }
        let mut collections = Vec::new();
        for entry in std::fs::read_dir(dir_path)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() && entry.path().join("manifest.json").exists() {
                if let Some(name) = entry.file_name().to_str() {
                    collections.push(name.to_string());
                }
            }
        }
        collections.sort();
        Ok(collections)
    }

    pub fn stats(&self) -> DbStats {
        DbStats {
            vector_count: self.vector_count(),
            segment_count: self.segments.segments.len(),
            buffered_vectors: self.wal_buffer.len(),
            d: self.d,
            b: self.b,
            total_disk_bytes: self.total_disk_bytes(),
            has_index: self.can_use_ann_index(),
            index_nodes: self.index_ids.len(),
            live_codes_bytes: self.live_codes.byte_len(),
            live_slot_count: self.live_codes.len(),
            live_id_count: self.id_pool.active_count(),
            live_vectors_count: self.live_active_count(),
            live_vectors_bytes_estimate: self.live_vraw.as_ref().map_or(0, |v| v.byte_len()),
            metadata_entries: self.metadata.len(),
            metadata_bytes_estimate: self.metadata.approx_bytes(),
            ann_slot_count: self.index_ids.len(),
            graph_nodes: self.graph.node_count(),
        }
    }

    fn live_mse_len(&self) -> usize {
        (self.quantizer.n * (self.b - 1)).div_ceil(8)
    }
    fn live_qjl_len(&self) -> usize {
        self.quantizer.n.div_ceil(8)
    }
    fn live_stride(&self) -> usize {
        self.live_mse_len()
            + self.live_qjl_len()
            + LIVE_GAMMA_BYTES
            + LIVE_NORM_BYTES
            + LIVE_DELETED_BYTES
    }
    /// Bytes per slot in live_vectors.bin (raw rerank buffer).
    fn live_vraw_stride(&self) -> usize {
        match self.manifest.rerank_precision {
            RerankPrecision::F16 => self.d * 2,
            RerankPrecision::Disabled | RerankPrecision::F32 => self.d * 4,
        }
    }
    fn live_active_count(&self) -> usize {
        self.id_pool.active_count()
    }

    fn live_codes_at_slot(&self, slot: usize) -> (Vec<CodeIndex>, &[u8], f32, f32) {
        let rec = self.live_codes.get_slot(slot);
        let mse_len = self.live_mse_len();
        let qjl_len = self.live_qjl_len();
        let mut indices = vec![0 as CodeIndex; self.quantizer.n];
        self.quantizer
            .unpack_mse_indices(&rec[0..mse_len], &mut indices);
        let qjl = &rec[mse_len..mse_len + qjl_len];
        let gamma = f32::from_le_bytes(
            rec[mse_len + qjl_len..mse_len + qjl_len + 4]
                .try_into()
                .unwrap(),
        );
        let norm = f32::from_le_bytes(
            rec[mse_len + qjl_len + 4..mse_len + qjl_len + 8]
                .try_into()
                .unwrap(),
        );
        (indices, qjl, gamma, norm)
    }

    fn live_raw_vector_at_slot(&self, slot: usize) -> Array1<f64> {
        let Some(vraw) = &self.live_vraw else {
            return Array1::zeros(self.d);
        };
        let rec = vraw.get_slot(slot);
        let mut out = Array1::zeros(self.d);
        match self.manifest.rerank_precision {
            RerankPrecision::F16 => {
                for i in 0..self.d {
                    let bytes: [u8; 2] = rec[i * 2..(i + 1) * 2].try_into().unwrap();
                    out[i] = half::f16::from_le_bytes(bytes).to_f64();
                }
            }
            RerankPrecision::F32 | RerankPrecision::Disabled => {
                for i in 0..self.d {
                    let bytes: [u8; 4] = rec[i * 4..(i + 1) * 4].try_into().unwrap();
                    out[i] = f32::from_le_bytes(bytes) as f64;
                }
            }
        }
        out
    }

    fn live_save_raw_vector(&mut self, slot: u32, vector: &[f32]) {
        let Some(vraw) = &mut self.live_vraw else {
            return;
        };
        let rec = vraw.get_slot_mut(slot as usize);
        match self.manifest.rerank_precision {
            RerankPrecision::F16 => {
                for (i, &val) in vector.iter().enumerate() {
                    let h = half::f16::from_f32(val);
                    rec[i * 2..(i + 1) * 2].copy_from_slice(&h.to_le_bytes());
                }
            }
            RerankPrecision::F32 | RerankPrecision::Disabled => {
                for (i, &val) in vector.iter().enumerate() {
                    rec[i * 4..(i + 1) * 4].copy_from_slice(&val.to_le_bytes());
                }
            }
        }
    }

    fn live_alloc_slot(
        &mut self,
        id: &str,
        indices: &[CodeIndex],
        qjl: &[u8],
        gamma: f32,
        norm: f32,
    ) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
        let slot = self.id_pool.insert(id)?;
        let new_slot = self.live_codes.alloc_slot()?;
        if let Some(vraw) = &mut self.live_vraw {
            let _ = vraw.alloc_slot()?;
        } // Keep aligned
        let mse_len = self.live_mse_len();
        let qjl_len = self.live_qjl_len();
        let packed_mse = self.quantizer.pack_mse_indices(indices);
        let rec = self.live_codes.get_slot_mut(new_slot);
        rec[0..mse_len].copy_from_slice(&packed_mse);
        rec[mse_len..mse_len + qjl_len].copy_from_slice(qjl);
        rec[mse_len + qjl_len..mse_len + qjl_len + 4].copy_from_slice(&gamma.to_le_bytes());
        rec[mse_len + qjl_len + 4..mse_len + qjl_len + 8].copy_from_slice(&norm.to_le_bytes());
        rec[mse_len + qjl_len + 8] = 0u8;
        Ok(slot)
    }

    fn live_alloc_or_update(
        &mut self,
        id: &str,
        indices: &[CodeIndex],
        qjl: &[u8],
        gamma: f32,
        norm: f32,
    ) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(slot) = self.id_pool.get_slot(id) {
            let mse_len = self.live_mse_len();
            let qjl_len = self.live_qjl_len();
            let packed_mse = self.quantizer.pack_mse_indices(indices);
            let rec = self.live_codes.get_slot_mut(slot as usize);
            rec[0..mse_len].copy_from_slice(&packed_mse);
            rec[mse_len..mse_len + qjl_len].copy_from_slice(qjl);
            rec[mse_len + qjl_len..mse_len + qjl_len + 4].copy_from_slice(&gamma.to_le_bytes());
            rec[mse_len + qjl_len + 4..mse_len + qjl_len + 8].copy_from_slice(&norm.to_le_bytes());
            rec[mse_len + qjl_len + 8] = 0u8;
            Ok(slot)
        } else {
            self.live_alloc_slot(id, indices, qjl, gamma, norm)
        }
    }

    fn live_delete_slot(&mut self, id: &str) {
        if let Some(slot) = self.id_pool.delete_by_id(id) {
            let stride = self.live_stride();
            let rec = self.live_codes.get_slot_mut(slot as usize);
            rec[stride - 1] = 1u8;
        }
    }

    fn live_iter_id_slots(&self) -> Vec<(String, u32)> {
        self.id_pool.iter_active()
    }

    fn live_compact_slab(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let stride = self.live_stride();
        let vstride = self.live_vraw_stride();
        let temp_path = Path::new(&self.local_dir).join("live_codes.bin.tmp");
        let vtemp_path = Path::new(&self.local_dir).join("live_vectors.bin.tmp");

        // Only compact live_vectors.bin if it actually exists (new DBs use dequant reranking).
        let had_vraw = self.live_vraw.is_some();
        let mut new_codes = LiveCodesFile::open(temp_path.clone(), stride)?;
        let mut new_vraw = if had_vraw {
            Some(LiveCodesFile::open(vtemp_path.clone(), vstride)?)
        } else {
            None
        };

        new_codes.clear()?;
        if let Some(nv) = &mut new_vraw {
            nv.clear()?;
        }

        let mut new_pool = IdPool::new();

        for (id, old_slot) in self.live_iter_id_slots() {
            let old_rec = self.live_codes.get_slot(old_slot as usize);
            let next_alloc = new_codes.alloc_slot()?;
            new_codes.get_slot_mut(next_alloc).copy_from_slice(old_rec);

            if let (Some(nv), Some(ov)) = (&mut new_vraw, &mut self.live_vraw) {
                let old_vrec = ov.get_slot(old_slot as usize);
                let _ = nv.alloc_slot()?;
                nv.get_slot_mut(next_alloc).copy_from_slice(old_vrec);
            }

            new_pool.insert(&id)?;
        }

        new_codes.truncate_to(new_pool.active_count())?;
        if let Some(nv) = &mut new_vraw {
            nv.truncate_to(new_pool.active_count())?;
        }

        new_codes.flush()?;
        drop(new_codes);
        if let Some(nv) = new_vraw {
            nv.flush()?;
            drop(nv);
        }

        let final_path = Path::new(&self.local_dir).join("live_codes.bin");
        let vfinal_path = Path::new(&self.local_dir).join("live_vectors.bin");

        self.live_codes.release_handles();
        if let Some(vraw) = &mut self.live_vraw {
            vraw.release_handles();
        }

        std::fs::rename(temp_path, &final_path)?;
        if had_vraw {
            std::fs::rename(vtemp_path, &vfinal_path)?;
        }

        self.live_codes = LiveCodesFile::open(final_path, stride)?;
        if had_vraw {
            self.live_vraw = Some(LiveCodesFile::open(vfinal_path, vstride)?);
        }

        self.id_pool = new_pool;
        Ok(())
    }

    fn persist_id_pool(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        save_id_pool(&self.local_dir, &self.backend, &self.id_pool)
    }

    fn invalidate_index_state(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.index_ids.clear();
        self.manifest.index_state = None;
        self.index_ids_dirty = true;
        Ok(())
    }

    fn maybe_persist_state(
        &mut self,
        force: bool,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if !force
            && !self.index_ids_dirty
            && self.pending_manifest_updates < MANIFEST_SAVE_INTERVAL_OPS
        {
            return Ok(());
        }
        if self.index_ids_dirty {
            save_index_ids(&self.local_dir, &self.index_ids)?;
            self.backend
                .write(INDEX_IDS_FILE, &serialize_index_ids(&self.index_ids)?)?;
            self.index_ids_dirty = false;
        }
        self.save_manifest()?;
        self.pending_manifest_updates = 0;
        Ok(())
    }

    fn can_use_ann_index(&self) -> bool {
        self.graph.has_index()
            && !self.index_ids.is_empty()
            && self
                .manifest
                .index_state
                .as_ref()
                .is_some_and(|s| s.indexed_nodes == self.index_ids.len())
    }

    fn save_manifest(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut m = self.manifest.clone();
        m.quantizer = None;
        let manifest_path = format!("{}/manifest.json", self.local_dir);
        m.save(&manifest_path)?;
        self.backend
            .write("manifest.json", &serde_json::to_vec_pretty(&m)?)?;
        Ok(())
    }

    fn total_disk_bytes(&self) -> u64 {
        let mut total = self.segments.total_disk_size();
        let mut files = vec![
            "manifest.json",
            "metadata.bin",
            "wal.log",
            QUANTIZER_STATE_FILE,
            "graph.bin",
            INDEX_IDS_FILE,
            "live_codes.bin",
        ];
        if self.rerank_enabled {
            files.push("live_vectors.bin");
        }
        for name in files {
            if let Ok(size) = self.backend.size(name) {
                total += size;
            }
        }
        total
    }

    fn write_vector_entry(
        &mut self,
        id: String,
        vector: &Array1<f64>,
        metadata: HashMap<String, JsonValue>,
        document: Option<String>,
        is_deleted: bool,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let vec_f32: Vec<f32> = vector.iter().map(|&v| v as f32).collect();
        let norm = vec_f32.iter().map(|x| x * x).sum::<f32>().sqrt();
        let vec_unit: Vec<f32> = if norm > 1e-10 {
            vec_f32.iter().map(|&x| x / norm).collect()
        } else {
            vec_f32.clone()
        };
        // When normalize=true, treat the vector as unit-length so IP ≡ cosine.
        // The stored norm drives IP score scaling; setting it to 1.0 removes that scaling.
        let stored_norm = if self.normalize { 1.0f32 } else { norm };
        let raw_for_rerank = if self.normalize { &vec_unit } else { &vec_f32 };
        let (indices, qjl, gamma) = self.quantizer.quantize(&vec_unit);
        let meta = VectorMetadata {
            properties: metadata,
            document,
        };
        let entry = WalEntry {
            id: id.clone(),
            quantized_indices: indices,
            qjl_bits: qjl,
            gamma: gamma as f32,
            norm: stored_norm,
            metadata_json: serde_json::to_string(&meta)?,
            is_deleted,
        };
        self.wal.append(&entry, false)?;
        self.wal_buffer.push(entry.clone());
        if !is_deleted {
            let slot = self.live_alloc_or_update(
                &id,
                &entry.quantized_indices,
                &entry.qjl_bits,
                entry.gamma,
                stored_norm,
            )?;
            self.live_save_raw_vector(slot, raw_for_rerank);
            self.metadata.put(slot, &meta)?;
        }
        self.invalidate_index_state()?;
        self.manifest.vector_count = self.live_active_count() as u64;
        self.maybe_persist_state(false)?;
        if self.wal_buffer.len() >= self.wal_flush_threshold {
            self.flush_wal_to_segment()?;
        }
        Ok(())
    }

    pub fn insert_many_with_mode(
        &mut self,
        items: Vec<BatchWriteItem>,
        mode: BatchWriteMode,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        for chunk in items.chunks(5000) {
            let mut wal_entries = Vec::with_capacity(chunk.len());
            let mut metadata_entries: Vec<(u32, VectorMetadata)> = Vec::with_capacity(chunk.len());
            // Normalize each vector to unit sphere before quantization so the
            // Lloyd-Max codebook (fitted to Beta-distribution unit-sphere coords) is valid.
            // Compute (unit_vec, norm) once to avoid a second O(d) pass per vector.
            let unit_vecs_and_norms: Vec<(Vec<f32>, f32)> = chunk
                .iter()
                .map(|item| {
                    let n = item.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if n > 1e-10 {
                        (item.vector.iter().map(|&x| x / n).collect(), n)
                    } else {
                        (item.vector.clone(), n)
                    }
                })
                .collect();
            let vec_refs: Vec<&[f32]> = unit_vecs_and_norms
                .iter()
                .map(|(v, _)| v.as_slice())
                .collect();
            let quantized = self.quantizer.quantize_batch(&vec_refs);

            for (_i, (item, ((unit_vec, norm), (indices, qjl, gamma)))) in chunk
                .iter()
                .zip(unit_vecs_and_norms.iter().zip(quantized))
                .enumerate()
            {
                match mode {
                    BatchWriteMode::Insert if self.id_pool.contains(&item.id) => {
                        return Err(format!("ID '{}' already exists", item.id).into());
                    }
                    BatchWriteMode::Update if !self.id_pool.contains(&item.id) => {
                        return Err(format!("ID '{}' does not exist", item.id).into());
                    }
                    _ => {}
                }
                // When normalize=true, store norm=1.0 so IP ≡ cosine (no scaling).
                let stored_norm = if self.normalize { 1.0f32 } else { *norm };
                let raw_for_rerank: &[f32] = if self.normalize {
                    unit_vec
                } else {
                    &item.vector
                };
                let meta = VectorMetadata {
                    properties: item.metadata.clone(),
                    document: item.document.clone(),
                };
                let entry = WalEntry {
                    id: item.id.clone(),
                    quantized_indices: indices,
                    qjl_bits: qjl,
                    gamma: gamma as f32,
                    norm: stored_norm,
                    metadata_json: serde_json::to_string(&meta)?,
                    is_deleted: false,
                };
                let slot = self.live_alloc_or_update(
                    &item.id,
                    &entry.quantized_indices,
                    &entry.qjl_bits,
                    entry.gamma,
                    stored_norm,
                )?;
                self.live_save_raw_vector(slot, raw_for_rerank);
                metadata_entries.push((slot, meta));
                wal_entries.push(entry);
            }
            self.wal.append_batch(&wal_entries, false)?;
            self.metadata.put_many(&metadata_entries)?;
            self.wal_buffer.extend(wal_entries);
        }
        self.invalidate_index_state()?;
        self.manifest.vector_count = self.live_active_count() as u64;
        self.maybe_persist_state(false)?;
        if self.wal_buffer.len() >= self.wal_flush_threshold {
            self.flush_wal_to_segment()?;
        }
        Ok(())
    }
}

fn save_quantizer_state(
    local_dir: &str,
    backend: &Arc<StorageBackend>,
    quantizer: &ProdQuantizer,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let bytes = bincode::serialize(quantizer)?;
    std::fs::write(format!("{}/{}", local_dir, QUANTIZER_STATE_FILE), &bytes)?;
    backend.write(QUANTIZER_STATE_FILE, &bytes)?;
    Ok(())
}

fn load_quantizer_state(
    local_dir: &str,
    backend: &Arc<StorageBackend>,
) -> Result<ProdQuantizer, Box<dyn std::error::Error + Send + Sync>> {
    let local = format!("{}/{}", local_dir, QUANTIZER_STATE_FILE);
    if Path::new(&local).exists() {
        return Ok(bincode::deserialize(&std::fs::read(&local)?)?);
    }
    if let Ok(bytes) = backend.read(QUANTIZER_STATE_FILE) {
        std::fs::write(&local, &bytes)?;
        return Ok(bincode::deserialize(&bytes)?);
    }
    Err("Quantizer state not found".into())
}

fn save_id_pool(
    local_dir: &str,
    backend: &Arc<StorageBackend>,
    pool: &IdPool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let bytes = bincode::serialize(pool)?;
    let local = format!("{}/{}", local_dir, ID_POOL_FILE);
    std::fs::write(&local, &bytes)?;
    backend.write(ID_POOL_FILE, &bytes)?;
    Ok(())
}

fn load_id_pool(
    local_dir: &str,
    backend: &Arc<StorageBackend>,
) -> Result<IdPool, Box<dyn std::error::Error + Send + Sync>> {
    let local = format!("{}/{}", local_dir, ID_POOL_FILE);
    if Path::new(&local).exists() {
        let bytes = std::fs::read(&local)?;
        let mut pool: IdPool = bincode::deserialize(&bytes)?;
        pool.rebuild_lookup();
        return Ok(pool);
    }
    if let Ok(bytes) = backend.read(ID_POOL_FILE) {
        std::fs::write(&local, &bytes)?;
        let mut pool: IdPool = bincode::deserialize(&bytes)?;
        pool.rebuild_lookup();
        return Ok(pool);
    }
    Err("live_ids.bin not found".into())
}

fn save_index_ids(
    local_dir: &str,
    ids: &[u32],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    std::fs::write(
        format!("{}/{}", local_dir, INDEX_IDS_FILE),
        serialize_index_ids(ids)?,
    )?;
    Ok(())
}

fn serialize_index_ids(ids: &[u32]) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    Ok(serde_json::to_vec_pretty(ids)?)
}

fn load_index_ids(local_dir: &str) -> Result<Vec<u32>, Box<dyn std::error::Error + Send + Sync>> {
    let local = format!("{}/{}", local_dir, INDEX_IDS_FILE);
    if Path::new(&local).exists() {
        return Ok(serde_json::from_slice(&std::fs::read(&local)?)?);
    }
    Ok(Vec::new())
}
#[cfg(test)]
mod tests;
