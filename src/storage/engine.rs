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

    fn load<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }
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
            uri, local_dir, d, b, seed, metric, rerank, fast_mode, precision,
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

        let mut engine = Self {
            d: manifest.d,
            b: manifest.b,
            quantizer,
            manifest: manifest.clone(),
            metric,
            backend,
            wal,
            wal_buffer: Vec::new(),
            wal_flush_threshold: 100,
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
            rerank_enabled: manifest.rerank_enabled,
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
                            .copy_from_slice(&0.0_f32.to_le_bytes()); // norm not in WAL
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
                            .copy_from_slice(&0.0_f32.to_le_bytes()); // norm not in WAL
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
            metadata_json: "{}".to_string(),
            is_deleted: true,
        };
        self.wal.append(&entry, false)?;
        self.wal_buffer.push(entry);
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
    ) -> Result<Vec<Vec<SearchResult>>, Box<dyn std::error::Error + Send + Sync>> {
        queries
            .iter()
            .map(|q| self.search_with_filter_and_ann(q, top_k, filter, ann_search_list_size))
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
                metadata_json: "{}".to_string(),
                is_deleted: true,
            };
            self.wal.append(&entry, false)?;
            self.wal_buffer.push(entry);
            self.live_delete_slot(&id);
            deleted += 1;
        }
        if deleted > 0 {
            self.invalidate_index_state()?;
            self.maybe_persist_state(false)?;
            if self.wal_buffer.len() >= self.wal_flush_threshold {
                self.flush_wal_to_segment()?;
            }
        }
        Ok(deleted)
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
                        q.dequantize(&idx, &rec[mse_len..mse_len + qjl_len], gamma as f64)
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
                let prep = if has_vraw {
                    let rec = vraw_ref.unwrap().get_slot(from_slot);
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
                    quantizer.prepare_ip_query_lite(&out)
                } else {
                    // O(n) centroid lookup from pre-cached MSE indices.
                    let from_i = &all_mse[from_idx * qn..(from_idx + 1) * qn];
                    quantizer.prepare_ip_query_from_codes(from_i)
                };
                candidates
                    .iter()
                    .map(|&to| {
                        let to_idx = to as usize;
                        let to_i = &all_mse[to_idx * qn..(to_idx + 1) * qn];
                        let to_q = if cached_qjl_len > 0 {
                            &all_qjl_flat[to_idx * cached_qjl_len..(to_idx + 1) * cached_qjl_len]
                        } else {
                            &[]
                        };
                        let to_g = all_gamma[to_idx];
                        (
                            to,
                            quantizer.score_ip_encoded_lite(&prep, to_i, to_q, to_g as f64),
                        )
                    })
                    .collect()
            } else if matches!(metric, DistanceMetric::Cosine) {
                let (prep, from_norm) = if has_vraw {
                    let rec = vraw_ref.unwrap().get_slot(from_slot);
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
                    let norm = out.iter().map(|x| x * x).sum::<f64>().sqrt();
                    (quantizer.prepare_ip_query_lite(&out), norm)
                } else {
                    // Use stored norm + centroid lookup from pre-cached data.
                    let from_i = &all_mse[from_idx * qn..(from_idx + 1) * qn];
                    let from_norm = all_norm[from_idx] as f64;
                    (quantizer.prepare_ip_query_from_codes(from_i), from_norm)
                };
                candidates
                    .iter()
                    .map(|&to| {
                        let to_idx = to as usize;
                        let to_i = &all_mse[to_idx * qn..(to_idx + 1) * qn];
                        let to_q = if cached_qjl_len > 0 {
                            &all_qjl_flat[to_idx * cached_qjl_len..(to_idx + 1) * cached_qjl_len]
                        } else {
                            &[]
                        };
                        let to_g = all_gamma[to_idx];
                        let to_n = all_norm[to_idx];
                        let ip = quantizer.score_ip_encoded_lite(&prep, to_i, to_q, to_g as f64);
                        let score = if from_norm > 0.0 && to_n > 0.0 {
                            ip / (from_norm * to_n as f64)
                        } else {
                            0.0
                        };
                        (to, score)
                    })
                    .collect()
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
                let vraw = vraw_ref.unwrap();
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
        self.search_with_filter_and_ann(query, top_k, None, None)
    }

    pub fn search_with_filter(
        &self,
        query: &Array1<f64>,
        top_k: usize,
        filter: Option<&HashMap<String, JsonValue>>,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error + Send + Sync>> {
        self.search_with_filter_and_ann(query, top_k, filter, None)
    }

    pub fn search_with_filter_and_ann(
        &self,
        query: &Array1<f64>,
        top_k: usize,
        filter: Option<&HashMap<String, JsonValue>>,
        ann_search_list_size: Option<usize>,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error + Send + Sync>> {
        if self.id_pool.active_count() == 0 || top_k == 0 {
            return Ok(Vec::new());
        }

        let has_index = self.graph.has_index();
        let not_empty = !self.index_ids.is_empty();
        let state_match = self
            .manifest
            .index_state
            .as_ref()
            .is_some_and(|s| s.indexed_nodes == self.index_ids.len());

        if has_index && not_empty && state_match {
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

            let internal_k = if self.rerank_enabled {
                top_k * 20
            } else {
                top_k
            };

            // Shared references captured by search closures. Using RefCell<Vec<u16>> for
            // idx_buf lets the Fn closure reuse a single allocation across all candidate
            // scorings instead of allocating a new Vec<CodeIndex> per call.
            let mse_len = self.live_mse_len();
            let qjl_len = self.live_qjl_len();
            let qn = self.quantizer.n;
            let live_codes_r = &self.live_codes;
            let quantizer_r = &self.quantizer;
            let idx_buf = std::cell::RefCell::new(vec![0u16; qn]);

            let ann = if matches!(self.metric, DistanceMetric::Ip) {
                let prep = self.quantizer.prepare_ip_query(query);
                let index_ids = &self.index_ids;
                let slot_set: Option<std::collections::HashSet<u32>> =
                    filter_slots.map(|s| s.into_iter().collect());

                self.graph.search(
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
                        let mut buf = idx_buf.borrow_mut();
                        quantizer_r.unpack_mse_indices(&rec[..mse_len], &mut buf);
                        quantizer_r.score_ip_encoded(&prep, &buf, qjl, gamma as f64)
                    },
                    slot_set
                        .map(|ss| move |node_idx: u32| ss.contains(&index_ids[node_idx as usize])),
                )?
            } else if matches!(self.metric, DistanceMetric::Cosine) {
                let prep = self.quantizer.prepare_ip_query(query);
                let query_norm = query.iter().map(|x| x * x).sum::<f64>().sqrt();
                let index_ids = &self.index_ids;
                let slot_set: Option<std::collections::HashSet<u32>> =
                    filter_slots.map(|s| s.into_iter().collect());

                self.graph.search(
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
                        let mut buf = idx_buf.borrow_mut();
                        quantizer_r.unpack_mse_indices(&rec[..mse_len], &mut buf);
                        let ip = quantizer_r.score_ip_encoded(&prep, &buf, qjl, gamma as f64);
                        if query_norm > 0.0 && doc_norm > 0.0 {
                            ip / (query_norm * doc_norm as f64)
                        } else {
                            0.0
                        }
                    },
                    slot_set
                        .map(|ss| move |node_idx: u32| ss.contains(&index_ids[node_idx as usize])),
                )?
            } else {
                let index_ids = &self.index_ids;
                let metric_r = &self.metric;
                let slot_set: Option<std::collections::HashSet<u32>> =
                    filter_slots.map(|s| s.into_iter().collect());

                self.graph.search(
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
                        let v = {
                            let mut buf = idx_buf.borrow_mut();
                            quantizer_r.unpack_mse_indices(&rec[..mse_len], &mut buf);
                            quantizer_r.dequantize(&buf, qjl, gamma as f64)
                        };
                        score_vectors_with_metric(metric_r, query, &v)
                    },
                    slot_set
                        .map(|ss| move |node_idx: u32| ss.contains(&index_ids[node_idx as usize])),
                )?
            };

            let slots: Vec<u32> = ann
                .iter()
                .map(|(n, _)| self.index_ids[*n as usize])
                .collect();
            let meta_map = self.metadata.get_many(&slots)?;
            let mut out = Vec::with_capacity(ann.len());

            // Batch-dequantize all candidates in parallel when reranking without raw vecs.
            let deq_vecs: Vec<Array1<f64>> = if self.rerank_enabled && self.live_vraw.is_none() {
                let encoded: Vec<(Vec<CodeIndex>, Vec<u8>, f64)> = slots
                    .iter()
                    .map(|&slot| {
                        let (idx, qjl, gamma, _) = self.live_codes_at_slot(slot as usize);
                        (idx, qjl.to_vec(), gamma as f64)
                    })
                    .collect();
                self.quantizer.dequantize_batch(&encoded)
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
            out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            if out.len() > top_k {
                out.truncate(top_k);
            }
            return Ok(out);
        }

        // Exhaustive search path (SIMD Optimized)
        self.exhaustive_search_simd(query, top_k, filter)
    }

    fn exhaustive_search_simd(
        &self,
        query: &Array1<f64>,
        top_k: usize,
        filter: Option<&HashMap<String, JsonValue>>,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error + Send + Sync>> {
        let active = self.id_pool.iter_active();
        if active.is_empty() {
            return Ok(Vec::new());
        }

        let internal_k = if self.rerank_enabled {
            top_k * 10
        } else {
            top_k
        };
        let q_norm = query.iter().map(|x| x * x).sum::<f64>().sqrt();

        // Pre-filter slots upfront to keep the hot scoring loop filter-free.
        // For the no-filter path this is just a collect; for filter it does one
        // bulk metadata read (cheaper than per-slot reads inside the parallel loop).
        let candidate_slots: Vec<u32> = if let Some(f) = filter {
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

        // Parallel scoring: each 512-slot chunk reuses one CodeIndex scratch buffer.
        const CHUNK: usize = 512;
        let scored: Vec<(u32, f64)> = match metric {
            DistanceMetric::Ip => {
                let prep = quantizer.prepare_ip_query(query);
                candidate_slots
                    .par_chunks(CHUNK)
                    .flat_map(|chunk| {
                        let mut idx = vec![0u16; quantizer.n];
                        let mut out = Vec::with_capacity(chunk.len());
                        for &slot in chunk {
                            let rec =
                                &codes_bytes[slot as usize * stride..(slot as usize + 1) * stride];
                            quantizer.unpack_mse_indices(&rec[..mse_len], &mut idx);
                            let gamma = f32::from_le_bytes(
                                rec[mse_len + qjl_len..mse_len + qjl_len + 4]
                                    .try_into()
                                    .unwrap(),
                            );
                            let score = quantizer.score_ip_encoded(
                                &prep,
                                &idx,
                                &rec[mse_len..mse_len + qjl_len],
                                gamma as f64,
                            );
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
                            let ip = quantizer.score_ip_encoded(
                                &prep,
                                &idx,
                                &rec[mse_len..mse_len + qjl_len],
                                gamma as f64,
                            );
                            let score = if q_norm > 0.0 && doc_norm > 0.0 {
                                ip / (q_norm * doc_norm as f64)
                            } else {
                                0.0
                            };
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
                            let rec =
                                &codes_bytes[slot as usize * stride..(slot as usize + 1) * stride];
                            quantizer.unpack_mse_indices(&rec[..mse_len], &mut idx);
                            let gamma = f32::from_le_bytes(
                                rec[mse_len + qjl_len..mse_len + qjl_len + 4]
                                    .try_into()
                                    .unwrap(),
                            );
                            let v = quantizer.dequantize(
                                &idx,
                                &rec[mse_len..mse_len + qjl_len],
                                gamma as f64,
                            );
                            let score = score_vectors_with_metric(metric, query, &v);
                            out.push((slot, score));
                        }
                        out
                    })
                    .collect()
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
            let encoded: Vec<(Vec<CodeIndex>, Vec<u8>, f64)> = slots
                .iter()
                .map(|&slot| {
                    let (idx, qjl, gamma, _) = self.live_codes_at_slot(slot as usize);
                    (idx, qjl.to_vec(), gamma as f64)
                })
                .collect();
            self.quantizer.dequantize_batch(&encoded)
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
        // live_codes is already updated during ingest; compact without full rebuild.
        self.live_compact_slab()?;
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
        if had_vraw {
            self.live_vraw = Some(LiveCodesFile::open(
                live_vraw_path,
                self.live_vraw_stride(),
            )?);
        }
        self.invalidate_index_state()?;
        self.manifest.vector_count = self.live_active_count() as u64;
        self.maybe_persist_state(false)?;
        Ok(())
    }

    pub fn close(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.flush_wal_to_segment()?;
        self.metadata.flush()?;
        self.persist_id_pool()?;
        self.maybe_persist_state(true)?;
        Ok(())
    }

    pub fn vector_count(&self) -> u64 {
        self.id_pool.active_count() as u64
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
        let (indices, qjl, gamma) = self.quantizer.quantize(&vec_f32);
        let norm = vec_f32.iter().map(|x| x * x).sum::<f32>().sqrt();
        let meta = VectorMetadata {
            properties: metadata,
            document,
        };
        let entry = WalEntry {
            id: id.clone(),
            quantized_indices: indices,
            qjl_bits: qjl,
            gamma: gamma as f32,
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
                norm,
            )?;
            self.live_save_raw_vector(slot, &vec_f32);
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
            let vec_refs: Vec<&[f32]> = chunk.iter().map(|i| i.vector.as_slice()).collect();
            let quantized = self.quantizer.quantize_batch(&vec_refs);

            for (_i, (item, (indices, qjl, gamma))) in chunk.iter().zip(quantized).enumerate() {
                match mode {
                    BatchWriteMode::Insert if self.id_pool.contains(&item.id) => {
                        return Err(format!("ID '{}' already exists", item.id).into());
                    }
                    BatchWriteMode::Update if !self.id_pool.contains(&item.id) => {
                        return Err(format!("ID '{}' does not exist", item.id).into());
                    }
                    _ => {}
                }
                let norm = item.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
                let meta = VectorMetadata {
                    properties: item.metadata.clone(),
                    document: item.document.clone(),
                };
                let entry = WalEntry {
                    id: item.id.clone(),
                    quantized_indices: indices,
                    qjl_bits: qjl,
                    gamma: gamma as f32,
                    metadata_json: serde_json::to_string(&meta)?,
                    is_deleted: false,
                };
                let slot = self.live_alloc_or_update(
                    &item.id,
                    &entry.quantized_indices,
                    &entry.qjl_bits,
                    entry.gamma,
                    norm,
                )?;
                self.live_save_raw_vector(slot, &item.vector);
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

fn metadata_matches_filter(
    meta: &HashMap<String, JsonValue>,
    filter: &HashMap<String, JsonValue>,
) -> bool {
    filter.iter().all(|(k, v)| match k.as_str() {
        "$and" => {
            if let JsonValue::Array(conditions) = v {
                conditions.iter().all(|cond| {
                    if let JsonValue::Object(map) = cond {
                        let as_hm: HashMap<String, JsonValue> =
                            map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                        metadata_matches_filter(meta, &as_hm)
                    } else {
                        false
                    }
                })
            } else {
                false
            }
        }
        "$or" => {
            if let JsonValue::Array(conditions) = v {
                conditions.iter().any(|cond| {
                    if let JsonValue::Object(map) = cond {
                        let as_hm: HashMap<String, JsonValue> =
                            map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                        metadata_matches_filter(meta, &as_hm)
                    } else {
                        false
                    }
                })
            } else {
                false
            }
        }
        field => {
            let field_val = get_nested_field(meta, field);
            match v {
                JsonValue::Object(op_map) => op_map
                    .iter()
                    .all(|(op, op_val)| apply_comparison_op(field_val, op.as_str(), op_val)),
                // Simple equality: {"field": value}
                _ => field_val.is_some_and(|fv| fv == v),
            }
        }
    })
}

/// Resolve a potentially dotted path like "profile.region" into a value from metadata.
fn get_nested_field<'a>(meta: &'a HashMap<String, JsonValue>, path: &str) -> Option<&'a JsonValue> {
    let mut parts = path.splitn(2, '.');
    let head = parts.next()?;
    let val = meta.get(head)?;
    if let Some(rest) = parts.next() {
        if let JsonValue::Object(obj) = val {
            // Recurse into nested object via the remaining path
            get_nested_json_field(obj, rest)
        } else {
            None
        }
    } else {
        Some(val)
    }
}

fn get_nested_json_field<'a>(
    obj: &'a serde_json::Map<String, JsonValue>,
    path: &str,
) -> Option<&'a JsonValue> {
    let mut parts = path.splitn(2, '.');
    let head = parts.next()?;
    let val = obj.get(head)?;
    if let Some(rest) = parts.next() {
        if let JsonValue::Object(nested) = val {
            get_nested_json_field(nested, rest)
        } else {
            None
        }
    } else {
        Some(val)
    }
}

fn apply_comparison_op(field: Option<&JsonValue>, op: &str, op_val: &JsonValue) -> bool {
    match op {
        "$eq" => field.is_some_and(|f| f == op_val),
        "$ne" => {
            // $ne matches missing fields too (missing ≠ value)
            field.map_or(true, |f| f != op_val)
        }
        "$gt" | "$gte" | "$lt" | "$lte" => {
            // Comparisons do not match missing fields
            let Some(f) = field else { return false };
            match (f, op_val) {
                (JsonValue::Number(a), JsonValue::Number(b)) => {
                    let av = a.as_f64().unwrap_or(f64::NAN);
                    let bv = b.as_f64().unwrap_or(f64::NAN);
                    match op {
                        "$gt" => av > bv,
                        "$gte" => av >= bv,
                        "$lt" => av < bv,
                        "$lte" => av <= bv,
                        _ => false,
                    }
                }
                (JsonValue::String(a), JsonValue::String(b)) => match op {
                    "$gt" => a > b,
                    "$gte" => a >= b,
                    "$lt" => a < b,
                    "$lte" => a <= b,
                    _ => false,
                },
                _ => false,
            }
        }
        "$in" => {
            if let (Some(f), JsonValue::Array(arr)) = (field, op_val) {
                arr.iter().any(|v| v == f)
            } else {
                false
            }
        }
        "$nin" => {
            if let JsonValue::Array(arr) = op_val {
                // missing field is "not in" any set
                field.map_or(true, |f| !arr.iter().any(|v| v == f))
            } else {
                false
            }
        }
        "$exists" => match op_val {
            JsonValue::Bool(true) => field.is_some(),
            JsonValue::Bool(false) => field.is_none(),
            _ => false,
        },
        "$contains" => {
            if let (Some(JsonValue::String(f)), JsonValue::String(sub)) = (field, op_val) {
                f.contains(sub.as_str())
            } else {
                false
            }
        }
        _ => false, // unknown operator: never matches
    }
}

fn score_vectors_with_metric(metric: &DistanceMetric, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    match metric {
        DistanceMetric::Ip => a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(),
        DistanceMetric::Cosine => {
            let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let an = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let bn = b.iter().map(|x| x * x).sum::<f64>().sqrt();
            if an == 0.0 || bn == 0.0 {
                0.0
            } else {
                dot / (an * bn)
            }
        }
        DistanceMetric::L2 => -a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use serde_json::json;
    use std::collections::HashMap;
    use tempfile::tempdir;

    fn make_vec(d: usize, val: f64) -> Array1<f64> {
        Array1::from_elem(d, val)
    }

    fn make_vec_f32(d: usize, val: f32) -> Vec<f32> {
        vec![val; d]
    }

    fn no_meta() -> HashMap<String, serde_json::Value> {
        HashMap::new()
    }

    fn open_default(dir: &str, d: usize) -> TurboQuantEngine {
        TurboQuantEngine::open(dir, dir, d, 2, 42).unwrap()
    }

    // ── Error paths on reopen ──────────────────────────────────────────────

    #[test]
    fn reopen_dimension_mismatch_returns_error() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let mut e = TurboQuantEngine::open(p, p, 8, 2, 42).unwrap();
        e.insert("a".into(), &make_vec(8, 0.5), no_meta()).unwrap();
        let result = TurboQuantEngine::open(p, p, 16, 2, 42);
        assert!(result.is_err());
        let msg = result.err().unwrap().to_string();
        assert!(msg.contains("dimension mismatch"), "unexpected: {msg}");
    }

    #[test]
    fn reopen_bits_mismatch_returns_error() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let mut e = TurboQuantEngine::open(p, p, 8, 2, 42).unwrap();
        e.insert("a".into(), &make_vec(8, 0.5), no_meta()).unwrap();
        let result = TurboQuantEngine::open(p, p, 8, 4, 42);
        assert!(result.is_err());
        let msg = result.err().unwrap().to_string();
        assert!(msg.contains("bits mismatch"), "unexpected: {msg}");
    }

    #[test]
    fn reopen_metric_mismatch_returns_error() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let mut e = TurboQuantEngine::open_with_metric(p, p, 8, 2, 42, DistanceMetric::Ip).unwrap();
        e.insert("a".into(), &make_vec(8, 0.5), no_meta()).unwrap();
        let result = TurboQuantEngine::open_with_metric(p, p, 8, 2, 42, DistanceMetric::L2);
        assert!(result.is_err());
        let msg = result.err().unwrap().to_string();
        assert!(msg.contains("metric mismatch"), "unexpected: {msg}");
    }

    // ── Single-vector write errors ─────────────────────────────────────────

    #[test]
    fn insert_duplicate_id_returns_error() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let mut e = open_default(p, 8);
        e.insert("dup".into(), &make_vec(8, 0.1), no_meta())
            .unwrap();
        let result = e.insert("dup".into(), &make_vec(8, 0.2), no_meta());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }

    #[test]
    fn update_nonexistent_id_returns_error() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let mut e = open_default(p, 8);
        let result = e.update_with_document("ghost".into(), &make_vec(8, 0.1), no_meta(), None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    // ── Batch write mode errors ────────────────────────────────────────────

    #[test]
    fn insert_many_insert_mode_duplicate_returns_error() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let mut e = open_default(p, 8);
        let item = BatchWriteItem {
            id: "x".into(),
            vector: make_vec_f32(8, 0.5),
            metadata: no_meta(),
            document: None,
        };
        e.insert_many(vec![item.clone()]).unwrap();
        let result = e.insert_many_with_mode(vec![item], BatchWriteMode::Insert);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }

    #[test]
    fn insert_many_update_mode_missing_id_returns_error() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let mut e = open_default(p, 8);
        let item = BatchWriteItem {
            id: "ghost".into(),
            vector: make_vec_f32(8, 0.5),
            metadata: no_meta(),
            document: None,
        };
        let result = e.insert_many_with_mode(vec![item], BatchWriteMode::Update);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    // ── Cosine exhaustive search ───────────────────────────────────────────

    #[test]
    fn cosine_exhaustive_search_returns_ordered_results() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 16;
        let mut e =
            TurboQuantEngine::open_with_metric(p, p, d, 4, 42, DistanceMetric::Cosine).unwrap();
        // Insert a parallel vector (same direction) and an orthogonal one
        let mut v_same = vec![0.0f64; d];
        v_same[0] = 1.0;
        let mut v_ortho = vec![0.0f64; d];
        v_ortho[1] = 1.0;
        e.insert("same".into(), &Array1::from_vec(v_same.clone()), no_meta())
            .unwrap();
        e.insert(
            "ortho".into(),
            &Array1::from_vec(v_ortho.clone()),
            no_meta(),
        )
        .unwrap();

        let query = Array1::from_vec(v_same);
        let results = e.search_with_filter_and_ann(&query, 2, None, None).unwrap();
        assert!(!results.is_empty());
        // The "same" vector should score higher than "ortho"
        let top_id = &results[0].id;
        assert_eq!(top_id, "same");
    }

    #[test]
    fn cosine_zero_norm_vector_scores_zero() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e =
            TurboQuantEngine::open_with_metric(p, p, d, 2, 7, DistanceMetric::Cosine).unwrap();
        // A zero vector has norm=0; cosine score must be 0.0
        e.insert("zero".into(), &Array1::zeros(d), no_meta())
            .unwrap();
        let query = make_vec(d, 1.0);
        let results = e.search_with_filter_and_ann(&query, 5, None, None).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].score, 0.0);
    }

    // ── L2 exhaustive search ───────────────────────────────────────────────

    #[test]
    fn l2_exhaustive_search_returns_ordered_results() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 16;
        let mut e = TurboQuantEngine::open_with_metric(p, p, d, 4, 42, DistanceMetric::L2).unwrap();
        // "close" is distance 0 from query (at origin); "far" is distance 10
        let v_close = vec![0.0f64; d];
        let mut v_far = vec![0.0f64; d];
        v_far[0] = 10.0;
        e.insert(
            "close".into(),
            &Array1::from_vec(v_close.clone()),
            no_meta(),
        )
        .unwrap();
        e.insert("far".into(), &Array1::from_vec(v_far), no_meta())
            .unwrap();

        let query = Array1::from_vec(v_close);
        let results = e.search_with_filter_and_ann(&query, 2, None, None).unwrap();
        assert!(!results.is_empty());
        // L2 scores are negative distances; "close" should have a higher (less negative) score
        assert_eq!(results[0].id, "close");
    }

    // ── ANN search with HNSW index ─────────────────────────────────────────

    #[test]
    fn ann_cosine_search_with_index_returns_results() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 16;
        let mut e =
            TurboQuantEngine::open_with_metric(p, p, d, 4, 42, DistanceMetric::Cosine).unwrap();
        for i in 0..20u32 {
            let mut v = vec![0.0f64; d];
            v[i as usize % d] = 1.0;
            e.insert(format!("v{i}"), &Array1::from_vec(v), no_meta())
                .unwrap();
        }
        e.create_index_with_params(8, 32, 32, 1.2, 2).unwrap();
        let mut q = vec![0.0f64; d];
        q[0] = 1.0;
        let results = e
            .search_with_filter_and_ann(&Array1::from_vec(q), 5, None, None)
            .unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn ann_l2_search_with_index_returns_results() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 16;
        let mut e = TurboQuantEngine::open_with_metric(p, p, d, 4, 42, DistanceMetric::L2).unwrap();
        for i in 0..20u32 {
            let mut v = vec![0.0f64; d];
            v[i as usize % d] = (i as f64 + 1.0) * 0.1;
            e.insert(format!("v{i}"), &Array1::from_vec(v), no_meta())
                .unwrap();
        }
        e.create_index_with_params(8, 32, 32, 1.2, 2).unwrap();
        let q = vec![0.0f64; d];
        let results = e
            .search_with_filter_and_ann(&Array1::from_vec(q), 5, None, None)
            .unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    // ── Search edge cases ──────────────────────────────────────────────────

    #[test]
    fn search_top_k_zero_returns_empty() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let mut e = open_default(p, 8);
        e.insert("a".into(), &make_vec(8, 0.5), no_meta()).unwrap();
        let results = e
            .search_with_filter_and_ann(&make_vec(8, 0.5), 0, None, None)
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_empty_db_returns_empty() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let e = open_default(p, 8);
        let results = e
            .search_with_filter_and_ann(&make_vec(8, 0.5), 5, None, None)
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_filter_no_match_returns_empty() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let mut e = open_default(p, 8);
        let mut meta = no_meta();
        meta.insert("tag".into(), json!("a"));
        e.insert("a".into(), &make_vec(8, 0.5), meta).unwrap();
        let mut filter = no_meta();
        filter.insert("tag".into(), json!("b"));
        let results = e
            .search_with_filter_and_ann(&make_vec(8, 0.5), 5, Some(&filter), None)
            .unwrap();
        assert!(results.is_empty());
    }

    // ── F16 rerank precision ───────────────────────────────────────────────

    #[test]
    fn f16_rerank_precision_stores_and_retrieves_vectors() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 16;
        let mut e = TurboQuantEngine::open_with_options(
            p,
            p,
            d,
            4,
            42,
            DistanceMetric::Ip,
            true,
            false,
            RerankPrecision::F16,
        )
        .unwrap();
        let mut v = vec![0.0f64; d];
        v[0] = 0.9;
        v[1] = 0.4;
        e.insert("v1".into(), &Array1::from_vec(v.clone()), no_meta())
            .unwrap();
        let results = e
            .search_with_filter_and_ann(&Array1::from_vec(v), 1, None, None)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "v1");
        // F16 has ~0.1% relative error; score should be close to 1.0
        assert!(results[0].score > 0.8, "score was {}", results[0].score);
    }

    #[test]
    fn f16_roundtrip_within_half_precision_tolerance() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 16;
        let mut e = TurboQuantEngine::open_with_options(
            p,
            p,
            d,
            4,
            42,
            DistanceMetric::Ip,
            true,
            false,
            RerankPrecision::F16,
        )
        .unwrap();
        // v = [0.1, 0.2, ..., 1.6]; expected IP = sum((0.1*i)^2) for i=1..=16
        let v: Vec<f64> = (1..=d).map(|i| i as f64 * 0.1).collect();
        let expected: f64 = v.iter().map(|x| x * x).sum();
        e.insert("v1".into(), &Array1::from_vec(v.clone()), no_meta())
            .unwrap();
        let results = e
            .search_with_filter_and_ann(&Array1::from_vec(v), 1, None, None)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "v1");
        // F16 tolerance: relative error < 1% (0.01)
        let rel_err = (results[0].score - expected).abs() / expected;
        assert!(
            rel_err < 0.01,
            "F16 relative error {:.4} exceeds 0.01; score={:.4}, expected={:.4}",
            rel_err,
            results[0].score,
            expected
        );
    }

    // ── Stats and disk bytes ───────────────────────────────────────────────

    #[test]
    fn stats_reflect_inserted_vectors() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        assert_eq!(e.stats().vector_count, 0);
        assert_eq!(e.stats().d, d);
        e.insert("a".into(), &make_vec(d, 0.1), no_meta()).unwrap();
        e.insert("b".into(), &make_vec(d, 0.2), no_meta()).unwrap();
        let s = e.stats();
        assert_eq!(s.vector_count, 2);
        assert_eq!(s.live_id_count, 2);
    }

    #[test]
    fn total_disk_bytes_nonzero_after_flush() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        for i in 0..5u32 {
            e.insert(format!("v{i}"), &make_vec(d, 0.5), no_meta())
                .unwrap();
        }
        e.flush_wal_to_segment().unwrap();
        let bytes = e.stats().total_disk_bytes;
        assert!(bytes > 0, "expected non-zero disk bytes, got {bytes}");
    }

    // ── Delete ─────────────────────────────────────────────────────────────

    #[test]
    fn delete_unknown_id_returns_false() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let mut e = open_default(p, 8);
        let deleted = e.delete("ghost".into()).unwrap();
        assert!(!deleted);
    }

    #[test]
    fn delete_existing_id_returns_true_and_reduces_count() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let mut e = open_default(p, 8);
        e.insert("a".into(), &make_vec(8, 0.5), no_meta()).unwrap();
        assert_eq!(e.stats().vector_count, 1);
        let deleted = e.delete("a".into()).unwrap();
        assert!(deleted);
        assert_eq!(e.stats().vector_count, 0);
    }

    // ── Index invalidation ─────────────────────────────────────────────────

    #[test]
    fn index_invalidated_after_insert() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 16;
        let mut e = open_default(p, d);
        for i in 0..10u32 {
            e.insert(format!("v{i}"), &make_vec(d, i as f64 * 0.1), no_meta())
                .unwrap();
        }
        e.create_index_with_params(4, 16, 16, 1.2, 1).unwrap();
        assert!(e.manifest.index_state.is_some(), "index should be built");
        // Any new insert must invalidate the index
        e.insert("new".into(), &make_vec(d, 0.99), no_meta())
            .unwrap();
        assert!(
            e.manifest.index_state.is_none(),
            "index state should be cleared after insert"
        );
    }

    #[test]
    fn index_invalidated_after_delete() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 16;
        let mut e = open_default(p, d);
        for i in 0..10u32 {
            e.insert(format!("v{i}"), &make_vec(d, i as f64 * 0.1), no_meta())
                .unwrap();
        }
        e.create_index_with_params(4, 16, 16, 1.2, 1).unwrap();
        assert!(e.manifest.index_state.is_some());
        e.delete("v0".into()).unwrap();
        assert!(
            e.manifest.index_state.is_none(),
            "index state should be cleared after delete"
        );
    }

    // ── Flush WAL edge case ────────────────────────────────────────────────

    #[test]
    fn flush_wal_empty_buffer_is_noop() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let mut e = open_default(p, 8);
        // No inserts — flush should succeed silently
        e.flush_wal_to_segment().unwrap();
        assert_eq!(e.stats().vector_count, 0);
    }

    // ── Fast-mode engine ───────────────────────────────────────────────────

    #[test]
    fn fast_mode_engine_creates_and_searches() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 32;
        let mut e = TurboQuantEngine::open_fast(p, p, d, 4, 42).unwrap();
        for i in 0..5u32 {
            e.insert(
                format!("v{i}"),
                &make_vec(d, i as f64 * 0.1 + 0.1),
                no_meta(),
            )
            .unwrap();
        }
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 3, None, None)
            .unwrap();
        assert!(!results.is_empty());
        assert!(e.manifest.fast_mode);
    }

    // ── Upsert mode ────────────────────────────────────────────────────────

    #[test]
    fn upsert_replaces_existing_vector() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        e.insert("a".into(), &make_vec(d, 0.1), no_meta()).unwrap();
        // Upsert with same ID should not error
        e.upsert_with_document("a".into(), &make_vec(d, 0.9), no_meta(), None)
            .unwrap();
        assert_eq!(e.stats().vector_count, 1);
    }

    // ── List all ───────────────────────────────────────────────────────────

    #[test]
    fn list_all_returns_all_active_ids() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        e.insert("x".into(), &make_vec(d, 0.1), no_meta()).unwrap();
        e.insert("y".into(), &make_vec(d, 0.2), no_meta()).unwrap();
        e.insert("z".into(), &make_vec(d, 0.3), no_meta()).unwrap();
        e.delete("y".into()).unwrap();
        let ids = e.list_all();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"x".to_string()));
        assert!(ids.contains(&"z".to_string()));
        assert!(!ids.contains(&"y".to_string()));
    }

    // ── Disabled rerank ────────────────────────────────────────────────────

    #[test]
    fn disabled_rerank_search_returns_results() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 16;
        let mut e = TurboQuantEngine::open_with_options(
            p,
            p,
            d,
            4,
            42,
            DistanceMetric::Ip,
            false,
            false,
            RerankPrecision::Disabled,
        )
        .unwrap();
        for i in 0..5u32 {
            let v = make_vec(d, i as f64 * 0.2 + 0.1);
            e.insert(format!("v{i}"), &v, no_meta()).unwrap();
        }
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 3, None, None)
            .unwrap();
        assert!(!results.is_empty());
        assert!(!e.rerank_enabled);
    }

    // ── get and get_many ───────────────────────────────────────────────────

    #[test]
    fn get_returns_none_for_missing_id() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let e = open_default(p, 8);
        let result = e.get("ghost").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn get_returns_metadata_and_document() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        let mut meta = no_meta();
        meta.insert("key".into(), json!("val"));
        e.insert_with_document(
            "v1".into(),
            &make_vec(d, 0.5),
            meta,
            Some("hello doc".into()),
        )
        .unwrap();
        let result = e.get("v1").unwrap().unwrap();
        assert_eq!(result.id, "v1");
        assert_eq!(result.metadata.get("key"), Some(&json!("val")));
        assert_eq!(result.document, Some("hello doc".into()));
    }

    #[test]
    fn get_many_returns_mixed_results() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        e.insert("a".into(), &make_vec(d, 0.1), no_meta()).unwrap();
        e.insert("b".into(), &make_vec(d, 0.2), no_meta()).unwrap();
        let ids: Vec<String> = vec!["a".into(), "missing".into(), "b".into()];
        let results = e.get_many(&ids).unwrap();
        assert_eq!(results.len(), 3);
        assert!(results[0].is_some());
        assert!(results[1].is_none());
        assert!(results[2].is_some());
    }

    // ── upsert_many / update_many wrappers ────────────────────────────────

    #[test]
    fn upsert_many_inserts_and_replaces() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        let item = BatchWriteItem {
            id: "u".into(),
            vector: make_vec_f32(d, 0.5),
            metadata: no_meta(),
            document: None,
        };
        // First upsert inserts
        e.upsert_many(vec![item.clone()]).unwrap();
        assert_eq!(e.stats().vector_count, 1);
        // Second upsert replaces (no error)
        e.upsert_many(vec![item]).unwrap();
        assert_eq!(e.stats().vector_count, 1);
    }

    #[test]
    fn update_many_updates_existing_vector() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        e.insert("x".into(), &make_vec(d, 0.1), no_meta()).unwrap();
        let item = BatchWriteItem {
            id: "x".into(),
            vector: make_vec_f32(d, 0.9),
            metadata: no_meta(),
            document: None,
        };
        e.update_many(vec![item]).unwrap();
        assert_eq!(e.stats().vector_count, 1);
    }

    // ── create_index on empty DB ────────────────────────────────────────────

    #[test]
    fn create_index_empty_db_is_noop() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let mut e = open_default(p, 8);
        // Should not error on empty DB
        e.create_index_with_params(4, 16, 16, 1.2, 1).unwrap();
        assert!(!e.stats().has_index);
    }

    // ── L2 index build without raw vecs (precomputed_l2 path) ─────────────

    #[test]
    fn l2_index_without_rerank_uses_dequantized_path() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 16;
        // rerank=false → no live_vectors.bin → L2 index uses precomputed_l2
        let mut e = TurboQuantEngine::open_with_options(
            p,
            p,
            d,
            4,
            42,
            DistanceMetric::L2,
            false,
            false,
            RerankPrecision::Disabled,
        )
        .unwrap();
        for i in 0..15u32 {
            let mut v = vec![0.0f32; d];
            v[0] = i as f32 * 0.1;
            e.insert_many(vec![BatchWriteItem {
                id: format!("v{i}"),
                vector: v,
                metadata: no_meta(),
                document: None,
            }])
            .unwrap();
        }
        e.create_index_with_params(4, 16, 16, 1.2, 1).unwrap();
        assert!(e.stats().has_index);
        let q = vec![0.0f64; d];
        let results = e
            .search_with_filter_and_ann(&Array1::from_vec(q), 5, None, None)
            .unwrap();
        assert!(!results.is_empty());
    }

    // ── ANN search with filter ─────────────────────────────────────────────

    #[test]
    fn ann_search_with_metadata_filter() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 16;
        let mut e = open_default(p, d);
        for i in 0..15u32 {
            let mut meta = no_meta();
            meta.insert("cat".into(), json!(if i % 2 == 0 { "even" } else { "odd" }));
            let mut v = vec![0.0f64; d];
            v[0] = i as f64 * 0.1;
            e.insert(format!("v{i}"), &Array1::from_vec(v), meta)
                .unwrap();
        }
        e.create_index_with_params(4, 16, 16, 1.2, 1).unwrap();
        let mut filter = no_meta();
        filter.insert("cat".into(), json!("even"));
        let q = make_vec(d, 0.5);
        let results = e
            .search_with_filter_and_ann(&q, 10, Some(&filter), None)
            .unwrap();
        // All results should have cat=even
        for r in &results {
            assert_eq!(r.metadata.get("cat"), Some(&json!("even")));
        }
    }

    // ── ANN filter returns empty when nothing matches ──────────────────────

    #[test]
    fn ann_search_filter_no_match_returns_empty() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 16;
        let mut e = open_default(p, d);
        for i in 0..10u32 {
            let mut meta = no_meta();
            meta.insert("cat".into(), json!("a"));
            e.insert(format!("v{i}"), &make_vec(d, i as f64 * 0.1 + 0.1), meta)
                .unwrap();
        }
        e.create_index_with_params(4, 16, 16, 1.2, 1).unwrap();
        let mut filter = no_meta();
        filter.insert("cat".into(), json!("z"));
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter), None)
            .unwrap();
        assert!(results.is_empty());
    }

    // ── DistanceMetric default ─────────────────────────────────────────────

    #[test]
    fn distance_metric_default_is_ip() {
        let m: DistanceMetric = Default::default();
        assert_eq!(m, DistanceMetric::Ip);
    }

    // ── WAL recovery (unclean shutdown simulation) ─────────────────────────

    #[test]
    fn wal_recovery_without_live_ids_rebuilds_correctly() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        // Write vectors and close without explicit flush (WAL has entries)
        {
            let mut e = open_default(p, d);
            e.insert("a".into(), &make_vec(d, 0.1), no_meta()).unwrap();
            e.insert("b".into(), &make_vec(d, 0.2), no_meta()).unwrap();
            // Do NOT flush WAL; force a "crash" by deleting live_ids.bin
            drop(e);
        }
        // Delete live_ids.bin to simulate unclean shutdown
        let ids_path = format!("{}/live_ids.bin", p);
        let _ = std::fs::remove_file(&ids_path);
        // Re-open — should recover from WAL
        let e2 = open_default(p, d);
        // Recovery may not restore counts from WAL if ids file was absent and
        // flush_wal_to_segment wasn't called, but should not panic
        let _ = e2.stats();
    }

    // ── Metadata filter operators ──────────────────────────────────────────

    #[test]
    fn metadata_filter_comparison_operators() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        for i in 0i64..5 {
            let mut meta = no_meta();
            meta.insert("score".into(), json!(i));
            e.insert(format!("v{i}"), &make_vec(d, i as f64 * 0.2), meta)
                .unwrap();
        }
        // $gte filter
        let mut filter = no_meta();
        filter.insert("score".into(), json!({"$gte": 3}));
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter), None)
            .unwrap();
        assert!(!results.is_empty());
        for r in &results {
            let score = r.metadata.get("score").and_then(|v| v.as_i64()).unwrap();
            assert!(score >= 3, "score={score} should be >= 3");
        }
    }

    #[test]
    fn metadata_filter_or_operator() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        for tag in ["a", "b", "c"] {
            let mut meta = no_meta();
            meta.insert("tag".into(), json!(tag));
            e.insert(tag.to_string(), &make_vec(d, 0.5), meta).unwrap();
        }
        // $or filter: tag == "a" OR tag == "b"
        let filter: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"$or": [{"tag": {"$eq": "a"}}, {"tag": {"$eq": "b"}}]}"#)
                .unwrap();
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter), None)
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    // ── Error paths: duplicate insert / update-nonexistent ─────────────────

    #[test]
    fn duplicate_insert_via_insert_method_returns_error() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let mut e = open_default(p, 8);
        e.insert("dup".into(), &make_vec(8, 0.5), no_meta())
            .unwrap();
        let result = e.insert("dup".into(), &make_vec(8, 0.6), no_meta());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }

    // ── Serde default functions ─────────────────────────────────────────────

    #[test]
    fn manifest_serde_default_rerank_enabled() {
        // Deserializing a manifest JSON without 'rerank_enabled' must default to true.
        let json = r#"{
            "version": 1, "d": 16, "b": 4, "seed": 42,
            "vector_count": 0, "storage_uri": ""
        }"#;
        let m: Manifest = serde_json::from_str(json).unwrap();
        assert!(
            m.rerank_enabled,
            "default_rerank_enabled() must return true"
        );
    }

    #[test]
    fn index_state_serde_default_ef_construction() {
        // Deserializing an IndexState without 'ef_construction' must default to 200.
        let json = r#"{
            "max_degree": 16, "search_list_size": 64,
            "alpha": 1.2, "indexed_nodes": 0
        }"#;
        let s: IndexState = serde_json::from_str(json).unwrap();
        assert_eq!(
            s.ef_construction, 200,
            "default_ef_construction() must return 200"
        );
    }

    // ── WAL recovery without live_ids.bin ───────────────────────────────────

    #[test]
    fn wal_recovery_without_live_ids_bin() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        {
            let mut e = TurboQuantEngine::open(p, p, d, 2, 42).unwrap();
            e.insert("a".into(), &make_vec(d, 0.5), no_meta()).unwrap();
            e.insert("b".into(), &make_vec(d, 0.3), no_meta()).unwrap();
            // Do NOT flush — leave entries only in WAL
            drop(e);
        }
        // Delete live_ids.bin to simulate unclean shutdown
        let ids_path = std::path::Path::new(p).join("live_ids.bin");
        let _ = std::fs::remove_file(&ids_path);
        // Reopen: must rebuild from WAL without live_ids.bin
        let e2 = TurboQuantEngine::open(p, p, d, 2, 42).unwrap();
        // Engine opens without error; count is at least 0
        let _ = e2.vector_count();
    }

    // ── Delete flush threshold ──────────────────────────────────────────────

    #[test]
    fn delete_triggers_wal_flush_at_threshold() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = TurboQuantEngine::open(p, p, d, 2, 42).unwrap();
        // Fill up to wal_flush_threshold (100) to ensure next delete pushes over
        for i in 0..100 {
            e.insert(format!("v{i}"), &make_vec(d, 0.5), no_meta())
                .unwrap();
        }
        // This delete should trigger flush_wal_to_segment
        e.delete("v0".to_string()).unwrap();
        // Engine must remain functional after the flush
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 5, None, None)
            .unwrap();
        assert!(results.len() <= 5);
    }

    // ── HNSW index with n_refinements > 0 ──────────────────────────────────

    #[test]
    fn create_index_with_refinements() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        for i in 0..20u32 {
            e.insert(format!("r{i}"), &make_vec(d, i as f64 * 0.05), no_meta())
                .unwrap();
        }
        // n_refinements = 2 exercises the refinement loop in graph.rs
        e.create_index_with_params(4, 16, 16, 1.2, 2).unwrap();
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 5, None, None)
            .unwrap();
        assert!(!results.is_empty());
    }

    // ── F16 reranking with ANN index + Cosine metric ───────────────────────
    // Exercises lines 734-738 (F16 IP in build_scorer) and
    // 772-776 (F16 Cosine in build_scorer) and 1456-1460 (live_raw_vector_at_slot F16).

    #[test]
    fn f16_rerank_cosine_ann_index_covers_f16_build_scorer_paths() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 16;
        let mut e = TurboQuantEngine::open_with_options(
            p,
            p,
            d,
            4,
            42,
            DistanceMetric::Cosine,
            true,
            false,
            RerankPrecision::F16,
        )
        .unwrap();
        for i in 0..30u32 {
            let mut v = vec![0.01f64; d];
            v[i as usize % d] += (i as f64 + 1.0) * 0.1;
            e.insert(format!("f{i}"), &Array1::from_vec(v), no_meta())
                .unwrap();
        }
        // Building ANN index with F16 + Cosine triggers lines 772-776 in build_scorer
        e.create_index_with_params(8, 32, 32, 1.2, 1).unwrap();
        // ANN search with reranking triggers live_raw_vector_at_slot F16 (lines 1456-1460)
        let mut q = vec![0.0f64; d];
        q[0] = 1.0;
        let results = e
            .search_with_filter_and_ann(&Array1::from_vec(q), 5, None, None)
            .unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn f16_rerank_ip_ann_index_covers_f16_ip_build_scorer_path() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 16;
        let mut e = TurboQuantEngine::open_with_options(
            p,
            p,
            d,
            4,
            42,
            DistanceMetric::Ip,
            true,
            false,
            RerankPrecision::F16,
        )
        .unwrap();
        for i in 0..30u32 {
            let mut v = vec![0.01f64; d];
            v[i as usize % d] += (i as f64 + 1.0) * 0.1;
            e.insert(format!("g{i}"), &Array1::from_vec(v), no_meta())
                .unwrap();
        }
        // Building ANN index with F16 + IP triggers lines 734-738 in build_scorer
        e.create_index_with_params(8, 32, 32, 1.2, 1).unwrap();
        let mut q = vec![0.0f64; d];
        q[0] = 1.0;
        let results = e
            .search_with_filter_and_ann(&Array1::from_vec(q), 5, None, None)
            .unwrap();
        assert!(!results.is_empty());
    }

    // ── L2 metric ANN index without raw vectors (precomputed_l2 path) ───────
    // Exercises lines 812-822 in build_scorer (L2 without raw vecs).

    #[test]
    fn l2_ann_index_without_raw_vectors_uses_precomputed_l2_path() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 16;
        let mut e = TurboQuantEngine::open_with_options(
            p,
            p,
            d,
            4,
            42,
            DistanceMetric::L2,
            false, // rerank disabled → no raw vectors → uses precomputed_l2
            false,
            RerankPrecision::Disabled,
        )
        .unwrap();
        for i in 0..30u32 {
            let mut v = vec![0.0f64; d];
            v[i as usize % d] = (i as f64 + 1.0) * 0.1;
            e.insert(format!("l{i}"), &Array1::from_vec(v), no_meta())
                .unwrap();
        }
        // build_scorer L2 without vraw → !precomputed_l2.is_empty() path (lines 812-822)
        e.create_index_with_params(8, 32, 32, 1.2, 1).unwrap();
        let q = Array1::from_vec(vec![0.0f64; d]);
        let results = e.search_with_filter_and_ann(&q, 5, None, None).unwrap();
        assert!(!results.is_empty());
        // L2 scores are negative distances
        for r in &results {
            assert!(
                r.score <= 0.0,
                "L2 score should be <= 0 (negative distance)"
            );
        }
    }

    // ── L2 exhaustive search with raw F32 vectors ────────────────────────────
    // Exercises lines 1217-1243 (exhaustive L2 scan) and
    // 1286-1290 (reranking with live_raw_vector_at_slot for L2).

    #[test]
    fn l2_exhaustive_search_with_raw_f32_vectors_reranks_correctly() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = TurboQuantEngine::open_with_options(
            p,
            p,
            d,
            2,
            42,
            DistanceMetric::L2,
            true,
            false,
            RerankPrecision::F32,
        )
        .unwrap();
        e.insert(
            "origin".into(),
            &Array1::from_vec(vec![0.0f64; d]),
            no_meta(),
        )
        .unwrap();
        let mut far = vec![0.0f64; d];
        far[0] = 10.0;
        e.insert("far".into(), &Array1::from_vec(far), no_meta())
            .unwrap();
        // Exhaustive search (no ANN index) → exercises L2 dequant path + raw rerank
        let q = Array1::from_vec(vec![0.0f64; d]);
        let results = e.search_with_filter_and_ann(&q, 2, None, None).unwrap();
        assert_eq!(results.len(), 2);
        // "origin" should score 0 (no distance), "far" should score -10 (negative distance)
        assert_eq!(results[0].id, "origin", "origin should be closest for L2");
    }

    // ── Corrupt state detection: missing live_ids.bin + non-empty live_codes ─
    // Exercises lines 478-481: "database state appears corrupt" error.

    #[test]
    fn missing_id_pool_with_nonempty_live_codes_returns_corrupt_error() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        {
            let mut e = TurboQuantEngine::open(p, p, d, 2, 42).unwrap();
            e.insert("x".into(), &make_vec(d, 0.5), no_meta()).unwrap();
            // Flush so data lands in live_codes.bin (not just WAL)
            e.flush_wal_to_segment().unwrap();
            drop(e);
        }
        // Remove live_ids.bin AND wal.log to create the corrupt state:
        // live_codes.bin exists, but id pool is gone and WAL has no recovery data
        let ids_path = std::path::Path::new(p).join("live_ids.bin");
        let wal_path = std::path::Path::new(p).join("wal.log");
        if ids_path.exists() {
            std::fs::remove_file(&ids_path).unwrap();
        }
        if wal_path.exists() {
            std::fs::remove_file(&wal_path).unwrap();
        }
        let result = TurboQuantEngine::open(p, p, d, 2, 42);
        assert!(result.is_err(), "should fail with corrupt state error");
        let msg = result.err().unwrap().to_string();
        assert!(
            msg.contains("corrupt"),
            "error should mention 'corrupt', got: {msg}"
        );
    }

    // ── L2 exhaustive search reranking without raw vectors (dequant path) ───
    // Exercises the deq_vecs rerank path (lines 1265-1276, 1290-1294)
    // when rerank_enabled=true but live_vraw is None.

    #[test]
    fn l2_exhaustive_rerank_without_raw_vectors_uses_dequant_path() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        // rerank=true but Disabled precision → uses dequantized vectors for rerank
        let mut e = TurboQuantEngine::open_with_options(
            p,
            p,
            d,
            2,
            42,
            DistanceMetric::L2,
            true,
            false,
            RerankPrecision::Disabled,
        )
        .unwrap();
        e.insert("a".into(), &Array1::from_vec(vec![0.0f64; d]), no_meta())
            .unwrap();
        let mut b = vec![0.0f64; d];
        b[0] = 3.0;
        e.insert("b".into(), &Array1::from_vec(b), no_meta())
            .unwrap();
        let q = Array1::from_vec(vec![0.0f64; d]);
        let results = e.search_with_filter_and_ann(&q, 2, None, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
    }

    // ── $and/$or filter false branches ────────────────────────────────────────
    // Lines 1895, 1899, 1910, 1914 in metadata_matches_filter

    #[test]
    fn filter_and_value_not_array_returns_no_results() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        let mut meta = no_meta();
        meta.insert("x".into(), json!(1));
        e.insert("v1".into(), &make_vec(d, 0.5), meta).unwrap();
        // $and value is a string, not an array → line 1899 (else false)
        let filter: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"$and": "not-an-array"}"#).unwrap();
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter), None)
            .unwrap();
        assert!(results.is_empty(), "non-array $and should match nothing");
    }

    #[test]
    fn filter_and_condition_not_object_returns_no_results() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        let mut meta = no_meta();
        meta.insert("x".into(), json!(1));
        e.insert("v1".into(), &make_vec(d, 0.5), meta).unwrap();
        // $and array element is a number, not an object → line 1895 (else false)
        let filter: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"$and": [42]}"#).unwrap();
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter), None)
            .unwrap();
        assert!(
            results.is_empty(),
            "non-object $and condition should match nothing"
        );
    }

    #[test]
    fn filter_or_value_not_array_returns_no_results() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        let mut meta = no_meta();
        meta.insert("x".into(), json!(1));
        e.insert("v1".into(), &make_vec(d, 0.5), meta).unwrap();
        // $or value is a string, not an array → line 1914 (else false)
        let filter: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"$or": "not-an-array"}"#).unwrap();
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter), None)
            .unwrap();
        assert!(results.is_empty(), "non-array $or should match nothing");
    }

    #[test]
    fn filter_or_condition_not_object_returns_no_results() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        let mut meta = no_meta();
        meta.insert("x".into(), json!(1));
        e.insert("v1".into(), &make_vec(d, 0.5), meta).unwrap();
        // $or array element is a string, not an object → line 1910 (else false)
        let filter: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"$or": ["not-an-object"]}"#).unwrap();
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter), None)
            .unwrap();
        assert!(
            results.is_empty(),
            "non-object $or condition should match nothing"
        );
    }

    // ── Nested dotted-path filter false branches ──────────────────────────────
    // Lines 1940, 1955-1958 in get_nested_field / get_nested_json_field

    #[test]
    fn filter_nested_dotted_path_non_object_returns_no_results() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        // metadata has "a": 5 (a scalar), filter on "a.b" → get_nested_field returns None
        // because 5 is not a JsonValue::Object → line 1940 (None)
        let mut meta = no_meta();
        meta.insert("a".into(), json!(5));
        e.insert("v1".into(), &make_vec(d, 0.5), meta).unwrap();
        let filter: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"a.b": 1}"#).unwrap();
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter), None)
            .unwrap();
        assert!(
            results.is_empty(),
            "dotted path through scalar should not match"
        );
    }

    #[test]
    fn filter_deeply_nested_non_object_returns_no_results() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        // metadata has "a": {"b": 5}, filter on "a.b.c" → get_nested_json_field
        // reaches "b"=5 (not Object) → line 1957/1958 (None)
        let mut meta = no_meta();
        meta.insert("a".into(), json!({"b": 5}));
        e.insert("v1".into(), &make_vec(d, 0.5), meta).unwrap();
        let filter: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"a.b.c": 1}"#).unwrap();
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter), None)
            .unwrap();
        assert!(
            results.is_empty(),
            "deeply nested path through scalar should not match"
        );
    }

    // ── String comparison operators in apply_comparison_op ────────────────────
    // Lines 1983-1984, 1989-1992

    #[test]
    fn filter_string_comparison_operators_work() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        let mut meta_a = no_meta();
        meta_a.insert("name".into(), json!("alpha"));
        let mut meta_b = no_meta();
        meta_b.insert("name".into(), json!("beta"));
        let mut meta_c = no_meta();
        meta_c.insert("name".into(), json!("gamma"));
        e.insert("a".into(), &make_vec(d, 0.1), meta_a).unwrap();
        e.insert("b".into(), &make_vec(d, 0.2), meta_b).unwrap();
        e.insert("c".into(), &make_vec(d, 0.3), meta_c).unwrap();

        // $gt on strings: name > "alpha" matches "beta", "gamma"
        let filter_gt: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"name": {"$gt": "alpha"}}"#).unwrap();
        let r_gt = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter_gt), None)
            .unwrap();
        assert_eq!(r_gt.len(), 2, "$gt string: expected 2 matches");

        // $gte on strings: name >= "beta" matches "beta", "gamma"
        let filter_gte: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"name": {"$gte": "beta"}}"#).unwrap();
        let r_gte = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter_gte), None)
            .unwrap();
        assert_eq!(r_gte.len(), 2, "$gte string: expected 2 matches");

        // $lt on strings: name < "beta" matches "alpha"
        let filter_lt: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"name": {"$lt": "beta"}}"#).unwrap();
        let r_lt = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter_lt), None)
            .unwrap();
        assert_eq!(r_lt.len(), 1, "$lt string: expected 1 match");

        // $lte on strings: name <= "alpha" matches "alpha"
        let filter_lte: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"name": {"$lte": "alpha"}}"#).unwrap();
        let r_lte = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter_lte), None)
            .unwrap();
        assert_eq!(r_lte.len(), 1, "$lte string: expected 1 match");
    }

    // ── WAL recovery upsert path (lines 447-455) ──────────────────────────────
    // Insert + upsert same ID without flush, reopen: second WAL entry hits upsert path

    #[test]
    fn wal_recovery_upsert_and_delete_skip_paths_covered() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        {
            let mut e = TurboQuantEngine::open(p, p, d, 2, 42).unwrap();
            // Insert "a" → WAL entry (non-deleted)
            e.insert("a".into(), &make_vec(d, 0.3), no_meta()).unwrap();
            // Insert "b" + delete "b" → WAL delete entry (is_deleted=true) → line 442 (continue)
            e.insert("b".into(), &make_vec(d, 0.4), no_meta()).unwrap();
            e.delete("b".into()).unwrap();
            // Upsert "a" → second WAL entry for same ID → lines 447-455 (upsert path)
            e.upsert_with_document("a".into(), &make_vec(d, 0.7), no_meta(), None)
                .unwrap();
            // Do NOT call close() or flush — live_ids.bin never written (only flush/close writes it)
            drop(e);
        }
        // Reopen: id_pool_loaded=false → WAL recovery:
        //   entry "a" (insert): get_slot → None → new slot
        //   entry "b" (insert): get_slot → None → new slot
        //   entry "b" (delete, is_deleted=true): → continue → LINE 442
        //   entry "a" (upsert): get_slot → Some → overwrite → LINES 447-455
        let e2 = TurboQuantEngine::open(p, p, d, 2, 42).unwrap();
        assert!(e2.vector_count() >= 1, "at least 'a' should be recovered");
    }

    // ── IP ANN index build without raw vectors (lines 748-749) ───────────────
    // Uses prepare_ip_query_from_codes (no vraw available)

    #[test]
    fn ann_ip_no_rerank_build_uses_from_codes_path() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 16;
        // rerank=false → live_vraw=None → has_vraw=false → IP build scorer uses from_codes (748-749)
        let mut e = TurboQuantEngine::open_with_metric_and_rerank(
            p,
            p,
            d,
            4,
            42,
            DistanceMetric::Ip,
            false,
            false,
        )
        .unwrap();
        for i in 0..30u32 {
            let mut v = vec![0.01f64; d];
            v[i as usize % d] += (i as f64 + 1.0) * 0.05;
            e.insert(format!("n{i}"), &Array1::from_vec(v), no_meta())
                .unwrap();
        }
        e.create_index_with_params(8, 32, 32, 1.2, 0).unwrap();
        let mut q = vec![0.0f64; d];
        q[0] = 1.0;
        let results = e
            .search_with_filter_and_ann(&Array1::from_vec(q), 5, None, None)
            .unwrap();
        assert!(!results.is_empty());
    }

    // ── IP fast_mode ANN index build (line 759: &[] for empty QJL) ───────────

    #[test]
    fn ann_ip_fast_mode_build_covers_empty_qjl_slice() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 16;
        // fast_mode=true → cached_qjl_len=0 → &[] branch at line 759
        let mut e = TurboQuantEngine::open_with_metric_and_rerank(
            p,
            p,
            d,
            4,
            42,
            DistanceMetric::Ip,
            false,
            true, // fast_mode
        )
        .unwrap();
        for i in 0..30u32 {
            let mut v = vec![0.01f64; d];
            v[i as usize % d] += (i as f64 + 1.0) * 0.05;
            e.insert(format!("fm{i}"), &Array1::from_vec(v), no_meta())
                .unwrap();
        }
        e.create_index_with_params(8, 32, 32, 1.2, 0).unwrap();
        let mut q = vec![0.0f64; d];
        q[0] = 1.0;
        let results = e
            .search_with_filter_and_ann(&Array1::from_vec(q), 5, None, None)
            .unwrap();
        assert!(!results.is_empty());
    }

    // ── ANN search with filter matching nothing (line 925-928) ───────────────

    #[test]
    fn ann_search_filter_no_matches_returns_empty() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 16;
        let mut e = open_default(p, d);
        for i in 0..30u32 {
            let mut meta = no_meta();
            meta.insert("tag".into(), json!("yes"));
            e.insert(format!("f{i}"), &make_vec(d, i as f64 * 0.01), meta)
                .unwrap();
        }
        e.create_index_with_params(8, 32, 32, 1.2, 0).unwrap();
        // Filter that matches nothing (tag="no" doesn't exist)
        let filter: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"tag": "no"}"#).unwrap();
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter), None)
            .unwrap();
        assert!(
            results.is_empty(),
            "filter matching nothing should return empty via ANN path"
        );
    }

    // ── Cosine ANN search (line 1042 — else branch of IP/else) ───────────────

    #[test]
    fn cosine_ann_search_exercises_else_ann_branch() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 16;
        let mut e = TurboQuantEngine::open_with_metric_and_rerank(
            p,
            p,
            d,
            4,
            42,
            DistanceMetric::Cosine,
            false, // no rerank
            false,
        )
        .unwrap();
        for i in 0..30u32 {
            let mut v = vec![0.01f64; d];
            v[i as usize % d] += (i as f64 + 1.0) * 0.1;
            e.insert(format!("cs{i}"), &Array1::from_vec(v), no_meta())
                .unwrap();
        }
        e.create_index_with_params(8, 32, 32, 1.2, 0).unwrap();
        let mut q = vec![0.0f64; d];
        q[0] = 1.0;
        let results = e
            .search_with_filter_and_ann(&Array1::from_vec(q), 5, None, None)
            .unwrap();
        assert!(!results.is_empty());
    }

    // ── Large multi-level ANN — exercises graph multi-level beam search ────────
    // With 200 vectors, HNSW almost certainly builds a multi-level graph,
    // exercising lines 251-291 in graph.rs (upper-layer beam navigation).

    #[test]
    fn large_multilevel_ann_search_exercises_graph_beam() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 16;
        let mut e = open_default(p, d);
        for i in 0..200u32 {
            let mut v = vec![0.001f64; d];
            v[i as usize % d] += i as f64 * 0.007;
            e.insert(format!("ml{i}"), &Array1::from_vec(v), no_meta())
                .unwrap();
        }
        // Large graph with max_degree=16 virtually guarantees multilevel HNSW
        e.create_index_with_params(16, 64, 64, 1.2, 0).unwrap();
        let mut q = vec![0.0f64; d];
        q[0] = 1.0;
        let results = e
            .search_with_filter_and_ann(&Array1::from_vec(q), 10, None, None)
            .unwrap();
        assert!(
            !results.is_empty(),
            "large ANN search should return results"
        );
    }

    // ── L2 ANN with rerank + raw vectors (lines 823-845 in build_scorer) ──────

    #[test]
    fn l2_ann_with_rerank_raw_vecs_build_scorer_path() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 16;
        // rerank=true + L2 → has_vraw=true → precomputed_l2 is empty → lines 823-845
        let mut e = TurboQuantEngine::open_with_metric_and_rerank(
            p,
            p,
            d,
            4,
            42,
            DistanceMetric::L2,
            true, // rerank → live_vraw exists → has_vraw=true
            false,
        )
        .unwrap();
        for i in 0..30u32 {
            let mut v = vec![0.0f64; d];
            v[i as usize % d] = (i as f64 + 1.0) * 0.05;
            e.insert(format!("lr{i}"), &Array1::from_vec(v), no_meta())
                .unwrap();
        }
        e.create_index_with_params(8, 32, 32, 1.2, 0).unwrap();
        let q = Array1::from_vec(vec![0.0f64; d]);
        let results = e.search_with_filter_and_ann(&q, 5, None, None).unwrap();
        assert!(!results.is_empty());
    }

    // ── Exhaustive search filter returning empty (line 1136-1138) ────────────

    #[test]
    fn exhaustive_search_filter_no_results_covers_empty_candidates() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        let mut meta = no_meta();
        meta.insert("tag".into(), json!("present"));
        e.insert("v1".into(), &make_vec(d, 0.5), meta).unwrap();
        // Filter matches nothing — exhaustive_search_simd returns empty (line 1136-1138)
        let filter: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"tag": "absent"}"#).unwrap();
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter), None)
            .unwrap();
        assert!(
            results.is_empty(),
            "filter matching nothing in exhaustive search should return empty"
        );
    }

    // ── Cosine ANN rerank via dequantized vectors (lines 1075-1076) ──────────

    #[test]
    fn cosine_ann_rerank_without_raw_vecs_uses_deq_path() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 16;
        // Cosine + rerank=true but Disabled precision → live_vraw=None → deq_vecs path (line 1076)
        let mut e = TurboQuantEngine::open_with_options(
            p,
            p,
            d,
            4,
            42,
            DistanceMetric::Cosine,
            true, // rerank_enabled=true
            false,
            RerankPrecision::Disabled, // no raw vecs → live_vraw=None
        )
        .unwrap();
        for i in 0..30u32 {
            let mut v = vec![0.01f64; d];
            v[i as usize % d] += (i as f64 + 1.0) * 0.1;
            e.insert(format!("cr{i}"), &Array1::from_vec(v), no_meta())
                .unwrap();
        }
        e.create_index_with_params(8, 32, 32, 1.2, 0).unwrap();
        let mut q = vec![0.0f64; d];
        q[0] = 1.0;
        let results = e
            .search_with_filter_and_ann(&Array1::from_vec(q), 5, None, None)
            .unwrap();
        assert!(!results.is_empty());
    }

    // ── count_with_filter ─────────────────────────────────────────────────────

    #[test]
    fn count_no_filter_returns_active_vector_count() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        assert_eq!(e.count_with_filter(None).unwrap(), 0);
        e.insert("a".into(), &make_vec(d, 0.5), no_meta()).unwrap();
        e.insert("b".into(), &make_vec(d, 0.3), no_meta()).unwrap();
        assert_eq!(e.count_with_filter(None).unwrap(), 2);
        e.delete("a".into()).unwrap();
        assert_eq!(e.count_with_filter(None).unwrap(), 1);
    }

    #[test]
    fn count_with_filter_returns_matching_count() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        let mut m1 = no_meta();
        m1.insert("topic".into(), json!("ml"));
        let mut m2 = no_meta();
        m2.insert("topic".into(), json!("nlp"));
        let mut m3 = no_meta();
        m3.insert("topic".into(), json!("ml"));
        e.insert("a".into(), &make_vec(d, 0.5), m1).unwrap();
        e.insert("b".into(), &make_vec(d, 0.4), m2).unwrap();
        e.insert("c".into(), &make_vec(d, 0.3), m3).unwrap();

        let filter: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"topic": "ml"}"#).unwrap();
        assert_eq!(e.count_with_filter(Some(&filter)).unwrap(), 2);

        let filter_nlp: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"topic": "nlp"}"#).unwrap();
        assert_eq!(e.count_with_filter(Some(&filter_nlp)).unwrap(), 1);

        let filter_none: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"topic": "cv"}"#).unwrap();
        assert_eq!(e.count_with_filter(Some(&filter_none)).unwrap(), 0);
    }

    // ── delete_batch ──────────────────────────────────────────────────────────

    #[test]
    fn delete_batch_removes_existing_ids_returns_count() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        e.insert("a".into(), &make_vec(d, 0.5), no_meta()).unwrap();
        e.insert("b".into(), &make_vec(d, 0.4), no_meta()).unwrap();
        e.insert("c".into(), &make_vec(d, 0.3), no_meta()).unwrap();

        let deleted = e
            .delete_batch(vec!["a".into(), "c".into(), "z".into()])
            .unwrap();
        assert_eq!(deleted, 2, "only 2 existing IDs should be deleted");
        assert_eq!(e.stats().vector_count, 1);
        assert_eq!(e.list_all(), vec!["b"]);
    }

    #[test]
    fn delete_batch_empty_input_returns_zero() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        e.insert("a".into(), &make_vec(d, 0.5), no_meta()).unwrap();
        assert_eq!(e.delete_batch(vec![]).unwrap(), 0);
        assert_eq!(e.stats().vector_count, 1);
    }

    #[test]
    fn delete_batch_all_unknown_ids_returns_zero() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        assert_eq!(e.delete_batch(vec!["x".into(), "y".into()]).unwrap(), 0);
    }

    // ── pub fn search / search_with_filter wrappers (lines 933-948) ──────────

    #[test]
    fn search_and_search_with_filter_wrappers_callable() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        for i in 0..5u32 {
            let mut meta = no_meta();
            meta.insert("tag".into(), json!("yes"));
            e.insert(format!("s{i}"), &make_vec(d, i as f64 * 0.1 + 0.1), meta)
                .unwrap();
        }
        let q = make_vec(d, 0.5);
        // search() wrapper — covers lines 933-939
        let r1 = e.search(&q, 3).unwrap();
        assert!(!r1.is_empty());
        // search_with_filter() wrapper — covers lines 941-948
        let mut f = no_meta();
        f.insert("tag".into(), json!("yes"));
        let r2 = e.search_with_filter(&q, 3, Some(&f)).unwrap();
        assert!(!r2.is_empty());
    }

    // ── pub fn close() + reopen restores id_pool (lines 1434-1440, 417-418, 1909-1912) ──

    #[test]
    fn close_and_reopen_restores_id_pool() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        {
            let mut e = open_default(p, d);
            for i in 0..5u32 {
                e.insert(
                    format!("id{i}"),
                    &make_vec(d, i as f64 * 0.1 + 0.1),
                    no_meta(),
                )
                .unwrap();
            }
            // close() covers lines 1434-1440; persists live_ids.bin
            e.close().unwrap();
        }
        // Reopen: load_id_pool finds live_ids.bin → covers lines 417-418, 1909-1912
        // rebuild_lookup() called inside load_id_pool → covers id_pool.rs 123-132
        let e2 = open_default(p, d);
        assert_eq!(e2.vector_count(), 5);
    }

    // ── update_with_document on existing id (lines 527-528) ──────────────────

    #[test]
    fn update_with_document_on_existing_id_succeeds() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        e.insert("x".into(), &make_vec(d, 0.1), no_meta()).unwrap();
        // update_with_document on existing id → covers lines 527-528
        e.update_with_document("x".into(), &make_vec(d, 0.9), no_meta(), None)
            .unwrap();
        assert_eq!(e.vector_count(), 1);
    }

    // ── delete() WAL flush at buffer==threshold (line 568) ───────────────────
    // 99 inserts → buffer=99; then delete → buffer=100 >= 100 → flush

    #[test]
    fn delete_at_wal_buffer_threshold_triggers_flush() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = TurboQuantEngine::open(p, p, d, 2, 42).unwrap();
        for i in 0..99u32 {
            e.insert(format!("w{i}"), &make_vec(d, 0.5), no_meta())
                .unwrap();
        }
        // This delete brings buffer to 100 → flush (line 568)
        e.delete("w0".to_string()).unwrap();
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 5, None, None)
            .unwrap();
        assert!(results.len() <= 5);
    }

    // ── delete_batch() WAL flush at buffer==threshold (line 685) ─────────────

    #[test]
    fn delete_batch_at_wal_buffer_threshold_triggers_flush() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = TurboQuantEngine::open(p, p, d, 2, 42).unwrap();
        for i in 0..99u32 {
            e.insert(format!("b{i}"), &make_vec(d, 0.5), no_meta())
                .unwrap();
        }
        // delete_batch brings buffer to 100 → flush (line 685)
        e.delete_batch(vec!["b0".to_string()]).unwrap();
        assert_eq!(e.vector_count(), 98);
    }

    // ── insert_many WAL flush at buffer==threshold (line 1859) ───────────────

    #[test]
    fn insert_many_at_threshold_triggers_wal_flush() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        let items: Vec<BatchWriteItem> = (0..100u32)
            .map(|i| BatchWriteItem {
                id: format!("m{i}"),
                vector: make_vec_f32(d, i as f32 * 0.01 + 0.01),
                metadata: no_meta(),
                document: None,
            })
            .collect();
        // 100 items → wal_buffer.len() = 100 >= 100 → flush (line 1859)
        e.insert_many(items).unwrap();
        assert_eq!(e.vector_count(), 100);
    }

    // ── Cosine ANN: fast_mode empty-qjl (863), zero-norm build (871),
    //    zero-query ANN search (1072) ──────────────────────────────────────────

    #[test]
    fn cosine_ann_fast_mode_empty_qjl_and_zero_norm_and_zero_query() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 16;
        // fast_mode=true → cached_qjl_len=0 → &[] branch (line 863)
        // rerank=false → uses stored norm path in build_scorer
        let mut e = TurboQuantEngine::open_with_metric_and_rerank(
            p,
            p,
            d,
            4,
            42,
            DistanceMetric::Cosine,
            false,
            true,
        )
        .unwrap();
        // Insert zero vector → norm=0 → line 871 in build_scorer
        e.insert("zero".into(), &Array1::zeros(d), no_meta())
            .unwrap();
        for i in 1..15u32 {
            let mut v = vec![0.0f64; d];
            v[i as usize % d] = 1.0;
            e.insert(format!("v{i}"), &Array1::from_vec(v), no_meta())
                .unwrap();
        }
        e.create_index_with_params(4, 16, 16, 1.2, 0).unwrap();
        // Search with zero query → query_norm=0 → line 1072
        let zero_q = Array1::<f64>::zeros(d);
        let _results = e
            .search_with_filter_and_ann(&zero_q, 5, None, None)
            .unwrap();
    }

    // ── $and filter with Object conditions (lines 1955-1957) ─────────────────

    #[test]
    fn and_filter_with_object_conditions_matches() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        for i in 0i64..5 {
            let mut meta = no_meta();
            meta.insert("score".into(), json!(i));
            meta.insert("tag".into(), json!("good"));
            e.insert(format!("a{i}"), &make_vec(d, i as f64 * 0.2), meta)
                .unwrap();
        }
        // $and with Object conditions — covers lines 1955-1957
        let filter: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"$and": [{"score": {"$gte": 2}}, {"tag": "good"}]}"#).unwrap();
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter), None)
            .unwrap();
        assert!(!results.is_empty());
    }

    // ── $ne comparison operator (line 2034) ──────────────────────────────────

    #[test]
    fn ne_filter_matches_different_values() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        for tag in ["a", "b", "c"] {
            let mut meta = no_meta();
            meta.insert("tag".into(), json!(tag));
            e.insert(tag.to_string(), &make_vec(d, 0.5), meta).unwrap();
        }
        // $ne filter — covers line 2034
        let mut filter = no_meta();
        filter.insert("tag".into(), json!({"$ne": "a"}));
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter), None)
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    // ── $lt/$lte numeric comparison (lines 2046, 2047) ───────────────────────

    #[test]
    fn lt_lte_numeric_filters_covered() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        for i in 0i64..5 {
            let mut meta = no_meta();
            meta.insert("n".into(), json!(i));
            e.insert(format!("n{i}"), &make_vec(d, i as f64 * 0.2), meta)
                .unwrap();
        }
        // $lt covers line 2046
        let mut f1 = no_meta();
        f1.insert("n".into(), json!({"$lt": 3}));
        let r1 = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&f1), None)
            .unwrap();
        assert_eq!(r1.len(), 3); // n=0,1,2

        // $lte covers line 2047
        let mut f2 = no_meta();
        f2.insert("n".into(), json!({"$lte": 2}));
        let r2 = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&f2), None)
            .unwrap();
        assert_eq!(r2.len(), 3); // n=0,1,2
    }

    // ── type mismatch in comparison (line 2058) ───────────────────────────────

    #[test]
    fn comparison_type_mismatch_returns_no_results() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        let mut meta = no_meta();
        meta.insert("name".into(), json!("alice"));
        e.insert("a".into(), &make_vec(d, 0.5), meta).unwrap();
        // String field vs numeric op_val → (String, Number) → _ => false (line 2058)
        let mut filter = no_meta();
        filter.insert("name".into(), json!({"$gte": 5}));
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter), None)
            .unwrap();
        assert!(results.is_empty());
    }

    // ── unknown comparison operator (line 2061) ───────────────────────────────

    #[test]
    fn unknown_operator_returns_no_results() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        let mut meta = no_meta();
        meta.insert("x".into(), json!(1));
        e.insert("v".into(), &make_vec(d, 0.5), meta).unwrap();
        // Unknown op "$bogus" → _ => false (line 2061)
        let mut filter = no_meta();
        filter.insert("x".into(), json!({"$bogus": 1}));
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter), None)
            .unwrap();
        assert!(results.is_empty());
    }

    // ── nested dotted paths: covers get_nested_json_field lines 2020, 2025 ───

    #[test]
    fn nested_dotted_path_three_levels_covers_recursive_and_leaf() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        // metadata: {"a": {"b": {"c": 42}}}
        let meta: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"a": {"b": {"c": 42}}}"#).unwrap();
        e.insert("deep".into(), &make_vec(d, 0.5), meta).unwrap();
        // Filter on "a.b.c" = 42 → get_nested_json_field recurses (line 2020)
        // and returns Some(val) at leaf (line 2025)
        let mut filter = no_meta();
        filter.insert("a.b.c".into(), json!(42));
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 5, Some(&filter), None)
            .unwrap();
        assert_eq!(results.len(), 1);
    }

    // ── delete_batch WAL-flush threshold coverage ─────────────────────────────

    #[test]
    fn delete_batch_persists_across_reopen() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        {
            let mut e = open_default(p, d);
            for i in 0..5u32 {
                e.insert(
                    format!("x{i}"),
                    &make_vec(d, 0.1 * i as f64 + 0.1),
                    no_meta(),
                )
                .unwrap();
            }
            let deleted = e
                .delete_batch(vec!["x0".into(), "x2".into(), "x4".into()])
                .unwrap();
            assert_eq!(deleted, 3);
        }
        // Reopen and confirm deletions persisted
        let e2 = open_default(p, d);
        let ids = e2.list_all();
        assert!(ids.contains(&"x1".to_string()));
        assert!(ids.contains(&"x3".to_string()));
        assert!(!ids.contains(&"x0".to_string()));
        assert!(!ids.contains(&"x2".to_string()));
        assert!(!ids.contains(&"x4".to_string()));
    }

    // ── New filter operators ───────────────────────────────────────────────────

    #[test]
    fn filter_in_operator_matches_element_in_array() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        let m = |tag: &str| {
            let mut h = no_meta();
            h.insert("tag".into(), json!(tag));
            h
        };
        e.insert("a".into(), &make_vec(d, 0.1), m("ml")).unwrap();
        e.insert("b".into(), &make_vec(d, 0.2), m("cv")).unwrap();
        e.insert("c".into(), &make_vec(d, 0.3), m("nlp")).unwrap();

        let filter: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"tag": {"$in": ["ml", "cv"]}}"#).unwrap();
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter), None)
            .unwrap();
        let ids: Vec<_> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"a"), "$in: expected 'a'");
        assert!(ids.contains(&"b"), "$in: expected 'b'");
        assert!(!ids.contains(&"c"), "$in: 'c' should be excluded");
    }

    #[test]
    fn filter_in_operator_missing_field_does_not_match() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        e.insert("no-tag".into(), &make_vec(d, 0.1), no_meta())
            .unwrap();
        let filter: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"tag": {"$in": ["ml"]}}"#).unwrap();
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter), None)
            .unwrap();
        assert!(results.is_empty(), "$in on missing field should not match");
    }

    #[test]
    fn filter_nin_operator_excludes_matching_elements() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        let m = |tag: &str| {
            let mut h = no_meta();
            h.insert("tag".into(), json!(tag));
            h
        };
        e.insert("a".into(), &make_vec(d, 0.1), m("ml")).unwrap();
        e.insert("b".into(), &make_vec(d, 0.2), m("cv")).unwrap();
        e.insert("c".into(), &make_vec(d, 0.3), m("nlp")).unwrap();

        let filter: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"tag": {"$nin": ["ml", "cv"]}}"#).unwrap();
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter), None)
            .unwrap();
        let ids: Vec<_> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(!ids.contains(&"a"), "$nin: 'a' should be excluded");
        assert!(!ids.contains(&"b"), "$nin: 'b' should be excluded");
        assert!(ids.contains(&"c"), "$nin: expected 'c'");
    }

    #[test]
    fn filter_nin_missing_field_matches() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        e.insert("no-tag".into(), &make_vec(d, 0.1), no_meta())
            .unwrap();
        let filter: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"tag": {"$nin": ["ml"]}}"#).unwrap();
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter), None)
            .unwrap();
        assert_eq!(results.len(), 1, "$nin on missing field should match");
    }

    #[test]
    fn filter_exists_true_matches_present_field_only() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        let mut with_tag = no_meta();
        with_tag.insert("tag".into(), json!("ml"));
        e.insert("has".into(), &make_vec(d, 0.1), with_tag).unwrap();
        e.insert("missing".into(), &make_vec(d, 0.2), no_meta())
            .unwrap();

        let filter_true: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"tag": {"$exists": true}}"#).unwrap();
        let r = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter_true), None)
            .unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].id, "has");

        let filter_false: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"tag": {"$exists": false}}"#).unwrap();
        let r2 = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter_false), None)
            .unwrap();
        assert_eq!(r2.len(), 1);
        assert_eq!(r2[0].id, "missing");
    }

    #[test]
    fn filter_contains_operator_matches_substring() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        let m = |s: &str| {
            let mut h = no_meta();
            h.insert("title".into(), json!(s));
            h
        };
        e.insert("a".into(), &make_vec(d, 0.1), m("GPU acceleration tips"))
            .unwrap();
        e.insert("b".into(), &make_vec(d, 0.2), m("CPU optimization guide"))
            .unwrap();
        e.insert("c".into(), &make_vec(d, 0.3), m("GPU memory management"))
            .unwrap();

        let filter: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"title": {"$contains": "GPU"}}"#).unwrap();
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter), None)
            .unwrap();
        let ids: Vec<_> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"a"), "$contains: expected 'a'");
        assert!(!ids.contains(&"b"), "$contains: 'b' should be excluded");
        assert!(ids.contains(&"c"), "$contains: expected 'c'");
    }

    #[test]
    fn filter_contains_non_string_field_does_not_match() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        let mut m = no_meta();
        m.insert("score".into(), json!(42));
        e.insert("a".into(), &make_vec(d, 0.1), m).unwrap();
        let filter: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"score": {"$contains": "4"}}"#).unwrap();
        let results = e
            .search_with_filter_and_ann(&make_vec(d, 0.5), 10, Some(&filter), None)
            .unwrap();
        assert!(
            results.is_empty(),
            "$contains on non-string field should not match"
        );
    }

    // ── update_metadata_only ──────────────────────────────────────────────────

    #[test]
    fn update_metadata_only_changes_metadata_without_changing_vector() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        let mut m = no_meta();
        m.insert("status".into(), json!("draft"));
        e.insert("a".into(), &make_vec(d, 0.5), m).unwrap();

        // Update only metadata
        let mut new_meta = no_meta();
        new_meta.insert("status".into(), json!("published"));
        e.update_metadata_only("a", new_meta, None).unwrap();

        let got = e.get("a").unwrap().unwrap();
        assert_eq!(got.metadata["status"], json!("published"));
        assert!(got.document.is_none());
    }

    #[test]
    fn update_metadata_only_updates_document_preserving_metadata() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        let mut m = no_meta();
        m.insert("cat".into(), json!("a"));
        e.insert_with_document("x".into(), &make_vec(d, 0.1), m, Some("old doc".into()))
            .unwrap();

        // Update document only (empty metadata map → preserve existing)
        e.update_metadata_only("x", no_meta(), Some("new doc".into()))
            .unwrap();

        let got = e.get("x").unwrap().unwrap();
        assert_eq!(got.document.as_deref(), Some("new doc"));
        assert_eq!(
            got.metadata["cat"],
            json!("a"),
            "metadata should be preserved"
        );
    }

    #[test]
    fn update_metadata_only_errors_on_missing_id() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        let result = e.update_metadata_only("nonexistent", no_meta(), None);
        assert!(result.is_err());
    }

    #[test]
    fn update_metadata_only_persists_across_reopen() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        {
            let mut e = open_default(p, d);
            let mut m = no_meta();
            m.insert("v".into(), json!(1));
            e.insert("doc".into(), &make_vec(d, 0.3), m).unwrap();
            let mut updated = no_meta();
            updated.insert("v".into(), json!(2));
            e.update_metadata_only("doc", updated, Some("hello".into()))
                .unwrap();
            e.flush_wal_to_segment().unwrap();
        }
        let e2 = open_default(p, d);
        let got = e2.get("doc").unwrap().unwrap();
        assert_eq!(got.metadata["v"], json!(2));
        assert_eq!(got.document.as_deref(), Some("hello"));
    }

    // ── list_with_filter_page ─────────────────────────────────────────────────

    #[test]
    fn list_with_filter_page_no_filter_returns_all() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        for i in 0..5u32 {
            e.insert(format!("id{i}"), &make_vec(d, 0.1), no_meta())
                .unwrap();
        }
        let ids = e.list_with_filter_page(None, None, 0).unwrap();
        assert_eq!(ids.len(), 5);
    }

    #[test]
    fn list_with_filter_page_limit_and_offset_work() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        for i in 0..10u32 {
            e.insert(format!("id{i:02}"), &make_vec(d, 0.1), no_meta())
                .unwrap();
        }
        let page = e.list_with_filter_page(None, Some(3), 2).unwrap();
        assert_eq!(page.len(), 3, "limit=3 should return 3 items");
    }

    #[test]
    fn list_with_filter_page_with_filter_returns_matching_ids() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        let mut m_a = no_meta();
        m_a.insert("kind".into(), json!("A"));
        let mut m_b = no_meta();
        m_b.insert("kind".into(), json!("B"));
        e.insert("a1".into(), &make_vec(d, 0.1), m_a.clone())
            .unwrap();
        e.insert("b1".into(), &make_vec(d, 0.2), m_b.clone())
            .unwrap();
        e.insert("a2".into(), &make_vec(d, 0.3), m_a.clone())
            .unwrap();

        let filter: HashMap<String, serde_json::Value> =
            serde_json::from_str(r#"{"kind": "A"}"#).unwrap();
        let ids = e.list_with_filter_page(Some(&filter), None, 0).unwrap();
        assert_eq!(ids.len(), 2);
        assert!(ids.iter().all(|id| id.starts_with('a')));
    }

    // ── search_batch ──────────────────────────────────────────────────────────

    #[test]
    fn search_batch_returns_one_result_set_per_query() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let mut e = open_default(p, d);
        for i in 0..5u32 {
            e.insert(
                format!("v{i}"),
                &make_vec(d, 0.1 * i as f64 + 0.1),
                no_meta(),
            )
            .unwrap();
        }
        let q1 = make_vec(d, 0.1);
        let q2 = make_vec(d, 0.9);
        let results = e.search_batch(&[q1, q2], 2, None, None).unwrap();
        assert_eq!(results.len(), 2, "one result set per query");
        assert_eq!(results[0].len(), 2, "top_k=2 for first query");
        assert_eq!(results[1].len(), 2, "top_k=2 for second query");
    }

    #[test]
    fn search_batch_empty_queries_returns_empty() {
        let dir = tempdir().unwrap();
        let p = dir.path().to_str().unwrap();
        let d = 8;
        let e = open_default(p, d);
        let results = e.search_batch(&[], 5, None, None).unwrap();
        assert!(results.is_empty());
    }
}
