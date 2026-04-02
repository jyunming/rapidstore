use ndarray::Array1;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;
use std::sync::Arc;

use super::backend::StorageBackend;
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
        Self::open_with_metric_and_rerank(uri, local_dir, d, b, seed, DistanceMetric::Ip, true, true)
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
        Self::open_with_options(uri, local_dir, d, b, seed, metric, rerank, fast_mode, RerankPrecision::F32)
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
            let q = load_quantizer_state(local_dir, &backend)?;
            (m, q)
        } else if let Ok(data) = backend.read("manifest.json") {
            let m: Manifest = serde_json::from_slice(&data)?;
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
            if matches!(rerank_precision, RerankPrecision::F16 | RerankPrecision::F32) {
                let vraw_path = Path::new(local_dir).join("live_vectors.bin");
                if !vraw_path.exists() {
                    std::fs::File::create(&vraw_path)?;
                }
            }
            (m, q)
        };

        let mut wal = Wal::open(&wal_path)?;
        wal.set_quantizer(std::sync::Arc::new(quantizer.clone()));
        let segments = SegmentManager::open(backend.clone())?;
        let metadata = MetadataStore::open(&metadata_path)?;
        let graph = GraphManager::open(backend.clone(), local_dir)?;

        let n = manifest.d.next_power_of_two();
        let qjl_len = n.div_ceil(8);
        let mse_len = (n * (manifest.b - 1)).div_ceil(8);
        let stride = mse_len + qjl_len + LIVE_GAMMA_BYTES + LIVE_NORM_BYTES + LIVE_DELETED_BYTES;
        let live_codes = LiveCodesFile::open(Path::new(local_dir).join("live_codes.bin"), stride)?;
        let live_vraw_path = Path::new(local_dir).join("live_vectors.bin");
        let live_vraw = if manifest.rerank_enabled && live_vraw_path.exists() && !matches!(manifest.rerank_precision, RerankPrecision::Disabled) {
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

        if let Ok(id_pool) = load_id_pool(local_dir, &engine.backend) {
            engine.id_pool = id_pool;
        } else if engine.live_codes.len() > 0 {
            return Err(
                "live_ids.bin missing; cannot rebuild id_pool without legacy segment data".into(),
            );
        }

        let pending = Wal::replay(&wal_path, Some(&engine.quantizer))?;
        if !pending.is_empty() {
            engine.wal_buffer.extend(pending);
            engine.flush_wal_to_segment()?;
        } else if engine.id_pool.active_count() == 0 && engine.live_codes.len() > 0 {
            return Err(
                "live_ids.bin missing; cannot rebuild id_pool without legacy segment data".into(),
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
            if !matches!(self.metric, DistanceMetric::Ip | DistanceMetric::Cosine)
                && !has_vraw
            {
                let codes_bytes = live_codes.as_bytes();
                let stride = self.live_stride();
                let q = &self.quantizer;
                indexed_slots
                    .par_iter()
                    .map(|&slot| {
                        let rec = &codes_bytes
                            [slot as usize * stride..(slot as usize + 1) * stride];
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
                        (to, quantizer.score_ip_encoded_lite(&prep, to_i, to_q, to_g as f64))
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
                        let ip =
                            quantizer.score_ip_encoded_lite(&prep, to_i, to_q, to_g as f64);
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
                            rec[mse_len + qjl_len..mse_len + qjl_len + 4].try_into().unwrap(),
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
                            rec[mse_len + qjl_len..mse_len + qjl_len + 4].try_into().unwrap(),
                        );
                        let doc_norm = f32::from_le_bytes(
                            rec[mse_len + qjl_len + 4..mse_len + qjl_len + 8].try_into().unwrap(),
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
                            rec[mse_len + qjl_len..mse_len + qjl_len + 4].try_into().unwrap(),
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
            let deq_vecs: Vec<Array1<f64>> =
                if self.rerank_enabled && self.live_vraw.is_none() {
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
                            let rec = &codes_bytes
                                [slot as usize * stride..(slot as usize + 1) * stride];
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
                            let rec = &codes_bytes
                                [slot as usize * stride..(slot as usize + 1) * stride];
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
        let deq_vecs: Vec<Array1<f64>> =
            if self.rerank_enabled && self.live_vraw.is_none() {
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

        self.live_codes.release_mmap();
        if let Some(vraw) = &mut self.live_vraw {
            vraw.release_mmap();
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
            self.live_vraw = Some(LiveCodesFile::open(live_vraw_path, self.live_vraw_stride())?);
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
    ) -> u32 {
        let slot = self.id_pool.insert(id);
        let new_slot = self.live_codes.alloc_slot().unwrap();
        if let Some(vraw) = &mut self.live_vraw {
            let _ = vraw.alloc_slot().unwrap();
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
        slot
    }

    fn live_alloc_or_update(
        &mut self,
        id: &str,
        indices: &[CodeIndex],
        qjl: &[u8],
        gamma: f32,
        norm: f32,
    ) -> u32 {
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
            slot
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

            new_pool.insert(&id);
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

        self.live_codes.release_mmap();
        if let Some(vraw) = &mut self.live_vraw {
            vraw.release_mmap();
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

    fn rebuild_live_codes_cache(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Err("rebuild_live_codes_cache is disabled; live_ids.bin is required for recovery".into())
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
            );
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
                );
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
    filter
        .iter()
        .all(|(k, v)| meta.get(k).is_some_and(|mv| mv == v))
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
