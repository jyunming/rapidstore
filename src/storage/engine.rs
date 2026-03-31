use ndarray::Array1;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use super::backend::StorageBackend;
use super::graph::GraphManager;
use super::id_pool::IdPool;
use super::live_codes::LiveCodesFile;
use super::metadata::{MetadataStore, VectorMetadata};
use super::segment::{SegmentManager, SegmentRecord};
use super::wal::{Wal, WalEntry};
use crate::quantizer::prod::ProdQuantizer;

const QUANTIZER_STATE_FILE: &str = "quantizer.bin";
const INDEX_IDS_FILE: &str = "graph_ids.json";
const MANIFEST_SAVE_INTERVAL_OPS: usize = 64;

const LIVE_GAMMA_BYTES: usize = 4;
const LIVE_NORM_BYTES: usize = 4;
const LIVE_DELETED_BYTES: usize = 1;

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
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct IndexState {
    pub max_degree: usize,
    pub search_list_size: usize,
    pub alpha: f64,
    pub indexed_nodes: usize,
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
    pub vector: Array1<f64>,
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
    id_pool: IdPool,
    live_vectors: HashMap<u32, Array1<f64>>,
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

    pub fn open_with_metric(
        uri: &str,
        local_dir: &str,
        d: usize,
        b: usize,
        seed: u64,
        metric: DistanceMetric,
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
            let q = ProdQuantizer::new(d, b, seed);
            let m = Manifest {
                version: 2, d, b, seed, vector_count: 0, storage_uri: uri.to_string(),
                quantizer: None, metric: metric.clone(), index_state: None,
            };
            save_quantizer_state(local_dir, &backend, &q)?;
            m.save(&manifest_path)?;
            backend.write("manifest.json", serde_json::to_vec_pretty(&m)?)?;
            (m, q)
        };

        let wal = Wal::open(&wal_path)?;
        let segments = SegmentManager::open(backend.clone())?;
        let metadata = MetadataStore::open(&metadata_path)?;
        let graph = GraphManager::open(backend.clone(), local_dir)?;

        let qjl_len = manifest.d.div_ceil(8);
        let stride = manifest.d + qjl_len + LIVE_GAMMA_BYTES + LIVE_NORM_BYTES + LIVE_DELETED_BYTES;
        let live_codes = LiveCodesFile::open(Path::new(local_dir).join("live_codes.bin"), stride)?;

        let mut engine = Self {
            d: manifest.d, b: manifest.b, quantizer, manifest, metric, backend, wal, wal_buffer: Vec::new(),
            wal_flush_threshold: 100, segments, metadata, graph, local_dir: local_dir.to_string(),
            index_ids: load_index_ids(local_dir).unwrap_or_default(), live_codes, id_pool: IdPool::new(),
            live_vectors: HashMap::new(), index_ids_dirty: false, pending_manifest_updates: 0,
        };

        let pending = Wal::replay(&wal_path)?;
        if !pending.is_empty() {
            engine.wal_buffer.extend(pending);
            engine.flush_wal_to_segment()?;
        } else {
            engine.rebuild_live_codes_cache()?;
        }
        Ok(engine)
    }

    pub fn insert(&mut self, id: String, vector: &Array1<f64>, metadata: HashMap<String, JsonValue>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.insert_with_document(id, vector, metadata, None)
    }

    pub fn insert_with_document(&mut self, id: String, vector: &Array1<f64>, metadata: HashMap<String, JsonValue>, document: Option<String>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.id_pool.get_slot(&id).is_some() { return Err(format!("ID '{}' already exists", id).into()); }
        self.write_vector_entry(id, vector, metadata, document, false)
    }

    pub fn upsert_with_document(&mut self, id: String, vector: &Array1<f64>, metadata: HashMap<String, JsonValue>, document: Option<String>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.write_vector_entry(id, vector, metadata, document, false)
    }

    pub fn update_with_document(&mut self, id: String, vector: &Array1<f64>, metadata: HashMap<String, JsonValue>, document: Option<String>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.id_pool.get_slot(&id).is_none() { return Err(format!("ID '{}' does not exist", id).into()); }
        self.write_vector_entry(id, vector, metadata, document, false)
    }

    pub fn insert_many(&mut self, items: Vec<BatchWriteItem>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> { self.insert_many_with_mode(items, BatchWriteMode::Insert) }
    pub fn upsert_many(&mut self, items: Vec<BatchWriteItem>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> { self.insert_many_with_mode(items, BatchWriteMode::Upsert) }
    pub fn update_many(&mut self, items: Vec<BatchWriteItem>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> { self.insert_many_with_mode(items, BatchWriteMode::Update) }

    pub fn delete(&mut self, id: String) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        if self.id_pool.get_slot(&id).is_none() { return Ok(false); }
        let entry = WalEntry { id: id.clone(), quantized_indices: Vec::new(), qjl_bits: Vec::new(), gamma: 0.0, metadata_json: "{}".to_string(), is_deleted: true };
        self.wal.append(&entry, false)?;
        self.wal_buffer.push(entry);
        self.live_delete_slot(&id);
        self.invalidate_index_state()?;
        self.maybe_persist_state(false)?;
        if self.wal_buffer.len() >= self.wal_flush_threshold { self.flush_wal_to_segment()?; }
        Ok(true)
    }

    pub fn get(&self, id: &str) -> Result<Option<GetResult>, Box<dyn std::error::Error + Send + Sync>> {
        let Some(slot) = self.id_pool.get_slot(id) else { return Ok(None); };
        let meta = self.metadata.get(slot)?.unwrap_or_default();
        Ok(Some(GetResult { id: id.to_string(), metadata: meta.properties, document: meta.document }))
    }

    pub fn get_many(&self, ids: &[String]) -> Result<Vec<Option<GetResult>>, Box<dyn std::error::Error + Send + Sync>> {
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
        self.id_pool.iter_active().into_iter().map(|(id, _)| id).collect()
    }

    pub fn create_index_with_params(&mut self, max_degree: usize, search_list_size: usize, alpha: f64) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.flush_wal_to_segment()?;
        let mut id_slot_pairs = self.live_iter_id_slots();
        if id_slot_pairs.is_empty() { return Ok(()); }
        id_slot_pairs.sort_by(|a, b| a.0.cmp(&b.0));
        let indexed_slots: Vec<u32> = id_slot_pairs.iter().map(|(_, slot)| *slot).collect();
        let live_codes = &self.live_codes;
        let d = self.d;
        let qjl_len = self.live_qjl_len();
        let metric = self.metric.clone();
        let quantizer = self.quantizer.clone();
        let slots_ref = indexed_slots.clone();

        let build_scorer = move |from: u32, candidates: &[u32]| -> Vec<(u32, f64)> {
            let from_slot = slots_ref[from as usize] as usize;
            let (from_i, from_q, from_g, from_n) = live_codes_at_slot_raw(live_codes, d, qjl_len, from_slot);
            let from_vec = quantizer.dequantize_single_no_parallel(from_i, from_q, from_g as f64);
            if matches!(metric, DistanceMetric::Ip) {
                let prep = quantizer.prepare_ip_query_lite(&from_vec);
                candidates.iter().map(|&to| {
                    let to_slot = slots_ref[to as usize] as usize;
                    let (to_i, to_q, to_g, _) = live_codes_at_slot_raw(live_codes, d, qjl_len, to_slot);
                    (to, quantizer.score_ip_encoded_lite(&prep, to_i, to_q, to_g as f64))
                }).collect()
            } else if matches!(metric, DistanceMetric::Cosine) {
                let prep = quantizer.prepare_ip_query_lite(&from_vec);
                let from_norm = from_n;
                candidates.iter().map(|&to| {
                    let to_slot = slots_ref[to as usize] as usize;
                    let (to_i, to_q, to_g, to_n) = live_codes_at_slot_raw(live_codes, d, qjl_len, to_slot);
                    let ip = quantizer.score_ip_encoded_lite(&prep, to_i, to_q, to_g as f64);
                    let score = if from_norm > 0.0 && to_n > 0.0 {
                        ip / (from_norm as f64 * to_n as f64)
                    } else { 0.0 };
                    (to, score)
                }).collect()
            } else {
                candidates.iter().map(|&to| {
                    let to_slot = slots_ref[to as usize] as usize;
                    let (to_i, to_q, to_g, _) = live_codes_at_slot_raw(live_codes, d, qjl_len, to_slot);
                    let to_vec = quantizer.dequantize_single_no_parallel(to_i, to_q, to_g as f64);
                    (to, score_vectors_with_metric(&metric, &from_vec, &to_vec))
                }).collect()
            }
        };

        self.graph.build(indexed_slots.len(), max_degree, alpha, build_scorer)?;
        self.index_ids = indexed_slots;
        self.index_ids_dirty = true;
        self.manifest.index_state = Some(IndexState { max_degree, search_list_size, alpha, indexed_nodes: self.index_ids.len() });
        self.maybe_persist_state(true)?;
        Ok(())
    }

    pub fn search(&self, query: &Array1<f64>, top_k: usize) -> Result<Vec<SearchResult>, Box<dyn std::error::Error + Send + Sync>> {
        self.search_with_filter_and_ann(query, top_k, None, None)
    }

    pub fn search_with_filter(&self, query: &Array1<f64>, top_k: usize, filter: Option<&HashMap<String, JsonValue>>) -> Result<Vec<SearchResult>, Box<dyn std::error::Error + Send + Sync>> {
        self.search_with_filter_and_ann(query, top_k, filter, None)
    }

    pub fn search_with_filter_and_ann(&self, query: &Array1<f64>, top_k: usize, filter: Option<&HashMap<String, JsonValue>>, ann_search_list_size: Option<usize>) -> Result<Vec<SearchResult>, Box<dyn std::error::Error + Send + Sync>> {
        if self.id_pool.active_count() == 0 || top_k == 0 { return Ok(Vec::new()); }
        
        let has_index = self.graph.has_index();
        let not_empty = !self.index_ids.is_empty();
        let state_match = self.manifest.index_state.as_ref().is_some_and(|s| s.indexed_nodes == self.index_ids.len());
        
        if has_index && not_empty && state_match {
            let sls = ann_search_list_size.unwrap_or_else(|| self.manifest.index_state.as_ref().map(|s| s.search_list_size).unwrap_or(32));
            
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
                if matches.is_empty() { return Ok(Vec::new()); }
                Some(matches)
            } else { None };

            let ann = if matches!(self.metric, DistanceMetric::Ip) {
                let prep = self.quantizer.prepare_ip_query(query);
                let index_ids = &self.index_ids;
                let slot_set: Option<std::collections::HashSet<u32>> = filter_slots.map(|s| s.into_iter().collect());
                
                self.graph.search(0, top_k, sls.max(top_k), |node| {
                    let slot = index_ids[node as usize];
                    let (indices, qjl, gamma, _) = self.live_codes_at_slot(slot as usize);
                    self.quantizer.score_ip_encoded(&prep, indices, qjl, gamma as f64)
                }, slot_set.map(|ss| move |node_idx: u32| ss.contains(&index_ids[node_idx as usize])))?
            } else if matches!(self.metric, DistanceMetric::Cosine) {
                let prep = self.quantizer.prepare_ip_query(query);
                let query_norm = query.iter().map(|x| x * x).sum::<f64>().sqrt();
                let index_ids = &self.index_ids;
                let slot_set: Option<std::collections::HashSet<u32>> = filter_slots.map(|s| s.into_iter().collect());

                self.graph.search(0, top_k, sls.max(top_k), |node| {
                    let slot = index_ids[node as usize];
                    let (indices, qjl, gamma, doc_norm) = self.live_codes_at_slot(slot as usize);
                    let ip = self.quantizer.score_ip_encoded(&prep, indices, qjl, gamma as f64);
                    if query_norm > 0.0 && doc_norm > 0.0 {
                        ip / (query_norm * doc_norm as f64)
                    } else { 0.0 }
                }, slot_set.map(|ss| move |node_idx: u32| ss.contains(&index_ids[node_idx as usize])))?
            } else {
                let index_ids = &self.index_ids;
                let slot_set: Option<std::collections::HashSet<u32>> = filter_slots.map(|s| s.into_iter().collect());

                self.graph.search(0, top_k, sls.max(top_k), |node| {
                    let slot = index_ids[node as usize];
                    let (indices, qjl, gamma, _) = self.live_codes_at_slot(slot as usize);
                    let v = self.quantizer.dequantize(indices, qjl, gamma as f64);
                    score_vectors_with_metric(&self.metric, query, &v)
                }, slot_set.map(|ss| move |node_idx: u32| ss.contains(&index_ids[node_idx as usize])))?
            };

            let slots: Vec<u32> = ann.iter().map(|(n, _)| self.index_ids[*n as usize]).collect();
            let meta_map = self.metadata.get_many(&slots)?;
            let mut out = Vec::with_capacity(ann.len());
            for (node, score) in ann {
                let slot = self.index_ids[node as usize];
                let id = self.id_pool.get_str(slot).unwrap_or_default().to_string();
                let meta = meta_map.get(&slot).cloned().unwrap_or_default();
                out.push(SearchResult { id, score, metadata: meta.properties, document: meta.document });
            }
            out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            return Ok(out);
        }

        // Exhaustive search path
        let mut results = Vec::new();
        let pairs = self.live_iter_id_slots();
        let slots: Vec<u32> = pairs.iter().map(|(_, s)| *s).collect();
        let meta_map = self.metadata.get_many(&slots)?;
        for (id, slot) in pairs {
            let meta = meta_map.get(&slot).cloned().unwrap_or_default();
            if let Some(f) = filter { if !metadata_matches_filter(&meta.properties, f) { continue; } }
            let (indices, qjl, gamma, _) = self.live_codes_at_slot(slot as usize);
            let v = self.quantizer.dequantize(indices, qjl, gamma as f64);
            results.push(SearchResult { id, score: score_vectors_with_metric(&self.metric, query, &v), metadata: meta.properties, document: meta.document });
        }
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(top_k);
        Ok(results)
    }

    pub fn flush_wal_to_segment(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.wal_buffer.is_empty() { return Ok(()); }
        let records: Vec<SegmentRecord> = self.wal_buffer.drain(..).map(|e| SegmentRecord {
            id: e.id, quantized_indices: e.quantized_indices, qjl_bits: e.qjl_bits, gamma: e.gamma, is_deleted: e.is_deleted,
        }).collect();
        for r in &records {
            if r.is_deleted { self.live_delete_slot(&r.id); }
        }
        self.live_compact_slab()?;
        self.live_codes.flush()?;
        self.segments.flush_batch(records)?;
        self.wal.truncate()?;
        self.metadata.flush()?;

        self.live_codes.release_mmap();
        let live_codes_path = Path::new(&self.local_dir).join("live_codes.bin");
        let live_codes_data = std::fs::read(&live_codes_path)?;
        self.backend.write("live_codes.bin", live_codes_data)?;
        self.live_codes = LiveCodesFile::open(live_codes_path, self.live_stride())?;
        self.invalidate_index_state()?;
        self.manifest.vector_count = self.live_active_count() as u64;
        self.maybe_persist_state(false)?;
        Ok(())
    }

    pub fn close(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.flush_wal_to_segment()?;
        self.metadata.flush()?;
        self.maybe_persist_state(true)?;
        Ok(())
    }

    pub fn vector_count(&self) -> u64 { self.id_pool.active_count() as u64 }

    pub fn stats(&self) -> DbStats {
        DbStats {
            vector_count: self.vector_count(), segment_count: self.segments.segments.len(), buffered_vectors: self.wal_buffer.len(),
            d: self.d, b: self.b, total_disk_bytes: self.total_disk_bytes(), has_index: self.can_use_ann_index(),
            index_nodes: self.index_ids.len(), live_codes_bytes: self.live_codes.byte_len(), live_slot_count: self.live_codes.len(),
            live_id_count: self.id_pool.active_count(), live_vectors_count: self.live_vectors.len(), live_vectors_bytes_estimate: 0,
            metadata_entries: self.metadata.len(), metadata_bytes_estimate: self.metadata.approx_bytes(),
            ann_slot_count: self.index_ids.len(), graph_nodes: self.graph.node_count(),
        }
    }

    fn live_qjl_len(&self) -> usize { self.d.div_ceil(8) }
    fn live_stride(&self) -> usize { self.d + self.live_qjl_len() + LIVE_GAMMA_BYTES + LIVE_NORM_BYTES + LIVE_DELETED_BYTES }
    fn live_active_count(&self) -> usize { self.id_pool.active_count() }

    fn live_codes_at_slot(&self, slot: usize) -> (&[u8], &[u8], f32, f32) {
        let rec = self.live_codes.get_slot(slot);
        let qjl_len = self.live_qjl_len();
        let gamma = f32::from_le_bytes(rec[self.d + qjl_len..self.d + qjl_len + 4].try_into().unwrap());
        let norm = f32::from_le_bytes(rec[self.d + qjl_len + 4..self.d + qjl_len + 8].try_into().unwrap());
        (&rec[0..self.d], &rec[self.d..self.d + qjl_len], gamma, norm)
    }

    fn live_alloc_slot(&mut self, id: &str, indices: &[u8], qjl: &[u8], gamma: f32, norm: f32) -> u32 {
        let slot = self.id_pool.insert(id);
        let new_slot = self.live_codes.alloc_slot().unwrap();
        let qjl_len = self.live_qjl_len();
        let rec = self.live_codes.get_slot_mut(new_slot);
        rec[0..self.d].copy_from_slice(indices);
        rec[self.d..self.d + qjl_len].copy_from_slice(qjl);
        rec[self.d + qjl_len..self.d + qjl_len + 4].copy_from_slice(&gamma.to_le_bytes());
        rec[self.d + qjl_len + 4..self.d + qjl_len + 8].copy_from_slice(&norm.to_le_bytes());
        rec[self.d + qjl_len + 8] = 0u8;
        slot
    }

    fn live_alloc_or_update(&mut self, id: &str, indices: &[u8], qjl: &[u8], gamma: f32, norm: f32) -> u32 {
        if let Some(slot) = self.id_pool.get_slot(id) {
            let qjl_len = self.live_qjl_len();
            let rec = self.live_codes.get_slot_mut(slot as usize);
            rec[0..self.d].copy_from_slice(indices);
            rec[self.d..self.d + qjl_len].copy_from_slice(qjl);
            rec[self.d + qjl_len..self.d + qjl_len + 4].copy_from_slice(&gamma.to_le_bytes());
            rec[self.d + qjl_len + 4..self.d + qjl_len + 8].copy_from_slice(&norm.to_le_bytes());
            rec[self.d + qjl_len + 8] = 0u8;
            slot
        } else { self.live_alloc_slot(id, indices, qjl, gamma, norm) }
    }

    fn live_delete_slot(&mut self, id: &str) {
        if let Some(slot) = self.id_pool.delete_by_id(id) {
            let stride = self.live_stride();
            let rec = self.live_codes.get_slot_mut(slot as usize);
            rec[stride - 1] = 1u8;
        }
    }

    fn live_iter_id_slots(&self) -> Vec<(String, u32)> { self.id_pool.iter_active() }

    fn live_compact_slab(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let stride = self.live_stride();
        let temp_path = Path::new(&self.local_dir).join("live_codes.bin.tmp");
        let mut new_codes = LiveCodesFile::open(temp_path.clone(), stride)?;
        new_codes.clear()?;
        let mut new_pool = IdPool::new();
        for (id, old_slot) in self.live_iter_id_slots() {
            let old_rec = self.live_codes.get_slot(old_slot as usize);
            let next_alloc = new_codes.alloc_slot()?;
            new_codes.get_slot_mut(next_alloc).copy_from_slice(old_rec);
            new_pool.insert(&id);
        }
        new_codes.truncate_to(new_pool.active_count())?;
        new_codes.flush()?; drop(new_codes);
        let final_path = Path::new(&self.local_dir).join("live_codes.bin");
        self.live_codes.release_mmap();
        std::fs::rename(temp_path, &final_path)?;
        self.live_codes = LiveCodesFile::open(final_path, stride)?;
        self.id_pool = new_pool;
        Ok(())
    }

    fn rebuild_live_codes_cache(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut deleted = std::collections::HashSet::new();
        for res in self.segments.iter_records_streaming() {
            let r = res?;
            if r.is_deleted { deleted.insert(r.id); }
        }
        self.live_codes.clear()?; self.id_pool.clear();
        let mut latest = HashMap::new();
        for res in self.segments.iter_records_streaming() {
            let r = res?;
            if !r.is_deleted && !deleted.contains(&r.id) { latest.insert(r.id.clone(), r); }
        }
        let mut ids: Vec<_> = latest.keys().cloned().collect();
        ids.sort();
        for id in ids {
            let r = latest.get(&id).unwrap();
            let v = self.quantizer.dequantize(&r.quantized_indices, &r.qjl_bits, r.gamma as f64);
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt() as f32;
            self.live_alloc_or_update(&r.id, &r.quantized_indices, &r.qjl_bits, r.gamma, norm);
        }
        Ok(())
    }

    fn invalidate_index_state(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.index_ids.clear(); self.manifest.index_state = None; self.index_ids_dirty = true;
        Ok(())
    }

    fn maybe_persist_state(&mut self, force: bool) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if !force && !self.index_ids_dirty && self.pending_manifest_updates < MANIFEST_SAVE_INTERVAL_OPS { return Ok(()); }
        if self.index_ids_dirty {
            save_index_ids(&self.local_dir, &self.index_ids)?;
            self.backend.write(INDEX_IDS_FILE, serialize_index_ids(&self.index_ids)?)?;
            self.index_ids_dirty = false;
        }
        self.save_manifest()?; self.pending_manifest_updates = 0;
        Ok(())
    }

    fn can_use_ann_index(&self) -> bool {
        self.graph.has_index() && !self.index_ids.is_empty() && 
        self.manifest.index_state.as_ref().is_some_and(|s| s.indexed_nodes == self.index_ids.len())
    }

    fn save_manifest(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut m = self.manifest.clone(); m.quantizer = None;
        let manifest_path = format!("{}/manifest.json", self.local_dir);
        m.save(&manifest_path)?;
        self.backend.write("manifest.json", serde_json::to_vec_pretty(&m)?)?;
        Ok(())
    }

    fn total_disk_bytes(&self) -> u64 {
        let mut total = self.segments.total_disk_size();
        for name in ["manifest.json", "metadata.bin", "wal.log", QUANTIZER_STATE_FILE, "graph.bin", INDEX_IDS_FILE, "live_codes.bin"] {
            if let Ok(data) = self.backend.read(name) { total += data.len() as u64; }
        }
        total
    }

    fn write_vector_entry(&mut self, id: String, vector: &Array1<f64>, metadata: HashMap<String, JsonValue>, document: Option<String>, is_deleted: bool) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let (indices, qjl, gamma) = self.quantizer.quantize(vector);
        let norm = vector.iter().map(|x| x * x).sum::<f64>().sqrt() as f32;
        let meta = VectorMetadata { properties: metadata, document };
        let entry = WalEntry { id: id.clone(), quantized_indices: indices, qjl_bits: qjl, gamma: gamma as f32, metadata_json: serde_json::to_string(&meta)?, is_deleted };
        self.wal.append(&entry, false)?; self.wal_buffer.push(entry.clone());
        if !is_deleted {
            let slot = self.live_alloc_or_update(&id, &entry.quantized_indices, &entry.qjl_bits, entry.gamma, norm);
            self.metadata.put(slot, &meta)?;
        }
        self.invalidate_index_state()?;
        self.manifest.vector_count = self.live_active_count() as u64;
        self.maybe_persist_state(false)?;
        if self.wal_buffer.len() >= self.wal_flush_threshold { self.flush_wal_to_segment()?; }
        Ok(())
    }

    pub fn insert_many_with_mode(&mut self, items: Vec<BatchWriteItem>, mode: BatchWriteMode) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        for chunk in items.chunks(5000) {
            let mut wal_entries = Vec::with_capacity(chunk.len());
            let mut metadata_entries: Vec<(u32, VectorMetadata)> = Vec::with_capacity(chunk.len());

            let vectors: Vec<_> = chunk.iter().map(|i| i.vector.clone()).collect();
            let quantized = self.quantizer.quantize_batch(&vectors);

            for (i, (item, (indices, qjl, gamma))) in chunk.iter().zip(quantized).enumerate() {
                match mode {
                    BatchWriteMode::Insert if self.id_pool.contains(&item.id) => {
                        return Err(format!("ID '{}' already exists", item.id).into());
                    }
                    BatchWriteMode::Update if !self.id_pool.contains(&item.id) => {
                        return Err(format!("ID '{}' does not exist", item.id).into());
                    }
                    _ => {}
                }

                let norm = vectors[i].iter().map(|x| x * x).sum::<f64>().sqrt() as f32;
                let meta = VectorMetadata { properties: item.metadata.clone(), document: item.document.clone() };
                let entry = WalEntry { id: item.id.clone(), quantized_indices: indices, qjl_bits: qjl, gamma: gamma as f32, metadata_json: serde_json::to_string(&meta)?, is_deleted: false };
                
                let slot = self.live_alloc_or_update(&item.id, &entry.quantized_indices, &entry.qjl_bits, entry.gamma, norm);
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
        if self.wal_buffer.len() >= self.wal_flush_threshold { self.flush_wal_to_segment()?; }
        Ok(())
    }
}

fn save_quantizer_state(local_dir: &str, backend: &Arc<StorageBackend>, quantizer: &ProdQuantizer) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let bytes = bincode::serialize(quantizer)?;
    std::fs::write(format!("{}/{}", local_dir, QUANTIZER_STATE_FILE), &bytes)?;
    backend.write(QUANTIZER_STATE_FILE, bytes)?; Ok(())
}

fn load_quantizer_state(local_dir: &str, backend: &Arc<StorageBackend>) -> Result<ProdQuantizer, Box<dyn std::error::Error + Send + Sync>> {
    let local = format!("{}/{}", local_dir, QUANTIZER_STATE_FILE);
    if Path::new(&local).exists() { return Ok(bincode::deserialize(&std::fs::read(&local)?)?); }
    if let Ok(bytes) = backend.read(QUANTIZER_STATE_FILE) {
        std::fs::write(&local, &bytes)?; return Ok(bincode::deserialize(&bytes)?);
    }
    Err("Quantizer state not found".into())
}

fn save_index_ids(local_dir: &str, ids: &[u32]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    std::fs::write(format!("{}/{}", local_dir, INDEX_IDS_FILE), serialize_index_ids(ids)?)?; Ok(())
}

fn serialize_index_ids(ids: &[u32]) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> { Ok(serde_json::to_vec_pretty(ids)?) }

fn load_index_ids(local_dir: &str) -> Result<Vec<u32>, Box<dyn std::error::Error + Send + Sync>> {
    let local = format!("{}/{}", local_dir, INDEX_IDS_FILE);
    if Path::new(&local).exists() { return Ok(serde_json::from_slice(&std::fs::read(&local)?)?); }
    Ok(Vec::new())
}

fn live_codes_at_slot_raw<'a>(live_codes: &'a LiveCodesFile, d: usize, qjl_len: usize, slot: usize) -> (&'a [u8], &'a [u8], f32, f32) {
    let rec = live_codes.get_slot(slot);
    let gamma = f32::from_le_bytes(rec[d + qjl_len..d + qjl_len + 4].try_into().unwrap());
    let norm = f32::from_le_bytes(rec[d + qjl_len + 4..d + qjl_len + 8].try_into().unwrap());
    (&rec[0..d], &rec[d..d + qjl_len], gamma, norm)
}

fn metadata_matches_filter(meta: &HashMap<String, JsonValue>, filter: &HashMap<String, JsonValue>) -> bool {
    filter.iter().all(|(k, v)| meta.get(k).is_some_and(|mv| mv == v))
}

fn score_vectors_with_metric(metric: &DistanceMetric, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    match metric {
        DistanceMetric::Ip => a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(),
        DistanceMetric::Cosine => {
            let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let an = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let bn = b.iter().map(|x| x * x).sum::<f64>().sqrt();
            if an == 0.0 || bn == 0.0 { 0.0 } else { dot / (an * bn) }
        }
        DistanceMetric::L2 => -a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt(),
    }
}
