use ndarray::Array1;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use super::backend::StorageBackend;
use super::compaction::Compactor;
use super::graph::GraphManager;
use super::metadata::{MetadataStore, VectorMetadata};
use super::segment::{SegmentManager, SegmentRecord};
use super::wal::{Wal, WalEntry};
use crate::quantizer::prod::ProdQuantizer;
use crate::quantizer::CodeIndex;

const QUANTIZER_STATE_FILE: &str = "quantizer.bin";
const INDEX_IDS_FILE: &str = "graph_ids.json";
const MANIFEST_SAVE_INTERVAL_OPS: usize = 64;

const LIVE_GAMMA_BYTES: usize = 4;
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
    compactor: Compactor,
    local_dir: String,

    index_ids: Vec<String>,
    ann_slots: Vec<u32>,
    live_codes: Vec<u8>,
    live_slot_to_id: Vec<Option<String>>,
    live_id_to_slot: HashMap<String, u32>,
    live_vectors: HashMap<String, Array1<f64>>,
    index_ids_dirty: bool,
    pending_manifest_updates: usize,
}


enum BatchWriteMode {
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
            if m.d != d || m.b != b {
                return Err(format!(
                    "Schema mismatch: existing manifest has d={}, b={} but open() requested d={}, b={}",
                    m.d, m.b, d, b
                )
                .into());
            }
            if m.metric != metric {
                return Err(format!(
                    "Metric mismatch: existing manifest has metric={:?} but open() requested metric={:?}",
                    m.metric, metric
                )
                .into());
            }
            let q = load_quantizer_state(local_dir, &backend, &m)?;
            (m, q)
        } else if let Ok(data) = backend.read("manifest.json") {
            let m: Manifest = serde_json::from_slice(&data)?;
            if m.d != d || m.b != b {
                return Err(format!(
                    "Schema mismatch: existing remote manifest has d={}, b={} but open() requested d={}, b={}",
                    m.d, m.b, d, b
                )
                .into());
            }
            if m.metric != metric {
                return Err(format!(
                    "Metric mismatch: existing remote manifest has metric={:?} but open() requested metric={:?}",
                    m.metric, metric
                )
                .into());
            }
            let q = load_quantizer_state(local_dir, &backend, &m)?;
            (m, q)
        } else {
            let q = ProdQuantizer::new(d, b, seed);
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
            };
            save_quantizer_state(local_dir, &backend, &q)?;
            m.save(&manifest_path)?;
            backend.write("manifest.json", serde_json::to_vec_pretty(&m)?)?;
            (m, q)
        };

        let wal = Wal::open(&wal_path)?;
        let compactor = Compactor::new(backend.clone());
        let _ = compactor.recover_if_needed()?;
        let segments = SegmentManager::open(backend.clone())?;
        let metadata = MetadataStore::open(&metadata_path)?;
        let graph = GraphManager::open(backend.clone(), local_dir)?;

        let mut engine = Self {
            d: manifest.d,
            b: manifest.b,
            quantizer,
            manifest,
            metric,
            backend,
            wal,
            wal_buffer: Vec::new(),
            wal_flush_threshold: 1000,
            segments,
            metadata,
            graph,
            compactor,
            local_dir: local_dir.to_string(),
            index_ids: load_index_ids(local_dir).unwrap_or_default(),
            ann_slots: Vec::new(),
            live_codes: Vec::new(),
            live_slot_to_id: Vec::new(),
            live_id_to_slot: HashMap::new(),
            live_vectors: HashMap::new(),
            index_ids_dirty: false,
            pending_manifest_updates: 0,
        };

        let pending = Wal::replay(&wal_path)?;
        if !pending.is_empty() {
            engine.wal_buffer.extend(pending);
            engine.flush_wal_to_segment()?;
        } else {
            engine.rebuild_live_codes_cache()?;
            engine.manifest.vector_count = engine.live_active_count() as u64;
            engine.save_manifest()?;
        }
        engine.rebuild_ann_slots_from_index_ids();

        save_quantizer_state(local_dir, &engine.backend, &engine.quantizer)?;
        if engine.manifest.quantizer.is_some() {
            engine.manifest.quantizer = None;
            engine.save_manifest()?;
        }

        Ok(engine)
    }

    pub fn insert(
        &mut self,
        id: String,
        vector: &Array1<f64>,
        metadata_props: HashMap<String, JsonValue>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.insert_with_document(id, vector, metadata_props, None)
    }

    pub fn insert_with_document(
        &mut self,
        id: String,
        vector: &Array1<f64>,
        metadata_props: HashMap<String, JsonValue>,
        document: Option<String>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.live_id_to_slot.contains_key(&id) {
            return Err(format!("ID '{}' already exists; use upsert/update", id).into());
        }
        self.write_vector_entry(id, vector, metadata_props, document, false)
    }

    pub fn upsert_with_document(
        &mut self,
        id: String,
        vector: &Array1<f64>,
        metadata_props: HashMap<String, JsonValue>,
        document: Option<String>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.write_vector_entry(id, vector, metadata_props, document, false)
    }

    pub fn update_with_document(
        &mut self,
        id: String,
        vector: &Array1<f64>,
        metadata_props: HashMap<String, JsonValue>,
        document: Option<String>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if !self.live_id_to_slot.contains_key(&id) {
            return Err(format!("ID '{}' does not exist; use insert/upsert", id).into());
        }
        self.write_vector_entry(id, vector, metadata_props, document, false)
    }

    pub fn insert_many(
        &mut self,
        items: Vec<BatchWriteItem>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.write_many(items, BatchWriteMode::Insert)
    }

    pub fn upsert_many(
        &mut self,
        items: Vec<BatchWriteItem>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.write_many(items, BatchWriteMode::Upsert)
    }

    pub fn update_many(
        &mut self,
        items: Vec<BatchWriteItem>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.write_many(items, BatchWriteMode::Update)
    }

    pub fn delete(
        &mut self,
        id: String,
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        if !self.live_id_to_slot.contains_key(&id) {
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
        self.metadata.delete(&id)?;
        self.live_delete_slot(&id);
        self.live_vectors.remove(&id);
        self.invalidate_index_state()?;
        self.manifest.vector_count = self.live_active_count() as u64;
        self.pending_manifest_updates += 1;
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
        if !self.live_id_to_slot.contains_key(id) {
            return Ok(None);
        }
        let meta = self.metadata.get(id)?.unwrap_or_default();
        Ok(Some(GetResult {
            id: id.to_string(),
            metadata: meta.properties,
            document: meta.document,
        }))
    }

    pub fn create_index_with_params(
        &mut self,
        max_degree: usize,
        search_list_size: usize,
        alpha: f64,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.flush_wal_to_segment()?;
        let mut id_slot_pairs = self.live_iter_id_slots();
        if id_slot_pairs.is_empty() {
            self.invalidate_index_state()?;
            self.maybe_persist_state(true)?;
            return Ok(());
        }
        id_slot_pairs.sort_by(|a, b| a.0.cmp(&b.0));

        let indexed_ids: Vec<String> = id_slot_pairs.iter().map(|(id, _)| id.clone()).collect();
        let indexed_slots: Vec<u32> = id_slot_pairs.iter().map(|(_, slot)| *slot).collect();

        let all_vectors: Vec<Array1<f64>> = id_slot_pairs
            .par_iter()
            .map(|(id, slot)| {
                self.live_vectors.get(id).cloned().unwrap_or_else(|| {
                    let (indices, qjl, gamma) = self.live_codes_at_slot(*slot as usize);
                    self.quantizer.dequantize(indices, qjl, gamma as f64)
                })
            })
            .collect();

        let metric = self.metric.clone();
        let build_scorer = |from: u32, to: u32| {
            score_vectors_with_metric(&metric, &all_vectors[from as usize], &all_vectors[to as usize])
        };

        self.graph
            .build(indexed_ids.len(), max_degree, alpha, build_scorer)?;

        self.index_ids = indexed_ids;
        self.ann_slots = indexed_slots;
        self.index_ids_dirty = true;

        self.manifest.index_state = Some(IndexState {
            max_degree,
            search_list_size,
            alpha,
            indexed_nodes: self.index_ids.len(),
        });
        self.pending_manifest_updates += 1;
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
        if self.live_id_to_slot.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        if filter.is_none() && self.can_use_ann_index() {
            let search_list_size = ann_search_list_size.unwrap_or_else(|| {
                self.manifest
                    .index_state
                    .as_ref()
                    .map(|s| s.search_list_size)
                    .unwrap_or(100)
            });
            let ann = if matches!(self.metric, DistanceMetric::Ip) {
                let prep = self.quantizer.prepare_ip_query(query);
                self.graph.search(0, top_k, search_list_size.max(top_k.max(1)), |node| {
                    let slot = self.ann_slots[node as usize];
                    let (indices, qjl, gamma) = self.live_codes_at_slot(slot as usize);
                    self.quantizer
                        .score_ip_encoded(&prep, indices, qjl, gamma as f64)
                })?
            } else {
                self.graph.search(0, top_k, search_list_size.max(top_k.max(1)), |node| {
                    let slot = self.ann_slots[node as usize];
                    let (indices, qjl, gamma) = self.live_codes_at_slot(slot as usize);
                    let v = self.quantizer.dequantize(indices, qjl, gamma as f64);
                    self.score_vectors(query, &v)
                })?
            };

            let ann_ids: Vec<String> = ann
                .iter()
                .map(|(node, _)| self.index_ids[*node as usize].clone())
                .collect();
            let meta_map = self.metadata.get_many(&ann_ids)?;

            let mut out = Vec::with_capacity(ann.len());
            for (idx, (node, score)) in ann.into_iter().enumerate() {
                let id = self.index_ids[node as usize].clone();
                let meta = meta_map
                    .get(&ann_ids[idx])
                    .cloned()
                    .unwrap_or_default();
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
                    .then_with(|| a.id.cmp(&b.id))
            });
            out.truncate(top_k);
            return Ok(out);
        }

        let live_ids: Vec<String> = self.live_id_to_slot.keys().cloned().collect();
        let meta_map = self.metadata.get_many(&live_ids)?;

        let mut candidates = Vec::new();
        for (id, slot) in self.live_iter_id_slots() {
            let meta = meta_map.get(&id).cloned().unwrap_or_default();
            if let Some(f) = filter {
                if !metadata_matches_filter(&meta.properties, f) {
                    continue;
                }
            }
            let v = self.live_vectors.get(&id).cloned().unwrap_or_else(|| {
                let (indices, qjl, gamma) = self.live_codes_at_slot(slot as usize);
                self.quantizer.dequantize(indices, qjl, gamma as f64)
            });
            candidates.push(SearchResult {
                id: id.clone(),
                score: self.score_vectors(query, &v),
                metadata: meta.properties,
                document: meta.document,
            });
        }

        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        candidates.truncate(top_k);
        Ok(candidates)
    }
    pub fn flush_wal_to_segment(
        &mut self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.wal_buffer.is_empty() {
            return Ok(());
        }

        let records: Vec<SegmentRecord> = self
            .wal_buffer
            .drain(..)
            .map(|e| SegmentRecord {
                id: e.id,
                quantized_indices: e.quantized_indices,
                qjl_bits: e.qjl_bits,
                gamma: e.gamma,
                is_deleted: e.is_deleted,
            })
            .collect();

        for record in &records {
            if record.is_deleted {
                self.live_delete_slot(&record.id);
                self.live_vectors.remove(&record.id);
                continue;
            }
            self.live_alloc_or_update(
                record.id.clone(),
                &record.quantized_indices,
                &record.qjl_bits,
                record.gamma,
            );
            self.live_vectors.remove(&record.id);
        }
        self.live_compact_slab();

        self.segments.flush_batch(records)?;
        self.wal.truncate()?;
        self.metadata.flush()?;

        if self.compactor.should_compact(self.segments.segments.len()) {
            self.compact_segments()?;
        }

        self.invalidate_index_state()?;
        self.manifest.vector_count = self.live_active_count() as u64;
        self.pending_manifest_updates += 1;
        self.maybe_persist_state(false)?;
        Ok(())
    }

    pub fn close(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.flush_wal_to_segment()?;
        save_quantizer_state(&self.local_dir, &self.backend, &self.quantizer)?;
        self.metadata.flush()?;
        self.pending_manifest_updates += 1;
        self.maybe_persist_state(true)?;
        Ok(())
    }

    pub fn vector_count(&self) -> u64 {
        self.live_active_count() as u64
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
        }
    }

    fn live_qjl_len(&self) -> usize {
        self.d.div_ceil(8)
    }

    fn live_stride(&self) -> usize {
        self.d + self.live_qjl_len() + LIVE_GAMMA_BYTES + LIVE_DELETED_BYTES
    }

    fn live_active_count(&self) -> usize {
        self.live_id_to_slot.len()
    }

    fn live_slot_base(&self, slot: usize) -> usize {
        slot * self.live_stride()
    }

    fn live_codes_at_slot(&self, slot: usize) -> (&[u8], &[u8], f32) {
        let base = self.live_slot_base(slot);
        let qjl_len = self.live_qjl_len();
        let gamma_off = base + self.d + qjl_len;
        let deleted_off = gamma_off + LIVE_GAMMA_BYTES;

        let indices = &self.live_codes[base..base + self.d];
        let qjl = &self.live_codes[base + self.d..base + self.d + qjl_len];
        let gamma = f32::from_le_bytes(self.live_codes[gamma_off..gamma_off + 4].try_into().unwrap());
        debug_assert_eq!(self.live_codes[deleted_off], 0u8);
        (indices, qjl, gamma)
    }

    fn live_alloc_slot(&mut self, id: String, indices: &[u8], qjl: &[u8], gamma: f32) -> u32 {
        let slot = self.live_slot_to_id.len() as u32;
        self.live_codes.extend_from_slice(indices);
        self.live_codes.extend_from_slice(qjl);
        self.live_codes.extend_from_slice(&gamma.to_le_bytes());
        self.live_codes.push(0u8);
        self.live_id_to_slot.insert(id.clone(), slot);
        self.live_slot_to_id.push(Some(id));
        slot
    }

    fn live_alloc_or_update(&mut self, id: String, indices: &[u8], qjl: &[u8], gamma: f32) -> u32 {
        if let Some(&slot) = self.live_id_to_slot.get(&id) {
            let base = self.live_slot_base(slot as usize);
            let qjl_len = self.live_qjl_len();
            let gamma_off = base + self.d + qjl_len;
            let deleted_off = gamma_off + LIVE_GAMMA_BYTES;
            self.live_codes[base..base + self.d].copy_from_slice(indices);
            self.live_codes[base + self.d..base + self.d + qjl_len].copy_from_slice(qjl);
            self.live_codes[gamma_off..gamma_off + 4].copy_from_slice(&gamma.to_le_bytes());
            self.live_codes[deleted_off] = 0u8;
            self.live_slot_to_id[slot as usize] = Some(id);
            slot
        } else {
            self.live_alloc_slot(id, indices, qjl, gamma)
        }
    }

    fn live_delete_slot(&mut self, id: &str) {
        if let Some(slot) = self.live_id_to_slot.remove(id) {
            let base = self.live_slot_base(slot as usize);
            let deleted_off = base + self.d + self.live_qjl_len() + LIVE_GAMMA_BYTES;
            self.live_codes[deleted_off] = 1u8;
            self.live_slot_to_id[slot as usize] = None;
        }
    }

    fn live_iter_id_slots(&self) -> Vec<(String, u32)> {
        self.live_id_to_slot
            .iter()
            .map(|(id, slot)| (id.clone(), *slot))
            .collect()
    }

    fn live_compact_slab(&mut self) {
        let stride = self.live_stride();
        let mut new_codes = Vec::with_capacity(self.live_active_count() * stride);
        let mut new_slots: Vec<Option<String>> = Vec::with_capacity(self.live_active_count());
        let mut new_map: HashMap<String, u32> = HashMap::with_capacity(self.live_active_count());

        for (old_slot, maybe_id) in self.live_slot_to_id.iter().enumerate() {
            if let Some(id) = maybe_id {
                let new_slot = new_slots.len() as u32;
                new_map.insert(id.clone(), new_slot);
                new_slots.push(Some(id.clone()));
                let base = self.live_slot_base(old_slot);
                new_codes.extend_from_slice(&self.live_codes[base..base + stride]);
            }
        }

        self.live_codes = new_codes;
        self.live_slot_to_id = new_slots;
        self.live_id_to_slot = new_map;
    }

    fn rebuild_ann_slots_from_index_ids(&mut self) {
        if self.index_ids.is_empty() {
            self.ann_slots.clear();
            return;
        }
        let mut slots = Vec::with_capacity(self.index_ids.len());
        for id in &self.index_ids {
            let Some(slot) = self.live_id_to_slot.get(id) else {
                self.index_ids.clear();
                self.ann_slots.clear();
                self.manifest.index_state = None;
                self.index_ids_dirty = true;
                return;
            };
            slots.push(*slot);
        }
        self.ann_slots = slots;
    }

    fn write_many(
        &mut self,
        items: Vec<BatchWriteItem>,
        mode: BatchWriteMode,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if items.is_empty() {
            return Ok(());
        }

        for item in &items {
            if item.vector.len() != self.d {
                return Err(format!(
                    "Vector dimension mismatch for '{}': got {}, expected {}",
                    item.id,
                    item.vector.len(),
                    self.d
                )
                .into());
            }
            match mode {
                BatchWriteMode::Insert if self.live_id_to_slot.contains_key(&item.id) => {
                    return Err(format!("ID '{}' already exists; use upsert/update", item.id).into())
                }
                BatchWriteMode::Update if !self.live_id_to_slot.contains_key(&item.id) => {
                    return Err(format!("ID '{}' does not exist; use insert/upsert", item.id).into())
                }
                _ => {}
            }
        }

        let mut wal_entries = Vec::with_capacity(items.len());
        let mut metadata_entries = Vec::with_capacity(items.len());
        let mut live_updates: Vec<(String, Vec<u8>, Vec<u8>, f32)> = Vec::with_capacity(items.len());

        let vectors: Vec<Array1<f64>> = items.iter().map(|item| item.vector.clone()).collect();
        let quantized: Vec<(Vec<CodeIndex>, Vec<u8>, f64)> =
            self.quantizer.quantize_batch(&vectors);

        for (item, (indices, qjl, gamma)) in items.into_iter().zip(quantized.into_iter()) {
            let meta = VectorMetadata {
                properties: item.metadata,
                document: item.document,
            };
            let metadata_json = serde_json::to_string(&meta)?;
            wal_entries.push(WalEntry {
                id: item.id.clone(),
                quantized_indices: indices.clone(),
                qjl_bits: qjl.clone(),
                gamma: gamma as f32,
                metadata_json,
                is_deleted: false,
            });
            metadata_entries.push((item.id.clone(), meta));
            live_updates.push((item.id, indices, qjl, gamma as f32));
        }

        self.wal.append_batch(&wal_entries, false)?;
        self.wal_buffer.extend(wal_entries);
        self.metadata.put_many(&metadata_entries)?;
        for (id, indices, qjl, gamma) in live_updates {
            self.live_alloc_or_update(id, &indices, &qjl, gamma);
        }

        self.invalidate_index_state()?;
        self.manifest.vector_count = self.live_active_count() as u64;
        self.pending_manifest_updates += 1;
        self.maybe_persist_state(false)?;
        if self.wal_buffer.len() >= self.wal_flush_threshold {
            self.flush_wal_to_segment()?;
        }
        Ok(())
    }

    fn write_vector_entry(
        &mut self,
        id: String,
        vector: &Array1<f64>,
        metadata_props: HashMap<String, JsonValue>,
        document: Option<String>,
        is_deleted: bool,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if vector.len() != self.d {
            return Err(format!("Vector dimension mismatch: got {}, expected {}", vector.len(), self.d).into());
        }

        let (indices, qjl, gamma) = self.quantizer.quantize(vector);
        let meta = VectorMetadata {
            properties: metadata_props,
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
            self.metadata.put(&id, &meta)?;
            self.live_alloc_or_update(id, &entry.quantized_indices, &entry.qjl_bits, entry.gamma);
        }

        self.invalidate_index_state()?;
        self.manifest.vector_count = self.live_active_count() as u64;
        self.pending_manifest_updates += 1;
        self.maybe_persist_state(false)?;
        if self.wal_buffer.len() >= self.wal_flush_threshold {
            self.flush_wal_to_segment()?;
        }
        Ok(())
    }

    fn rebuild_live_codes_cache(
        &mut self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut by_id: HashMap<String, SegmentRecord> = HashMap::new();
        for record in self.segments.iter_all_records()? {
            by_id.insert(record.id.clone(), record);
        }
        for entry in &self.wal_buffer {
            by_id.insert(
                entry.id.clone(),
                SegmentRecord {
                    id: entry.id.clone(),
                    quantized_indices: entry.quantized_indices.clone(),
                    qjl_bits: entry.qjl_bits.clone(),
                    gamma: entry.gamma,
                    is_deleted: entry.is_deleted,
                },
            );
        }
        by_id.retain(|_, r| !r.is_deleted);

        self.live_codes.clear();
        self.live_slot_to_id.clear();
        self.live_id_to_slot.clear();
        self.live_vectors.clear();

        let mut records: Vec<_> = by_id.into_values().collect();
        records.sort_by(|a, b| a.id.cmp(&b.id));
        for record in records {
            self.live_alloc_or_update(record.id, &record.quantized_indices, &record.qjl_bits, record.gamma);
        }
        Ok(())
    }

    fn invalidate_index_state(
        &mut self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.index_ids.is_empty() && self.ann_slots.is_empty() && self.manifest.index_state.is_none() {
            return Ok(());
        }
        self.index_ids.clear();
        self.ann_slots.clear();
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
                .write(INDEX_IDS_FILE, serde_json::to_vec_pretty(&self.index_ids)?)?;
            self.index_ids_dirty = false;
        }

        self.save_manifest()?;
        self.pending_manifest_updates = 0;
        Ok(())
    }

    fn can_use_ann_index(&self) -> bool {
        self.graph.has_index()
            && !self.index_ids.is_empty()
            && self.index_ids.len() == self.ann_slots.len()
            && self
                .manifest
                .index_state
                .as_ref()
                .is_some_and(|s| s.indexed_nodes == self.index_ids.len())
    }

    fn compact_segments(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.segments.segments.len() <= 1 {
            return Ok(());
        }

        let old_segment_names: Vec<String> = self
            .segments
            .segments
            .iter()
            .map(|s| s.name.clone())
            .collect();

        let mut live_records: Vec<SegmentRecord> = self
            .live_iter_id_slots()
            .into_iter()
            .map(|(id, slot)| {
                let (indices, qjl, gamma) = self.live_codes_at_slot(slot as usize);
                SegmentRecord {
                    id,
                    quantized_indices: indices.to_vec(),
                    qjl_bits: qjl.to_vec(),
                    gamma,
                    is_deleted: false,
                }
            })
            .collect();
        live_records.sort_by(|a, b| a.id.cmp(&b.id));

        let new_segment_name = self.segments.next_segment_name();
        self.compactor
            .begin_compaction(&old_segment_names, &new_segment_name)?;
        self.segments
            .flush_batch_named(new_segment_name.clone(), live_records)?;
        for old in &old_segment_names {
            self.backend.delete(old)?;
        }
        self.segments.remove_segments(&old_segment_names);
        self.compactor.finish_compaction()?;
        Ok(())
    }

    fn save_manifest(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut m = self.manifest.clone();
        m.version = m.version.max(2);
        m.quantizer = None;
        let manifest_path = format!("{}/manifest.json", self.local_dir);
        m.save(&manifest_path)?;
        self.backend
            .write("manifest.json", serde_json::to_vec_pretty(&m)?)?;
        Ok(())
    }

    fn score_vectors(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        score_vectors_with_metric(&self.metric, a, b)
    }

    fn total_disk_bytes(&self) -> u64 {
        let mut total = self.segments.total_disk_size();
        for name in ["manifest.json", "metadata.bin", "metadata.redb", "wal.log", QUANTIZER_STATE_FILE, "graph.bin", INDEX_IDS_FILE] {
            if let Ok(data) = self.backend.read(name) {
                total += data.len() as u64;
            }
        }
        total
    }
}

fn save_quantizer_state(
    local_dir: &str,
    backend: &Arc<StorageBackend>,
    quantizer: &ProdQuantizer,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let bytes = bincode::serialize(quantizer)?;
    std::fs::write(format!("{}/{}", local_dir, QUANTIZER_STATE_FILE), &bytes)?;
    backend.write(QUANTIZER_STATE_FILE, bytes)?;
    Ok(())
}

fn load_quantizer_state(
    local_dir: &str,
    backend: &Arc<StorageBackend>,
    manifest: &Manifest,
) -> Result<ProdQuantizer, Box<dyn std::error::Error + Send + Sync>> {
    if let Some(q) = manifest.quantizer.clone() {
        return Ok(q);
    }

    let local = format!("{}/{}", local_dir, QUANTIZER_STATE_FILE);
    if Path::new(&local).exists() {
        let bytes = std::fs::read(&local)?;
        return Ok(bincode::deserialize(&bytes)?);
    }

    if let Ok(bytes) = backend.read(QUANTIZER_STATE_FILE) {
        std::fs::write(&local, &bytes)?;
        return Ok(bincode::deserialize(&bytes)?);
    }

    Err("Manifest missing quantizer state and quantizer.bin not found".into())
}
fn save_index_ids(
    local_dir: &str,
    ids: &[String],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    std::fs::write(
        format!("{}/{}", local_dir, INDEX_IDS_FILE),
        serde_json::to_vec_pretty(ids)?,
    )?;
    Ok(())
}

fn load_index_ids(
    local_dir: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
    let local = format!("{}/{}", local_dir, INDEX_IDS_FILE);
    if Path::new(&local).exists() {
        let raw = std::fs::read(&local)?;
        return Ok(serde_json::from_slice(&raw)?);
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
            let an: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let bn: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
            if an == 0.0 || bn == 0.0 {
                0.0
            } else {
                dot / (an * bn)
            }
        }
        DistanceMetric::L2 => {
            -a.iter()
                .zip(b.iter())
                .map(|(x, y)| {
                    let d = x - y;
                    d * d
                })
                .sum::<f64>()
                .sqrt()
        }
    }
}







































