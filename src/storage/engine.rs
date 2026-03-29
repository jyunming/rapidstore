use ndarray::Array1;
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

const QUANTIZER_STATE_FILE: &str = "quantizer.bin";
const INDEX_IDS_FILE: &str = "graph_ids.json";

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
    index_vectors: Vec<Array1<f64>>, // cached once per index build/open
    live_records: HashMap<String, SegmentRecord>,
    live_vectors: HashMap<String, Array1<f64>>,
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
        let metadata_path = format!("{}/metadata.redb", local_dir);

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
            index_vectors: Vec::new(),
            live_records: HashMap::new(),
            live_vectors: HashMap::new(),
        };

        let pending = Wal::replay(&wal_path)?;
        if !pending.is_empty() {
            engine.wal_buffer.extend(pending);
            engine.flush_wal_to_segment()?;
        } else {
            engine.rebuild_live_records_cache()?;
            engine.manifest.vector_count = engine.live_records.len() as u64;
            engine.save_manifest()?;
        }

        engine.refresh_index_vectors_cache()?;
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
        if self.live_records.contains_key(&id) {
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
        if !self.live_records.contains_key(&id) {
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
        if !self.live_records.contains_key(&id) {
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
        self.wal.append(&entry)?;
        self.wal_buffer.push(entry);
        self.metadata.delete(&id)?;
        self.live_records.remove(&id);
        self.live_vectors.remove(&id);
        self.invalidate_index_state()?;
        self.manifest.vector_count = self.live_records.len() as u64;
        if self.wal_buffer.len() >= self.wal_flush_threshold {
            self.flush_wal_to_segment()?;
        }
        Ok(true)
    }

    pub fn get(
        &self,
        id: &str,
    ) -> Result<Option<GetResult>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.live_records.contains_key(id) {
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
        let mut records: Vec<SegmentRecord> = self.live_records.values().cloned().collect();
        if records.is_empty() {
            self.invalidate_index_state()?;
            self.save_manifest()?;
            return Ok(());
        }
        records.sort_by(|a, b| a.id.cmp(&b.id));
        let indexed_ids: Vec<String> = records.iter().map(|r| r.id.clone()).collect();
        let all_vectors: Vec<Array1<f64>> = indexed_ids
            .iter()
            .map(|id| {
                self.live_vectors.get(id).cloned().unwrap_or_else(|| {
                    let r = &self.live_records[id];
                    self.quantizer.dequantize(&r.quantized_indices, &r.qjl_bits, r.gamma as f64)
                })
            })
            .collect();

        let metric = self.metric.clone();
        let build_scorer = |from: u32, to: u32| {
            score_vectors_with_metric(&metric, &all_vectors[from as usize], &all_vectors[to as usize])
        };

        self.graph
            .build(records.len(), max_degree, alpha, build_scorer)?;

        self.index_ids = indexed_ids;
        self.index_vectors = all_vectors;
        save_index_ids(&self.local_dir, &self.index_ids)?;
        self.backend
            .write(INDEX_IDS_FILE, serde_json::to_vec_pretty(&self.index_ids)?)?;

        self.manifest.index_state = Some(IndexState {
            max_degree,
            search_list_size,
            alpha,
            indexed_nodes: self.index_ids.len(),
        });
        self.save_manifest()?;
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
        if self.live_records.is_empty() || top_k == 0 {
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
            let ann = self
                .graph
                .search(0, top_k, search_list_size.max(top_k.max(1)), |node| {
                    self.score_vectors(query, &self.index_vectors[node as usize])
                })?;

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

        let live_ids: Vec<String> = self.live_records.keys().cloned().collect();
        let meta_map = self.metadata.get_many(&live_ids)?;

        let mut candidates = Vec::new();
        for (id, record) in &self.live_records {
            let meta = meta_map.get(id).cloned().unwrap_or_default();
            if let Some(f) = filter {
                if !metadata_matches_filter(&meta.properties, f) {
                    continue;
                }
            }
            let v = self.live_vectors.get(id).cloned().unwrap_or_else(|| {
                self.quantizer
                    .dequantize(&record.quantized_indices, &record.qjl_bits, record.gamma as f64)
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

        self.segments.flush_batch(records)?;
        self.wal.truncate()?;

        if self.compactor.should_compact(self.segments.segments.len()) {
            self.compact_segments()?;
        }

        self.rebuild_live_records_cache()?;
        self.invalidate_index_state()?;
        self.manifest.vector_count = self.live_records.len() as u64;
        self.save_manifest()?;
        Ok(())
    }

    pub fn close(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.flush_wal_to_segment()?;
        save_quantizer_state(&self.local_dir, &self.backend, &self.quantizer)?;
        self.save_manifest()?;
        Ok(())
    }

    pub fn vector_count(&self) -> u64 {
        self.live_records.len() as u64
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
                BatchWriteMode::Insert if self.live_records.contains_key(&item.id) => {
                    return Err(format!("ID '{}' already exists; use upsert/update", item.id).into())
                }
                BatchWriteMode::Update if !self.live_records.contains_key(&item.id) => {
                    return Err(format!("ID '{}' does not exist; use insert/upsert", item.id).into())
                }
                _ => {}
            }
        }

        let mut wal_entries = Vec::with_capacity(items.len());
        let mut metadata_entries = Vec::with_capacity(items.len());
        let mut live_updates = Vec::with_capacity(items.len());

        for item in items {
            let meta = VectorMetadata {
                properties: item.metadata,
                document: item.document,
            };
            let (indices, qjl, gamma) = self.quantizer.quantize(&item.vector);
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
            live_updates.push((
                item.id,
                item.vector,
                SegmentRecord {
                    id: wal_entries.last().unwrap().id.clone(),
                    quantized_indices: indices,
                    qjl_bits: qjl,
                    gamma: gamma as f32,
                    is_deleted: false,
                },
            ));
        }

        self.wal.append_batch(&wal_entries)?;
        self.wal_buffer.extend(wal_entries.clone());
        self.metadata.put_many(&metadata_entries)?;
        for (id, vec, record) in live_updates {
            self.live_vectors.insert(id.clone(), vec);
            self.live_records.insert(id, record);
        }


        self.invalidate_index_state()?;
        self.manifest.vector_count = self.live_records.len() as u64;

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

        self.wal.append(&entry)?;
        self.wal_buffer.push(entry.clone());

        if !is_deleted {
        if !is_deleted {
            self.metadata.put(&id, &meta)?;
            self.live_vectors.insert(id.clone(), vector.clone());
            self.live_records.insert(
                id,
                SegmentRecord {
                    id: entry.id,
                    quantized_indices: entry.quantized_indices,
                    qjl_bits: entry.qjl_bits,
                    gamma: entry.gamma,
                    is_deleted: false,
                },
            );
        }
        }

        self.invalidate_index_state()?;
        self.manifest.vector_count = self.live_records.len() as u64;

        if self.wal_buffer.len() >= self.wal_flush_threshold {
            self.flush_wal_to_segment()?;
        }
        Ok(())
    }

    fn rebuild_live_records_cache(
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
        let by_vec: HashMap<String, Array1<f64>> = by_id
            .iter()
            .map(|(id, r)| {
                (
                    id.clone(),
                    self.quantizer
                        .dequantize(&r.quantized_indices, &r.qjl_bits, r.gamma as f64),
                )
            })
            .collect();
        self.live_records = by_id;
        self.live_vectors = by_vec;
        Ok(())
    }
    fn refresh_index_vectors_cache(
        &mut self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.index_vectors.clear();
        if self.index_ids.is_empty() {
            return Ok(());
        }
        for id in &self.index_ids {
            let Some(record) = self.live_records.get(id) else {
                self.index_ids.clear();
                self.manifest.index_state = None;
                return Ok(());
            };
            self.index_vectors.push(self.quantizer.dequantize(
                &record.quantized_indices,
                &record.qjl_bits,
                record.gamma as f64,
            ));
        }
        Ok(())
    }

    fn invalidate_index_state(
        &mut self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.index_ids.is_empty() && self.manifest.index_state.is_none() {
            self.index_vectors.clear();
            return Ok(());
        }
        self.index_ids.clear();
        self.index_vectors.clear();
        self.manifest.index_state = None;
        save_index_ids(&self.local_dir, &self.index_ids)?;
        self.backend
            .write(INDEX_IDS_FILE, serde_json::to_vec_pretty(&self.index_ids)?)?;
        Ok(())
    }

    fn can_use_ann_index(&self) -> bool {
        self.graph.has_index()
            && !self.index_ids.is_empty()
            && self.index_ids.len() == self.index_vectors.len()
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

        let mut live_records: Vec<SegmentRecord> = self.live_records.values().cloned().collect();
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
        for name in ["manifest.json", "metadata.redb", "wal.log", QUANTIZER_STATE_FILE, "graph.bin", INDEX_IDS_FILE] {
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

















