use std::path::PathBuf;
use std::sync::Arc;
use ndarray::Array1;
use serde::{Serialize, Deserialize};
use serde_json::Value as JsonValue;

use crate::quantizer::prod::ProdQuantizer;
use super::wal::{Wal, WalEntry};
use super::segment::{SegmentManager, SegmentRecord};
use super::metadata::{MetadataStore, VectorMetadata};
use super::compaction::Compactor;

/// Database manifest persisted to `manifest.json`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Manifest {
    pub version: u32,
    pub d: usize,
    pub b: usize,
    pub seed: u64,
    pub vector_count: u64,
    pub storage_uri: String,
    pub quantizer: Option<ProdQuantizer>, // Serialized quantization state for reproducibility
}

impl Manifest {
    fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }
}

/// The core disk-native TurboQuant engine.
/// Insert path:    Python → WAL (fsync) → memory buffer → SegmentManager → disk
/// Search path:    SegmentManager (scan/graph) → quantizer.dequantize → dot product
pub struct TurboQuantEngine {
    pub d: usize,
    pub b: usize,
    pub quantizer: ProdQuantizer,
    pub manifest: Manifest,

    // Storage layers
    wal: Wal,
    wal_buffer: Vec<WalEntry>,
    wal_flush_threshold: usize,
    segments: SegmentManager,
    metadata: MetadataStore,
    compactor: Compactor,
    db_dir: String,
}

impl TurboQuantEngine {
    /// Create or open a TurboQuantDB database at `db_dir`.
    /// If `manifest.json` exists, the existing database is opened.
    /// Otherwise a new database is created with the given `d`, `b`, `seed`.
    pub fn open(db_dir: &str, d: usize, b: usize, seed: u64) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        std::fs::create_dir_all(db_dir)?;
        let manifest_path = format!("{}/manifest.json", db_dir);
        let wal_path = format!("{}/wal.log", db_dir);
        let segments_dir = format!("{}/segments", db_dir);
        let metadata_path = format!("{}/metadata.redb", db_dir);

        // Load or create manifest
        let (manifest, quantizer) = if std::path::Path::new(&manifest_path).exists() {
            let m = Manifest::load(&manifest_path)?;
            if m.d != d || m.b != b {
                return Err(format!("Schema mismatch: DB has d={}, b={} but got d={}, b={}", m.d, m.b, d, b).into());
            }
            let q = m.quantizer.clone().ok_or("Manifest missing quantizer state")?;
            (m, q)
        } else {
            let q = ProdQuantizer::new(d, b, seed);
            let m = Manifest { 
                version: 1, 
                d, 
                b, 
                seed, 
                vector_count: 0, 
                storage_uri: db_dir.to_string(),
                quantizer: Some(q.clone()),
            };
            m.save(&manifest_path)?;
            (m, q)
        };

        let segments = SegmentManager::open(&segments_dir)?;
        let metadata = MetadataStore::open(&metadata_path)?;

        // Replay WAL on startup for crash recovery
        let pending = Wal::replay(&wal_path)?;

        let wal = Wal::open(&wal_path)?;
        let compactor = Compactor::new(&segments_dir);

        let mut engine = Self {
            d,
            b,
            quantizer,
            manifest,
            wal,
            wal_buffer: Vec::new(),
            wal_flush_threshold: 1024, // Flush to segment every 1024 vectors
            segments,
            metadata,
            compactor,
            db_dir: db_dir.to_string(),
        };

        // Replay pending WAL entries into memory buffer then flush them 
        if !pending.is_empty() {
            for entry in pending {
                engine.wal_buffer.push(entry);
            }
            engine.flush_wal_to_segment()?;
        }

        Ok(engine)
    }

    /// Insert a vector into the database.
    pub fn insert(
        &mut self,
        id: String,
        vector: &Array1<f64>,
        metadata_props: std::collections::HashMap<String, JsonValue>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        assert_eq!(vector.len(), self.d, "Vector dimension mismatch");

        // 1. Quantize
        let (indices, qjl, gamma) = self.quantizer.quantize(vector);
        let gamma_f32 = gamma as f32;

        // 2. Build metadata
        let meta = VectorMetadata { properties: metadata_props.clone() };
        let meta_json = serde_json::to_string(&meta)?;

        // 3. Write to WAL (crash-safe)
        let entry = WalEntry {
            id: id.clone(),
            quantized_indices: indices,
            qjl_bits: qjl,
            gamma: gamma_f32,
            metadata_json: meta_json,
        };
        self.wal.append(&entry)?;

        // 4. Store metadata separately in redb
        self.metadata.put(&id, &meta)?;

        // 5. Buffer the entry
        self.wal_buffer.push(entry);
        self.manifest.vector_count += 1;

        // 6. Flush buffer to segment when threshold is reached
        if self.wal_buffer.len() >= self.wal_flush_threshold {
            self.flush_wal_to_segment()?;
        }

        Ok(())
    }

    /// Force-flush WAL buffer to a new immutable segment.
    pub fn flush_wal_to_segment(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.wal_buffer.is_empty() {
            return Ok(());
        }

        let records: Vec<SegmentRecord> = self.wal_buffer.drain(..).map(|e| SegmentRecord {
            id: e.id,
            quantized_indices: e.quantized_indices,
            qjl_bits: e.qjl_bits,
            gamma: e.gamma,
        }).collect();

        self.segments.flush_batch(records)?;
        self.wal.truncate()?;

        // Sync manifest count from actual on-disk segments (handles WAL replay accurately)
        self.manifest.vector_count = self.segments.total_vectors() as u64;

        // Save updated manifest
        let manifest_path = format!("{}/manifest.json", self.db_dir);
        self.manifest.save(&manifest_path)?;

        Ok(())
    }

    /// Search for the top_k most similar vectors using brute-force scan over all segments.
    /// Phase 3 will replace this with Vamana graph traversal.
    pub fn search(
        &self,
        query: &Array1<f64>,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error + Send + Sync>> {
        // Load WAL buffer entries for in-flight inserts
        let mut candidates: Vec<(String, f64)> = Vec::new();

        // Score buffered (not-yet-flushed) entries
        for entry in &self.wal_buffer {
            let x_tilde = self.quantizer.dequantize(
                &entry.quantized_indices,
                &entry.qjl_bits,
                entry.gamma as f64,
            );
            let score: f64 = query.iter().zip(x_tilde.iter()).map(|(a, b)| a * b).sum();
            candidates.push((entry.id.clone(), score));
        }

        // Score all segment records
        for record in self.segments.iter_all_records()? {
            let x_tilde = self.quantizer.dequantize(
                &record.quantized_indices,
                &record.qjl_bits,
                record.gamma as f64,
            );
            let score: f64 = query.iter().zip(x_tilde.iter()).map(|(a, b)| a * b).sum();
            candidates.push((record.id, score));
        }

        // Sort by score descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(top_k);

        // Fetch metadata for top hits
        let mut results = Vec::with_capacity(candidates.len());
        for (id, score) in candidates {
            let meta = self.metadata.get(&id)?.unwrap_or_default();
            results.push(SearchResult { id, score, metadata: meta.properties });
        }

        Ok(results)
    }

    /// Close the database, flushing any remaining WAL buffer to disk.
    pub fn close(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.flush_wal_to_segment()?;
        Ok(())
    }

    /// Returns total vector count including buffered (unflushed) entries
    pub fn vector_count(&self) -> u64 {
        self.manifest.vector_count + self.wal_buffer.len() as u64
    }

    pub fn stats(&self) -> DbStats {
        DbStats {
            vector_count: self.manifest.vector_count,
            segment_count: self.segments.segments.len(),
            buffered_vectors: self.wal_buffer.len(),
            d: self.d,
            b: self.b,
            total_disk_bytes: self.segments.total_disk_size(),
        }
    }
}

/// A single search result with score and metadata.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f64,
    pub metadata: std::collections::HashMap<String, JsonValue>,
}

/// Database statistics for monitoring.
#[derive(Debug, Clone)]
pub struct DbStats {
    pub vector_count: u64,
    pub segment_count: usize,
    pub buffered_vectors: usize,
    pub d: usize,
    pub b: usize,
    pub total_disk_bytes: u64,
}
