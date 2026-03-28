use serde::{Serialize, Deserialize};
use std::io::{Write, Read};
use std::fs::{File, OpenOptions};
use std::path::Path;

/// Fixed-size record stored in a segment file.
/// Each record is exactly RECORD_SIZE bytes.
/// Layout: [id_len: u16][id_bytes: 128 bytes][indices_len: u16][indices_bytes: variable][qjl_len: u16][qjl_bytes: variable][gamma: f32]
/// For simplicity, we use length-prefixed bincode (variable size records with u32 length prefix).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SegmentRecord {
    pub id: String,
    pub quantized_indices: Vec<usize>,
    pub qjl_bits: Vec<i8>,
    pub gamma: f32,
}

/// An immutable, append-only segment file containing quantized vector records.
/// Once written, segments are never mutated — only compacted into larger segments.
pub struct Segment {
    pub path: String,
    pub record_count: usize,
}

impl Segment {
    /// Write a batch of records to a new segment file.
    pub fn write_batch<P: AsRef<Path>>(
        path: P,
        records: &[SegmentRecord],
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let path_str = path.as_ref().to_string_lossy().into_owned();
        let mut file = BufWriter::new(File::create(&path)?);
        let count = records.len() as u64;
        // Write header: record count
        file.write_all(&count.to_le_bytes())?;
        for record in records {
            let encoded = bincode::serialize(record)?;
            let len = encoded.len() as u32;
            file.write_all(&len.to_le_bytes())?;
            file.write_all(&encoded)?;
        }
        file.flush()?;
        Ok(Self {
            path: path_str,
            record_count: records.len(),
        })
    }

    /// Read all records from an existing segment file.
    pub fn read_all<P: AsRef<Path>>(
        path: P,
    ) -> Result<Vec<SegmentRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let mut file = File::open(path)?;
        let mut count_buf = [0u8; 8];
        file.read_exact(&mut count_buf)?;
        let count = u64::from_le_bytes(count_buf) as usize;
        let mut records = Vec::with_capacity(count);
        for _ in 0..count {
            let mut len_buf = [0u8; 4];
            if file.read_exact(&mut len_buf).is_err() { break; }
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut payload = vec![0u8; len];
            if file.read_exact(&mut payload).is_err() { break; }
            let record: SegmentRecord = bincode::deserialize(&payload)?;
            records.push(record);
        }
        Ok(records)
    }
}

/// Manages the directory of segment files on disk.
pub struct SegmentManager {
    pub dir: String,
    pub segments: Vec<Segment>,
    next_id: u64,
}

impl SegmentManager {
    /// Open a segment directory, loading existing segment metadata.
    pub fn open(dir: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        std::fs::create_dir_all(dir)?;
        let mut segments = Vec::new();
        let mut next_id = 0u64;

        // Scan for existing seg-*.bin files
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with("seg-") && name.ends_with(".bin") {
                // Count records by reading header
                let path = entry.path();
                let mut file = File::open(&path)?;
                let mut count_buf = [0u8; 8];
                if file.read_exact(&mut count_buf).is_ok() {
                    let count = u64::from_le_bytes(count_buf) as usize;
                    segments.push(Segment {
                        path: path.to_string_lossy().into_owned(),
                        record_count: count,
                    });
                }
                // Track highest segment ID
                if let Some(id_str) = name.strip_prefix("seg-").and_then(|s| s.strip_suffix(".bin")) {
                    if let Ok(id) = id_str.parse::<u64>() {
                        next_id = next_id.max(id + 1);
                    }
                }
            }
        }

        Ok(Self { dir: dir.to_string(), segments, next_id })
    }

    /// Flush a batch of WAL entries into a new immutable segment.
    pub fn flush_batch(
        &mut self,
        records: Vec<SegmentRecord>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if records.is_empty() {
            return Ok(());
        }
        let filename = format!("{}/seg-{:08}.bin", self.dir, self.next_id);
        let seg = Segment::write_batch(&filename, &records)?;
        self.segments.push(seg);
        self.next_id += 1;
        Ok(())
    }

    /// Iterate over all records across all segments (for brute-force scan / graph build).
    pub fn iter_all_records(
        &self,
    ) -> Result<Vec<SegmentRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let mut all = Vec::new();
        for seg in &self.segments {
            let records = Segment::read_all(&seg.path)?;
            all.extend(records);
        }
        Ok(all)
    }

    pub fn total_vectors(&self) -> usize {
        self.segments.iter().map(|s| s.record_count).sum()
    }

    /// Calculate total bytes occupied by all segment files.
    pub fn total_disk_size(&self) -> u64 {
        self.segments.iter()
            .map(|s| {
                std::fs::metadata(&s.path).map(|m| m.len()).unwrap_or(0)
            })
            .sum()
    }
}

use std::io::BufWriter;
