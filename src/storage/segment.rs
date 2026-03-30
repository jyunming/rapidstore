use super::backend::StorageBackend;
use serde::{Deserialize, Serialize};
use std::io::{Cursor, Read, Write};
use std::sync::Arc;

/// Fixed-size record stored in a segment file.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SegmentRecord {
    pub id: String,
    pub quantized_indices: Vec<u8>,
    pub qjl_bits: Vec<u8>,
    pub gamma: f32,
    #[serde(default)]
    pub is_deleted: bool,
}

/// An immutable, append-only segment file containing quantized vector records.
/// Phase 4: Now works with a generic StorageBackend (Local, S3, GCS).
#[derive(Clone)]
pub struct Segment {
    pub name: String,
    pub record_count: usize,
}

impl Segment {
    /// Write a batch of records to a new segment in the backend.
    pub fn write_batch(
        backend: &StorageBackend,
        name: &str,
        records: &[SegmentRecord],
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut buf = Vec::new();
        let count = records.len() as u64;

        // Write header: record count
        buf.write_all(&count.to_le_bytes())?;
        for record in records {
            let encoded = bincode::serialize(record)?;
            let len = encoded.len() as u32;
            buf.write_all(&len.to_le_bytes())?;
            buf.write_all(&encoded)?;
        }

        backend.write(name, buf)?;

        Ok(Self {
            name: name.to_string(),
            record_count: records.len(),
        })
    }

    /// Read all records from a segment in the backend.
    pub fn read_all(
        backend: &StorageBackend,
        name: &str,
    ) -> Result<Vec<SegmentRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let data = backend.read(name)?;
        let mut cursor = Cursor::new(data);

        let mut count_buf = [0u8; 8];
        cursor.read_exact(&mut count_buf)?;
        let count = u64::from_le_bytes(count_buf) as usize;

        let mut records = Vec::with_capacity(count);
        for _ in 0..count {
            let mut len_buf = [0u8; 4];
            if cursor.read_exact(&mut len_buf).is_err() {
                break;
            }
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut payload = vec![0u8; len];
            if cursor.read_exact(&mut payload).is_err() {
                break;
            }
            let record: SegmentRecord = bincode::deserialize(&payload)?;
            records.push(record);
        }
        Ok(records)
    }
}

/// Manages segment files using the StorageBackend.
pub struct SegmentManager {
    pub backend: Arc<StorageBackend>,
    pub segments: Vec<Segment>,
    next_id: u64,
}

impl SegmentManager {
    /// Open a segment storage, listing existing segments from the backend.
    pub fn open(
        backend: Arc<StorageBackend>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut segments = Vec::new();
        let mut next_id = 0u64;

        // List files in the backend root
        let files = backend.list("")?;
        for name in files {
            if name.starts_with("seg-") && name.ends_with(".bin") {
                let data = backend.read(&name)?;
                if data.len() >= 8 {
                    let count = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
                    segments.push(Segment {
                        name: name.clone(),
                        record_count: count,
                    });
                }

                // Track highest segment ID
                if let Some(id_str) = name
                    .strip_prefix("seg-")
                    .and_then(|s| s.strip_suffix(".bin"))
                {
                    if let Ok(id) = id_str.parse::<u64>() {
                        next_id = next_id.max(id + 1);
                    }
                }
            }
        }

        segments.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Self {
            backend,
            segments,
            next_id,
        })
    }

    /// Flush a batch of WAL entries into a new immutable segment in the backend.
    pub fn flush_batch(
        &mut self,
        records: Vec<SegmentRecord>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let name = self.next_segment_name();
        self.flush_batch_named(name, records)
    }

    /// Generate the next monotonic segment file name.
    pub fn next_segment_name(&self) -> String {
        format!("seg-{:08}.bin", self.next_id)
    }

    /// Flush a batch into a specific segment file name.
    pub fn flush_batch_named(
        &mut self,
        name: String,
        records: Vec<SegmentRecord>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if records.is_empty() {
            return Ok(());
        }
        let seg = Segment::write_batch(&self.backend, &name, &records)?;
        self.segments.push(seg);

        // Keep next_id monotonic whenever the name follows `seg-<u64>.bin`.
        if let Some(id_str) = name
            .strip_prefix("seg-")
            .and_then(|s| s.strip_suffix(".bin"))
        {
            if let Ok(id) = id_str.parse::<u64>() {
                self.next_id = self.next_id.max(id + 1);
            } else {
                self.next_id += 1;
            }
        } else {
            self.next_id += 1;
        }
        Ok(())
    }

    /// Iterate over all records across all segments.
    pub fn iter_all_records(
        &self,
    ) -> Result<Vec<SegmentRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let mut all = Vec::new();
        for seg in &self.segments {
            let records = Segment::read_all(&self.backend, &seg.name)?;
            all.extend(records);
        }
        Ok(all)
    }

    pub fn total_vectors(&self) -> usize {
        self.segments.iter().map(|s| s.record_count).sum()
    }

    /// Remove segments from the internal list.
    pub fn remove_segments(&mut self, names: &[String]) {
        self.segments.retain(|s| !names.contains(&s.name));
    }

    /// Add a segment to the internal list.
    pub fn add_segment(&mut self, segment: Segment) {
        self.segments.push(segment);
    }

    pub fn total_disk_size(&self) -> u64 {
        self.segments
            .iter()
            .map(|s| {
                self.backend
                    .read(&s.name)
                    .map(|d| d.len() as u64)
                    .unwrap_or(0)
            })
            .sum()
    }

    pub fn segment_sizes_bytes(&self) -> Vec<u64> {
        self.segments
            .iter()
            .map(|s| {
                self.backend
                    .read(&s.name)
                    .map(|d| d.len() as u64)
                    .unwrap_or(0)
            })
            .collect()
    }
}


