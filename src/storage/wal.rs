use serde::{Deserialize, Serialize};

use crate::quantizer::CodeIndex;
use crate::quantizer::prod::ProdQuantizer;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const WAL_MAGIC: &[u8; 4] = b"TQWV";
const WAL_VERSION: u32 = 3;

/// On-disk V3 entry — MSE indices stored as bit-packed bytes.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct WalEntryPacked {
    pub id: String,
    pub packed_mse: Vec<u8>,
    pub qjl_bits: Vec<u8>,
    pub gamma: f32,
    pub metadata_json: String,
    #[serde(default)]
    pub is_deleted: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WalEntry {
    pub id: String,
    pub quantized_indices: Vec<CodeIndex>,
    pub qjl_bits: Vec<u8>,
    pub gamma: f32,
    pub metadata_json: String,
    #[serde(default)]
    pub is_deleted: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[doc(hidden)]
struct WalEntryLegacy {
    pub id: String,
    pub quantized_indices: Vec<CodeIndex>,
    pub qjl_bits: Vec<u8>,
    pub gamma: f32,
    pub metadata_json: String,
    #[serde(default)]
    pub is_deleted: bool,
    #[serde(default)]
    pub original_vector: Option<Vec<f32>>,
}

pub struct Wal {
    path: PathBuf,
    writer: BufWriter<File>,
    entry_count: u64,
    quantizer: Option<std::sync::Arc<ProdQuantizer>>,
}

impl Wal {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let path = path.as_ref().to_path_buf();
        let exists = path.exists();
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;

        if !exists || file.metadata()?.len() == 0 {
            file.write_all(WAL_MAGIC)?;
            file.write_all(&WAL_VERSION.to_le_bytes())?;
        }

        Ok(Self {
            path,
            writer: BufWriter::new(file),
            entry_count: 0,
            quantizer: None,
        })
    }

    /// Set the quantizer used for packing/unpacking indices (V3 format).
    pub fn set_quantizer(&mut self, q: std::sync::Arc<ProdQuantizer>) {
        self.quantizer = Some(q);
    }

    pub fn append(
        &mut self,
        entry: &WalEntry,
        force_sync: bool,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.append_batch(std::slice::from_ref(entry), force_sync)
    }

    pub fn append_batch(
        &mut self,
        entries: &[WalEntry],
        force_sync: bool,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if entries.is_empty() {
            if force_sync {
                self.sync()?;
            }
            return Ok(());
        }
        for entry in entries {
            let encoded = if let Some(q) = &self.quantizer {
                // Delete tombstones carry empty quantized_indices; skip packing for them.
                let packed_mse = if entry.is_deleted {
                    Vec::new()
                } else {
                    q.pack_mse_indices(&entry.quantized_indices)
                };
                let packed = WalEntryPacked {
                    id: entry.id.clone(),
                    packed_mse,
                    qjl_bits: entry.qjl_bits.clone(),
                    gamma: entry.gamma,
                    metadata_json: entry.metadata_json.clone(),
                    is_deleted: entry.is_deleted,
                };
                bincode::serialize(&packed)?
            } else {
                bincode::serialize(entry)?
            };
            let len = encoded.len() as u64;
            self.writer.write_all(&len.to_le_bytes())?;
            self.writer.write_all(&encoded)?;
            self.entry_count += 1;
        }
        self.writer.flush()?;
        if force_sync {
            self.writer.get_ref().sync_data()?;
        }
        Ok(())
    }

    pub fn sync(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.writer.flush()?;
        self.writer.get_ref().sync_data()?;
        Ok(())
    }

    pub fn replay<P: AsRef<Path>>(
        path: P,
        quantizer: Option<&ProdQuantizer>,
    ) -> Result<Vec<WalEntry>, Box<dyn std::error::Error + Send + Sync>> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Vec::new());
        }
        let mut file = File::open(path)?;
        let mut magic = [0u8; 4];
        let mut version = 0u32;

        let has_header = if file.read_exact(&mut magic).is_ok() && &magic == WAL_MAGIC {
            let mut v_buf = [0u8; 4];
            file.read_exact(&mut v_buf)?;
            version = u32::from_le_bytes(v_buf);
            true
        } else {
            // No magic, reset to start for legacy read
            file = File::open(path)?;
            false
        };

        let mut entries = Vec::new();
        loop {
            let mut len_buf = [0u8; 8];
            match file.read_exact(&mut len_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            let len = u64::from_le_bytes(len_buf) as usize;
            let mut payload = vec![0u8; len];
            match file.read_exact(&mut payload) {
                Ok(_) => {}
                Err(_) => break,
            }

            if has_header && version == 3 {
                // V3: packed MSE indices
                if let Ok(packed) = bincode::deserialize::<WalEntryPacked>(&payload) {
                    let quantized_indices = if packed.is_deleted || packed.packed_mse.is_empty() {
                        Vec::new()
                    } else if let Some(q) = quantizer {
                        let mut idx = vec![0 as CodeIndex; q.n];
                        q.unpack_mse_indices(&packed.packed_mse, &mut idx);
                        idx
                    } else {
                        Vec::new()
                    };
                    entries.push(WalEntry {
                        id: packed.id,
                        quantized_indices,
                        qjl_bits: packed.qjl_bits,
                        gamma: packed.gamma,
                        metadata_json: packed.metadata_json,
                        is_deleted: packed.is_deleted,
                    });
                }
            } else if has_header && version == 2 {
                if let Ok(entry) = bincode::deserialize::<WalEntry>(&payload) {
                    entries.push(entry);
                }
            } else {
                // Legacy v1 path
                if let Ok(legacy) = bincode::deserialize::<WalEntryLegacy>(&payload) {
                    entries.push(WalEntry {
                        id: legacy.id,
                        quantized_indices: legacy.quantized_indices,
                        qjl_bits: legacy.qjl_bits,
                        gamma: legacy.gamma,
                        metadata_json: legacy.metadata_json,
                        is_deleted: legacy.is_deleted,
                    });
                }
            }
        }
        Ok(entries)
    }

    pub fn truncate(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut file = File::create(&self.path)?;
        file.write_all(WAL_MAGIC)?;
        file.write_all(&WAL_VERSION.to_le_bytes())?;
        Ok(())
    }

    pub fn entry_count(&self) -> u64 {
        self.entry_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantizer::prod::ProdQuantizer;
    use std::sync::Arc;
    use tempfile::tempdir;

    // Build a WalEntry with empty vectors (no quantizer needed; empty vecs are
    // bincode-compatible between Vec<u16> and Vec<u8> so V3 replay works).
    fn tombstone_entry(id: &str) -> WalEntry {
        WalEntry {
            id: id.to_string(),
            quantized_indices: vec![],
            qjl_bits: vec![],
            gamma: 0.0,
            metadata_json: "{}".to_string(),
            is_deleted: true,
        }
    }

    // Build a WalEntry with real quantized data (requires quantizer for V3 pack/unpack).
    fn real_entry(id: &str, pq: &ProdQuantizer) -> WalEntry {
        let x: Vec<f32> = (0..pq.d).map(|i| i as f32 * 0.01).collect();
        let (idx, qjl, gamma) = pq.quantize(&x);
        WalEntry {
            id: id.to_string(),
            quantized_indices: idx,
            qjl_bits: qjl,
            gamma: gamma as f32,
            metadata_json: r#"{"key":"val"}"#.to_string(),
            is_deleted: false,
        }
    }

    fn make_pq() -> Arc<ProdQuantizer> {
        Arc::new(ProdQuantizer::new(32, 4, 42))
    }

    // -----------------------------------------------------------------------
    // Header / open
    // -----------------------------------------------------------------------

    #[test]
    fn open_creates_file_with_magic_header() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.bin");
        let _wal = Wal::open(&path).unwrap();
        drop(_wal);

        let data = std::fs::read(&path).unwrap();
        assert!(data.len() >= 8, "file should have header");
        assert_eq!(&data[0..4], b"TQWV");
        assert_eq!(u32::from_le_bytes(data[4..8].try_into().unwrap()), 3u32);
    }

    #[test]
    fn open_existing_file_does_not_duplicate_header() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.bin");
        {
            let _w = Wal::open(&path).unwrap();
        }
        let size1 = std::fs::metadata(&path).unwrap().len();
        {
            let _w = Wal::open(&path).unwrap();
        }
        let size2 = std::fs::metadata(&path).unwrap().len();
        assert_eq!(size1, size2, "reopening should not add a second header");
    }

    // -----------------------------------------------------------------------
    // append / replay with quantizer (V3 packed path)
    // -----------------------------------------------------------------------

    #[test]
    fn append_and_replay_with_quantizer_roundtrip() {
        let pq = make_pq();
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.bin");
        let entry = real_entry("id1", &pq);

        {
            let mut wal = Wal::open(&path).unwrap();
            wal.set_quantizer(Arc::clone(&pq));
            wal.append(&entry, true).unwrap();
        }

        let replayed = Wal::replay(&path, Some(&pq)).unwrap();
        assert_eq!(replayed.len(), 1);
        assert_eq!(replayed[0].id, "id1");
        assert!(!replayed[0].is_deleted);
        assert_eq!(replayed[0].metadata_json, r#"{"key":"val"}"#);
        assert_eq!(replayed[0].quantized_indices.len(), pq.n);
    }

    #[test]
    fn append_batch_and_replay_with_quantizer() {
        let pq = make_pq();
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.bin");

        let entries_in = vec![
            real_entry("a", &pq),
            real_entry("b", &pq),
            tombstone_entry("c"),
        ];
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.set_quantizer(Arc::clone(&pq));
            wal.append_batch(&entries_in, true).unwrap();
        }

        let replayed = Wal::replay(&path, Some(&pq)).unwrap();
        assert_eq!(replayed.len(), 3);
        assert_eq!(replayed[2].id, "c");
        assert!(replayed[2].is_deleted);
        assert!(replayed[2].quantized_indices.is_empty());
    }

    #[test]
    fn append_batch_empty_is_no_op() {
        let pq = make_pq();
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.bin");
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.set_quantizer(Arc::clone(&pq));
            wal.append_batch(&[], true).unwrap();
        }
        let replayed = Wal::replay(&path, Some(&pq)).unwrap();
        assert!(replayed.is_empty());
    }

    // -----------------------------------------------------------------------
    // delete tombstones (empty vectors – work without quantizer in V3)
    // -----------------------------------------------------------------------

    #[test]
    fn delete_tombstone_survives_replay() {
        let pq = make_pq();
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.bin");

        {
            let mut wal = Wal::open(&path).unwrap();
            wal.set_quantizer(Arc::clone(&pq));
            wal.append(&real_entry("vec1", &pq), false).unwrap();
            wal.append(&tombstone_entry("vec1"), true).unwrap();
        }

        let replayed = Wal::replay(&path, Some(&pq)).unwrap();
        assert_eq!(replayed.len(), 2);
        assert!(!replayed[0].is_deleted);
        assert!(replayed[1].is_deleted);
        assert_eq!(replayed[1].id, "vec1");
    }

    // -----------------------------------------------------------------------
    // truncate
    // -----------------------------------------------------------------------

    #[test]
    fn truncate_clears_all_entries() {
        let pq = make_pq();
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.bin");
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.set_quantizer(Arc::clone(&pq));
            wal.append(&real_entry("x", &pq), true).unwrap();
            wal.truncate().unwrap();
        }
        let replayed = Wal::replay(&path, Some(&pq)).unwrap();
        assert!(replayed.is_empty(), "truncate should remove all entries");
    }

    // -----------------------------------------------------------------------
    // entry_count
    // -----------------------------------------------------------------------

    #[test]
    fn entry_count_tracks_appends() {
        let pq = make_pq();
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.bin");
        let mut wal = Wal::open(&path).unwrap();
        wal.set_quantizer(Arc::clone(&pq));
        assert_eq!(wal.entry_count(), 0);
        wal.append(&real_entry("a", &pq), false).unwrap();
        assert_eq!(wal.entry_count(), 1);
        wal.append(&real_entry("b", &pq), false).unwrap();
        assert_eq!(wal.entry_count(), 2);
    }

    // -----------------------------------------------------------------------
    // replay nonexistent file
    // -----------------------------------------------------------------------

    #[test]
    fn replay_nonexistent_path_returns_empty() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("no_such_file.bin");
        let entries = Wal::replay(&path, None).unwrap();
        assert!(entries.is_empty());
    }

    // -----------------------------------------------------------------------
    // replay without quantizer yields empty quantized_indices
    // -----------------------------------------------------------------------

    #[test]
    fn replay_without_quantizer_gives_empty_indices() {
        let pq = make_pq();
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.bin");

        {
            let mut wal = Wal::open(&path).unwrap();
            wal.set_quantizer(Arc::clone(&pq));
            wal.append(&real_entry("v", &pq), true).unwrap();
        }

        // Replay without quantizer: packed_mse is non-empty but quantizer=None
        // → quantized_indices will be Vec::new()
        let replayed = Wal::replay(&path, None).unwrap();
        assert_eq!(replayed.len(), 1);
        assert_eq!(replayed[0].id, "v");
        assert!(
            replayed[0].quantized_indices.is_empty(),
            "no quantizer → cannot unpack, so indices are empty"
        );
    }

    // -----------------------------------------------------------------------
    // sync
    // -----------------------------------------------------------------------

    #[test]
    fn sync_flushes_without_error() {
        let pq = make_pq();
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal_sync.bin");
        let mut wal = Wal::open(&path).unwrap();
        wal.set_quantizer(Arc::clone(&pq));
        wal.append(&real_entry("s1", &pq), false).unwrap();
        wal.sync().unwrap(); // explicit sync should not error
    }

    // -----------------------------------------------------------------------
    // Legacy V1 format (no header) — entries without magic bytes
    // -----------------------------------------------------------------------

    #[test]
    fn replay_legacy_v1_format_no_header() {
        // Build a V1 WAL manually: no TQWV header, just length-prefixed WalEntryLegacy structs.
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal_v1.bin");

        use std::fs::File;
        use std::io::Write;
        // Use WalEntryLegacy directly (accessible via use super::*) to guarantee format match.
        let entry = WalEntryLegacy {
            id: "legacy1".to_string(),
            quantized_indices: vec![0u16, 1u16],
            qjl_bits: vec![0b10101010u8],
            gamma: 0.5,
            metadata_json: "{}".to_string(),
            is_deleted: false,
            original_vector: None,
        };
        let encoded = bincode::serialize(&entry).unwrap();
        let mut f = File::create(&path).unwrap();
        f.write_all(&(encoded.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&encoded).unwrap();
        drop(f);

        // Replay: no magic → falls back to V1 path
        let replayed = Wal::replay(&path, None).unwrap();
        assert_eq!(replayed.len(), 1, "V1 format should produce one entry");
        assert_eq!(replayed[0].id, "legacy1");
        assert_eq!(replayed[0].gamma, 0.5);
    }

    // -----------------------------------------------------------------------
    // V2 format (header version=2, plain WalEntry without bit-packing)
    // -----------------------------------------------------------------------

    #[test]
    fn replay_v2_format() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal_v2.bin");

        use std::fs::File;
        use std::io::Write;
        let entry = WalEntry {
            id: "v2entry".to_string(),
            quantized_indices: vec![5u16, 10u16],
            qjl_bits: vec![0xFFu8],
            gamma: 1.0,
            metadata_json: r#"{"x":1}"#.to_string(),
            is_deleted: false,
        };
        let encoded = bincode::serialize(&entry).unwrap();

        let mut f = File::create(&path).unwrap();
        f.write_all(b"TQWV").unwrap();
        f.write_all(&2u32.to_le_bytes()).unwrap(); // version = 2
        f.write_all(&(encoded.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&encoded).unwrap();

        let replayed = Wal::replay(&path, None).unwrap();
        assert_eq!(replayed.len(), 1);
        assert_eq!(replayed[0].id, "v2entry");
        assert_eq!(replayed[0].quantized_indices, vec![5, 10]);
        assert_eq!(replayed[0].gamma, 1.0);
    }
}
