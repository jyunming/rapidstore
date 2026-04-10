use serde::{Deserialize, Serialize};

use crate::quantizer::CodeIndex;
use crate::quantizer::prod::ProdQuantizer;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const WAL_MAGIC: &[u8; 4] = b"TQWV";
const WAL_VERSION: u32 = 5;

/// On-disk V3 entry — MSE indices stored as bit-packed bytes.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct WalEntryPacked {
    pub id: String,
    pub packed_mse: Vec<u8>,
    pub qjl_bits: Vec<u8>,
    pub gamma: f32,
    pub norm: f32,
    pub metadata_json: String,
    pub is_deleted: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WalEntry {
    pub id: String,
    pub quantized_indices: Vec<CodeIndex>,
    pub qjl_bits: Vec<u8>,
    pub gamma: f32,
    pub norm: f32,
    pub metadata_json: String,
    pub is_deleted: bool,
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
            writer: BufWriter::with_capacity(4 * 1024 * 1024, file),
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

        // Pre-allocate a single buffer for the whole batch (rough estimate: 512 bytes/entry).
        // Serialising all entries first then issuing one write_all eliminates per-entry
        // syscall overhead and avoids partial BufWriter flushes mid-batch.
        let mut buf: Vec<u8> = Vec::with_capacity(entries.len() * 512);

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
                    norm: entry.norm,
                    metadata_json: entry.metadata_json.clone(),
                    is_deleted: entry.is_deleted,
                };
                bincode::serialize(&packed)?
            } else {
                bincode::serialize(entry)?
            };

            let len = encoded.len() as u64;
            let len_bytes = len.to_le_bytes();
            let mut hasher = crc32fast::Hasher::new();
            hasher.update(&len_bytes);
            hasher.update(&encoded);

            buf.extend_from_slice(&len_bytes);
            buf.extend_from_slice(&encoded);
            buf.extend_from_slice(&hasher.finalize().to_le_bytes());
            self.entry_count += 1;
        }

        // One write_all + one flush for the entire batch.
        self.writer.write_all(&buf)?;
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
        let version;

        if file.read_exact(&mut magic).is_ok() && &magic == WAL_MAGIC {
            let mut v_buf = [0u8; 4];
            file.read_exact(&mut v_buf)?;
            version = u32::from_le_bytes(v_buf);
        } else {
            // No magic — not a valid WAL file; return empty rather than corrupt read.
            return Ok(Vec::new());
        };

        if version < 4 || version > WAL_VERSION {
            return Err(format!(
                "Unsupported WAL version {} (supported: 4–{}). \
                 The WAL format changed; delete wal.log to start fresh.",
                version, WAL_VERSION
            )
            .into());
        }
        let use_crc = version >= 5;

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
                // Partial write at end of file — treat as truncated, stop here.
                Err(_) => break,
            }

            if use_crc {
                let mut crc_buf = [0u8; 4];
                if file.read_exact(&mut crc_buf).is_err() {
                    // CRC bytes not fully written (truncated entry) — stop here.
                    break;
                }
                let stored = u32::from_le_bytes(crc_buf);
                let mut hasher = crc32fast::Hasher::new();
                hasher.update(&len_buf);
                hasher.update(&payload);
                if stored != hasher.finalize() {
                    eprintln!(
                        "WAL CRC32 mismatch at entry {}; truncating replay here.",
                        entries.len()
                    );
                    break;
                }
            }

            // V4/V5: packed MSE indices with norm field.
            match bincode::deserialize::<WalEntryPacked>(&payload) {
                Ok(packed) => {
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
                        norm: packed.norm,
                        metadata_json: packed.metadata_json,
                        is_deleted: packed.is_deleted,
                    });
                }
                // Decode failure on a complete payload means a schema mismatch or corruption —
                // treat as a truncated final entry and stop replaying.
                Err(_) => break,
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
            norm: 0.0,
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
            norm: 1.0,
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
        assert_eq!(u32::from_le_bytes(data[4..8].try_into().unwrap()), 5u32);
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

        let mut e_with_norm = real_entry("a", &pq);
        e_with_norm.norm = 3.14;
        let entries_in = vec![e_with_norm, real_entry("b", &pq), tombstone_entry("c")];
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.set_quantizer(Arc::clone(&pq));
            wal.append_batch(&entries_in, true).unwrap();
        }

        let replayed = Wal::replay(&path, Some(&pq)).unwrap();
        assert_eq!(replayed.len(), 3);
        assert_eq!(replayed[0].id, "a");
        assert!(
            (replayed[0].norm - 3.14).abs() < 1e-5,
            "norm should survive roundtrip"
        );
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
    // Legacy V1 and V2 formats no longer supported; tests removed.
    // -----------------------------------------------------------------------

    // ── append_batch empty + force_sync=false covers line 96 ─────────────────
    // When entries is empty AND force_sync=false, the `if force_sync` block is
    // not entered. LLVM marks this "not-taken" path at line 96.

    #[test]
    fn append_batch_empty_no_sync_covers_branch_at_line_96() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.bin");
        let mut wal = Wal::open(&path).unwrap();
        // entries=[] + force_sync=false → the `if force_sync { sync() }` block skipped
        wal.append_batch(&[], false).unwrap();
        // Nothing was written beyond the header
        let replayed = Wal::replay(&path, None).unwrap();
        assert!(replayed.is_empty());
    }

    // ── append_batch without quantizer uses plain serialize (line 117) ────────
    // When quantizer=None, the else branch at line 117 serializes WalEntry directly.

    #[test]
    fn append_batch_without_quantizer_uses_plain_serialize() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.bin");
        let mut wal = Wal::open(&path).unwrap();
        // No set_quantizer() call → self.quantizer=None → else branch at line 117
        wal.append_batch(&[tombstone_entry("x")], false).unwrap();
        let file_size = std::fs::metadata(&path).unwrap().len();
        assert!(
            file_size > 8,
            "WAL should have data after append: size={file_size}"
        );
    }

    // -----------------------------------------------------------------------
    // CRC32 checksums (v5)
    // -----------------------------------------------------------------------

    /// A v4 WAL file (no CRC32) should still replay without errors.
    #[test]
    fn wal_v4_legacy_file_replays_without_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal_v4.bin");

        // Construct a v4 file manually (header v=4, two tombstone entries, no CRC).
        let mut buf = Vec::new();
        buf.extend_from_slice(b"TQWV");
        buf.extend_from_slice(&4u32.to_le_bytes());
        for id in &["a", "b"] {
            let packed = WalEntryPacked {
                id: id.to_string(),
                packed_mse: vec![],
                qjl_bits: vec![],
                gamma: 0.0,
                norm: 0.0,
                metadata_json: "{}".to_string(),
                is_deleted: true,
            };
            let encoded = bincode::serialize(&packed).unwrap();
            let len = encoded.len() as u64;
            buf.extend_from_slice(&len.to_le_bytes());
            buf.extend_from_slice(&encoded);
        }
        std::fs::write(&path, &buf).unwrap();

        let replayed = Wal::replay(&path, None).unwrap();
        assert_eq!(replayed.len(), 2);
        assert_eq!(replayed[0].id, "a");
        assert_eq!(replayed[1].id, "b");
    }

    /// A v5 entry with a corrupted CRC32 must stop replay at that entry.
    #[test]
    fn wal_crc32_mismatch_stops_replay_at_corrupt_entry() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal_crc.bin");

        let mut buf = Vec::new();
        buf.extend_from_slice(b"TQWV");
        buf.extend_from_slice(&5u32.to_le_bytes());
        for (i, id) in ["e1", "e2", "e3"].iter().enumerate() {
            let packed = WalEntryPacked {
                id: id.to_string(),
                packed_mse: vec![],
                qjl_bits: vec![],
                gamma: 0.0,
                norm: 0.0,
                metadata_json: "{}".to_string(),
                is_deleted: true,
            };
            let encoded = bincode::serialize(&packed).unwrap();
            let len = encoded.len() as u64;
            let len_bytes = len.to_le_bytes();
            buf.extend_from_slice(&len_bytes);
            buf.extend_from_slice(&encoded);
            let mut hasher = crc32fast::Hasher::new();
            hasher.update(&len_bytes);
            hasher.update(&encoded);
            let crc = hasher.finalize();
            if i == 1 {
                // Corrupt entry 2's CRC by flipping bits.
                buf.extend_from_slice(&(crc ^ 0xDEAD_BEEF).to_le_bytes());
            } else {
                buf.extend_from_slice(&crc.to_le_bytes());
            }
        }
        std::fs::write(&path, &buf).unwrap();

        let replayed = Wal::replay(&path, None).unwrap();
        // Only entry 1 should be recovered; replay stops at entry 2's bad CRC.
        assert_eq!(replayed.len(), 1, "corrupt CRC should stop replay");
        assert_eq!(replayed[0].id, "e1");
    }

    /// A truncated payload (incomplete write) must still allow earlier entries to replay.
    #[test]
    fn wal_truncated_entry_allows_prior_entries_to_replay() {
        let pq = make_pq();
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal_trunc.bin");
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.set_quantizer(Arc::clone(&pq));
            wal.append(&real_entry("ok1", &pq), false).unwrap();
            wal.append(&real_entry("ok2", &pq), false).unwrap();
        }
        // Append a truncated entry (write len but only half the payload + no CRC).
        let mut raw = std::fs::read(&path).unwrap();
        raw.extend_from_slice(&999u64.to_le_bytes()); // claims 999-byte payload
        raw.extend_from_slice(&[0xAB; 10]); // only 10 bytes of garbage
        std::fs::write(&path, &raw).unwrap();

        let replayed = Wal::replay(&path, Some(&pq)).unwrap();
        assert_eq!(
            replayed.len(),
            2,
            "two complete entries should survive despite truncated tail"
        );
    }
}
