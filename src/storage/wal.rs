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
                    let quantized_indices = if let Some(q) = quantizer {
                        let mut idx = vec![0 as CodeIndex; q.n];
                        q.unpack_mse_indices(&packed.packed_mse, &mut idx);
                        idx
                    } else {
                        // No quantizer: store raw packed bytes as zero-length (shouldn't happen)
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



