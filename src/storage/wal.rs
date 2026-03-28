use std::io::{BufWriter, Write};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use serde::{Serialize, Deserialize};

/// A single WAL entry representing a vector insertion
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WalEntry {
    pub id: String,
    pub quantized_indices: Vec<usize>,
    pub qjl_bits: Vec<i8>,
    pub gamma: f32,
    pub metadata_json: String,
}

/// Segmented Write-Ahead Log for crash-safe insertions.
/// Entries are appended as length-prefixed bincode records.
/// On restart, the WAL is replayed before any segment data is trusted.
pub struct Wal {
    path: PathBuf,
    writer: BufWriter<File>,
    entry_count: u64,
}

impl Wal {
    /// Open (or create) a WAL file at the given path.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(Self {
            path,
            writer: BufWriter::new(file),
            entry_count: 0,
        })
    }

    /// Append an entry to the WAL — guaranteed persistent before returning.
    pub fn append(&mut self, entry: &WalEntry) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let encoded = bincode::serialize(entry)?;
        let len = encoded.len() as u64;
        // Write length prefix then payload
        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(&encoded)?;
        // fsync ensures data survives a crash
        self.writer.flush()?;
        self.writer.get_ref().sync_data()?;
        self.entry_count += 1;
        Ok(())
    }

    /// Replay all entries from the WAL file (used on startup for crash recovery).
    pub fn replay<P: AsRef<Path>>(path: P) -> Result<Vec<WalEntry>, Box<dyn std::error::Error + Send + Sync>> {
        use std::io::Read;
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Vec::new());
        }
        let mut file = File::open(path)?;
        let mut entries = Vec::new();
        loop {
            let mut len_buf = [0u8; 8];
            match file.read_exact(&mut len_buf) {
                Ok(_) => {},
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            let len = u64::from_le_bytes(len_buf) as usize;
            let mut payload = vec![0u8; len];
            match file.read_exact(&mut payload) {
                Ok(_) => {},
                Err(_) => break, // Partial write at tail — safe to truncate
            }
            if let Ok(entry) = bincode::deserialize::<WalEntry>(&payload) {
                entries.push(entry);
            }
        }
        Ok(entries)
    }

    /// Truncate the WAL after all pending entries have been flushed to a segment.
    pub fn truncate(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        File::create(&self.path)?; // Overwrite with empty file
        Ok(())
    }

    pub fn entry_count(&self) -> u64 {
        self.entry_count
    }
}
