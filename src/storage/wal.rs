use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WalEntry {
    pub id: String,
    pub quantized_indices: Vec<u16>,
    pub qjl_bits: Vec<i8>,
    pub gamma: f32,
    pub metadata_json: String,
    #[serde(default)]
    pub is_deleted: bool,
}

pub struct Wal {
    path: PathBuf,
    writer: BufWriter<File>,
    entry_count: u64,
}

impl Wal {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self {
            path,
            writer: BufWriter::new(file),
            entry_count: 0,
        })
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
            let encoded = bincode::serialize(entry)?;
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
    ) -> Result<Vec<WalEntry>, Box<dyn std::error::Error + Send + Sync>> {
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
            if let Ok(entry) = bincode::deserialize::<WalEntry>(&payload) {
                entries.push(entry);
            }
        }
        Ok(entries)
    }

    pub fn truncate(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        File::create(&self.path)?;
        Ok(())
    }

    pub fn entry_count(&self) -> u64 {
        self.entry_count
    }
}

