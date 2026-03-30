use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Cursor, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct VectorMetadata {
    pub properties: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub document: Option<String>,
}

pub struct MetadataStore {
    path: PathBuf,
    data: HashMap<String, VectorMetadata>,
    dirty: bool,
}

impl MetadataStore {
    pub fn open(path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let path = PathBuf::from(path);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let data = if path.exists() {
            Self::load_from_file(&path)?
        } else {
            HashMap::new()
        };
        Ok(Self {
            path,
            data,
            dirty: false,
        })
    }

    pub fn put(
        &mut self,
        id: &str,
        meta: &VectorMetadata,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.data.insert(id.to_string(), meta.clone());
        self.dirty = true;
        Ok(())
    }

    pub fn put_many(
        &mut self,
        entries: &[(String, VectorMetadata)],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if entries.is_empty() {
            return Ok(());
        }
        for (id, meta) in entries {
            self.data.insert(id.clone(), meta.clone());
        }
        self.dirty = true;
        Ok(())
    }

    pub fn get(
        &self,
        id: &str,
    ) -> Result<Option<VectorMetadata>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(self.data.get(id).cloned())
    }

    pub fn get_many(
        &self,
        ids: &[String],
    ) -> Result<HashMap<String, VectorMetadata>, Box<dyn std::error::Error + Send + Sync>> {
        let mut out = HashMap::with_capacity(ids.len());
        for id in ids {
            if let Some(meta) = self.data.get(id) {
                out.insert(id.clone(), meta.clone());
            }
        }
        Ok(out)
    }

    pub fn delete(&mut self, id: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.data.remove(id);
        self.dirty = true;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if !self.dirty {
            return Ok(());
        }

        let tmp = self.path.with_extension("tmp");
        let mut writer = BufWriter::new(File::create(&tmp)?);
        writer.write_all(&(self.data.len() as u64).to_le_bytes())?;

        for (id, meta) in &self.data {
            let id_bytes = id.as_bytes();
            let meta_bytes = serde_json::to_vec(meta)?;

            writer.write_all(&(id_bytes.len() as u32).to_le_bytes())?;
            writer.write_all(id_bytes)?;
            writer.write_all(&(meta_bytes.len() as u32).to_le_bytes())?;
            writer.write_all(&meta_bytes)?;
        }

        writer.flush()?;
        std::fs::rename(&tmp, &self.path)?;
        self.dirty = false;
        Ok(())
    }

    fn load_from_file(
        path: &Path,
    ) -> Result<HashMap<String, VectorMetadata>, Box<dyn std::error::Error + Send + Sync>> {
        let bytes = std::fs::read(path)?;
        let mut cur = Cursor::new(bytes);

        let mut count_buf = [0u8; 8];
        cur.read_exact(&mut count_buf)?;
        let count = u64::from_le_bytes(count_buf) as usize;

        let mut map = HashMap::with_capacity(count);
        for _ in 0..count {
            let mut id_len_buf = [0u8; 4];
            cur.read_exact(&mut id_len_buf)?;
            let id_len = u32::from_le_bytes(id_len_buf) as usize;
            let mut id_bytes = vec![0u8; id_len];
            cur.read_exact(&mut id_bytes)?;
            let id = String::from_utf8(id_bytes)?;

            let mut meta_len_buf = [0u8; 4];
            cur.read_exact(&mut meta_len_buf)?;
            let meta_len = u32::from_le_bytes(meta_len_buf) as usize;
            let mut meta_bytes = vec![0u8; meta_len];
            cur.read_exact(&mut meta_bytes)?;
            let meta: VectorMetadata = serde_json::from_slice(&meta_bytes)?;

            map.insert(id, meta);
        }

        Ok(map)
    }
}
