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
    data: HashMap<u32, VectorMetadata>,
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
        slot: u32,
        meta: &VectorMetadata,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.data.insert(slot, meta.clone());
        self.dirty = true;
        Ok(())
    }

    pub fn put_many(
        &mut self,
        entries: &[(u32, VectorMetadata)],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if entries.is_empty() {
            return Ok(());
        }
        for (slot, meta) in entries {
            self.data.insert(*slot, meta.clone());
        }
        self.dirty = true;
        Ok(())
    }

    pub fn get(
        &self,
        slot: u32,
    ) -> Result<Option<VectorMetadata>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(self.data.get(&slot).cloned())
    }

    pub fn get_many(
        &self,
        slots: &[u32],
    ) -> Result<HashMap<u32, VectorMetadata>, Box<dyn std::error::Error + Send + Sync>> {
        let mut out = HashMap::with_capacity(slots.len());
        for slot in slots {
            if let Some(meta) = self.data.get(slot) {
                out.insert(*slot, meta.clone());
            }
        }
        Ok(out)
    }

    pub fn delete(&mut self, slot: u32) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.data.remove(&slot);
        self.dirty = true;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn approx_bytes(&self) -> usize {
        self.data
            .iter()
            .map(|(_slot, meta)| {
                let payload = serde_json::to_vec(meta).map(|v| v.len()).unwrap_or(0);
                4 + payload
            })
            .sum()
    }

    pub fn flush(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if !self.dirty {
            return Ok(());
        }

        let tmp = self.path.with_extension("tmp");
        let mut writer = BufWriter::new(File::create(&tmp)?);
        writer.write_all(b"M2S1")?;
        writer.write_all(&(self.data.len() as u64).to_le_bytes())?;

        for (slot, meta) in &self.data {
            let meta_bytes = serde_json::to_vec(meta)?;
            writer.write_all(&slot.to_le_bytes())?;
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
    ) -> Result<HashMap<u32, VectorMetadata>, Box<dyn std::error::Error + Send + Sync>> {
        let bytes = std::fs::read(path)?;
        let mut cur = Cursor::new(bytes);

        let mut magic = [0u8; 4];
        cur.read_exact(&mut magic)?;
        if &magic != b"M2S1" {
            return Ok(HashMap::new());
        }

        let mut count_buf = [0u8; 8];
        cur.read_exact(&mut count_buf)?;
        let count = u64::from_le_bytes(count_buf) as usize;

        let mut map = HashMap::with_capacity(count);
        for _ in 0..count {
            let mut slot_buf = [0u8; 4];
            cur.read_exact(&mut slot_buf)?;
            let slot = u32::from_le_bytes(slot_buf);

            let mut meta_len_buf = [0u8; 4];
            cur.read_exact(&mut meta_len_buf)?;
            let meta_len = u32::from_le_bytes(meta_len_buf) as usize;
            let mut meta_bytes = vec![0u8; meta_len];
            cur.read_exact(&mut meta_bytes)?;
            let meta: VectorMetadata = serde_json::from_slice(&meta_bytes)?;
            map.insert(slot, meta);
        }

        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    fn make_meta(key: &str, val: &str) -> VectorMetadata {
        let mut props = HashMap::new();
        props.insert(key.to_string(), serde_json::Value::String(val.to_string()));
        VectorMetadata {
            properties: props,
            document: None,
        }
    }

    fn open_store(dir: &tempfile::TempDir, filename: &str) -> MetadataStore {
        let path = dir.path().join(filename).to_str().unwrap().to_string();
        MetadataStore::open(&path).unwrap()
    }

    // -----------------------------------------------------------------------
    // open
    // -----------------------------------------------------------------------

    #[test]
    fn open_creates_empty_store() {
        let dir = tempdir().unwrap();
        let store = open_store(&dir, "meta.bin");
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn open_creates_parent_directory() {
        let dir = tempdir().unwrap();
        let nested = dir.path().join("sub").join("meta.bin");
        MetadataStore::open(nested.to_str().unwrap()).unwrap();
        assert!(
            dir.path().join("sub").is_dir(),
            "parent dir should be created"
        );
    }

    // -----------------------------------------------------------------------
    // put / get
    // -----------------------------------------------------------------------

    #[test]
    fn put_and_get_roundtrip() {
        let dir = tempdir().unwrap();
        let mut store = open_store(&dir, "meta.bin");
        let meta = make_meta("color", "blue");
        store.put(0, &meta).unwrap();
        let got = store.get(0).unwrap().expect("should exist");
        assert_eq!(got.properties["color"], json!("blue"));
    }

    #[test]
    fn get_missing_key_returns_none() {
        let dir = tempdir().unwrap();
        let store = open_store(&dir, "meta.bin");
        assert!(store.get(999).unwrap().is_none());
    }

    #[test]
    fn put_overwrites_existing() {
        let dir = tempdir().unwrap();
        let mut store = open_store(&dir, "meta.bin");
        store.put(0, &make_meta("k", "v1")).unwrap();
        store.put(0, &make_meta("k", "v2")).unwrap();
        let got = store.get(0).unwrap().unwrap();
        assert_eq!(got.properties["k"], json!("v2"));
    }

    // -----------------------------------------------------------------------
    // put_many / get_many
    // -----------------------------------------------------------------------

    #[test]
    fn put_many_and_get_many() {
        let dir = tempdir().unwrap();
        let mut store = open_store(&dir, "meta.bin");
        let entries = vec![
            (0u32, make_meta("a", "1")),
            (1u32, make_meta("b", "2")),
            (2u32, make_meta("c", "3")),
        ];
        store.put_many(&entries).unwrap();
        assert_eq!(store.len(), 3);
        let got = store.get_many(&[0, 1, 2, 99]).unwrap();
        assert_eq!(got.len(), 3);
        assert!(got.contains_key(&0));
        assert!(!got.contains_key(&99));
    }

    #[test]
    fn put_many_empty_is_no_op() {
        let dir = tempdir().unwrap();
        let mut store = open_store(&dir, "meta.bin");
        store.put_many(&[]).unwrap();
        assert_eq!(store.len(), 0);
    }

    // -----------------------------------------------------------------------
    // delete
    // -----------------------------------------------------------------------

    #[test]
    fn delete_removes_entry() {
        let dir = tempdir().unwrap();
        let mut store = open_store(&dir, "meta.bin");
        store.put(0, &make_meta("k", "v")).unwrap();
        assert_eq!(store.len(), 1);
        store.delete(0).unwrap();
        assert_eq!(store.len(), 0);
        assert!(store.get(0).unwrap().is_none());
    }

    #[test]
    fn delete_missing_slot_is_no_op() {
        let dir = tempdir().unwrap();
        let mut store = open_store(&dir, "meta.bin");
        // Should not error
        store.delete(42).unwrap();
        assert_eq!(store.len(), 0);
    }

    // -----------------------------------------------------------------------
    // flush / persistence
    // -----------------------------------------------------------------------

    #[test]
    fn flush_and_reload_persists_data() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meta.bin").to_str().unwrap().to_string();
        {
            let mut store = MetadataStore::open(&path).unwrap();
            store.put(0, &make_meta("field", "hello")).unwrap();
            let with_doc = VectorMetadata {
                properties: Default::default(),
                document: Some("doc text".to_string()),
            };
            store.put(1, &with_doc).unwrap();
            store.flush().unwrap();
        }
        // Reload from disk
        let store2 = MetadataStore::open(&path).unwrap();
        assert_eq!(store2.len(), 2);
        let m0 = store2.get(0).unwrap().unwrap();
        assert_eq!(m0.properties["field"], json!("hello"));
        let m1 = store2.get(1).unwrap().unwrap();
        assert_eq!(m1.document, Some("doc text".to_string()));
    }

    #[test]
    fn flush_is_idempotent_when_not_dirty() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meta.bin").to_str().unwrap().to_string();
        let mut store = MetadataStore::open(&path).unwrap();
        store.put(0, &make_meta("k", "v")).unwrap();
        store.flush().unwrap();
        let size1 = std::fs::metadata(&path).unwrap().len();
        // Not dirty — second flush should not touch the file
        store.flush().unwrap();
        let size2 = std::fs::metadata(&path).unwrap().len();
        assert_eq!(
            size1, size2,
            "flush on non-dirty store should not modify file"
        );
    }

    #[test]
    fn flush_empty_store_creates_valid_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meta.bin").to_str().unwrap().to_string();
        let mut store = MetadataStore::open(&path).unwrap();
        store.put(0, &make_meta("x", "y")).unwrap();
        store.delete(0).unwrap(); // mark dirty again after delete
        store.flush().unwrap();
        // Reload should succeed and return empty
        let store2 = MetadataStore::open(&path).unwrap();
        assert_eq!(store2.len(), 0);
    }

    // -----------------------------------------------------------------------
    // approx_bytes
    // -----------------------------------------------------------------------

    #[test]
    fn approx_bytes_increases_with_entries() {
        let dir = tempdir().unwrap();
        let mut store = open_store(&dir, "meta.bin");
        let before = store.approx_bytes();
        store.put(0, &make_meta("key", "value")).unwrap();
        assert!(
            store.approx_bytes() > before,
            "approx_bytes should grow after put"
        );
    }

    // -----------------------------------------------------------------------
    // VectorMetadata default / document
    // -----------------------------------------------------------------------

    #[test]
    fn vector_metadata_default_has_empty_properties_and_no_document() {
        let meta = VectorMetadata::default();
        assert!(meta.properties.is_empty());
        assert!(meta.document.is_none());
    }
}
