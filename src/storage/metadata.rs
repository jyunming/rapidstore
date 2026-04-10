use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufWriter, Cursor, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct VectorMetadata {
    pub properties: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub document: Option<String>,
}

/// Convert a scalar JSON value to a typed index key. Object/Array -> None (not indexable).
///
/// Each key is prefixed with a one-character type tag so that values of different types
/// that stringify identically (e.g. boolean `true` vs string `"true"`, number `1` vs
/// string `"1"`, null vs string `"__null__"`) never collide in the eq_index.
fn value_to_index_key(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) => Some(format!("s:{}", s)),
        serde_json::Value::Number(n) => Some(format!("n:{}", n)),
        serde_json::Value::Bool(b) => Some(format!("b:{}", b)),
        serde_json::Value::Null => Some("z:".to_string()),
        _ => None,
    }
}

/// Map an f64 to a u64 that preserves numeric ordering for use as a BTreeMap key.
///
/// The encoding:
/// - Positive numbers: flip the sign bit → u64 values that sort ascending.
/// - Negative numbers: flip all bits → u64 values that sort ascending (most-negative first).
/// - NaN maps to the same key as positive infinity; callers should treat NaN fields as
///   non-indexed and let the scalar filter handle them.
pub(crate) fn f64_to_ord(v: f64) -> u64 {
    let bits = v.to_bits();
    if bits >> 63 == 0 {
        bits | (1u64 << 63)
    } else {
        !bits
    }
}

pub struct MetadataStore {
    path: PathBuf,
    data: HashMap<u32, VectorMetadata>,
    dirty: bool,
    /// Equality index: field -> value_key -> sorted slot list.
    /// Covers scalar (string/number/bool/null) top-level fields only.
    /// Enables O(1) pre-filter candidate lookup for $eq filter conditions.
    eq_index: HashMap<String, HashMap<String, Vec<u32>>>,
    /// Range index: field -> BTreeMap<ord_key, sorted slot list>.
    /// Covers numeric (f64) top-level fields only.
    /// Enables sub-linear pre-filter for $gt/$gte/$lt/$lte conditions.
    range_index: HashMap<String, BTreeMap<u64, Vec<u32>>>,
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
        let eq_index = Self::build_eq_index(&data);
        let range_index = Self::build_range_index(&data);
        Ok(Self {
            path,
            data,
            dirty: false,
            eq_index,
            range_index,
        })
    }

    pub fn put(
        &mut self,
        slot: u32,
        meta: &VectorMetadata,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.index_remove(slot);
        self.data.insert(slot, meta.clone());
        self.index_add(slot, meta);
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
            self.index_remove(*slot);
            self.data.insert(*slot, meta.clone());
            self.index_add(*slot, meta);
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
        self.index_remove(slot);
        self.data.remove(&slot);
        self.dirty = true;
        Ok(())
    }

    /// Return all slots where `field == value`.
    /// Only works for scalar values (string, number, bool, null).
    /// Returns `None` when the field is not indexed or `value` is not scalar.
    pub fn get_eq_candidates(&self, field: &str, value: &serde_json::Value) -> Option<&[u32]> {
        let key = value_to_index_key(value)?;
        // The field is indexed — return the candidate slice (may be empty, which
        // short-circuits the caller to zero results rather than falling back to a full scan).
        let field_map = self.eq_index.get(field)?;
        Some(field_map.get(&key).map(Vec::as_slice).unwrap_or(&[]))
    }

    /// Return all slots where `field` falls within a numeric range.
    ///
    /// `lo` and `hi` are `(ord_key, inclusive)` pairs produced by [`f64_to_ord`].
    /// Either bound may be `None` (open-ended). Returns `None` when the field has
    /// no range index (not a numeric field or never seen in any inserted vector).
    pub fn get_range_candidates(
        &self,
        field: &str,
        lo: Option<(u64, bool)>,
        hi: Option<(u64, bool)>,
    ) -> Option<Vec<u32>> {
        let btree = self.range_index.get(field)?;
        let iter: Box<dyn Iterator<Item = (&u64, &Vec<u32>)>> = match (lo, hi) {
            (Some((lo_k, lo_incl)), Some((hi_k, hi_incl))) => {
                let lo_bound = if lo_incl {
                    std::ops::Bound::Included(lo_k)
                } else {
                    std::ops::Bound::Excluded(lo_k)
                };
                let hi_bound = if hi_incl {
                    std::ops::Bound::Included(hi_k)
                } else {
                    std::ops::Bound::Excluded(hi_k)
                };
                Box::new(btree.range((lo_bound, hi_bound)))
            }
            (Some((lo_k, lo_incl)), None) => {
                let lo_bound = if lo_incl {
                    std::ops::Bound::Included(lo_k)
                } else {
                    std::ops::Bound::Excluded(lo_k)
                };
                Box::new(btree.range((lo_bound, std::ops::Bound::Unbounded)))
            }
            (None, Some((hi_k, hi_incl))) => {
                let hi_bound = if hi_incl {
                    std::ops::Bound::Included(hi_k)
                } else {
                    std::ops::Bound::Excluded(hi_k)
                };
                Box::new(btree.range((std::ops::Bound::Unbounded, hi_bound)))
            }
            (None, None) => return None,
        };

        let mut slots: Vec<u32> = iter.flat_map(|(_, v)| v.iter().copied()).collect();
        slots.sort_unstable();
        slots.dedup();
        if slots.is_empty() { None } else { Some(slots) }
    }

    pub fn clear(&mut self) {
        if self.data.is_empty() {
            return;
        }
        self.data.clear();
        self.eq_index.clear();
        self.range_index.clear();
        self.dirty = true;
    }

    pub fn replace_all(&mut self, data: HashMap<u32, VectorMetadata>) {
        self.eq_index = Self::build_eq_index(&data);
        self.range_index = Self::build_range_index(&data);
        self.data = data;
        self.dirty = true;
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
        // On Windows, rename fails if the destination already exists.
        #[cfg(target_os = "windows")]
        let _ = std::fs::remove_file(&self.path);
        std::fs::rename(&tmp, &self.path)?;
        self.dirty = false;
        Ok(())
    }

    fn index_add(&mut self, slot: u32, meta: &VectorMetadata) {
        for (field, val) in &meta.properties {
            // Equality index (all scalars).
            if let Some(key) = value_to_index_key(val) {
                let slots = self
                    .eq_index
                    .entry(field.clone())
                    .or_default()
                    .entry(key)
                    .or_default();
                let pos = slots.partition_point(|&s| s < slot);
                if slots.get(pos) != Some(&slot) {
                    slots.insert(pos, slot);
                }
            }
            // Range index (numeric fields only).
            if let Some(n) = val.as_f64() {
                if n.is_finite() {
                    let ord = f64_to_ord(n);
                    let slots = self
                        .range_index
                        .entry(field.clone())
                        .or_default()
                        .entry(ord)
                        .or_default();
                    let pos = slots.partition_point(|&s| s < slot);
                    if slots.get(pos) != Some(&slot) {
                        slots.insert(pos, slot);
                    }
                }
            }
        }
    }

    fn index_remove(&mut self, slot: u32) {
        let Some(meta) = self.data.get(&slot).cloned() else {
            return;
        };
        for (field, val) in &meta.properties {
            // Equality index removal.
            if let Some(key) = value_to_index_key(val) {
                if let Some(val_map) = self.eq_index.get_mut(field.as_str()) {
                    if let Some(slots) = val_map.get_mut(key.as_str()) {
                        if let Ok(pos) = slots.binary_search(&slot) {
                            slots.remove(pos);
                        }
                    }
                }
            }
            // Range index removal.
            if let Some(n) = val.as_f64() {
                if n.is_finite() {
                    let ord = f64_to_ord(n);
                    if let Some(btree) = self.range_index.get_mut(field.as_str()) {
                        if let Some(slots) = btree.get_mut(&ord) {
                            if let Ok(pos) = slots.binary_search(&slot) {
                                slots.remove(pos);
                            }
                        }
                    }
                }
            }
        }
    }

    fn build_eq_index(
        data: &HashMap<u32, VectorMetadata>,
    ) -> HashMap<String, HashMap<String, Vec<u32>>> {
        let mut idx: HashMap<String, HashMap<String, Vec<u32>>> = HashMap::new();
        let mut slots_sorted: Vec<u32> = data.keys().copied().collect();
        slots_sorted.sort_unstable();
        for slot in slots_sorted {
            let meta = &data[&slot];
            for (field, val) in &meta.properties {
                if let Some(key) = value_to_index_key(val) {
                    idx.entry(field.clone())
                        .or_default()
                        .entry(key)
                        .or_default()
                        .push(slot);
                }
            }
        }
        idx
    }

    fn build_range_index(
        data: &HashMap<u32, VectorMetadata>,
    ) -> HashMap<String, BTreeMap<u64, Vec<u32>>> {
        let mut idx: HashMap<String, BTreeMap<u64, Vec<u32>>> = HashMap::new();
        let mut slots_sorted: Vec<u32> = data.keys().copied().collect();
        slots_sorted.sort_unstable();
        for slot in slots_sorted {
            let meta = &data[&slot];
            for (field, val) in &meta.properties {
                if let Some(n) = val.as_f64() {
                    if n.is_finite() {
                        idx.entry(field.clone())
                            .or_default()
                            .entry(f64_to_ord(n))
                            .or_default()
                            .push(slot);
                    }
                }
            }
        }
        idx
    }

    fn load_from_file(
        path: &Path,
    ) -> Result<HashMap<u32, VectorMetadata>, Box<dyn std::error::Error + Send + Sync>> {
        let bytes = std::fs::read(path)?;
        let mut cur = Cursor::new(bytes);

        let mut magic = [0u8; 4];
        cur.read_exact(&mut magic)?;
        if &magic != b"M2S1" {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "invalid metadata magic in {}: expected {:?}, found {:?}",
                    path.display(),
                    b"M2S1",
                    magic
                ),
            )));
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

    // -----------------------------------------------------------------------
    // eq_index
    // -----------------------------------------------------------------------

    #[test]
    fn eq_index_basic_string_lookup() {
        let dir = tempdir().unwrap();
        let mut store = open_store(&dir, "meta.bin");
        store.put(0, &make_meta("status", "active")).unwrap();
        store.put(1, &make_meta("status", "inactive")).unwrap();
        store.put(2, &make_meta("status", "active")).unwrap();
        let active = store.get_eq_candidates("status", &json!("active")).unwrap();
        assert_eq!(active, &[0, 2]);
        let inactive = store
            .get_eq_candidates("status", &json!("inactive"))
            .unwrap();
        assert_eq!(inactive, &[1]);
    }

    #[test]
    fn eq_index_missing_field_returns_none() {
        let dir = tempdir().unwrap();
        let store = open_store(&dir, "meta.bin");
        assert!(
            store
                .get_eq_candidates("nonexistent", &json!("x"))
                .is_none()
        );
    }

    #[test]
    fn eq_index_missing_value_returns_empty_slice() {
        // The field "status" IS indexed (we inserted "active"), so querying a
        // value that has no matches returns Some(&[]) rather than None — this
        // lets callers short-circuit to zero results without a full scan.
        let dir = tempdir().unwrap();
        let mut store = open_store(&dir, "meta.bin");
        store.put(0, &make_meta("status", "active")).unwrap();
        let result = store.get_eq_candidates("status", &json!("pending"));
        assert!(
            result.is_some(),
            "indexed field should return Some, not None"
        );
        assert!(
            result.unwrap().is_empty(),
            "should be empty for an unmatched value"
        );
    }

    #[test]
    fn eq_index_object_value_not_indexable() {
        let dir = tempdir().unwrap();
        let store = open_store(&dir, "meta.bin");
        assert!(
            store
                .get_eq_candidates("nested", &json!({"a": 1}))
                .is_none()
        );
    }

    #[test]
    fn eq_index_number_value() {
        let dir = tempdir().unwrap();
        let mut store = open_store(&dir, "meta.bin");
        let mut p0 = HashMap::new();
        p0.insert("year".to_string(), json!(2024));
        let mut p1 = HashMap::new();
        p1.insert("year".to_string(), json!(2023));
        store
            .put(
                0,
                &VectorMetadata {
                    properties: p0,
                    document: None,
                },
            )
            .unwrap();
        store
            .put(
                1,
                &VectorMetadata {
                    properties: p1,
                    document: None,
                },
            )
            .unwrap();
        assert_eq!(store.get_eq_candidates("year", &json!(2024)).unwrap(), &[0]);
    }

    #[test]
    fn eq_index_updated_on_overwrite() {
        let dir = tempdir().unwrap();
        let mut store = open_store(&dir, "meta.bin");
        store.put(0, &make_meta("status", "active")).unwrap();
        store.put(0, &make_meta("status", "inactive")).unwrap();
        // "active" was removed; field is still indexed → Some(&[]) not None.
        let active = store.get_eq_candidates("status", &json!("active"));
        assert!(active.is_some_and(|v| v.is_empty()));
        assert_eq!(
            store
                .get_eq_candidates("status", &json!("inactive"))
                .unwrap(),
            &[0]
        );
    }

    #[test]
    fn eq_index_removed_on_delete() {
        let dir = tempdir().unwrap();
        let mut store = open_store(&dir, "meta.bin");
        store.put(0, &make_meta("status", "active")).unwrap();
        store.put(1, &make_meta("status", "active")).unwrap();
        store.delete(0).unwrap();
        assert_eq!(
            store.get_eq_candidates("status", &json!("active")).unwrap(),
            &[1]
        );
    }

    #[test]
    fn eq_index_rebuilt_on_reload() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meta.bin").to_str().unwrap().to_string();
        {
            let mut store = MetadataStore::open(&path).unwrap();
            store.put(0, &make_meta("color", "red")).unwrap();
            store.put(1, &make_meta("color", "blue")).unwrap();
            store.put(2, &make_meta("color", "red")).unwrap();
            store.flush().unwrap();
        }
        let store2 = MetadataStore::open(&path).unwrap();
        assert_eq!(
            store2.get_eq_candidates("color", &json!("red")).unwrap(),
            &[0, 2]
        );
    }
}
