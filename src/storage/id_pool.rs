use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct IdPool {
    bytes: Vec<u8>,
    offsets: Vec<u32>,
    lens: Vec<u16>,
    hashes: Vec<u64>,
    alive: Vec<bool>,
    lookup: HashMap<u64, Vec<u32>>,
    active_count: usize,
}

impl IdPool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn slot_count(&self) -> usize {
        self.offsets.len()
    }

    pub fn active_count(&self) -> usize {
        self.active_count
    }

    pub fn bytes_len(&self) -> usize {
        self.bytes.len()
    }

    pub fn contains(&self, id: &str) -> bool {
        self.get_slot(id).is_some()
    }

    pub fn insert(&mut self, id: &str) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(slot) = self.get_slot(id) {
            return Ok(slot);
        }

        if id.len() > u16::MAX as usize {
            return Err(format!("id is too long ({} bytes, max {})", id.len(), u16::MAX).into());
        }
        if self.bytes.len() > u32::MAX as usize {
            return Err("id pool byte buffer exceeds 4 GiB — too many unique IDs inserted".into());
        }

        let slot = self.offsets.len() as u32;
        let hash = fnv1a64(id.as_bytes());

        self.offsets.push(self.bytes.len() as u32);
        self.lens.push(id.len() as u16);
        self.hashes.push(hash);
        self.alive.push(true);
        self.bytes.extend_from_slice(id.as_bytes());

        self.lookup.entry(hash).or_default().push(slot);
        self.active_count += 1;
        Ok(slot)
    }

    pub fn get_slot(&self, id: &str) -> Option<u32> {
        let hash = fnv1a64(id.as_bytes());
        let candidates = self.lookup.get(&hash)?;
        for &slot in candidates {
            let i = slot as usize;
            if !self.alive.get(i).copied().unwrap_or(false) {
                continue;
            }
            if self.get_str(slot).is_some_and(|s| s == id) {
                return Some(slot);
            }
        }
        None
    }

    pub fn get_str(&self, slot: u32) -> Option<&str> {
        let i = slot as usize;
        if i >= self.offsets.len() || !self.alive.get(i).copied().unwrap_or(false) {
            return None;
        }
        let off = self.offsets[i] as usize;
        let len = self.lens[i] as usize;
        // Bounds check before slicing — guards against on-disk corruption or truncated files.
        if off > self.bytes.len() || off + len > self.bytes.len() {
            return None;
        }
        std::str::from_utf8(&self.bytes[off..off + len]).ok()
    }

    pub fn delete_by_slot(&mut self, slot: u32) -> bool {
        let i = slot as usize;
        if i >= self.alive.len() || !self.alive[i] {
            return false;
        }
        self.alive[i] = false;
        self.active_count = self.active_count.saturating_sub(1);
        // Prune the slot from the hash-bucket to prevent unbounded lookup growth under churn.
        let hash = self.hashes[i];
        if let Some(candidates) = self.lookup.get_mut(&hash) {
            candidates.retain(|&c| c != slot);
            if candidates.is_empty() {
                self.lookup.remove(&hash);
            }
        }
        true
    }

    pub fn delete_by_id(&mut self, id: &str) -> Option<u32> {
        let slot = self.get_slot(id)?;
        if self.delete_by_slot(slot) {
            Some(slot)
        } else {
            None
        }
    }

    /// Returns true if `slot` is within bounds and has not been deleted.
    pub fn is_slot_alive(&self, slot: u32) -> bool {
        (slot as usize) < self.alive.len() && self.alive[slot as usize]
    }

    pub fn iter_active(&self) -> Vec<(String, u32)> {
        let mut out = Vec::with_capacity(self.active_count);
        for slot in 0..self.offsets.len() {
            if self.alive[slot] {
                if let Some(id) = self.get_str(slot as u32) {
                    out.push((id.to_string(), slot as u32));
                }
            }
        }
        out
    }

    pub fn clear(&mut self) {
        self.bytes.clear();
        self.offsets.clear();
        self.lens.clear();
        self.hashes.clear();
        self.alive.clear();
        self.lookup.clear();
        self.active_count = 0;
    }

    pub fn rebuild_lookup(&mut self) {
        self.lookup.clear();
        self.active_count = 0;
        for (i, &hash) in self.hashes.iter().enumerate() {
            if self.alive.get(i).copied().unwrap_or(false) {
                self.lookup.entry(hash).or_default().push(i as u32);
                self.active_count += 1;
            }
        }
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::IdPool;

    #[test]
    fn insert_get_delete_roundtrip() {
        let mut p = IdPool::new();
        let a = p.insert("a").unwrap();
        let b = p.insert("b").unwrap();
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(p.get_slot("a"), Some(0));
        assert_eq!(p.get_slot("b"), Some(1));
        assert_eq!(p.get_str(0), Some("a"));
        assert_eq!(p.get_str(1), Some("b"));

        assert_eq!(p.delete_by_id("a"), Some(0));
        assert_eq!(p.get_slot("a"), None);
        assert_eq!(p.get_str(0), None);
        assert_eq!(p.active_count(), 1);
    }

    #[test]
    fn duplicate_insert_returns_existing_slot() {
        let mut p = IdPool::new();
        let a0 = p.insert("same").unwrap();
        let a1 = p.insert("same").unwrap();
        assert_eq!(a0, a1);
        assert_eq!(p.slot_count(), 1);
        assert_eq!(p.active_count(), 1);
    }

    #[test]
    fn bytes_len_reflects_stored_data() {
        let mut p = IdPool::new();
        assert_eq!(p.bytes_len(), 0);
        p.insert("hello").unwrap();
        assert_eq!(p.bytes_len(), 5);
        p.insert("world").unwrap();
        assert_eq!(p.bytes_len(), 10);
    }

    #[test]
    fn clear_resets_all_state() {
        let mut p = IdPool::new();
        p.insert("a").unwrap();
        p.insert("b").unwrap();
        assert_eq!(p.slot_count(), 2);
        assert_eq!(p.active_count(), 2);
        assert!(p.bytes_len() > 0);
        p.clear();
        assert_eq!(p.slot_count(), 0);
        assert_eq!(p.active_count(), 0);
        assert_eq!(p.bytes_len(), 0);
        assert_eq!(p.contains("a"), false);
        assert_eq!(p.contains("b"), false);
    }

    #[test]
    fn delete_by_slot_out_of_bounds_returns_false() {
        let mut p = IdPool::new();
        // Empty pool — any slot is out of bounds
        assert_eq!(p.delete_by_slot(0), false);
        assert_eq!(p.delete_by_slot(99), false);
    }

    #[test]
    fn delete_by_slot_already_dead_returns_false() {
        let mut p = IdPool::new();
        let slot = p.insert("x").unwrap();
        assert_eq!(p.delete_by_slot(slot), true);
        // Second delete on same slot — already dead
        assert_eq!(p.delete_by_slot(slot), false);
        assert_eq!(p.active_count(), 0);
    }

    #[test]
    fn get_str_on_out_of_bounds_slot_returns_none() {
        let p = IdPool::new();
        assert_eq!(p.get_str(0), None);
        assert_eq!(p.get_str(100), None);
    }

    #[test]
    fn get_str_on_deleted_slot_returns_none() {
        let mut p = IdPool::new();
        let slot = p.insert("ghost").unwrap();
        assert_eq!(p.get_str(slot), Some("ghost"));
        p.delete_by_slot(slot);
        assert_eq!(p.get_str(slot), None);
    }

    #[test]
    fn insert_id_too_long_returns_err() {
        let mut p = IdPool::new();
        let long_id = "x".repeat(u16::MAX as usize + 1);
        assert!(p.insert(&long_id).is_err());
    }
}
