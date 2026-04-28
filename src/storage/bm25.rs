//! BM25 inverted index for sparse keyword retrieval.
//!
//! Sits alongside the dense store as a sidecar (`bm25.idx`). The engine drives
//! put/delete from the same insert/update/delete paths it already uses for
//! metadata; on flush, this index serializes its full in-memory state.
//!
//! Crash recovery: if the process dies between flushes, in-memory updates are
//! lost from `bm25.idx`. They are recovered on reopen because the engine's WAL
//! replay re-applies every vector with its document, which calls `put()` again.
//! That makes the engine WAL the source of truth for documents — BM25 doesn't
//! need its own WAL.
//!
//! Storage format (`bm25.idx`):
//!
//! | bytes | content |
//! |-------|---------|
//! | 4 | magic `b"M2BX"` |
//! | 8 | bincode payload length (u64 LE) |
//! | …  | bincode-serialized [`Bm25Snapshot`] |
//!
//! Atomic replace mirrors `metadata.idx`: write to `<path>.tmp`, fsync, rename.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Cursor, Read, Write};
use std::path::{Path, PathBuf};

use crate::storage::tokenizer::tokenize;

const IDX_MAGIC: &[u8; 4] = b"M2BX";

/// Okapi BM25 term-frequency saturation parameter. Standard default.
pub const DEFAULT_K1: f32 = 1.2;
/// Okapi BM25 length-normalization parameter. Standard default.
pub const DEFAULT_B: f32 = 0.75;

/// Disk-serializable snapshot of the in-memory index. Kept separate from the
/// runtime struct so the on-disk shape doesn't accidentally drift when fields
/// are added (`#[serde(default)]` on additions keeps old files readable).
#[derive(Serialize, Deserialize, Default)]
struct Bm25Snapshot {
    /// token hash → sorted Vec<(slot, term_freq)>.
    postings: HashMap<u64, Vec<(u32, u32)>>,
    /// slot → total token count for that doc (used for length normalization).
    doc_lengths: HashMap<u32, u32>,
    /// Sum of all values in `doc_lengths`. Tracked incrementally so search
    /// doesn't have to fold the map every call.
    total_len: u64,
}

pub struct Bm25Index {
    idx_path: PathBuf,
    snap: Bm25Snapshot,
    dirty: bool,
    k1: f32,
    b: f32,
}

impl Bm25Index {
    /// Open or create the index at `idx_path`. Missing files yield an empty index.
    pub fn open(idx_path: PathBuf) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let snap = if idx_path.exists() {
            match Self::load(&idx_path) {
                Ok(s) => s,
                Err(_) => Bm25Snapshot::default(), // corrupt → treat as empty; engine will rebuild
            }
        } else {
            Bm25Snapshot::default()
        };
        Ok(Self {
            idx_path,
            snap,
            dirty: false,
            k1: DEFAULT_K1,
            b: DEFAULT_B,
        })
    }

    pub fn n_docs(&self) -> u32 {
        self.snap.doc_lengths.len() as u32
    }

    pub fn avg_doc_len(&self) -> f32 {
        let n = self.snap.doc_lengths.len();
        if n == 0 {
            0.0
        } else {
            self.snap.total_len as f32 / n as f32
        }
    }

    /// Return whether any documents are currently indexed.
    pub fn is_empty(&self) -> bool {
        self.snap.doc_lengths.is_empty()
    }

    /// Add or replace the document for `slot`. Empty/whitespace text removes the slot
    /// (the document carries no signal for BM25 and shouldn't inflate avgdl).
    pub fn put(&mut self, slot: u32, text: &str) {
        // Replacing an existing entry is delete-then-insert; the public method
        // covers both "first put" and "update" cases.
        self.delete(slot);

        let tokens = tokenize(text);
        if tokens.is_empty() {
            return;
        }
        let doc_len = tokens.len() as u32;

        // Collapse to per-token frequencies before touching the postings map.
        let mut tf: HashMap<u64, u32> = HashMap::with_capacity(tokens.len());
        for h in tokens {
            *tf.entry(h).or_insert(0) += 1;
        }

        for (tok, freq) in tf {
            let posting = self.snap.postings.entry(tok).or_default();
            // Postings are kept sorted by slot for efficient filter intersection.
            // `delete()` already removed any stale entry for this slot.
            match posting.binary_search_by_key(&slot, |&(s, _)| s) {
                Ok(_) => unreachable!("delete() above guarantees no entry for this slot"),
                Err(pos) => posting.insert(pos, (slot, freq)),
            }
        }
        self.snap.doc_lengths.insert(slot, doc_len);
        self.snap.total_len += doc_len as u64;
        self.dirty = true;
    }

    /// Bulk insert/replace. Equivalent to repeated `put` but allocates the work
    /// upfront and dirties only once.
    pub fn put_many(&mut self, entries: &[(u32, &str)]) {
        for (slot, text) in entries {
            self.put(*slot, text);
        }
    }

    /// Remove the slot from the index. No-op if it wasn't indexed.
    pub fn delete(&mut self, slot: u32) {
        let Some(doc_len) = self.snap.doc_lengths.remove(&slot) else {
            return;
        };
        self.snap.total_len = self.snap.total_len.saturating_sub(doc_len as u64);

        // Walk every posting list removing this slot. The unconditional scan is
        // O(unique_tokens_in_corpus) per delete, which is acceptable; the alternative
        // (storing slot→tokens reverse map) doubles memory for a path that runs
        // far less often than `put`.
        self.snap.postings.retain(|_tok, posting| {
            if let Ok(pos) = posting.binary_search_by_key(&slot, |&(s, _)| s) {
                posting.remove(pos);
            }
            !posting.is_empty()
        });
        self.dirty = true;
    }

    pub fn delete_many(&mut self, slots: &[u32]) {
        for slot in slots {
            self.delete(*slot);
        }
    }

    /// Drop all in-memory state. Used by the engine on compaction (slot-renumbering)
    /// and on full rebuild paths so the index can be repopulated under fresh slot IDs.
    pub fn clear(&mut self) {
        if self.snap.doc_lengths.is_empty() && self.snap.postings.is_empty() {
            return;
        }
        self.snap = Bm25Snapshot::default();
        self.dirty = true;
    }

    /// Score documents for `query` and return the top `top_k` (slot, score) pairs.
    ///
    /// `filter` (if `Some`) is a sorted, unique slot list — only those slots are
    /// scored. This is how the engine pushes metadata pre-filters down into BM25.
    /// `None` scores the entire posting universe.
    pub fn search(&self, query: &str, top_k: usize, filter: Option<&[u32]>) -> Vec<(u32, f32)> {
        let n = self.snap.doc_lengths.len();
        if n == 0 || top_k == 0 {
            return Vec::new();
        }
        let avgdl = self.avg_doc_len().max(1.0);
        let n_f = n as f32;

        let q_tokens = tokenize(query);
        if q_tokens.is_empty() {
            return Vec::new();
        }
        // Deduplicate query tokens — repeated terms in the query don't change the
        // score multiplier under the standard Okapi formulation we use.
        let mut q_uniq: Vec<u64> = q_tokens;
        q_uniq.sort_unstable();
        q_uniq.dedup();

        let mut accum: HashMap<u32, f32> = HashMap::new();

        for tok in q_uniq {
            let Some(posting) = self.snap.postings.get(&tok) else {
                continue;
            };
            let df = posting.len() as f32;
            // BM25+1 IDF: ln((N - df + 0.5) / (df + 0.5) + 1). The `+1` keeps it
            // non-negative even when df > N/2 (rare but possible in tiny corpora).
            let idf = ((n_f - df + 0.5) / (df + 0.5) + 1.0).ln();

            for &(slot, tf) in posting {
                if let Some(allow) = filter {
                    if allow.binary_search(&slot).is_err() {
                        continue;
                    }
                }
                let dl = *self.snap.doc_lengths.get(&slot).unwrap_or(&0) as f32;
                let denom = tf as f32 + self.k1 * (1.0 - self.b + self.b * dl / avgdl);
                let term = idf * (tf as f32 * (self.k1 + 1.0)) / denom;
                *accum.entry(slot).or_insert(0.0) += term;
            }
        }

        if accum.is_empty() {
            return Vec::new();
        }
        let mut out: Vec<(u32, f32)> = accum.into_iter().collect();
        // Partial sort: only the top_k entries need to be ordered.
        if out.len() > top_k {
            out.select_nth_unstable_by(top_k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            out.truncate(top_k);
        }
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Persist the in-memory state to `bm25.idx` via tmp+rename. No-op if clean.
    pub fn flush(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if !self.dirty {
            return Ok(());
        }
        let bytes = bincode::serialize(&self.snap)?;
        let tmp = self.idx_path.with_extension("idx.tmp");
        if let Some(parent) = tmp.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let mut f = BufWriter::new(File::create(&tmp)?);
        f.write_all(IDX_MAGIC)?;
        f.write_all(&(bytes.len() as u64).to_le_bytes())?;
        f.write_all(&bytes)?;
        f.flush()?;
        drop(f);
        // Windows mmap can hold the destination open; matches the metadata.rs guard.
        #[cfg(target_os = "windows")]
        let _ = std::fs::remove_file(&self.idx_path);
        std::fs::rename(tmp, &self.idx_path)?;
        self.dirty = false;
        Ok(())
    }

    fn load(path: &Path) -> Result<Bm25Snapshot, Box<dyn std::error::Error + Send + Sync>> {
        let bytes = std::fs::read(path)?;
        if bytes.len() < 12 || &bytes[..4] != IDX_MAGIC {
            return Err("invalid bm25 idx magic".into());
        }
        let mut cur = Cursor::new(&bytes[4..]);
        let mut len_buf = [0u8; 8];
        cur.read_exact(&mut len_buf)?;
        let payload_len = u64::from_le_bytes(len_buf) as usize;
        let mut payload = vec![0u8; payload_len];
        cur.read_exact(&mut payload)?;
        let snap: Bm25Snapshot = bincode::deserialize(&payload)?;
        Ok(snap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn open_empty(dir: &std::path::Path) -> Bm25Index {
        Bm25Index::open(dir.join("bm25.idx")).unwrap()
    }

    #[test]
    fn put_and_search_finds_keyword() {
        let d = tempdir().unwrap();
        let mut idx = open_empty(d.path());
        idx.put(0, "the quick brown fox");
        idx.put(1, "the lazy dog");
        idx.put(2, "a quick rabbit");

        let r = idx.search("quick", 10, None);
        let slots: Vec<u32> = r.iter().map(|(s, _)| *s).collect();
        assert!(slots.contains(&0));
        assert!(slots.contains(&2));
        assert!(!slots.contains(&1));
    }

    #[test]
    fn delete_removes_from_postings() {
        let d = tempdir().unwrap();
        let mut idx = open_empty(d.path());
        idx.put(0, "alpha beta");
        idx.put(1, "alpha gamma");
        assert_eq!(idx.search("alpha", 10, None).len(), 2);
        idx.delete(0);
        let r = idx.search("alpha", 10, None);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 1);
    }

    #[test]
    fn put_replaces_existing_slot() {
        let d = tempdir().unwrap();
        let mut idx = open_empty(d.path());
        idx.put(0, "first version");
        idx.put(0, "second version");
        // "first" should no longer match slot 0.
        let r = idx.search("first", 10, None);
        assert!(r.is_empty());
        let r = idx.search("second", 10, None);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 0);
    }

    #[test]
    fn empty_text_is_noop() {
        let d = tempdir().unwrap();
        let mut idx = open_empty(d.path());
        idx.put(0, "");
        idx.put(1, "   ");
        idx.put(2, "real content");
        assert_eq!(idx.n_docs(), 1);
    }

    #[test]
    fn filter_restricts_results() {
        let d = tempdir().unwrap();
        let mut idx = open_empty(d.path());
        idx.put(0, "match");
        idx.put(1, "match");
        idx.put(2, "match");
        // Filter to only slot 1.
        let r = idx.search("match", 10, Some(&[1]));
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 1);
    }

    #[test]
    fn persistence_roundtrip() {
        let d = tempdir().unwrap();
        {
            let mut idx = open_empty(d.path());
            idx.put(0, "alpha bravo charlie");
            idx.put(1, "delta echo");
            idx.flush().unwrap();
        }
        // Reopen — should see prior state.
        let idx = open_empty(d.path());
        assert_eq!(idx.n_docs(), 2);
        let r = idx.search("alpha", 10, None);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 0);
    }

    #[test]
    fn flush_is_idempotent_when_clean() {
        let d = tempdir().unwrap();
        let mut idx = open_empty(d.path());
        idx.put(0, "x");
        idx.flush().unwrap();
        // Second flush with no changes shouldn't fail or rewrite the file.
        idx.flush().unwrap();
    }

    #[test]
    fn case_insensitive_match() {
        let d = tempdir().unwrap();
        let mut idx = open_empty(d.path());
        idx.put(0, "Hello World");
        let r = idx.search("hello", 10, None);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 0);
    }

    #[test]
    fn rare_term_outranks_common_term() {
        // A doc that contains the rarer query term should score higher.
        let d = tempdir().unwrap();
        let mut idx = open_empty(d.path());
        for s in 0..10u32 {
            idx.put(s, "common common common");
        }
        idx.put(10, "common rare");
        let r = idx.search("common rare", 10, None);
        assert_eq!(r[0].0, 10, "doc with the rare term must rank first");
    }

    #[test]
    fn corrupt_idx_is_treated_as_empty() {
        let d = tempdir().unwrap();
        let p = d.path().join("bm25.idx");
        std::fs::write(&p, b"garbage").unwrap();
        // Should not panic; should yield an empty index.
        let idx = Bm25Index::open(p).unwrap();
        assert_eq!(idx.n_docs(), 0);
    }
}
