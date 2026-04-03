use super::backend::StorageBackend;
use super::segment::{Segment, SegmentRecord};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Compacts many immutable segments into one new segment.
pub struct Compactor {
    pub backend: Arc<StorageBackend>,
    pub min_segments: usize,
}

impl Compactor {
    pub fn new(backend: Arc<StorageBackend>) -> Self {
        Self {
            backend,
            min_segments: 10,
        }
    }

    pub fn should_compact(&self, segment_count: usize) -> bool {
        segment_count >= self.min_segments
    }

    /// Persist a compaction transaction marker before old segments are deleted.
    pub fn begin_compaction(
        &self,
        old_segment_names: &[String],
        new_segment_name: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let state = CompactionState {
            phase: "prepared".to_string(),
            old_segment_names: old_segment_names.to_vec(),
            new_segment_name: new_segment_name.to_string(),
        };
        self.backend
            .write(COMPACTION_STATE_FILE, &serde_json::to_vec_pretty(&state)?)?;
        Ok(())
    }

    /// Finalize compaction transaction marker after old segments were deleted.
    pub fn finish_compaction(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.backend.delete(COMPACTION_STATE_FILE)?;
        Ok(())
    }

    /// Recover from an interrupted compaction if a marker file exists.
    /// Returns true when a marker was found and processed.
    pub fn recover_if_needed(&self) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        if !self.backend.exists(COMPACTION_STATE_FILE) {
            return Ok(false);
        }
        let state = match self.read_state() {
            Ok(s) => s,
            Err(_) => {
                // Corrupt marker: best-effort cleanup to unblock startup.
                self.backend.delete(COMPACTION_STATE_FILE)?;
                return Ok(true);
            }
        };

        // If the compacted segment exists, complete old-segment deletion.
        if self.backend.exists(&state.new_segment_name) {
            for name in &state.old_segment_names {
                if name != &state.new_segment_name {
                    let _ = self.backend.delete(name);
                }
            }
        }

        // If the compacted segment does not exist, abandon transaction and keep old segments.
        self.backend.delete(COMPACTION_STATE_FILE)?;
        Ok(true)
    }

    /// Write a new compacted segment from already-resolved live records,
    /// then delete the provided old segment files.
    pub fn compact_live_records(
        &self,
        old_segment_names: &[String],
        new_segment_name: &str,
        live_records: &[SegmentRecord],
    ) -> Result<Segment, Box<dyn std::error::Error + Send + Sync>> {
        let new_seg = Segment::write_batch(&self.backend, new_segment_name, live_records)?;
        for name in old_segment_names {
            // Guard: never delete the freshly written compacted segment in case a caller
            // bug causes new_segment_name to appear in old_segment_names.
            if name == new_segment_name {
                continue;
            }
            self.backend.delete(name)?;
        }
        Ok(new_seg)
    }

    fn read_state(&self) -> Result<CompactionState, Box<dyn std::error::Error + Send + Sync>> {
        let raw = self.backend.read(COMPACTION_STATE_FILE)?;
        Ok(serde_json::from_slice::<CompactionState>(&raw)?)
    }
}

const COMPACTION_STATE_FILE: &str = "compaction_state.json";

#[derive(Serialize, Deserialize)]
struct CompactionState {
    phase: String,
    old_segment_names: Vec<String>,
    new_segment_name: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::segment::{Segment, SegmentRecord};
    use std::sync::Arc;
    use tempfile::tempdir;

    fn make_compactor_and_backend() -> (tempfile::TempDir, Arc<StorageBackend>, Compactor) {
        let dir = tempdir().unwrap();
        let backend = Arc::new(StorageBackend::from_uri(dir.path().to_str().unwrap()).unwrap());
        let compactor = Compactor::new(Arc::clone(&backend));
        (dir, backend, compactor)
    }

    #[test]
    fn should_compact_threshold() {
        let (_dir, _backend, compactor) = make_compactor_and_backend();
        assert!(!compactor.should_compact(9));
        assert!(compactor.should_compact(10));
        assert!(compactor.should_compact(100));
    }

    #[test]
    fn compact_live_records_creates_new_and_deletes_old() {
        let (_dir, backend, compactor) = make_compactor_and_backend();
        let r = vec![SegmentRecord {
            id: "a".to_string(),
            is_deleted: false,
        }];
        Segment::write_batch(&backend, "seg-00000000.bin", &r).unwrap();
        Segment::write_batch(&backend, "seg-00000001.bin", &r).unwrap();

        let live = vec![SegmentRecord {
            id: "a".to_string(),
            is_deleted: false,
        }];
        let old_names = vec![
            "seg-00000000.bin".to_string(),
            "seg-00000001.bin".to_string(),
        ];
        let new_seg = compactor
            .compact_live_records(&old_names, "seg-00000002.bin", &live)
            .unwrap();

        assert_eq!(new_seg.name, "seg-00000002.bin");
        assert_eq!(new_seg.record_count, 1);
        assert!(!backend.exists("seg-00000000.bin"));
        assert!(!backend.exists("seg-00000001.bin"));
        assert!(backend.exists("seg-00000002.bin"));
    }

    #[test]
    fn compact_live_records_guard_prevents_self_deletion() {
        let (_dir, backend, compactor) = make_compactor_and_backend();
        let r = vec![SegmentRecord {
            id: "a".to_string(),
            is_deleted: false,
        }];
        Segment::write_batch(&backend, "seg-00000000.bin", &r).unwrap();

        // new_segment_name also appears in old_segment_names — must NOT be deleted
        let old_names = vec![
            "seg-00000000.bin".to_string(),
            "seg-00000001.bin".to_string(),
        ];
        compactor
            .compact_live_records(&old_names, "seg-00000001.bin", &r)
            .unwrap();

        assert!(
            backend.exists("seg-00000001.bin"),
            "freshly compacted segment must survive"
        );
    }

    #[test]
    fn begin_and_finish_compaction_lifecycle() {
        let (_dir, backend, compactor) = make_compactor_and_backend();
        compactor
            .begin_compaction(&["seg-00000000.bin".to_string()], "seg-00000001.bin")
            .unwrap();
        assert!(backend.exists("compaction_state.json"));
        compactor.finish_compaction().unwrap();
        assert!(!backend.exists("compaction_state.json"));
    }

    #[test]
    fn recover_if_needed_no_state_file_returns_false() {
        let (_dir, _backend, compactor) = make_compactor_and_backend();
        let recovered = compactor.recover_if_needed().unwrap();
        assert!(!recovered);
    }

    #[test]
    fn recover_with_completed_new_segment_deletes_old() {
        let (_dir, backend, compactor) = make_compactor_and_backend();
        let r = vec![SegmentRecord {
            id: "a".to_string(),
            is_deleted: false,
        }];
        Segment::write_batch(&backend, "seg-00000000.bin", &r).unwrap();
        Segment::write_batch(&backend, "seg-00000001.bin", &r).unwrap();

        // Simulate crash after write_batch but before finish_compaction
        compactor
            .begin_compaction(&["seg-00000000.bin".to_string()], "seg-00000001.bin")
            .unwrap();

        let recovered = compactor.recover_if_needed().unwrap();
        assert!(recovered);
        assert!(!backend.exists("compaction_state.json"));
        assert!(
            !backend.exists("seg-00000000.bin"),
            "old segment must be cleaned up"
        );
        assert!(
            backend.exists("seg-00000001.bin"),
            "new segment must be preserved"
        );
    }

    #[test]
    fn recover_with_missing_new_segment_preserves_old() {
        let (_dir, backend, compactor) = make_compactor_and_backend();
        let r = vec![SegmentRecord {
            id: "a".to_string(),
            is_deleted: false,
        }];
        Segment::write_batch(&backend, "seg-00000000.bin", &r).unwrap();

        // Crash before write_batch: new segment doesn't exist
        compactor
            .begin_compaction(&["seg-00000000.bin".to_string()], "seg-00000001.bin")
            .unwrap();

        let recovered = compactor.recover_if_needed().unwrap();
        assert!(recovered);
        assert!(!backend.exists("compaction_state.json"));
        assert!(
            backend.exists("seg-00000000.bin"),
            "old segment must survive abandoned compaction"
        );
    }

    #[test]
    fn recover_with_corrupt_state_file_cleans_up() {
        let (_dir, backend, compactor) = make_compactor_and_backend();
        backend
            .write("compaction_state.json", b"not valid json!!")
            .unwrap();
        let recovered = compactor.recover_if_needed().unwrap();
        assert!(recovered);
        assert!(
            !backend.exists("compaction_state.json"),
            "corrupt state file must be removed"
        );
    }

    #[test]
    fn compactor_new_sets_min_segments_to_10() {
        let (_dir, _backend, compactor) = make_compactor_and_backend();
        assert_eq!(compactor.min_segments, 10);
    }
}
