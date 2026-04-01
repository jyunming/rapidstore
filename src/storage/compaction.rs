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
