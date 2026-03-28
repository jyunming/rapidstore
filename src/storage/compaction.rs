/// Stub for Phase 4 background compaction.
/// Merges small segment files into larger ones and rebuilds graph edges.
/// This is intentionally left as a stub in Phase 1 — the engine works
/// correctly without compaction; it only affects long-term read latency.
pub struct Compactor {
    pub segments_dir: String,
}

impl Compactor {
    pub fn new(segments_dir: &str) -> Self {
        Self { segments_dir: segments_dir.to_string() }
    }

    /// Compact segments larger than threshold into a single merged segment.
    /// Full implementation deferred to Phase 4.
    pub fn compact_if_needed(&self, segment_count: usize) -> bool {
        // Trigger compaction when there are > 100 segments
        segment_count > 100
    }
}
