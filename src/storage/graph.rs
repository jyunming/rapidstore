use memmap2::Mmap;
use rand::Rng;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::fs::File;
use std::sync::Arc;

use super::backend::StorageBackend;

/// Maximum adjacency list size per HNSW node across all layers.
pub const MAX_DEGREE: usize = 127;
const GRAPH_V3_MAGIC: &[u8; 4] = b"TQG3";

#[derive(Copy, Clone, PartialEq)]
pub struct SearchCandidate {
    pub id: u32,
    pub score: f64,
}
impl Eq for SearchCandidate {}
impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(PartialEq, Clone)]
pub struct OrderingWrapper(pub SearchCandidate);
impl Eq for OrderingWrapper {}
impl Ord for OrderingWrapper {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .0
            .score
            .partial_cmp(&self.0.score)
            .unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for OrderingWrapper {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Manages the HNSW graph index: construction, persistence, and approximate search.
///
/// The graph is stored in `graph.bin` (V3 binary: magic `TQG3` + header + offset
/// table + varint-delta-encoded adjacency lists) and memory-mapped at open time
/// for O(1) node lookup. Call [`GraphManager::build`] after loading vectors, then
/// [`GraphManager::search`] for sub-linear approximate nearest-neighbour queries.
pub struct GraphManager {
    backend: Arc<StorageBackend>,
    local_cache_path: String,
    mmap: Option<Mmap>,
    node_count: usize,
    offsets: Vec<u64>,
    data_start: usize,
    pub entry_point: u32,
    pub max_level: u32,
}

impl GraphManager {
    /// Open (or create) the graph manager. Reads `graph.bin` from `backend` into
    /// `cache_dir` and memory-maps it; if the file is absent the manager starts empty.
    pub fn open(
        backend: Arc<StorageBackend>,
        cache_dir: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let local_cache_path = format!("{}/graph.bin", cache_dir);
        let mut manager = Self {
            backend: backend.clone(),
            local_cache_path,
            mmap: None,
            node_count: 0,
            offsets: Vec::new(),
            data_start: 0,
            entry_point: 0,
            max_level: 0,
        };
        if let Ok(data) = backend.read("graph.bin") {
            std::fs::create_dir_all(cache_dir)?;
            std::fs::write(&manager.local_cache_path, data)?;
            let file = File::open(&manager.local_cache_path)?;
            manager.mmap = Some(unsafe { Mmap::map(&file)? });
            manager.parse_layout();
        }
        Ok(manager)
    }

    /// Number of nodes currently indexed. Returns 0 when no index has been built.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Returns `true` when a valid graph is loaded and ready for [`search`](Self::search).
    pub fn has_index(&self) -> bool {
        self.node_count > 0 && self.mmap.is_some()
    }

    fn parse_layout(&mut self) {
        self.offsets.clear();
        self.data_start = 0;

        let Some(mmap) = self.mmap.as_ref() else {
            self.node_count = 0;
            return;
        };

        if mmap.len() >= 16 && &mmap[0..4] == GRAPH_V3_MAGIC {
            let n = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;
            self.entry_point = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
            self.max_level = u32::from_le_bytes(mmap[12..16].try_into().unwrap());

            let table_bytes = (n + 1) * 8;
            let data_start = 16 + table_bytes;
            if mmap.len() >= data_start {
                let mut offsets = Vec::with_capacity(n + 1);
                for i in 0..=n {
                    let start = 16 + (i * 8);
                    let end = start + 8;
                    let off = u64::from_le_bytes(mmap[start..end].try_into().unwrap());
                    offsets.push(off);
                }

                self.node_count = n;
                self.offsets = offsets;
                self.data_start = data_start;
                return;
            }
        }
        self.node_count = 0;
    }

    /// Read the neighbour list of `node_id` at the given `level`. Returns an empty
    /// `Vec` when `level` exceeds the node's assigned HNSW level.
    pub fn get_neighbors_at_level(&self, node_id: u32, level: u32) -> Result<Vec<u32>, String> {
        let mmap = self.mmap.as_ref().ok_or("Graph not loaded")?;
        let i = node_id as usize;
        if i >= self.node_count {
            return Err("OOR".into());
        }

        let start = self.data_start + self.offsets[i] as usize;
        let end = self.data_start + self.offsets[i + 1] as usize;
        let rec = &mmap[start..end];

        if rec.len() < 4 {
            return Err(format!(
                "corrupt graph record for node {}: too short ({} bytes)",
                node_id,
                rec.len()
            ));
        }
        let num_levels = u32::from_le_bytes(rec[0..4].try_into().unwrap());
        if level >= num_levels {
            return Ok(Vec::new());
        }

        let mut curr = 4;
        for l in 0..num_levels {
            if curr + 4 > rec.len() {
                return Err(format!(
                    "corrupt graph record for node {}: truncated at level {}",
                    node_id, l
                ));
            }
            let count = u32::from_le_bytes(rec[curr..curr + 4].try_into().unwrap()) as usize;
            // Validate count is plausible given remaining bytes (each varint is at least 1 byte)
            let max_possible = (rec.len() - curr - 4).min(count);
            if count > rec.len() {
                return Err(format!(
                    "corrupt graph record for node {}: implausible neighbour count {}",
                    node_id, count
                ));
            }
            let _ = max_possible;
            curr += 4;
            if l == level {
                let mut nbs = Vec::with_capacity(count);
                let mut last_nb = 0u32;
                for _ in 0..count {
                    if curr >= rec.len() {
                        return Err(format!(
                            "corrupt graph record for node {}: varint overrun",
                            node_id
                        ));
                    }
                    let (delta, bytes_read) = decode_varint(&rec[curr..]);
                    let nb = last_nb + delta;
                    nbs.push(nb);
                    last_nb = nb;
                    curr += bytes_read;
                }
                return Ok(nbs);
            }
            // Skip level data
            for _ in 0..count {
                if curr >= rec.len() {
                    return Err(format!(
                        "corrupt graph record for node {}: varint overrun while skipping",
                        node_id
                    ));
                }
                let (_, bytes_read) = decode_varint(&rec[curr..]);
                curr += bytes_read;
            }
        }
        Ok(Vec::new())
    }

    /// Beam search from the entry point down to level 0. Returns at most `k` results
    /// ordered by descending `scorer` value, optionally filtered by `filter`.
    pub fn search(
        &self,
        _entry_node_unused: u32,
        k: usize,
        search_list_size: usize,
        scorer: impl Fn(u32) -> f64,
        filter: Option<impl Fn(u32) -> bool>,
    ) -> Result<Vec<(u32, f64)>, Box<dyn std::error::Error + Send + Sync>> {
        if self.node_count == 0 {
            return Ok(Vec::new());
        }

        // Upper-layer beam uses a fixed cap (not search_list_size) because upper layers
        // have far fewer nodes and the scoring LUT for large b/dim can be cache-cold.
        // A 32-wide beam is sufficient for greedy navigation to a good level-0 entry point.
        const EF_UPPER: usize = 32;
        let mut beam = BinaryHeap::new();
        let start_score = scorer(self.entry_point);
        beam.push(OrderingWrapper(SearchCandidate {
            id: self.entry_point,
            score: start_score,
        }));

        // Multi-level navigation with beam search
        for l in (1..=self.max_level).rev() {
            let mut candidates = BinaryHeap::new();
            let mut visited = HashSet::new();

            for OrderingWrapper(cand) in &beam {
                candidates.push(cand.clone());
                visited.insert(cand.id);
            }

            let mut layer_results = beam.clone();

            while let Some(current) = candidates.pop() {
                if layer_results.len() >= EF_UPPER
                    && current.score < layer_results.peek().unwrap().0.score
                {
                    break;
                }

                if let Ok(nbs) = self.get_neighbors_at_level(current.id, l) {
                    for nb in nbs {
                        if visited.contains(&nb) {
                            continue;
                        }
                        visited.insert(nb);

                        let score = scorer(nb);
                        let cand = SearchCandidate { id: nb, score };
                        if layer_results.len() < EF_UPPER
                            || score > layer_results.peek().unwrap().0.score
                        {
                            candidates.push(cand);
                            layer_results.push(OrderingWrapper(cand));
                            if layer_results.len() > EF_UPPER {
                                layer_results.pop();
                            }
                        }
                    }
                }
            }
            beam = layer_results;
        }

        // Level 0 search
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        for OrderingWrapper(seed) in beam {
            candidates.push(seed);
            visited.insert(seed.id);
            let matches = if let Some(f) = &filter {
                f(seed.id)
            } else {
                true
            };
            if matches {
                results.push(OrderingWrapper(seed));
            }
        }

        while let Some(current) = candidates.pop() {
            if results.len() >= search_list_size && current.score < results.peek().unwrap().0.score
            {
                break;
            }

            if let Ok(nbs) = self.get_neighbors_at_level(current.id, 0) {
                for nb in nbs {
                    if visited.contains(&nb) {
                        continue;
                    }
                    visited.insert(nb);

                    let score = scorer(nb);
                    let cand = SearchCandidate { id: nb, score };
                    candidates.push(cand);

                    let matches = if let Some(f) = &filter { f(nb) } else { true };
                    if matches {
                        if results.len() < search_list_size
                            || cand.score > results.peek().unwrap().0.score
                        {
                            results.push(OrderingWrapper(cand));
                            if results.len() > search_list_size {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut out: Vec<(u32, f64)> = results
            .into_iter()
            .map(|OrderingWrapper(c)| (c.id, c.score))
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        out.truncate(k);
        Ok(out)
    }

    /// Build the HNSW graph for `n` nodes, persist it to `graph.bin`, and memory-map
    /// the result. `build_scorer(i, candidates)` must return scored `(id, score)` pairs.
    pub fn build(
        &mut self,
        n: usize,
        max_degree: usize,
        ef_construction: usize,
        n_refinements: usize,
        _alpha: f64,
        build_scorer: impl Fn(u32, &[u32]) -> Vec<(u32, f64)> + Sync,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.mmap = None;
        let mut rng = rand::thread_rng();

        let m = (max_degree / 2).max(4);
        let m0 = max_degree.min(MAX_DEGREE);
        let level_mult = 1.0 / (m as f64).ln();
        let cand_pool = ef_construction;

        let mut node_levels = vec![0u32; n];
        let mut max_l = 0u32;
        let mut ep = 0u32;

        for i in 0..n {
            let r: f64 = rng.gen_range(0.0..1.0);
            let l = ((-r.ln() * level_mult) as u32).min(5);
            node_levels[i] = l;
            if l >= max_l {
                max_l = l;
                ep = i as u32;
            }
        }

        let mut adjacency: Vec<Vec<Vec<u32>>> = vec![vec![Vec::new(); 6]; n];

        for l in 0..=max_l {
            let level_nodes: Vec<u32> = (0..n)
                .filter(|&i| node_levels[i] >= l)
                .map(|i| i as u32)
                .collect();
            if level_nodes.len() < 2 {
                continue;
            }

            let ln = level_nodes.len();
            let degree_cap = if l == 0 { m0 } else { m };

            // Build adjacency for this level subset
            let level_adj: Vec<Vec<u32>> = level_nodes
                .par_iter()
                .map(|&i| {
                    let mut local_rng = rand::thread_rng();
                    let mut candidates = HashSet::new();
                    while candidates.len() < cand_pool && candidates.len() < ln - 1 {
                        let cand = level_nodes[local_rng.gen_range(0..ln)];
                        if cand != i {
                            candidates.insert(cand);
                        }
                    }

                    let cand_vec: Vec<u32> = candidates.into_iter().collect();
                    let mut scored = build_scorer(i, &cand_vec);
                    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                    let mut chosen: Vec<u32> = scored
                        .into_iter()
                        .take(degree_cap)
                        .map(|(id, _)| id)
                        .collect();
                    chosen.sort_unstable();
                    chosen
                })
                .collect();

            for (idx, &i) in level_nodes.iter().enumerate() {
                adjacency[i as usize][l as usize] = level_adj[idx].clone();
            }

            // Multiple refinement iterations (nn-descent style)
            for _ in 0..n_refinements {
                let next_level_adj: Vec<Vec<u32>> = level_nodes
                    .par_iter()
                    .map(|&i| {
                        let mut candidates = HashSet::new();
                        for &nb in &adjacency[i as usize][l as usize] {
                            candidates.insert(nb);
                            for &nbnb in &adjacency[nb as usize][l as usize] {
                                if nbnb != i {
                                    candidates.insert(nbnb);
                                }
                            }
                        }
                        let cand_vec: Vec<u32> = candidates.into_iter().collect();
                        let mut scored = build_scorer(i, &cand_vec);
                        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                        let mut chosen: Vec<u32> = scored
                            .into_iter()
                            .take(degree_cap)
                            .map(|(id, _)| id)
                            .collect();
                        chosen.sort_unstable();
                        chosen
                    })
                    .collect();

                for (idx, &i) in level_nodes.iter().enumerate() {
                    adjacency[i as usize][l as usize] = next_level_adj[idx].clone();
                }
            }
        }

        let mut payload = Vec::<u8>::new();
        let mut offsets = Vec::<u64>::with_capacity(n + 1);
        offsets.push(0);

        for i in 0..n {
            let l_count = node_levels[i] + 1;
            payload.extend_from_slice(&l_count.to_le_bytes());
            for l in 0..l_count {
                let neighbors = &adjacency[i][l as usize];
                let deg = neighbors.len() as u32;
                payload.extend_from_slice(&deg.to_le_bytes());
                let mut last_nb = 0u32;
                for &nb in neighbors {
                    let delta = nb - last_nb;
                    encode_varint(delta, &mut payload);
                    last_nb = nb;
                }
            }
            offsets.push(payload.len() as u64);
        }

        let mut out = Vec::<u8>::with_capacity(16 + (n + 1) * 8 + payload.len());
        out.extend_from_slice(GRAPH_V3_MAGIC);
        out.extend_from_slice(&(n as u32).to_le_bytes());
        out.extend_from_slice(&ep.to_le_bytes());
        out.extend_from_slice(&max_l.to_le_bytes());
        for off in &offsets {
            out.extend_from_slice(&off.to_le_bytes());
        }
        out.extend_from_slice(&payload);

        std::fs::write(&self.local_cache_path, &out)?;
        self.node_count = n;
        self.entry_point = ep;
        self.max_level = max_l;
        self.backend.write("graph.bin", &out)?;
        let file_ro = File::open(&self.local_cache_path)?;
        self.mmap = Some(unsafe { Mmap::map(&file_ro)? });
        self.parse_layout();
        Ok(())
    }
}

fn encode_varint(mut val: u32, out: &mut Vec<u8>) {
    while val >= 0x80 {
        out.push((val as u8) | 0x80);
        val >>= 7;
    }
    out.push(val as u8);
}

fn decode_varint(data: &[u8]) -> (u32, usize) {
    let mut val = 0u32;
    let mut shift = 0;
    for (i, &b) in data.iter().enumerate() {
        val |= ((b & 0x7F) as u32) << shift;
        if b < 0x80 {
            return (val, i + 1);
        }
        shift += 7;
    }
    (val, data.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn make_backend(dir: &tempfile::TempDir) -> Arc<StorageBackend> {
        Arc::new(StorageBackend::from_uri(dir.path().to_str().unwrap()).unwrap())
    }

    /// A simple identity scorer used in build() calls below.
    fn dot_scorer(from: u32, candidates: &[u32]) -> Vec<(u32, f64)> {
        candidates
            .iter()
            .map(|&to| {
                let score = 1.0 / (1.0 + (from as f64 - to as f64).abs());
                (to, score)
            })
            .collect()
    }

    // ── parse_layout with mmap=None (lines 122-124) ──────────────────────────
    // Fresh GraphManager (no graph.bin) → mmap is None → parse_layout returns early

    #[test]
    fn parse_layout_mmap_none_sets_node_count_zero() {
        let dir = tempdir().unwrap();
        let backend = make_backend(&dir);
        let mut mgr = GraphManager::open(backend, dir.path().to_str().unwrap()).unwrap();
        // No graph.bin exists → mmap is None → node_count must be 0
        assert_eq!(mgr.node_count(), 0);
        // Call parse_layout directly (accessible because we're in the same module)
        mgr.parse_layout();
        assert_eq!(
            mgr.node_count(),
            0,
            "parse_layout with no mmap must set node_count=0"
        );
    }

    // ── parse_layout V3 magic but header too short (line 149) ─────────────────
    // Write a file with TQG3 magic + n=100 but without enough bytes for the
    // offset table → data_start check fails → node_count set to 0

    #[test]
    fn parse_layout_v3_truncated_sets_node_count_zero() {
        let dir = tempdir().unwrap();
        let backend = make_backend(&dir);
        let cache = dir.path().to_str().unwrap();

        // Header: magic(4) + n=100 u32(4) + ep=0 u32(4) + max_level=0 u32(4) = 16 bytes
        // No offset table → data_start = 16 + (100+1)*8 = 824 > file_len(16) → truncated path
        let mut fake: Vec<u8> = Vec::new();
        fake.extend_from_slice(b"TQG3");
        fake.extend_from_slice(&100u32.to_le_bytes()); // n=100
        fake.extend_from_slice(&0u32.to_le_bytes()); // entry_point
        fake.extend_from_slice(&0u32.to_le_bytes()); // max_level

        // Write directly to local cache path and also to backend
        let local_path = format!("{}/graph.bin", cache);
        std::fs::write(&local_path, &fake).unwrap();
        backend.write("graph.bin", &fake).unwrap();

        // Open — should succeed but report node_count=0 because data_start > file size
        let mgr = GraphManager::open(backend, cache).unwrap();
        assert_eq!(
            mgr.node_count(),
            0,
            "truncated V3 header should result in node_count=0"
        );
    }

    // ── search on empty graph (line 236) ─────────────────────────────────────

    #[test]
    fn search_empty_graph_returns_empty() {
        let dir = tempdir().unwrap();
        let backend = make_backend(&dir);
        let mgr = GraphManager::open(backend, dir.path().to_str().unwrap()).unwrap();
        assert_eq!(mgr.node_count(), 0);
        let results = mgr
            .search(0, 5, 32, |_| 1.0, None::<fn(u32) -> bool>)
            .unwrap();
        assert!(
            results.is_empty(),
            "empty graph search must return empty vec"
        );
    }

    // ── get_neighbors_at_level: node_id >= node_count (line 156) ─────────────

    #[test]
    fn get_neighbors_node_out_of_range_returns_error() {
        let dir = tempdir().unwrap();
        let backend = make_backend(&dir);
        let cache = dir.path().to_str().unwrap();

        let mut mgr = GraphManager::open(backend, cache).unwrap();
        // Build a tiny 3-node graph
        mgr.build(3, 4, 8, 0, 1.2, dot_scorer).unwrap();

        // node_id=99 is >= node_count=3 → OOR error
        let result = mgr.get_neighbors_at_level(99, 0);
        assert!(result.is_err(), "out-of-range node_id must return Err");
    }

    // ── get_neighbors_at_level: level >= num_levels (line 172) ───────────────

    #[test]
    fn get_neighbors_level_exceeds_num_levels_returns_empty() {
        let dir = tempdir().unwrap();
        let backend = make_backend(&dir);
        let cache = dir.path().to_str().unwrap();

        let mut mgr = GraphManager::open(backend, cache).unwrap();
        mgr.build(5, 4, 8, 0, 1.2, dot_scorer).unwrap();

        // Level 99 far exceeds any stored num_levels → returns Ok(empty)
        let nbs = mgr.get_neighbors_at_level(0, 99).unwrap();
        assert!(
            nbs.is_empty(),
            "requesting a level beyond num_levels should return empty vec"
        );
    }

    // ── decode_varint: no terminator byte (line 521) ──────────────────────────
    // All bytes have the continuation bit set → loop exhausts slice → fallback return

    #[test]
    fn decode_varint_no_terminator_returns_full_length() {
        // All bytes have high bit set → no b < 0x80 found → returns data.len()
        let data = [0x80u8, 0x81, 0x82, 0x83];
        let (_, bytes_consumed) = decode_varint(&data);
        assert_eq!(
            bytes_consumed,
            data.len(),
            "varint with no terminator must consume all bytes"
        );
    }

    // ── Multi-level HNSW search (lines 251-291) ───────────────────────────────
    // With 200 nodes and max_degree=4 (level_mult≈0.721), ~50 nodes at level≥1,
    // which exceeds EF_UPPER=32 and reliably triggers beam saturation code paths.

    #[test]
    fn multilevel_graph_search_exercises_upper_layer_beam() {
        let dir = tempdir().unwrap();
        let backend = make_backend(&dir);
        let cache = dir.path().to_str().unwrap();

        let mut mgr = GraphManager::open(backend, cache).unwrap();
        let n = 200usize;
        // max_degree=4 → level_mult≈0.721 → ~50 nodes at level≥1 > EF_UPPER=32
        // This guarantees beam saturation: lines 257, 259, 272, 277, 279 covered.
        mgr.build(n, 4, 64, 1, 1.2, |from, candidates| {
            candidates
                .iter()
                .map(|&to| {
                    let diff = (from as f64) - (to as f64);
                    (to, -(diff * diff))
                })
                .collect()
        })
        .unwrap();

        assert!(
            mgr.max_level > 0,
            "200-node HNSW with max_degree=4 should virtually always produce max_level > 0"
        );

        let results = mgr
            .search(
                mgr.entry_point,
                10,
                64,
                |n| -(n as f64),
                None::<fn(u32) -> bool>,
            )
            .unwrap();
        assert!(
            !results.is_empty(),
            "multilevel search should return results"
        );
    }

    // ── Corrupt graph records: get_neighbors_at_level error paths ─────────────
    // Helper: write a valid graph.bin header with n=1 node but a custom (corrupt) record.

    fn write_corrupt_graph(dir: &tempfile::TempDir, max_level: u32, record: &[u8]) {
        let n: u32 = 1;
        let ep: u32 = 0;
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"TQG3");
        buf.extend_from_slice(&n.to_le_bytes());
        buf.extend_from_slice(&ep.to_le_bytes());
        buf.extend_from_slice(&max_level.to_le_bytes());
        // Offset table: offsets[0]=0, offsets[1]=record.len()
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&(record.len() as u64).to_le_bytes());
        buf.extend_from_slice(record);
        let path = dir.path().join("graph.bin");
        std::fs::write(&path, &buf).unwrap();
    }

    #[test]
    fn get_neighbors_corrupt_too_short_record_returns_error() {
        // rec.len()=3 < 4 → error "too short" (lines 155-159)
        let dir = tempdir().unwrap();
        let backend = make_backend(&dir);
        let cache = dir.path().to_str().unwrap();
        write_corrupt_graph(&dir, 0, &[0x01, 0x02, 0x03]);
        let mgr = GraphManager::open(backend, cache).unwrap();
        assert_eq!(mgr.node_count(), 1);
        let result = mgr.get_neighbors_at_level(0, 0);
        assert!(result.is_err(), "too-short record should return Err");
        assert!(result.unwrap_err().contains("too short"));
    }

    #[test]
    fn get_neighbors_corrupt_truncated_at_level_returns_error() {
        // num_levels=1 but no count bytes → curr+4 > rec.len() (lines 169-172)
        let dir = tempdir().unwrap();
        let backend = make_backend(&dir);
        let cache = dir.path().to_str().unwrap();
        let record = [1u8, 0, 0, 0]; // num_levels=1, no count data
        write_corrupt_graph(&dir, 0, &record);
        let mgr = GraphManager::open(backend, cache).unwrap();
        let result = mgr.get_neighbors_at_level(0, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("truncated at level"));
    }

    #[test]
    fn get_neighbors_corrupt_implausible_count_returns_error() {
        // count=200 > rec.len()=8 → implausible (lines 178-181)
        let dir = tempdir().unwrap();
        let backend = make_backend(&dir);
        let cache = dir.path().to_str().unwrap();
        let mut record = Vec::new();
        record.extend_from_slice(&1u32.to_le_bytes()); // num_levels=1
        record.extend_from_slice(&200u32.to_le_bytes()); // count=200 > rec.len()=8
        write_corrupt_graph(&dir, 0, &record);
        let mgr = GraphManager::open(backend, cache).unwrap();
        let result = mgr.get_neighbors_at_level(0, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("implausible"));
    }

    #[test]
    fn get_neighbors_corrupt_varint_overrun_reading_returns_error() {
        // count=3 but only 1 varint present → overrun reading (lines 190-193)
        let dir = tempdir().unwrap();
        let backend = make_backend(&dir);
        let cache = dir.path().to_str().unwrap();
        let mut record = Vec::new();
        record.extend_from_slice(&1u32.to_le_bytes()); // num_levels=1
        record.extend_from_slice(&3u32.to_le_bytes()); // count=3
        record.push(0x05u8); // one varint (delta=5), then overrun
        write_corrupt_graph(&dir, 0, &record);
        let mgr = GraphManager::open(backend, cache).unwrap();
        let result = mgr.get_neighbors_at_level(0, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("varint overrun"));
    }

    #[test]
    fn get_neighbors_corrupt_varint_overrun_skipping_returns_error() {
        // num_levels=2, level0_count=2, only 1 varint → overrun while skipping (lines 206-209)
        let dir = tempdir().unwrap();
        let backend = make_backend(&dir);
        let cache = dir.path().to_str().unwrap();
        let mut record = Vec::new();
        record.extend_from_slice(&2u32.to_le_bytes()); // num_levels=2
        record.extend_from_slice(&2u32.to_le_bytes()); // level0 count=2
        record.push(0x05u8); // one varint, then overrun while skipping level 0
        write_corrupt_graph(&dir, 1, &record);
        let mgr = GraphManager::open(backend, cache).unwrap();
        // Request level=1 → must skip level 0 → overrun
        let result = mgr.get_neighbors_at_level(0, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("varint overrun"));
    }

    #[test]
    fn search_with_corrupt_record_covers_err_arms_in_upper_and_base_level() {
        // max_level=1, record=[2,0,0,0] (num_levels=2, no count bytes)
        // get_neighbors_at_level(0, 1) → skip level 0 data → truncated → Err (line 281)
        // get_neighbors_at_level(0, 0) → read level 0 count → truncated → Err (line 333)
        let dir = tempdir().unwrap();
        let backend = make_backend(&dir);
        let cache = dir.path().to_str().unwrap();
        let record = [2u8, 0, 0, 0]; // num_levels=2, truncated after that
        write_corrupt_graph(&dir, 1, &record);
        let mgr = GraphManager::open(backend, cache).unwrap();
        assert_eq!(mgr.node_count(), 1);
        // search() internally calls get_neighbors_at_level and silently handles Err
        let results = mgr
            .search(0, 5, 16, |n| n as f64, None::<fn(u32) -> bool>)
            .unwrap();
        // Results may be non-empty (just the entry point), but must not panic
        let _ = results;
    }
}
