use memmap2::Mmap;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::fs::File;
use std::sync::Arc;

use super::backend::StorageBackend;

pub const MAX_DEGREE: usize = 127;
const CANDIDATE_MULTIPLIER: usize = 2;
const GRAPH_V2_MAGIC: &[u8; 4] = b"TQG2";

#[derive(Copy, Clone, PartialEq)]
pub struct SearchCandidate {
    pub id: u32,
    pub score: f64,
}
impl Eq for SearchCandidate {}
impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // High scores first (Max-Heap behavior for candidates heap)
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

#[derive(PartialEq)]
pub struct OrderingWrapper(pub SearchCandidate);
impl Eq for OrderingWrapper {}
impl Ord for OrderingWrapper {
    fn cmp(&self, other: &Self) -> Ordering {
        // Low scores first (Min-Heap behavior for results heap)
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

pub struct GraphManager {
    backend: Arc<StorageBackend>,
    local_cache_path: String,
    mmap: Option<Mmap>,
    node_count: usize,
    offsets: Vec<u64>,
    data_start: usize,
}

impl GraphManager {
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

    pub fn node_count(&self) -> usize {
        self.node_count
    }

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

        if mmap.len() >= 8 && &mmap[0..4] == GRAPH_V2_MAGIC {
            let n = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;
            let table_bytes = (n + 1) * 8;
            let data_start = 8 + table_bytes;
            if mmap.len() >= data_start {
                let mut offsets = Vec::with_capacity(n + 1);
                let mut ok = true;
                for i in 0..=n {
                    let start = 8 + (i * 8);
                    let end = start + 8;
                    let off = u64::from_le_bytes(mmap[start..end].try_into().unwrap());
                    offsets.push(off);
                }
                if offsets[0] != 0 {
                    ok = false;
                }
                if let Some(last) = offsets.last().copied() {
                    if (data_start as u64).saturating_add(last) > mmap.len() as u64 {
                        ok = false;
                    }
                }
                for w in offsets.windows(2) {
                    if w[0] > w[1] {
                        ok = false;
                        break;
                    }
                }

                if ok {
                    self.node_count = n;
                    self.offsets = offsets;
                    self.data_start = data_start;
                    return;
                }
            }
        }
        self.node_count = 0;
    }

    pub fn get_neighbors_raw(&self, node_id: u32) -> Result<&[u8], String> {
        let mmap = self.mmap.as_ref().ok_or("Graph not loaded")?;

        let i = node_id as usize;
        if i >= self.node_count {
            return Err(format!("Node ID {} out of range", node_id));
        }
        let start = self.data_start + self.offsets[i] as usize;
        let end = self.data_start + self.offsets[i + 1] as usize;
        if end > mmap.len() || start > end || end.saturating_sub(start) < 4 {
            return Err(format!("Corrupt graph record for node {}", node_id));
        }
        Ok(&mmap[start..end])
    }

    pub fn search(
        &self,
        entry_node: u32,
        k: usize,
        search_list_size: usize,
        scorer: impl Fn(u32) -> f64,
        filter: Option<impl Fn(u32) -> bool>,
    ) -> Result<Vec<(u32, f64)>, Box<dyn std::error::Error + Send + Sync>> {
        if self.node_count == 0 {
            return Ok(Vec::new());
        }
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        // Use multiple entry points to improve recall
        let mut entries = Vec::new();
        entries.push(entry_node);
        if self.node_count > 20 {
            for i in 1..10 {
                let rd = (entry_node as usize + i * 1234567) % self.node_count;
                entries.push(rd as u32);
            }
        }

        for en in entries {
            if visited.contains(&en) { continue; }
            let score = scorer(en);
            let cand = SearchCandidate { id: en, score };
            candidates.push(cand);
            visited.insert(en);
            let matches = if let Some(f) = &filter { f(en) } else { true };
            if matches {
                results.push(OrderingWrapper(cand));
            }
        }

        while let Some(current) = candidates.pop() {
            if results.len() >= search_list_size && current.score < results.peek().unwrap().0.score {
                break;
            }

            if let Ok(rec) = self.get_neighbors_raw(current.id) {
                let count = u32::from_le_bytes(rec[0..4].try_into().unwrap()) as usize;
                let cap_count = count.min(MAX_DEGREE);
                for i in 0..cap_count {
                    let p = 4 + i * 4;
                    let nb = u32::from_le_bytes(rec[p..p + 4].try_into().unwrap());
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

    pub fn select_neighbors_heuristic(
        &self,
        _node_id: u32,
        candidates: Vec<(u32, f64)>,
        max_degree: usize,
    ) -> Vec<u32> {
        if candidates.is_empty() { return Vec::new(); }
        // candidates is already sorted descending by score (closest first)
        
        let mut result = Vec::new();
        for (c_id, _c_score) in candidates {
            if result.len() >= max_degree { break; }
            // Simplification: In a truly data-oblivious approach, we can just pick the top K.
            // But we keep the diversity heuristic placeholder for future HNSW alignment.
            result.push(c_id);
        }
        result
    }

    pub fn build(
        &mut self,
        n: usize,
        max_degree: usize,
        _alpha: f64,
        build_scorer: impl Fn(u32, &[u32]) -> Vec<(u32, f64)> + Sync,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.mmap = None;

        let degree_cap = max_degree.min(MAX_DEGREE).max(1);
        let max_neighbors = n.saturating_sub(1);
        let candidate_cap = (degree_cap * CANDIDATE_MULTIPLIER)
            .max(degree_cap)
            .min(max_neighbors);

        let candidate_lists: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut candidate_ids: Vec<u32> = Vec::with_capacity(candidate_cap + 4);
                let mut seen = HashSet::with_capacity(candidate_cap + 4);

                if n > 1 {
                    let prev = if i == 0 { n - 1 } else { i - 1 };
                    let next = (i + 1) % n;
                    if prev != i && seen.insert(prev as u32) {
                        candidate_ids.push(prev as u32);
                    }
                    if next != i && seen.insert(next as u32) {
                        candidate_ids.push(next as u32);
                    }
                }

                let mut x = (i as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                while candidate_ids.len() < candidate_cap && (seen.len() + 1) < n {
                    x = x.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
                    let cand = (x % (n.max(1) as u64)) as usize;
                    if cand != i && seen.insert(cand as u32) {
                        candidate_ids.push(cand as u32);
                    }
                }
                candidate_ids
            })
            .collect();

        // Initial adjacency using Greedy search + Heuristic
        let mut adjacency: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut scored: Vec<(u32, f64)> = build_scorer(i as u32, &candidate_lists[i]);
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                self.select_neighbors_heuristic(i as u32, scored, degree_cap)
            })
            .collect();

        // Build Refinement (NN-style)
        for _iter in 0..1 {
            let next_adjacency: Vec<Vec<u32>> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut candidates = HashSet::new();
                    for &nb in &adjacency[i] {
                        candidates.insert(nb);
                        for &nbnb in &adjacency[nb as usize] {
                            if nbnb != i as u32 { candidates.insert(nbnb); }
                        }
                    }
                    let cand_vec: Vec<u32> = candidates.into_iter().collect();
                    let mut scored = build_scorer(i as u32, &cand_vec);
                    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                    self.select_neighbors_heuristic(i as u32, scored, degree_cap)
                })
                .collect();
            adjacency = next_adjacency;
        }

        let mut payload = Vec::<u8>::new();
        let mut offsets = Vec::<u64>::with_capacity(n + 1);
        offsets.push(0);
        for neighbors in adjacency {
            let deg = neighbors.len().min(MAX_DEGREE) as u32;
            payload.extend_from_slice(&deg.to_le_bytes());
            for nb in neighbors.into_iter().take(deg as usize) {
                payload.extend_from_slice(&nb.to_le_bytes());
            }
            offsets.push(payload.len() as u64);
        }

        let mut out = Vec::<u8>::with_capacity(8 + (n + 1) * 8 + payload.len());
        out.extend_from_slice(GRAPH_V2_MAGIC);
        out.extend_from_slice(&(n as u32).to_le_bytes());
        for off in &offsets {
            out.extend_from_slice(&off.to_le_bytes());
        }
        out.extend_from_slice(&payload);

        std::fs::write(&self.local_cache_path, &out)?;
        self.node_count = n;
        self.backend.write("graph.bin", out)?;
        let file_ro = File::open(&self.local_cache_path)?;
        self.mmap = Some(unsafe { Mmap::map(&file_ro)? });
        self.parse_layout();
        Ok(())
    }
}
