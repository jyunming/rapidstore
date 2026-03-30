use memmap2::Mmap;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::Arc;

use super::backend::StorageBackend;

const MAX_DEGREE: usize = 127;
const BLOCK_SIZE: usize = 512;
const CANDIDATE_MULTIPLIER: usize = 8;

#[derive(Clone, Debug, PartialEq)]
struct NodeRecord {
    pub neighbors: Vec<u32>,
}

impl NodeRecord {
    fn encode(&self) -> [u8; BLOCK_SIZE] {
        let mut buf = [0u8; BLOCK_SIZE];
        let count = (self.neighbors.len() as u32).min(MAX_DEGREE as u32);
        buf[0..4].copy_from_slice(&count.to_le_bytes());
        for i in 0..count as usize {
            let start = 4 + (i * 4);
            buf[start..start + 4].copy_from_slice(&self.neighbors[i].to_le_bytes());
        }
        buf
    }

    fn decode(buf: &[u8]) -> Self {
        let count = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
        let mut neighbors = Vec::with_capacity(count);
        for i in 0..count.min(MAX_DEGREE) {
            let start = 4 + (i * 4);
            neighbors.push(u32::from_le_bytes(buf[start..start + 4].try_into().unwrap()));
        }
        Self { neighbors }
    }
}

#[derive(Copy, Clone, PartialEq)]
struct SearchCandidate {
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

#[derive(PartialEq)]
struct OrderingWrapper(SearchCandidate);
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

pub struct GraphManager {
    backend: Arc<StorageBackend>,
    local_cache_path: String,
    mmap: Option<Mmap>,
    node_count: usize,
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
        };
        if let Ok(data) = backend.read("graph.bin") {
            std::fs::create_dir_all(cache_dir)?;
            std::fs::write(&manager.local_cache_path, data)?;
            let file = File::open(&manager.local_cache_path)?;
            let len = file.metadata()?.len();
            manager.mmap = Some(unsafe { Mmap::map(&file)? });
            manager.node_count = (len as usize) / BLOCK_SIZE;
        }
        Ok(manager)
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }

    pub fn has_index(&self) -> bool {
        self.node_count > 0 && self.mmap.is_some()
    }

    pub fn get_neighbors(&self, node_id: u32) -> Result<Vec<u32>, String> {
        let mmap = self.mmap.as_ref().ok_or("Graph not loaded")?;
        let offset = (node_id as usize) * BLOCK_SIZE;
        if offset + BLOCK_SIZE > mmap.len() {
            return Err(format!("Node ID {} out of range", node_id));
        }
        let record = NodeRecord::decode(&mmap[offset..offset + BLOCK_SIZE]);
        Ok(record.neighbors)
    }

    pub fn search(
        &self,
        entry_node: u32,
        k: usize,
        search_list_size: usize,
        scorer: impl Fn(u32) -> f64,
    ) -> Result<Vec<(u32, f64)>, Box<dyn std::error::Error + Send + Sync>> {
        if self.node_count == 0 {
            return Ok(Vec::new());
        }
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        let initial = SearchCandidate {
            id: entry_node,
            score: scorer(entry_node),
        };
        candidates.push(initial);
        results.push(OrderingWrapper(initial));
        visited.insert(entry_node);

        while let Some(current) = candidates.pop() {
            if results.len() >= search_list_size && current.score < results.peek().unwrap().0.score {
                break;
            }
            for nb in self.get_neighbors(current.id).unwrap_or_default() {
                if visited.contains(&nb) {
                    continue;
                }
                visited.insert(nb);
                let cand = SearchCandidate {
                    id: nb,
                    score: scorer(nb),
                };
                if results.len() < search_list_size || cand.score >= results.peek().unwrap().0.score {
                    candidates.push(cand);
                    results.push(OrderingWrapper(cand));
                    if results.len() > search_list_size {
                        results.pop();
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

    pub fn build(
        &mut self,
        n: usize,
        max_degree: usize,
        _alpha: f64,
        build_scorer: impl Fn(u32, &[u32]) -> Vec<(u32, f64)> + Sync,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.mmap = None;

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.local_cache_path)?;
        file.set_len((n * BLOCK_SIZE) as u64)?;

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
                    .wrapping_add(1442695040888963407)
                    % (n.max(1) as u64);
                while candidate_ids.len() < candidate_cap {
                    x = x
                        .wrapping_mul(2862933555777941757)
                        .wrapping_add(3037000493)
                        % (n.max(1) as u64);
                    let cand = x as usize;
                    if cand == i {
                        continue;
                    }
                    let cand_u32 = cand as u32;
                    if seen.insert(cand_u32) {
                        candidate_ids.push(cand_u32);
                    }
                }
                candidate_ids
            })
            .collect();

        let adjacency: Vec<Vec<u32>> = candidate_lists
            .into_iter()
            .enumerate()
            .map(|(i, candidate_ids)| {
                let mut scored: Vec<(u32, f64)> = build_scorer(i as u32, &candidate_ids);
                scored.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(Ordering::Equal)
                        .then_with(|| a.0.cmp(&b.0))
                });

                scored
                    .into_iter()
                    .take(degree_cap.min(n.saturating_sub(1)))
                    .map(|(id, _)| id)
                    .collect::<Vec<u32>>()
            })
            .collect();

        let mut mmap_mut = unsafe { memmap2::MmapMut::map_mut(&file)? };
        for (i, neighbors) in adjacency.into_iter().enumerate() {
            let record = NodeRecord { neighbors };
            mmap_mut[i * BLOCK_SIZE..(i + 1) * BLOCK_SIZE].copy_from_slice(&record.encode());
        }

        mmap_mut.flush()?;
        drop(mmap_mut);
        file.flush()?;

        self.node_count = n;
        let data = std::fs::read(&self.local_cache_path)?;
        self.backend.write("graph.bin", data)?;
        let file_ro = File::open(&self.local_cache_path)?;
        self.mmap = Some(unsafe { Mmap::map(&file_ro)? });
        Ok(())
    }
}


