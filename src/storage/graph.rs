use memmap2::Mmap;
use rand::Rng;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::fs::File;
use std::sync::Arc;

use super::backend::StorageBackend;

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

    pub fn get_neighbors_at_level(&self, node_id: u32, level: u32) -> Result<Vec<u32>, String> {
        let mmap = self.mmap.as_ref().ok_or("Graph not loaded")?;
        let i = node_id as usize;
        if i >= self.node_count { return Err("OOR".into()); }
        
        let start = self.data_start + self.offsets[i] as usize;
        let end = self.data_start + self.offsets[i + 1] as usize;
        let rec = &mmap[start..end];
        
        let num_levels = u32::from_le_bytes(rec[0..4].try_into().unwrap());
        if level >= num_levels { return Ok(Vec::new()); }
        
        let mut curr = 4;
        for l in 0..num_levels {
            let count = u32::from_le_bytes(rec[curr..curr+4].try_into().unwrap()) as usize;
            curr += 4;
            if l == level {
                let mut nbs = Vec::with_capacity(count);
                let mut last_nb = 0u32;
                for _ in 0..count {
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
                let (_, bytes_read) = decode_varint(&rec[curr..]);
                curr += bytes_read;
            }
        }
        Ok(Vec::new())
    }

    pub fn search(
        &self,
        _entry_node_unused: u32,
        k: usize,
        search_list_size: usize,
        scorer: impl Fn(u32) -> f64,
        filter: Option<impl Fn(u32) -> bool>,
    ) -> Result<Vec<(u32, f64)>, Box<dyn std::error::Error + Send + Sync>> {
        if self.node_count == 0 { return Ok(Vec::new()); }
        
        const EF_UPPER: usize = 32;
        let mut beam = BinaryHeap::new();
        let start_score = scorer(self.entry_point);
        beam.push(OrderingWrapper(SearchCandidate { id: self.entry_point, score: start_score }));
        
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
                if layer_results.len() >= EF_UPPER && current.score < layer_results.peek().unwrap().0.score {
                    break;
                }
                
                if let Ok(nbs) = self.get_neighbors_at_level(current.id, l) {
                    for nb in nbs {
                        if visited.contains(&nb) { continue; }
                        visited.insert(nb);
                        
                        let score = scorer(nb);
                        let cand = SearchCandidate { id: nb, score };
                        if layer_results.len() < EF_UPPER || score > layer_results.peek().unwrap().0.score {
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
            let matches = if let Some(f) = &filter { f(seed.id) } else { true };
            if matches {
                results.push(OrderingWrapper(seed));
            }
        }

        while let Some(current) = candidates.pop() {
            if results.len() >= search_list_size && current.score < results.peek().unwrap().0.score {
                break;
            }

            if let Ok(nbs) = self.get_neighbors_at_level(current.id, 0) {
                for nb in nbs {
                    if visited.contains(&nb) { continue; }
                    visited.insert(nb);
                    
                    let score = scorer(nb);
                    let cand = SearchCandidate { id: nb, score };
                    candidates.push(cand);

                    let matches = if let Some(f) = &filter { f(nb) } else { true };
                    if matches {
                        if results.len() < search_list_size || cand.score > results.peek().unwrap().0.score {
                            results.push(OrderingWrapper(cand));
                            if results.len() > search_list_size { results.pop(); }
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
            let level_nodes: Vec<u32> = (0..n).filter(|&i| node_levels[i] >= l).map(|i| i as u32).collect();
            if level_nodes.len() < 2 { continue; }
            
            let ln = level_nodes.len();
            let degree_cap = if l == 0 { m0 } else { m };
            
            // Build adjacency for this level subset
            let level_adj: Vec<Vec<u32>> = level_nodes.par_iter().map(|&i| {
                let mut local_rng = rand::thread_rng();
                let mut candidates = HashSet::new();
                while candidates.len() < cand_pool && candidates.len() < ln - 1 {
                    let cand = level_nodes[local_rng.gen_range(0..ln)];
                    if cand != i { candidates.insert(cand); }
                }
                
                let cand_vec: Vec<u32> = candidates.into_iter().collect();
                let mut scored = build_scorer(i, &cand_vec);
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                let mut chosen: Vec<u32> = scored.into_iter().take(degree_cap).map(|(id, _)| id).collect();
                chosen.sort_unstable();
                chosen
            }).collect();
            
            for (idx, &i) in level_nodes.iter().enumerate() {
                adjacency[i as usize][l as usize] = level_adj[idx].clone();
            }
            
            // Multiple refinement iterations (nn-descent style)
            for _ in 0..n_refinements {
                let next_level_adj: Vec<Vec<u32>> = level_nodes.par_iter().map(|&i| {
                    let mut candidates = HashSet::new();
                    for &nb in &adjacency[i as usize][l as usize] {
                        candidates.insert(nb);
                        for &nbnb in &adjacency[nb as usize][l as usize] {
                            if nbnb != i { candidates.insert(nbnb); }
                        }
                    }
                    let cand_vec: Vec<u32> = candidates.into_iter().collect();
                    let mut scored = build_scorer(i, &cand_vec);
                    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                    let mut chosen: Vec<u32> = scored.into_iter().take(degree_cap).map(|(id, _)| id).collect();
                    chosen.sort_unstable();
                    chosen
                }).collect();
                
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
