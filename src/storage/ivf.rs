/// IVF (Inverted File Index) coarse routing for large-scale search.
///
/// Partitions the corpus into k clusters. At query time, only the top-nprobe
/// clusters are scored — reducing candidate count from N to ~(nprobe/k)×N.
///
/// Centroids live in the n-dimensional SRHT-rotated space shared with the MSE
/// quantizer, so no separate rotation is needed at query time.
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

const IVF_MAGIC: &[u8; 4] = b"TIVF";

#[derive(Serialize, Deserialize)]
pub struct IvfIndex {
    /// k × n centroid matrix, row-major (centroid c starts at centroids[c * n]).
    pub centroids: Vec<f32>,
    /// n = SRHT-rotated dimension (== quantizer.n).
    pub n: usize,
    /// k = number of clusters.
    pub k: usize,
    /// cluster_map[slot] = cluster_id.  u32::MAX = unassigned.
    pub cluster_map: Vec<u32>,
    /// cluster_index[cluster_id] = sorted Vec<u32> of slot IDs.
    pub cluster_index: Vec<Vec<u32>>,
}

impl IvfIndex {
    /// Build an IVF index from existing quantized codes.
    ///
    /// `codes_bytes`: raw bytes of `live_codes.bin`.
    /// `active_slots`: the slots to index.
    /// `stride`, `mse_len`: per-slot layout (same as engine).
    /// `mse_centroids`: the Lloyd-Max codebook (centroid values, length n).
    pub fn build(
        codes_bytes: &[u8],
        active_slots: &[u32],
        stride: usize,
        mse_len: usize,
        n: usize,
        bits_per_idx: usize,
        mse_centroids: &[f32],
        k: usize,
        max_iter: usize,
        rng_seed: u64,
    ) -> Self {
        let n_slots = active_slots.len();
        if n_slots == 0 || k == 0 {
            return Self {
                centroids: vec![0.0; k * n],
                n,
                k,
                cluster_map: Vec::new(),
                cluster_index: vec![Vec::new(); k],
            };
        }
        let k = k.min(n_slots);

        // Helper: dequantize a code record into n float values (SRHT-rotated space).
        let dequant = |slot: u32| -> Vec<f32> {
            let rec = &codes_bytes[slot as usize * stride..(slot as usize + 1) * stride];
            let mut out = vec![0.0f32; n];
            unpack_and_lookup(rec, mse_len, bits_per_idx, n, mse_centroids, &mut out);
            out
        };

        // Seed-deterministic initial centroid selection (reservoir sample k slots).
        let mut seed_state = rng_seed;
        let mut initial_indices: Vec<usize> = (0..k).collect();
        for i in k..n_slots {
            seed_state = lcg_next(seed_state);
            let j = (seed_state as usize) % (i + 1);
            if j < k {
                initial_indices[j] = i;
            }
        }

        // Build initial centroids.
        let mut centroids: Vec<Vec<f32>> = initial_indices
            .iter()
            .map(|&i| dequant(active_slots[i]))
            .collect();

        // k-means iterations (parallel cluster assignment).
        let mut assignments = vec![0u32; n_slots];
        for _iter in 0..max_iter {
            // Assignment step: each slot → nearest centroid (max IP).
            let new_assignments: Vec<u32> = active_slots
                .par_iter()
                .map(|&slot| {
                    let v = dequant(slot);
                    nearest_centroid_ip(&v, &centroids)
                })
                .collect();

            let changed = new_assignments
                .iter()
                .zip(assignments.iter())
                .any(|(a, b)| a != b);
            assignments = new_assignments;
            if !changed {
                break;
            }

            // Update step: compute new centroid = mean of assigned vectors.
            let mut sums = vec![vec![0.0f32; n]; k];
            let mut counts = vec![0usize; k];
            for (i, &slot) in active_slots.iter().enumerate() {
                let c = assignments[i] as usize;
                let v = dequant(slot);
                for (d, &val) in v.iter().enumerate() {
                    sums[c][d] += val;
                }
                counts[c] += 1;
            }
            for c in 0..k {
                if counts[c] > 0 {
                    let inv = 1.0 / counts[c] as f32;
                    for d in 0..n {
                        centroids[c][d] = sums[c][d] * inv;
                    }
                }
            }
        }

        // Build cluster_map (indexed by slot) and cluster_index.
        let max_slot = active_slots.iter().copied().max().unwrap_or(0) as usize;
        let mut cluster_map = vec![u32::MAX; max_slot + 1];
        let mut cluster_index: Vec<Vec<u32>> = vec![Vec::new(); k];
        for (i, &slot) in active_slots.iter().enumerate() {
            let c = assignments[i] as usize;
            cluster_map[slot as usize] = c as u32;
            cluster_index[c].push(slot);
        }
        for slots in &mut cluster_index {
            slots.sort_unstable();
        }

        // Flatten centroids to row-major Vec<f32>.
        let flat_centroids: Vec<f32> = centroids.into_iter().flatten().collect();

        Self {
            centroids: flat_centroids,
            n,
            k,
            cluster_map,
            cluster_index,
        }
    }

    /// Return sorted candidate slots for a query using nprobe clusters.
    ///
    /// `y_query`: the query already rotated into SRHT space (n-dimensional).
    pub fn probe(&self, y_query: &[f32], nprobe: usize) -> Vec<u32> {
        debug_assert_eq!(y_query.len(), self.n);
        let nprobe = nprobe.min(self.k);

        // Score query against all k centroids.
        let mut centroid_scores: Vec<(u32, f32)> = (0..self.k as u32)
            .map(|c| {
                let row = &self.centroids[c as usize * self.n..(c as usize + 1) * self.n];
                let score: f32 = y_query.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
                (c, score)
            })
            .collect();

        // Partial sort: top-nprobe by score (descending).
        if nprobe < self.k {
            centroid_scores.select_nth_unstable_by(nprobe, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            centroid_scores.truncate(nprobe);
        }

        // Union slot lists from top-nprobe clusters.
        let mut candidates: Vec<u32> = centroid_scores
            .iter()
            .flat_map(|&(c, _)| self.cluster_index[c as usize].iter().copied())
            .collect();
        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }

    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let payload = bincode::serialize(self)?;
        let tmp = path.with_extension("ivf.tmp");
        let mut out = Vec::with_capacity(payload.len() + 4);
        out.extend_from_slice(IVF_MAGIC);
        out.extend_from_slice(&payload);
        std::fs::write(&tmp, &out)?;
        #[cfg(target_os = "windows")]
        let _ = std::fs::remove_file(path);
        std::fs::rename(tmp, path)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let bytes = std::fs::read(path)?;
        if bytes.len() < 4 || &bytes[..4] != IVF_MAGIC {
            return Err("invalid IVF file magic".into());
        }
        Ok(bincode::deserialize(&bytes[4..])?)
    }
}

/// Find the centroid with the highest IP score against `v`.
fn nearest_centroid_ip(v: &[f32], centroids: &[Vec<f32>]) -> u32 {
    let mut best_c = 0u32;
    let mut best_score = f32::NEG_INFINITY;
    for (c, cent) in centroids.iter().enumerate() {
        let score: f32 = v.iter().zip(cent.iter()).map(|(a, b)| a * b).sum();
        if score > best_score {
            best_score = score;
            best_c = c as u32;
        }
    }
    best_c
}

/// Unpack MSE indices from a code record and look up centroid values.
fn unpack_and_lookup(
    rec: &[u8],
    mse_len: usize,
    bits_per_idx: usize,
    n: usize,
    mse_centroids: &[f32],
    out: &mut Vec<f32>,
) {
    let lut_width = 1usize << bits_per_idx;
    let packed = &rec[..mse_len];
    match bits_per_idx {
        4 => {
            for (i, byte) in packed.iter().enumerate() {
                let lo = (byte & 0x0F) as usize;
                let hi = ((byte >> 4) & 0x0F) as usize;
                let base = i * 2;
                if base < n {
                    out[base] = mse_centroids[lo];
                }
                if base + 1 < n {
                    out[base + 1] = mse_centroids[hi];
                }
            }
        }
        8 => {
            for (i, &byte) in packed.iter().enumerate().take(n) {
                out[i] = mse_centroids[byte as usize];
            }
        }
        b => {
            // General bit-unpacking for other widths (1, 2, 3, 5, 6, 7).
            let mut bit_pos = 0usize;
            for i in 0..n {
                let byte_pos = bit_pos / 8;
                let bit_off = bit_pos % 8;
                let idx = if bit_off + b <= 8 {
                    let mask = (1u16 << b) - 1;
                    ((packed.get(byte_pos).copied().unwrap_or(0) as u16 >> bit_off) & mask) as usize
                } else {
                    let lo = packed.get(byte_pos).copied().unwrap_or(0) as u16;
                    let hi = packed.get(byte_pos + 1).copied().unwrap_or(0) as u16;
                    let mask = (1u16 << b) - 1;
                    ((lo | (hi << 8)) >> bit_off & mask) as usize
                };
                out[i] = mse_centroids[idx.min(lut_width - 1)];
                bit_pos += b;
            }
        }
    }
}

/// Simple LCG for deterministic pseudo-random selection.
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}
