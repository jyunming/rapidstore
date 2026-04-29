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
    ///
    /// `nprobe` is clamped to `[1, self.k]` — passing 0 still probes the single
    /// best cluster (treated as "minimum useful work"); passing > k probes all
    /// clusters. If the index has zero clusters (degenerate `build()` call),
    /// returns an empty vec.
    pub fn probe(&self, y_query: &[f32], nprobe: usize) -> Vec<u32> {
        debug_assert_eq!(y_query.len(), self.n);
        if self.k == 0 {
            return Vec::new();
        }
        // A4 (v0.8.2 audit): clamp nprobe to >=1 so we always probe at least
        // one cluster. nprobe=0 previously called select_nth_unstable_by(0, ...)
        // which is defined but useless (returns the smallest score); engine
        // callers depend on getting at least one cluster's worth of candidates.
        let nprobe = nprobe.max(1).min(self.k);

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
        let tmp = path.with_file_name("ivf.bin.tmp");
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Build a simple `codes_bytes` array with `n_slots` records, each carrying
    /// the same all-zero MSE codes. Tests that need varying codes should write
    /// directly into the returned buffer.
    fn make_codes(n_slots: usize, mse_len: usize) -> Vec<u8> {
        vec![0u8; n_slots * mse_len]
    }

    /// Centroids of size `2^bits` mapping index → centroid value.
    fn linear_centroids(bits: usize) -> Vec<f32> {
        let n = 1 << bits;
        (0..n).map(|i| i as f32 - (n as f32 / 2.0)).collect()
    }

    /// Tiny build-able fixture: 4-bit MSE, 8 dims, 4 byte stride per slot.
    fn fixture_params() -> (usize, usize, usize, usize, Vec<f32>) {
        let bits = 4;
        let n = 8;
        let mse_len = (n * bits + 7) / 8; // 4 bytes for n=8 b=4
        let stride = mse_len;
        (bits, n, mse_len, stride, linear_centroids(bits))
    }

    /// A4: zero-cluster IVF returns empty probe results without panicking.
    #[test]
    fn probe_on_zero_cluster_index_returns_empty() {
        let ivf = IvfIndex {
            centroids: Vec::new(),
            n: 8,
            k: 0,
            cluster_map: Vec::new(),
            cluster_index: Vec::new(),
        };
        let q = vec![0.0f32; 8];
        let probed = ivf.probe(&q, 4);
        assert!(probed.is_empty(), "k=0 IVF must return empty probe");
    }

    /// A4: probe with nprobe=0 still probes at least one cluster (clamped to 1).
    #[test]
    fn probe_with_zero_nprobe_clamps_to_one_cluster() {
        let (bits, n, mse_len, stride, centroids) = fixture_params();
        let n_slots = 16;
        let codes = make_codes(n_slots, mse_len);
        let active: Vec<u32> = (0..n_slots as u32).collect();
        let ivf = IvfIndex::build(
            &codes, &active, stride, mse_len, n, bits, &centroids, 4, 5, 42,
        );
        let q = vec![1.0f32; n];
        let probed = ivf.probe(&q, 0);
        assert!(
            !probed.is_empty(),
            "nprobe=0 should clamp to 1 and return at least one cluster's slots"
        );
    }

    /// C1: build with the same code for every slot puts everything in one cluster.
    #[test]
    fn build_assigns_all_slots_to_some_cluster() {
        let (bits, n, mse_len, stride, centroids) = fixture_params();
        let n_slots = 32;
        let codes = make_codes(n_slots, mse_len);
        let active: Vec<u32> = (0..n_slots as u32).collect();
        let ivf = IvfIndex::build(
            &codes, &active, stride, mse_len, n, bits, &centroids, 4, 10, 42,
        );

        // Every slot should be assigned to some cluster (no slot left unassigned).
        assert_eq!(ivf.cluster_map.len(), n_slots);
        for &cid in &ivf.cluster_map {
            assert!(
                (cid as usize) < ivf.k,
                "cluster id {cid} out of bounds (k={})",
                ivf.k
            );
        }

        // Every assigned slot appears in exactly one cluster's slot list.
        let total: usize = ivf.cluster_index.iter().map(|c| c.len()).sum();
        assert_eq!(
            total, n_slots,
            "every slot must appear in exactly one cluster"
        );
    }

    /// C1: build with k > n_slots silently clamps k to n_slots.
    #[test]
    fn build_clamps_k_to_n_slots() {
        let (bits, n, mse_len, stride, centroids) = fixture_params();
        let n_slots = 5;
        let codes = make_codes(n_slots, mse_len);
        let active: Vec<u32> = (0..n_slots as u32).collect();
        let ivf = IvfIndex::build(
            &codes, &active, stride, mse_len, n, bits, &centroids, 100, 5, 42,
        );
        // k clamps to n_slots when k > n_slots.
        assert!(
            ivf.k <= n_slots,
            "k must be clamped to n_slots; got k={}",
            ivf.k
        );
    }

    /// C1: empty active_slots returns a degenerate but non-panicking IVF.
    #[test]
    fn build_with_no_active_slots_returns_empty_clusters() {
        let (bits, n, mse_len, stride, centroids) = fixture_params();
        let codes: Vec<u8> = Vec::new();
        let active: Vec<u32> = Vec::new();
        let ivf = IvfIndex::build(
            &codes, &active, stride, mse_len, n, bits, &centroids, 4, 5, 42,
        );
        assert!(ivf.cluster_map.is_empty());
        assert_eq!(ivf.k, 4);
    }

    /// C1: probe results are unique and sorted.
    #[test]
    fn probe_returns_unique_sorted_slots() {
        let (bits, n, mse_len, stride, centroids) = fixture_params();
        let n_slots = 32;
        // Vary codes per slot so clusters actually separate.
        let mut codes = make_codes(n_slots, mse_len);
        for slot in 0..n_slots {
            codes[slot * mse_len] = (slot % 16) as u8;
        }
        let active: Vec<u32> = (0..n_slots as u32).collect();
        let ivf = IvfIndex::build(
            &codes, &active, stride, mse_len, n, bits, &centroids, 4, 10, 42,
        );
        let q = vec![1.0f32; n];
        let probed = ivf.probe(&q, 2);

        let mut sorted = probed.clone();
        sorted.sort_unstable();
        assert_eq!(probed, sorted, "probe results should be sorted ascending");

        let mut deduped = probed.clone();
        deduped.dedup();
        assert_eq!(probed.len(), deduped.len(), "probe results must be unique");
    }

    /// C1: probe with nprobe == k returns all slots in some order.
    #[test]
    fn probe_nprobe_at_k_returns_all_slots() {
        let (bits, n, mse_len, stride, centroids) = fixture_params();
        let n_slots = 16;
        let codes = make_codes(n_slots, mse_len);
        let active: Vec<u32> = (0..n_slots as u32).collect();
        let ivf = IvfIndex::build(
            &codes, &active, stride, mse_len, n, bits, &centroids, 4, 10, 42,
        );
        let q = vec![1.0f32; n];
        let probed = ivf.probe(&q, ivf.k);
        assert_eq!(probed.len(), n_slots, "nprobe=k must return every slot");
    }

    /// C1: save/load round-trip preserves the index.
    #[test]
    fn save_load_roundtrip_preserves_index() {
        let (bits, n, mse_len, stride, centroids) = fixture_params();
        let n_slots = 16;
        let codes = make_codes(n_slots, mse_len);
        let active: Vec<u32> = (0..n_slots as u32).collect();
        let ivf = IvfIndex::build(
            &codes, &active, stride, mse_len, n, bits, &centroids, 4, 5, 42,
        );
        let dir = tempdir().unwrap();
        let path = dir.path().join("ivf.bin");
        ivf.save(&path).unwrap();
        let loaded = IvfIndex::load(&path).unwrap();
        assert_eq!(loaded.k, ivf.k);
        assert_eq!(loaded.n, ivf.n);
        assert_eq!(loaded.cluster_map, ivf.cluster_map);
        assert_eq!(loaded.centroids, ivf.centroids);
    }

    /// C1: loading a file without IVF magic returns an error.
    #[test]
    fn load_rejects_bad_magic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("ivf.bin");
        std::fs::write(&path, b"NOTIVF__").unwrap();
        let result = IvfIndex::load(&path);
        assert!(result.is_err(), "bad magic should be rejected");
    }
}
