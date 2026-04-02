use nalgebra::DMatrix;
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::CodeIndex;
use super::codebook::lloyd_max;
use crate::linalg::hadamard::{srht, inverse_srht};

/// MSE (Mean Squared Error) scalar quantizer using SRHT rotation + Lloyd-Max codebook.
///
/// ## Algorithm
///
/// 1. **SRHT rotation**: apply `H * D * (x, 0)` where `H` is the Walsh-Hadamard matrix
///    and `D` is a seeded random diagonal ±1 matrix. This decorrelates the input and
///    makes the per-dimension variance uniform — a prerequisite for scalar quantization.
///
/// 2. **Lloyd-Max encoding**: each rotated dimension is quantized independently using
///    a precomputed optimal scalar codebook (1D centroids). For `b` bits of MSE budget,
///    each dimension gets `b-1` bits → `2^(b-1)` centroids.
///
/// 3. **Dequantization**: reverse by looking up centroids + inverse SRHT to recover
///    an approximation of the original vector in the original space.
///
/// The codebook is **data-oblivious**: generated once from a Gaussian prior using the
/// `lloyd_max` function, seeded deterministically. No training data required.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MseQuantizer {
    pub d: usize,
    pub n: usize,
    pub b: usize,
    /// SRHT diagonal sign vector (length n)
    pub rotation_signs: Vec<f32>,
    pub centroids: Vec<f32>,
}

impl MseQuantizer {
    pub fn new(d: usize, b: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = d.next_power_of_two();

        let mut rotation_signs = vec![0.0f32; n];
        for s in &mut rotation_signs {
            *s = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
        }

        // Variance after SRHT is roughly 1/n if we pad with zeros.
        let centroids: Vec<f32> = lloyd_max(b, n, 20_000)
            .into_iter()
            .map(|c| c as f32)
            .collect();
        assert!(
            centroids.len() <= u16::MAX as usize + 1,
            "codebook too large for u16 indices; reduce bits"
        );

        Self { d, n, b, rotation_signs, centroids }
    }

    /// Forward rotation: y = H * D * (x, 0)
    pub fn apply_rotation(&self, x: &[f32], out: &mut [f32]) {
        srht(x, &self.rotation_signs, out);
    }

    /// Inverse rotation: (x, 0) = D * H * y
    pub fn apply_rotation_transpose(&self, y: &[f32], out: &mut [f32]) {
        inverse_srht(y, &self.rotation_signs, out);
    }

    pub fn quantize(&self, x: &[f32]) -> Vec<CodeIndex> {
        assert_eq!(x.len(), self.d);
        let mut y = vec![0.0f32; self.n];
        self.apply_rotation(x, &mut y);

        let mut indices = vec![0 as CodeIndex; self.n];
        for i in 0..self.n {
            indices[i] = self.nearest_centroid_index(y[i]);
        }
        indices
    }

    pub fn quantize_batch(&self, xs: &DMatrix<f32>) -> Vec<Vec<CodeIndex>> {
        assert_eq!(xs.nrows(), self.d);
        let num_vecs = xs.ncols();
        if num_vecs == 0 {
            return Vec::new();
        }

        let mut all_indices = vec![vec![0 as CodeIndex; self.n]; num_vecs];
        all_indices.par_iter_mut().enumerate().for_each(|(col, idxs)| {
            let x_col = xs.column(col);
            let mut y = vec![0.0f32; self.n];
            self.apply_rotation(x_col.as_slice(), &mut y);
            for row in 0..self.n {
                idxs[row] = self.nearest_centroid_index(y[row]);
            }
        });
        all_indices
    }

    fn nearest_centroid_index(&self, val: f32) -> CodeIndex {
        let n = self.centroids.len();
        if n == 0 {
            return 0;
        }
        let pos = self.centroids.partition_point(|&c| c < val);
        if pos == 0 {
            0
        } else if pos >= n {
            (n - 1) as CodeIndex
        } else {
            let lo = pos - 1;
            let hi = pos;
            if (val - self.centroids[lo]).abs() <= (self.centroids[hi] - val).abs() {
                lo as CodeIndex
            } else {
                hi as CodeIndex
            }
        }
    }

    pub fn dequantize(&self, indices: &[CodeIndex]) -> Array1<f64> {
        assert_eq!(indices.len(), self.n);
        let mut y_tilde = vec![0.0f32; self.n];
        for i in 0..self.n {
            y_tilde[i] = self.centroids[indices[i] as usize];
        }
        let mut out_data = vec![0.0f32; self.d];
        self.apply_rotation_transpose(&y_tilde, &mut out_data);
        let mut out = Array1::zeros(self.d);
        for i in 0..self.d {
            out[i] = out_data[i] as f64;
        }
        out
    }

    pub fn dequantize_batch(&self, indices_batch: &[Vec<CodeIndex>]) -> DMatrix<f32> {
        let num_vecs = indices_batch.len();
        if num_vecs == 0 {
            return DMatrix::zeros(self.d, 0);
        }
        let mut out = DMatrix::zeros(self.d, num_vecs);
        let n = self.n;
        let signs = &self.rotation_signs;
        let centroids = &self.centroids;
        
        for (col, indices) in indices_batch.iter().enumerate() {
            let mut y_tilde = vec![0.0f32; n];
            for row in 0..n {
                y_tilde[row] = centroids[indices[row] as usize];
            }
            let mut x_rec = vec![0.0f32; self.d];
            inverse_srht(&y_tilde, signs, &mut x_rec);
            for row in 0..self.d {
                out[(row, col)] = x_rec[row];
            }
        }
        out
    }
}
