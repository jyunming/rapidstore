use nalgebra::DMatrix;
use ndarray::Array1;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::CodeIndex;
use super::codebook::lloyd_max;
use crate::linalg::hadamard::{inverse_srht, srht};

/// Generate a d×d Haar-random orthogonal matrix via modified Gram-Schmidt on i.i.d. N(0,1) columns.
/// Returns a row-major `Vec<f32>` of shape d×d.
fn random_orthogonal(d: usize, rng: &mut impl Rng) -> Vec<f32> {
    // Sample d columns, each of length d, from N(0,1).
    let mut cols: Vec<Vec<f32>> = (0..d)
        .map(|_| (0..d).map(|_| rng.sample(StandardNormal)).collect())
        .collect();
    // Modified Gram-Schmidt orthogonalization.
    for i in 0..d {
        let norm: f32 = cols[i].iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut cols[i] {
                *x /= norm;
            }
        }
        for j in (i + 1)..d {
            let dot: f32 = cols[i].iter().zip(cols[j].iter()).map(|(a, b)| a * b).sum();
            for k in 0..d {
                cols[j][k] -= dot * cols[i][k];
            }
        }
    }
    // Transpose to row-major so mat[row*d + col] = Q[row][col].
    let mut mat = vec![0.0f32; d * d];
    for row in 0..d {
        for col in 0..d {
            mat[row * d + col] = cols[col][row];
        }
    }
    mat
}

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
///    each dimension gets `b` bits → `2^b` centroids.
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
    /// SRHT diagonal sign vector (length n). Empty when `rotation_matrix` is `Some`.
    pub rotation_signs: Vec<f32>,
    pub centroids: Vec<f32>,
    /// Exact-paper mode: d×d Haar-random orthogonal matrix, row-major.
    /// `None` → use SRHT (default). `Some` → dense matrix-vector multiply.
    #[serde(default)]
    pub rotation_matrix: Option<Vec<f32>>,
}

impl MseQuantizer {
    /// Create a new `MseQuantizer` for `d`-dimensional vectors with `b`-bit codebook and the
    /// given `seed`. Generates the SRHT diagonal signs and Lloyd-Max centroids deterministically.
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

        Self {
            d,
            n,
            b,
            rotation_signs,
            centroids,
            rotation_matrix: None,
        }
    }

    /// Dense mode: Haar-uniform QR rotation (Haar measure on O(d)).
    /// Uses a full d×d matrix instead of SRHT. O(d²) apply time, O(d²) storage.
    pub fn new_dense(d: usize, b: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let rotation_matrix = Some(random_orthogonal(d, &mut rng));
        let centroids: Vec<f32> = lloyd_max(b, d, 20_000)
            .into_iter()
            .map(|c| c as f32)
            .collect();
        assert!(
            centroids.len() <= u16::MAX as usize + 1,
            "codebook too large for u16 indices; reduce bits"
        );
        Self {
            d,
            n: d, // no zero-padding: matrix is exactly d×d
            b,
            rotation_signs: vec![],
            centroids,
            rotation_matrix,
        }
    }

    /// Forward rotation: Q·x (exact mode) or H·D·(x,0) (SRHT mode).
    pub fn apply_rotation(&self, x: &[f32], out: &mut [f32]) {
        if let Some(mat) = &self.rotation_matrix {
            for row in 0..self.d {
                out[row] = mat[row * self.d..(row + 1) * self.d]
                    .iter()
                    .zip(x.iter())
                    .map(|(a, b)| a * b)
                    .sum();
            }
        } else {
            srht(x, &self.rotation_signs, out);
        }
    }

    /// Inverse rotation: Q^T·y (exact mode, Q orthogonal so Q^T = Q^{-1}) or inverse SRHT.
    pub fn apply_rotation_transpose(&self, y: &[f32], out: &mut [f32]) {
        if let Some(mat) = &self.rotation_matrix {
            for col in 0..self.d {
                out[col] = (0..self.d)
                    .map(|row| mat[row * self.d + col] * y[row])
                    .sum();
            }
        } else {
            inverse_srht(y, &self.rotation_signs, out);
        }
    }

    /// Quantize a single `d`-dimensional vector: apply SRHT rotation, then map each
    /// rotated dimension to the nearest codebook centroid. Returns `n` centroid indices.
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

    /// Batch version of [`quantize`](Self::quantize): each column of `xs` is an independent
    /// `d`-dimensional input vector. Rows are quantized in parallel via Rayon.
    pub fn quantize_batch(&self, xs: &DMatrix<f32>) -> Vec<Vec<CodeIndex>> {
        assert_eq!(xs.nrows(), self.d);
        let num_vecs = xs.ncols();
        if num_vecs == 0 {
            return Vec::new();
        }

        let mut all_indices = vec![vec![0 as CodeIndex; self.n]; num_vecs];
        all_indices
            .par_iter_mut()
            .enumerate()
            .for_each(|(col, idxs)| {
                let x_col = xs.column(col);
                let mut y = vec![0.0f32; self.n];
                self.apply_rotation(x_col.as_slice(), &mut y);
                for row in 0..self.n {
                    idxs[row] = self.nearest_centroid_index(y[row]);
                }
            });
        all_indices
    }

    pub(crate) fn nearest_centroid_index(&self, val: f32) -> CodeIndex {
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

    /// Reconstruct a `d`-dimensional vector from `n` centroid indices by looking up
    /// centroids and applying the inverse SRHT rotation.
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

    /// Batch dequantization: each element of `indices_batch` is a slice of `n` centroid
    /// indices. Returns a `d × num_vecs` matrix of reconstructed float32 vectors.
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
            if self.rotation_matrix.is_some() {
                self.apply_rotation_transpose(&y_tilde, &mut x_rec);
            } else {
                inverse_srht(&y_tilde, signs, &mut x_rec);
            }
            for row in 0..self.d {
                out[(row, col)] = x_rec[row];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    fn make_quantizer(d: usize) -> MseQuantizer {
        MseQuantizer::new(d, 4, 42)
    }

    #[test]
    fn new_creates_correct_dimensions() {
        let d = 128;
        let q = MseQuantizer::new(d, 4, 42);
        assert_eq!(q.d, d);
        assert_eq!(q.n, d.next_power_of_two());
        assert_eq!(q.b, 4);
        assert_eq!(q.rotation_signs.len(), q.n);
        // 2^b centroids for b=4 → 16 centroids
        assert_eq!(q.centroids.len(), 1usize << 4);
    }

    #[test]
    fn quantize_output_length_equals_n() {
        let d = 64;
        let q = make_quantizer(d);
        let x: Vec<f32> = (0..d).map(|i| i as f32 * 0.01).collect();
        let indices = q.quantize(&x);
        assert_eq!(indices.len(), q.n);
    }

    #[test]
    fn quantize_indices_are_in_bounds() {
        let d = 32;
        let q = make_quantizer(d);
        let x: Vec<f32> = vec![1.0f32; d];
        let indices = q.quantize(&x);
        let max_code = q.centroids.len() as u32;
        for &idx in &indices {
            assert!(
                (idx as u32) < max_code,
                "index {} out of bounds (max {})",
                idx,
                max_code
            );
        }
    }

    #[test]
    fn dequantize_output_length_equals_d() {
        let d = 64;
        let q = make_quantizer(d);
        let x: Vec<f32> = (0..d).map(|i| i as f32 * 0.01).collect();
        let indices = q.quantize(&x);
        let reconstructed = q.dequantize(&indices);
        assert_eq!(reconstructed.len(), d);
    }

    #[test]
    fn quantize_dequantize_reasonable_mse() {
        let d = 64;
        let q = make_quantizer(d);
        let x: Vec<f32> = (0..d).map(|i| i as f32 * 0.01 - 0.32).collect();
        let indices = q.quantize(&x);
        let reconstructed = q.dequantize(&indices);
        let mse: f64 = x
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| (a as f64 - b).powi(2))
            .sum::<f64>()
            / d as f64;
        assert!(mse < 1.0, "MSE should be reasonable, got: {}", mse);
    }

    #[test]
    fn quantize_batch_matches_single_quantize() {
        let d = 32;
        let q = make_quantizer(d);
        let x1: Vec<f32> = (0..d).map(|i| i as f32 * 0.01).collect();
        let x2: Vec<f32> = (0..d).map(|i| -(i as f32) * 0.01).collect();

        let single1 = q.quantize(&x1);
        let single2 = q.quantize(&x2);

        let mut mat = DMatrix::zeros(d, 2);
        for (i, &v) in x1.iter().enumerate() {
            mat[(i, 0)] = v;
        }
        for (i, &v) in x2.iter().enumerate() {
            mat[(i, 1)] = v;
        }
        let batch = q.quantize_batch(&mat);

        assert_eq!(batch[0], single1);
        assert_eq!(batch[1], single2);
    }

    #[test]
    fn quantize_batch_empty_returns_empty() {
        let d = 32;
        let q = make_quantizer(d);
        let mat = DMatrix::<f32>::zeros(d, 0);
        let batch = q.quantize_batch(&mat);
        assert!(batch.is_empty());
    }

    #[test]
    fn dequantize_batch_matches_single_dequantize() {
        let d = 32;
        let q = make_quantizer(d);
        let x: Vec<f32> = (0..d).map(|i| i as f32 * 0.02 - 0.32).collect();
        let indices = q.quantize(&x);
        let single = q.dequantize(&indices);
        let batch = q.dequantize_batch(&[indices]);
        assert_eq!(batch.ncols(), 1);
        for i in 0..d {
            let diff = (single[i] - batch[(i, 0)] as f64).abs();
            assert!(
                diff < 1e-5,
                "dequantize_batch mismatch at dim {}: {} vs {}",
                i,
                single[i],
                batch[(i, 0)]
            );
        }
    }

    #[test]
    fn dequantize_batch_empty_returns_zero_cols() {
        let d = 32;
        let q = make_quantizer(d);
        let result = q.dequantize_batch(&[]);
        assert_eq!(result.ncols(), 0);
        assert_eq!(result.nrows(), d);
    }

    #[test]
    fn apply_rotation_produces_nonzero_output() {
        let d = 64;
        let q = make_quantizer(d);
        let x: Vec<f32> = vec![1.0f32; d];
        let mut out = vec![0.0f32; q.n];
        q.apply_rotation(&x, &mut out);
        assert!(
            out.iter().any(|&v| v != 0.0),
            "rotation output should be non-zero"
        );
    }

    #[test]
    fn apply_rotation_transpose_approximately_inverts() {
        let d = 64;
        let q = make_quantizer(d);
        let x: Vec<f32> = (0..d).map(|i| i as f32 * 0.1).collect();
        let mut y = vec![0.0f32; q.n];
        q.apply_rotation(&x, &mut y);
        let mut x_rec = vec![0.0f32; q.n];
        q.apply_rotation_transpose(&y, &mut x_rec);
        for i in 0..d {
            let diff = (x[i] - x_rec[i]).abs();
            assert!(
                diff < 0.01,
                "rotation transpose mismatch at dim {}: {} vs {}",
                i,
                x[i],
                x_rec[i]
            );
        }
    }

    #[test]
    fn nearest_centroid_clamps_to_valid_range() {
        let d = 8;
        // b=2 → 2^2 = 4 centroids
        let q = MseQuantizer::new(d, 2, 42);
        assert_eq!(q.centroids.len(), 4);

        let big = vec![1e10f32; d];
        let idx_big = q.quantize(&big);
        for &idx in &idx_big {
            assert!((idx as usize) < q.centroids.len());
        }

        let small = vec![-1e10f32; d];
        let idx_small = q.quantize(&small);
        for &idx in &idx_small {
            assert!((idx as usize) < q.centroids.len());
        }
    }

    #[test]
    fn rotation_signs_are_plus_minus_one() {
        let d = 32;
        let q = make_quantizer(d);
        for &s in &q.rotation_signs {
            assert!(s == 1.0 || s == -1.0, "sign should be ±1, got {}", s);
        }
    }
}
