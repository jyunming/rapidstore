use nalgebra::DMatrix;
use ndarray::Array1;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::CodeIndex;
use super::codebook::lloyd_max;
use crate::linalg::hadamard::{fwht, srht};
use crate::linalg::rotation::generate_random_rotation;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MseQuantizer {
    pub d: usize,
    pub b: usize,
    /// Rotation state — two formats distinguished by length:
    ///   - len == d   : legacy SRHT diagonal sign vector (old databases)
    ///   - len == d*d : flattened d×d QR rotation matrix Π, column-major (paper-conformant)
    pub rotation_signs: Vec<f32>,
    pub centroids: Vec<f32>,
}

impl MseQuantizer {
    /// Construct with paper-exact QR rotation (O(d²) per vector, SIMD-accelerated).
    pub fn new(d: usize, b: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        // Paper Algorithm 1, step 2: Π = Q from QR(G), G_{ij} ~ N(0,1)
        let rotation = generate_random_rotation(d, &mut rng);
        // Store flattened column-major; len == d*d signals paper-conformant mode
        let rotation_signs: Vec<f32> = rotation.as_slice().to_vec();

        let centroids: Vec<f32> = lloyd_max(b, d, 20_000)
            .into_iter()
            .map(|c| c as f32)
            .collect();
        assert!(
            centroids.len() <= u16::MAX as usize + 1,
            "codebook too large for u16 indices; reduce bits"
        );

        Self { d, b, rotation_signs, centroids }
    }

    /// Construct with O(d log d) SRHT fast-path.
    /// Preserves JL-guarantee and Beta-marginal coordinate distribution;
    /// gives O(1/√d) approximation error in QJL estimator (negligible for d ≥ 512).
    /// `rotation_signs.len() == d` selects this path at runtime.
    pub fn new_srht(d: usize, b: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let rotation_signs: Vec<f32> = (0..d)
            .map(|_| if rng.gen_bool(0.5) { 1.0f32 } else { -1.0f32 })
            .collect();
        let centroids: Vec<f32> = lloyd_max(b, d, 20_000)
            .into_iter()
            .map(|c| c as f32)
            .collect();
        assert!(
            centroids.len() <= u16::MAX as usize + 1,
            "codebook too large for u16 indices; reduce bits"
        );
        Self { d, b, rotation_signs, centroids }
    }

    /// True when `rotation_signs` holds a full d×d QR matrix (paper-conformant).
    /// False when it holds the legacy d-element SRHT sign vector.
    #[inline]
    pub fn is_qr_mode(&self) -> bool {
        self.rotation_signs.len() == self.d * self.d
    }

    /// Forward rotation: y = Π · x  (used during quantization).
    pub fn apply_rotation(&self, x: &[f32], out: &mut [f32]) {
        let d = self.d;
        if self.is_qr_mode() {
            // y = Π · x via SIMD-optimized sgemm.
            // Π stored column-major: row_stride=1, col_stride=d.
            unsafe {
                matrixmultiply::sgemm(
                    d, d, 1,
                    1.0,
                    self.rotation_signs.as_ptr(), 1, d as isize,
                    x.as_ptr(), 1, 1,
                    0.0,
                    out.as_mut_ptr(), 1, 1,
                );
            }
        } else {
            srht(x, &self.rotation_signs, out);
        }
    }

    /// Inverse rotation: x̃ = Π^T · ỹ  (used during dequantization).
    pub fn apply_rotation_transpose(&self, y: &[f32], out: &mut [f32]) {
        let d = self.d;
        if self.is_qr_mode() {
            // x̃ = Π^T · ỹ — transpose by swapping row/col strides.
            unsafe {
                matrixmultiply::sgemm(
                    d, d, 1,
                    1.0,
                    self.rotation_signs.as_ptr(), d as isize, 1,
                    y.as_ptr(), 1, 1,
                    0.0,
                    out.as_mut_ptr(), 1, 1,
                );
            }
        } else {
            // Legacy SRHT inverse: x = D · H · (1/√n) · y
            let n = d.next_power_of_two();
            let mut temp = vec![0.0f32; n];
            temp[..d].copy_from_slice(y);
            fwht(&mut temp);
            let norm = 1.0 / (n as f32).sqrt();
            for i in 0..d {
                out[i] = temp[i] * self.rotation_signs[i] * norm;
            }
        }
    }

    pub fn quantize(&self, x: &Array1<f64>) -> Vec<CodeIndex> {
        assert_eq!(x.len(), self.d);
        let x_f32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let mut y = vec![0.0f32; self.d];
        self.apply_rotation(&x_f32, &mut y);

        let mut indices = vec![0 as CodeIndex; self.d];
        for i in 0..self.d {
            indices[i] = self.nearest_centroid_index(y[i]);
        }
        indices
    }

    pub fn quantize_batch(&self, xs: &DMatrix<f32>) -> Vec<Vec<CodeIndex>> {
        assert_eq!(xs.nrows(), self.d);
        let n = xs.ncols();
        if n == 0 {
            return Vec::new();
        }

        let mut all_indices = vec![vec![0 as CodeIndex; self.d]; n];
        all_indices.par_iter_mut().enumerate().for_each(|(col, idxs)| {
            let x_col: Vec<f32> = xs.column(col).iter().copied().collect();
            let mut y = vec![0.0f32; self.d];
            self.apply_rotation(&x_col, &mut y);
            for row in 0..self.d {
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
        assert_eq!(indices.len(), self.d);
        let d = self.d;
        let mut y_tilde = vec![0.0f32; d];
        for i in 0..d {
            y_tilde[i] = self.centroids[indices[i] as usize];
        }
        let mut out_data = vec![0.0f32; d];
        self.apply_rotation_transpose(&y_tilde, &mut out_data);
        let mut out = Array1::zeros(d);
        for i in 0..d {
            out[i] = out_data[i] as f64;
        }
        out
    }

    pub fn dequantize_batch(&self, indices_batch: &[Vec<CodeIndex>]) -> DMatrix<f32> {
        let n = indices_batch.len();
        if n == 0 {
            return DMatrix::zeros(self.d, 0);
        }
        let mut out = DMatrix::zeros(self.d, n);
        let d = self.d;
        for (col, indices) in indices_batch.iter().enumerate() {
            let mut y_tilde = vec![0.0f32; d];
            for row in 0..d {
                y_tilde[row] = self.centroids[indices[row] as usize];
            }
            let mut x_rec = vec![0.0f32; d];
            self.apply_rotation_transpose(&y_tilde, &mut x_rec);
            for row in 0..d {
                out[(row, col)] = x_rec[row];
            }
        }
        out
    }
}
