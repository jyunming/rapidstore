use nalgebra::DMatrix;
use ndarray::Array1;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

use crate::linalg::hadamard::srht;
use crate::linalg::rotation::generate_projection_matrix;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct QjlQuantizer {
    pub d: usize,
    /// Projection state — two formats distinguished by length:
    ///   - len == d   : legacy SRHT diagonal sign vector (old databases)
    ///   - len == d*d : flattened d×d Gaussian matrix S, column-major (paper-conformant)
    pub projection_signs: Vec<f32>,
}

impl QjlQuantizer {
    /// Construct with paper-exact dense Gaussian projection (O(d²) per vector, SIMD-accelerated).
    pub fn new(d: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        // Paper Algorithm 2, step 3: S ∈ R^{d×d}, S_{ij} ~ N(0,1) i.i.d.
        let projection = generate_projection_matrix(d, &mut rng);
        // Store flattened column-major; len == d*d signals paper-conformant mode
        let projection_signs: Vec<f32> = projection.as_slice().to_vec();

        Self { d, projection_signs }
    }

    /// Construct with O(d log d) SRHT fast-path.
    /// `projection_signs.len() == d` selects this path at runtime.
    pub fn new_srht(d: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let projection_signs: Vec<f32> = (0..d)
            .map(|_| if rng.gen_bool(0.5) { 1.0f32 } else { -1.0f32 })
            .collect();
        Self { d, projection_signs }
    }

    /// True when `projection_signs` holds a full d×d Gaussian matrix (paper-conformant).
    /// False when it holds the legacy d-element SRHT sign vector.
    #[inline]
    pub fn is_matrix_mode(&self) -> bool {
        self.projection_signs.len() == self.d * self.d
    }

    /// Forward projection: s = S · x  (used during quantization).
    pub fn apply_projection(&self, x: &[f32], out: &mut [f32]) {
        let d = self.d;
        if self.is_matrix_mode() {
            // s = S · x via SIMD-optimized sgemm.
            // S stored column-major: row_stride=1, col_stride=d.
            unsafe {
                matrixmultiply::sgemm(
                    d, d, 1,
                    1.0,
                    self.projection_signs.as_ptr(), 1, d as isize,
                    x.as_ptr(), 1, 1,
                    0.0,
                    out.as_mut_ptr(), 1, 1,
                );
            }
        } else {
            srht(x, &self.projection_signs, out);
        }
    }

    /// Transpose projection: v = S^T · y  (used during dequantization).
    pub fn apply_projection_transpose(&self, y: &[f32], out: &mut [f32]) {
        let d = self.d;
        if self.is_matrix_mode() {
            // v = S^T · y — transpose by swapping row/col strides.
            unsafe {
                matrixmultiply::sgemm(
                    d, d, 1,
                    1.0,
                    self.projection_signs.as_ptr(), d as isize, 1,
                    y.as_ptr(), 1, 1,
                    0.0,
                    out.as_mut_ptr(), 1, 1,
                );
            }
        } else {
            srht(y, &self.projection_signs, out);
        }
    }

    pub fn quantize(&self, r: &Array1<f64>) -> Vec<u8> {
        assert_eq!(r.len(), self.d);
        let r_f32: Vec<f32> = r.iter().map(|&v| v as f32).collect();
        let mut s_r = vec![0.0f32; self.d];
        self.apply_projection(&r_f32, &mut s_r);

        let packed_len = self.d.div_ceil(8);
        let mut packed = vec![0u8; packed_len];
        for row in 0..self.d {
            if s_r[row] >= 0.0 {
                packed[row / 8] |= 1u8 << (row % 8);
            }
        }
        packed
    }

    pub fn quantize_batch(&self, rs: &DMatrix<f32>) -> Vec<Vec<u8>> {
        assert_eq!(rs.nrows(), self.d);
        let n = rs.ncols();
        if n == 0 {
            return Vec::new();
        }

        let packed_len = self.d.div_ceil(8);
        let mut all_qjl = vec![vec![0u8; packed_len]; n];
        all_qjl.par_iter_mut().enumerate().for_each(|(col, packed)| {
            let r_col: Vec<f32> = rs.column(col).iter().copied().collect();
            let mut s_r = vec![0.0f32; self.d];
            self.apply_projection(&r_col, &mut s_r);
            for row in 0..self.d {
                if s_r[row] >= 0.0 {
                    packed[row / 8] |= 1u8 << (row % 8);
                }
            }
        });
        all_qjl
    }

    pub fn dequantize(&self, qjl: &[u8], gamma: f64) -> Array1<f64> {
        assert_eq!(qjl.len(), self.d.div_ceil(8));
        let mut out = self.dequantize_batch(&[(qjl.to_vec(), gamma)]);
        out.pop().unwrap_or_else(|| Array1::zeros(self.d))
    }

    pub fn dequantize_batch(&self, encoded: &[(Vec<u8>, f64)]) -> Vec<Array1<f64>> {
        if encoded.is_empty() {
            return Vec::new();
        }

        let d = self.d;
        // Unbiased reconstruction scale derived from QJL estimator:
        //   E[<S·y, sign(S·r)>] = d · sqrt(2/π) · <y,r>/‖r‖
        //   → scale = sqrt(π/2) / d  so that E[x̃_qjl] = r
        let scale_base = (PI / 2.0_f32).sqrt() / d as f32;

        let mut out = vec![Array1::zeros(d); encoded.len()];
        out.par_iter_mut().enumerate().for_each(|(col, result)| {
            let (qjl, gamma) = &encoded[col];
            let mut sign_vec = vec![0.0f32; d];
            for row in 0..d {
                let bit_set = ((qjl[row / 8] >> (row % 8)) & 1u8) == 1u8;
                sign_vec[row] = if bit_set { 1.0 } else { -1.0 };
            }
            let mut st_qjl = vec![0.0f32; d];
            self.apply_projection_transpose(&sign_vec, &mut st_qjl);
            let multiplier = (scale_base * *gamma as f32) as f64;
            for row in 0..d {
                result[row] = multiplier * st_qjl[row] as f64;
            }
        });
        out
    }
}
