use nalgebra::DMatrix;
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

use crate::linalg::hadamard::srht;

/// QJL (Quantized Johnson-Lindenstrauss) residual quantizer.
///
/// Captures the MSE residual using a single-bit random projection.
///
/// ## Algorithm
///
/// Given residual `r = x - dequant(mse(x))`:
///
/// 1. Apply SRHT: `s = H * D * (r, 0)` (pad to next power of two)
/// 2. Store `sign(s[i])` as a 1-bit value, bit-packed into `ceil(n/8)` bytes
/// 3. A scaling factor `gamma = sqrt(n / d) * ||x||₂` (stored as f32) unbiases the
///    inner-product estimate during scoring
///
/// ## Scoring
///
/// For query `q` and encoded vector `z`, the inner product estimate is:
/// ```text
/// <x, q> ≈ mse_score(mse(x), mse(q)) + gamma * qjl_score(z, qjl(q))
/// ```
/// where `qjl_score` is the normalized Hamming distance scaled by `1/sqrt(n)`.
///
/// See the TurboQuant paper (arXiv:2504.19874, Section 3) for the unbiasedness proof.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct QjlQuantizer {
    pub d: usize,
    pub n: usize,
    pub projection_signs: Vec<f32>,
}

impl QjlQuantizer {
    pub fn new(d: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = d.next_power_of_two();
        let mut projection_signs = vec![0.0f32; n];
        for s in &mut projection_signs {
            *s = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
        }
        Self { d, n, projection_signs }
    }

    /// Construct with O(d log d) SRHT fast-path.
    pub fn new_srht(d: usize, seed: u64) -> Self {
        Self::new(d, seed)
    }

    /// Forward projection: y = H * D * (x, 0)
    pub fn apply_projection(&self, x: &[f32], out: &mut [f32]) {
        srht(x, &self.projection_signs, out);
    }

    pub fn quantize(&self, r: &Array1<f64>) -> Vec<u8> {
        assert_eq!(r.len(), self.d);
        let r_f32: Vec<f32> = r.iter().map(|&v| v as f32).collect();
        let mut s_r = vec![0.0f32; self.n];
        self.apply_projection(&r_f32, &mut s_r);

        let packed_len = self.n.div_ceil(8);
        let mut packed = vec![0u8; packed_len];
        for row in 0..self.n {
            if s_r[row] >= 0.0 {
                let byte_idx = row / 8;
                let bit_idx = row % 8;
                packed[byte_idx] |= 1u8 << bit_idx;
            }
        }
        packed
    }

    pub fn quantize_batch(&self, rs: &DMatrix<f32>) -> Vec<Vec<u8>> {
        assert_eq!(rs.nrows(), self.d);
        let num_vecs = rs.ncols();
        if num_vecs == 0 { return Vec::new(); }

        let packed_len = self.n.div_ceil(8);
        let mut all_qjl = vec![vec![0u8; packed_len]; num_vecs];
        
        all_qjl.par_iter_mut().enumerate().for_each(|(col, packed)| {
            let r_col_view = rs.column(col);
            let r_col = r_col_view.as_slice();
            let mut s_r = vec![0.0f32; self.n];
            srht(r_col, &self.projection_signs, &mut s_r);
            for row in 0..self.n {
                if s_r[row] >= 0.0 {
                    let byte_idx = row / 8;
                    let bit_idx = row % 8;
                    packed[byte_idx] |= 1u8 << bit_idx;
                }
            }
        });

        all_qjl
    }

    pub fn dequantize(&self, qjl: &[u8], gamma: f64) -> Array1<f64> {
        assert_eq!(qjl.len(), self.n.div_ceil(8));
        let mut out = self.dequantize_batch(&[(qjl.to_vec(), gamma)]);
        out.pop().unwrap_or_else(|| Array1::zeros(self.d))
    }

    pub fn dequantize_batch(&self, encoded: &[(Vec<u8>, f64)]) -> Vec<Array1<f64>> {
        if encoded.is_empty() { return Vec::new(); }

        let n_vecs = encoded.len();
        let scale_base = (PI / (2.0 * self.n as f32)).sqrt();
        
        let mut out = vec![Array1::zeros(self.d); n_vecs];
        out.par_iter_mut().enumerate().for_each(|(col, result)| {
            let (qjl, gamma) = &encoded[col];
            let mut qjl_f32 = vec![0.0f32; self.n];
            for row in 0..self.n {
                let byte_idx = row / 8;
                let bit_idx = row % 8;
                let bit_set = ((qjl[byte_idx] >> bit_idx) & 1u8) == 1u8;
                qjl_f32[row] = if bit_set { 1.0 } else { -1.0 };
            }
            
            let mut st_qjl = vec![0.0f32; self.d];
            crate::linalg::hadamard::inverse_srht(&qjl_f32, &self.projection_signs, &mut st_qjl);
            
            let multiplier = (scale_base * *gamma as f32) as f64;
            for row in 0..self.d {
                result[row] = multiplier * st_qjl[row] as f64;
            }
        });

        out
    }
}
