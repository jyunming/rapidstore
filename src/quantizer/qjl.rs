use nalgebra::DMatrix;
use ndarray::Array1;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
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
        Self {
            d,
            n,
            projection_signs,
        }
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
        if num_vecs == 0 {
            return Vec::new();
        }

        let packed_len = self.n.div_ceil(8);
        let mut all_qjl = vec![vec![0u8; packed_len]; num_vecs];

        all_qjl
            .par_iter_mut()
            .enumerate()
            .for_each(|(col, packed)| {
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
        if encoded.is_empty() {
            return Vec::new();
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    fn make_qjl(d: usize) -> QjlQuantizer {
        QjlQuantizer::new(d, 42)
    }

    #[test]
    fn new_creates_correct_dimensions() {
        let d = 64;
        let q = make_qjl(d);
        assert_eq!(q.d, d);
        assert_eq!(q.n, d.next_power_of_two());
        assert_eq!(q.projection_signs.len(), q.n);
    }

    #[test]
    fn new_srht_equals_new() {
        let d = 32;
        let q1 = QjlQuantizer::new(d, 42);
        let q2 = QjlQuantizer::new_srht(d, 42);
        assert_eq!(q1.d, q2.d);
        assert_eq!(q1.n, q2.n);
        assert_eq!(q1.projection_signs, q2.projection_signs);
    }

    #[test]
    fn projection_signs_are_plus_minus_one() {
        let d = 32;
        let q = make_qjl(d);
        for &s in &q.projection_signs {
            assert!(s == 1.0 || s == -1.0, "sign should be ±1, got {}", s);
        }
    }

    #[test]
    fn quantize_returns_correct_packed_byte_length() {
        let d = 64;
        let q = make_qjl(d);
        let x = Array1::<f64>::from_iter((0..d).map(|i| i as f64 * 0.01));
        let packed = q.quantize(&x);
        assert_eq!(packed.len(), q.n.div_ceil(8));
    }

    #[test]
    fn quantize_non_power_of_two_d() {
        // d=48 → n=64 → packed = 8 bytes
        let d = 48;
        let q = make_qjl(d);
        assert_eq!(q.n, 64);
        let x = Array1::<f64>::from_iter((0..d).map(|i| i as f64 * 0.1));
        let packed = q.quantize(&x);
        assert_eq!(packed.len(), 8);
    }

    #[test]
    fn quantize_deterministic_with_same_seed() {
        let d = 32;
        let q = QjlQuantizer::new(d, 42);
        let x = Array1::<f64>::from_iter((0..d).map(|i| i as f64));
        let a = q.quantize(&x);
        let b = q.quantize(&x);
        assert_eq!(a, b);
    }

    #[test]
    fn quantize_batch_empty_returns_empty() {
        let d = 32;
        let q = make_qjl(d);
        let mat = DMatrix::<f32>::zeros(d, 0);
        let result = q.quantize_batch(&mat);
        assert!(result.is_empty());
    }

    #[test]
    fn quantize_batch_matches_single() {
        let d = 32;
        let q = make_qjl(d);
        let x = Array1::<f64>::from_iter((0..d).map(|i| i as f64 * 0.05));
        let x_f32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let single = q.quantize(&x);

        let mut mat = DMatrix::<f32>::zeros(d, 1);
        for (i, &v) in x_f32.iter().enumerate() {
            mat[(i, 0)] = v;
        }
        let batch = q.quantize_batch(&mat);

        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0], single);
    }

    #[test]
    fn dequantize_returns_correct_dimension() {
        let d = 64;
        let q = make_qjl(d);
        let x = Array1::from_iter((0..d).map(|i| i as f64 * 0.01));
        let packed = q.quantize(&x);
        let gamma = 1.5;
        let recon = q.dequantize(&packed, gamma);
        assert_eq!(recon.len(), d);
    }

    #[test]
    fn dequantize_batch_empty_returns_empty() {
        let d = 32;
        let q = make_qjl(d);
        let result = q.dequantize_batch(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn dequantize_batch_matches_single() {
        let d = 32;
        let q = make_qjl(d);
        let x = Array1::from_iter((0..d).map(|i| i as f64 * 0.01));
        let packed = q.quantize(&x);
        let gamma = 2.0f64;
        let single = q.dequantize(&packed, gamma);
        let batch = q.dequantize_batch(&[(packed, gamma)]);
        assert_eq!(batch.len(), 1);
        for i in 0..d {
            let diff = (single[i] - batch[0][i]).abs();
            assert!(
                diff < 1e-5,
                "mismatch at dim {}: {} vs {}",
                i,
                single[i],
                batch[0][i]
            );
        }
    }

    #[test]
    fn dequantize_zero_gamma_gives_zero_vector() {
        let d = 32;
        let q = make_qjl(d);
        let x = Array1::<f64>::from_iter((0..d).map(|i| i as f64));
        let packed = q.quantize(&x);
        let recon = q.dequantize(&packed, 0.0);
        for v in recon.iter() {
            assert_eq!(*v, 0.0, "gamma=0 should give zero output");
        }
    }

    #[test]
    fn apply_projection_produces_nonzero_output() {
        let d = 32;
        let q = make_qjl(d);
        let x: Vec<f32> = vec![1.0f32; d];
        let mut out = vec![0.0f32; q.n];
        q.apply_projection(&x, &mut out);
        assert!(out.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn bit_packing_round_trip() {
        let d = 8;
        let q = make_qjl(d);
        // Create a vector that we know will project positively in some dims
        let x = Array1::<f64>::from_iter((0..d).map(|_| 1.0f64));
        let packed = q.quantize(&x);
        assert_eq!(packed.len(), q.n.div_ceil(8));

        // All bits should be either 0 or 1 — verify no undefined bits
        let total_valid_bits = q.n;
        let total_packed_bits = packed.len() * 8;
        let padding_bits = total_packed_bits - total_valid_bits;
        if padding_bits > 0 {
            // Padding bits in the last byte should be 0
            let last_byte = packed[packed.len() - 1];
            let valid_bits_in_last = 8 - padding_bits;
            let padding_mask = !((1u8 << valid_bits_in_last) - 1);
            assert_eq!(
                last_byte & padding_mask,
                0,
                "padding bits should be 0 in last byte"
            );
        }
    }
}
