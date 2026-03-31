use nalgebra::DMatrix;
use ndarray::Array1;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use super::mse::MseQuantizer;
use super::qjl::QjlQuantizer;
use super::CodeIndex;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ProdQuantizer {
    pub d: usize,
    pub b: usize,
    pub mse_quantizer: MseQuantizer,
    pub qjl_quantizer: QjlQuantizer,
}

pub struct PreparedIpQuery {
    // Lookup table flattened as [dim0 c0..cK, dim1 c0..cK, ...]
    mse_lut: Vec<f32>,
    mse_lut_width: usize,
    sq: Vec<f32>,
    qjl_scale: f32,
}

pub struct PreparedIpQueryLite {
    y: Vec<f32>,
    sq: Vec<f32>,
    qjl_scale: f32,
}

impl ProdQuantizer {
    pub fn new(d: usize, b: usize, seed: u64) -> Self {
        assert!(b >= 2, "ProdQuantizer requires at least b=2");

        let mse_quantizer = MseQuantizer::new(d, b - 1, seed);
        let qjl_quantizer = QjlQuantizer::new(d, seed ^ 0xdeadbeef);

        Self {
            d,
            b,
            mse_quantizer,
            qjl_quantizer,
        }
    }

    pub fn prepare_ip_query(&self, query: &Array1<f64>) -> PreparedIpQuery {
        assert_eq!(query.len(), self.d);

        let mut y = vec![0.0f32; self.d];
        let mut sq = vec![0.0f32; self.d];

        // y = R * q
        for row in 0..self.d {
            let mut acc = 0.0;
            for col in 0..self.d {
                acc += self.mse_quantizer.rotation[(row, col)] * query[col];
            }
            y[row] = acc as f32;
        }

        // Precompute MSE lookup scores for each dimension/code.
        let centroids = &self.mse_quantizer.centroids;
        let lut_w = centroids.len();
        let mut mse_lut = vec![0.0f32; self.d * lut_w];
        for i in 0..self.d {
            let yi = y[i];
            let row_off = i * lut_w;
            for (k, &c) in centroids.iter().enumerate() {
                mse_lut[row_off + k] = (c * yi as f64) as f32;
            }
        }

        // sq = S * q
        for row in 0..self.d {
            let mut acc = 0.0;
            for col in 0..self.d {
                acc += self.qjl_quantizer.projection[(row, col)] * query[col];
            }
            sq[row] = acc as f32;
        }

        PreparedIpQuery {
            mse_lut,
            mse_lut_width: lut_w,
            sq,
            qjl_scale: (PI / (2.0 * self.d as f64)).sqrt() as f32,
        }
    }

    pub fn prepare_ip_query_lite(&self, query: &Array1<f64>) -> PreparedIpQueryLite {
        assert_eq!(query.len(), self.d);

        let mut y = vec![0.0f32; self.d];
        let mut sq = vec![0.0f32; self.d];

        for row in 0..self.d {
            let mut acc = 0.0;
            for col in 0..self.d {
                acc += self.mse_quantizer.rotation[(row, col)] * query[col];
            }
            y[row] = acc as f32;
        }

        for row in 0..self.d {
            let mut acc = 0.0;
            for col in 0..self.d {
                acc += self.qjl_quantizer.projection[(row, col)] * query[col];
            }
            sq[row] = acc as f32;
        }

        PreparedIpQueryLite {
            y,
            sq,
            qjl_scale: (PI / (2.0 * self.d as f64)).sqrt() as f32,
        }
    }

    pub fn score_ip_encoded_lite(
        &self,
        prep: &PreparedIpQueryLite,
        idx: &[CodeIndex],
        qjl: &[u8],
        gamma: f64,
    ) -> f64 {
        let mut mse_score = 0.0f32;
        let centroids = &self.mse_quantizer.centroids;
        let prep_y = &prep.y;

        for i in 0..self.d {
            let c = unsafe { *centroids.get_unchecked(idx[i] as usize) };
            mse_score += (c as f32) * unsafe { *prep_y.get_unchecked(i) };
        }

        let mut qjl_score = 0.0f32;
        let sq = &prep.sq;
        let qjl_len = qjl.len();

        for byte_idx in 0..qjl_len {
            let byte = unsafe { *qjl.get_unchecked(byte_idx) };
            let base_i = byte_idx << 3;

            for bit_idx in 0..8 {
                let i = base_i + bit_idx;
                if i >= self.d {
                    break;
                }
                let bit_set = ((byte >> bit_idx) & 1u8) == 1u8;
                let s = unsafe { *sq.get_unchecked(i) };
                qjl_score += if bit_set { s } else { -s };
            }
        }

        (mse_score + (gamma as f32) * prep.qjl_scale * qjl_score) as f64
    }

    pub fn score_ip_encoded(
        &self,
        prep: &PreparedIpQuery,
        idx: &[CodeIndex],
        qjl: &[u8],
        gamma: f64,
    ) -> f64 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { self.score_ip_encoded_simd(prep, idx, qjl, gamma) };
            }
        }
        self.score_ip_encoded_scalar(prep, idx, qjl, gamma)
    }

    fn score_ip_encoded_scalar(
        &self,
        prep: &PreparedIpQuery,
        idx: &[CodeIndex],
        qjl: &[u8],
        gamma: f64,
    ) -> f64 {
        let mut mse_score = 0.0f32;
        let lut_width = prep.mse_lut_width;
        let lut = &prep.mse_lut;

        let mut row_off = 0;
        for i in 0..self.d {
            let val = unsafe { *lut.get_unchecked(row_off + idx[i] as usize) };
            mse_score += val;
            row_off += lut_width;
        }

        let mut qjl_score = 0.0f32;
        let sq = &prep.sq;

        let qjl_len = qjl.len();
        for byte_idx in 0..qjl_len {
            let byte = unsafe { *qjl.get_unchecked(byte_idx) };
            let base_i = byte_idx << 3;

            for bit_idx in 0..8 {
                let i = base_i + bit_idx;
                if i >= self.d {
                    break;
                }
                let bit_set = ((byte >> bit_idx) & 1u8) == 1u8;
                let s = unsafe { *sq.get_unchecked(i) };
                qjl_score += if bit_set { s } else { -s };
            }
        }

        (mse_score + (gamma as f32) * prep.qjl_scale * qjl_score) as f64
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn score_ip_encoded_simd(
        &self,
        prep: &PreparedIpQuery,
        idx: &[CodeIndex],
        qjl: &[u8],
        gamma: f64,
    ) -> f64 {
        use std::arch::x86_64::*;

        let mut mse_acc = _mm256_setzero_ps();
        let lut = prep.mse_lut.as_ptr();
        let lut_width = prep.mse_lut_width;

        // MSE loop: 8 coordinates at a time
        let mut i = 0;
        while i + 7 < self.d {
            let mut vals = [0.0f32; 8];
            for j in 0..8 {
                let row = i + j;
                unsafe {
                    vals[j] = *lut.add(row * lut_width + idx[row] as usize);
                }
            }
            let v = _mm256_loadu_ps(vals.as_ptr());
            mse_acc = _mm256_add_ps(mse_acc, v);
            i += 8;
        }

        // Horizontal sum for MSE
        let mut mse_sum = 0.0f32;
        let mut res = [0.0f32; 8];
        _mm256_storeu_ps(res.as_mut_ptr(), mse_acc);
        for val in res {
            mse_sum += val;
        }
        // MSE remainder
        while i < self.d {
            unsafe {
                mse_sum += *lut.add(i * lut_width + idx[i] as usize);
            }
            i += 1;
        }

        // QJL loop: 8 bits at a time (one byte)
        let mut qjl_acc = _mm256_setzero_ps();
        let sq = prep.sq.as_ptr();
        let qjl_ptr = qjl.as_ptr();
        let qjl_len = qjl.len();

        let mut b = 0;
        while b < qjl_len {
            let byte = unsafe { *qjl_ptr.add(b) as i32 };
            let base_i = b << 3;
            if base_i + 7 < self.d {
                let s_vec = _mm256_loadu_ps(unsafe { sq.add(base_i) });
                
                // Expand bits to signs: 1.0 or -1.0
                let mut signs = [0.0f32; 8];
                for bit in 0..8 {
                    signs[bit] = if (byte & (1 << bit)) != 0 { 1.0 } else { -1.0 };
                }
                let sign_vec = _mm256_loadu_ps(signs.as_ptr());
                qjl_acc = _mm256_add_ps(qjl_acc, _mm256_mul_ps(s_vec, sign_vec));
            } else {
                // Remainder of QJL within the byte
                let mut qs = 0.0f32;
                for bit in 0..8 {
                    let idx = base_i + bit;
                    if idx >= self.d { break; }
                    let s = unsafe { *sq.add(idx) };
                    qs += if (byte & (1 << bit)) != 0 { s } else { -s };
                }
                mse_sum += (gamma as f32) * prep.qjl_scale * qs;
            }
            b += 1;
        }

        _mm256_storeu_ps(res.as_mut_ptr(), qjl_acc);
        let mut qjl_sum = 0.0f32;
        for val in res {
            qjl_sum += val;
        }

        (mse_sum + (gamma as f32) * prep.qjl_scale * qjl_sum) as f64
    }

    pub fn quantize(&self, x: &Array1<f64>) -> (Vec<CodeIndex>, Vec<u8>, f64) {
        let mut out = self.quantize_batch(&[x.clone()]);
        out.pop().unwrap_or_else(|| (Vec::new(), Vec::new(), 0.0))
    }

    pub fn quantize_batch(&self, xs: &[Array1<f64>]) -> Vec<(Vec<CodeIndex>, Vec<u8>, f64)> {
        let n = xs.len();
        if n == 0 {
            return Vec::new();
        }

        for x in xs {
            assert_eq!(x.len(), self.d);
        }

        let mut x_mat = DMatrix::zeros(self.d, n);
        for (col, x) in xs.iter().enumerate() {
            for row in 0..self.d {
                x_mat[(row, col)] = x[row];
            }
        }

        let mse_indices = self.mse_quantizer.quantize_batch(&x_mat);
        let x_tilde_mse_batch = self.mse_quantizer.dequantize_batch(&mse_indices);

        let residual_cols: Vec<(Vec<f64>, f64)> = (0..n)
            .into_par_iter()
            .map(|col| {
                let mut residual = vec![0.0f64; self.d];
                let mut gamma_sq = 0.0f64;
                for row in 0..self.d {
                    let rv = x_mat[(row, col)] - x_tilde_mse_batch[(row, col)];
                    residual[row] = rv;
                    gamma_sq += rv * rv;
                }
                (residual, gamma_sq.sqrt())
            })
            .collect();

        let mut r_mat = DMatrix::zeros(self.d, n);
        let mut gammas = vec![0.0f64; n];
        for (col, (residual, gamma)) in residual_cols.into_iter().enumerate() {
            gammas[col] = gamma;
            for row in 0..self.d {
                r_mat[(row, col)] = residual[row];
            }
        }

        let qjl_all = self.qjl_quantizer.quantize_batch(&r_mat);

        mse_indices
            .into_iter()
            .zip(qjl_all)
            .zip(gammas)
            .map(|((idx, qjl), gamma)| (idx, qjl, gamma))
            .collect()
    }

    pub fn dequantize(&self, idx: &[CodeIndex], qjl: &[u8], gamma: f64) -> Array1<f64> {
        let mut out = self.dequantize_batch(&[(idx.to_vec(), qjl.to_vec(), gamma)]);
        out.pop().unwrap_or_else(|| Array1::zeros(self.d))
    }

    pub fn dequantize_single_no_parallel(
        &self,
        idx: &[CodeIndex],
        qjl: &[u8],
        gamma: f64,
    ) -> Array1<f64> {
        assert_eq!(idx.len(), self.d);
        assert_eq!(qjl.len(), self.d.div_ceil(8));

        let mut y_tilde = vec![0.0f64; self.d];
        for i in 0..self.d {
            y_tilde[i] = self.mse_quantizer.centroids[idx[i] as usize];
        }

        // x_mse = R^T * y_tilde
        let mut x_mse = vec![0.0f64; self.d];
        for row in 0..self.d {
            let mut acc = 0.0;
            for col in 0..self.d {
                acc += self.mse_quantizer.rotation[(col, row)] * y_tilde[col];
            }
            x_mse[row] = acc;
        }

        // x_qjl = sqrt(pi/(2d)) * gamma * S^T * sign(qjl)
        let mut x_qjl = vec![0.0f64; self.d];
        let scale = (PI / (2.0 * self.d as f64)).sqrt() * gamma;
        for row in 0..self.d {
            let mut acc = 0.0;
            for col in 0..self.d {
                let byte_idx = col / 8;
                let bit_idx = col % 8;
                let bit_set = ((qjl[byte_idx] >> bit_idx) & 1u8) == 1u8;
                let sign = if bit_set { 1.0 } else { -1.0 };
                acc += self.qjl_quantizer.projection[(col, row)] * sign;
            }
            x_qjl[row] = scale * acc;
        }

        let mut out = Array1::zeros(self.d);
        for i in 0..self.d {
            out[i] = x_mse[i] + x_qjl[i];
        }
        out
    }

    pub fn dequantize_batch(&self, encoded: &[(Vec<CodeIndex>, Vec<u8>, f64)]) -> Vec<Array1<f64>> {
        if encoded.is_empty() {
            return Vec::new();
        }

        let idx_batch: Vec<Vec<CodeIndex>> = encoded.iter().map(|(idx, _, _)| idx.clone()).collect();
        let x_tilde_mse_batch = self.mse_quantizer.dequantize_batch(&idx_batch);

        let qjl_batch: Vec<(Vec<u8>, f64)> = encoded
            .iter()
            .map(|(_, qjl, gamma)| (qjl.clone(), *gamma))
            .collect();
        let x_tilde_qjl_batch = self.qjl_quantizer.dequantize_batch(&qjl_batch);

        (0..encoded.len())
            .into_par_iter()
            .map(|col| {
                let mut result = Array1::zeros(self.d);
                for row in 0..self.d {
                    result[row] = x_tilde_mse_batch[(row, col)] + x_tilde_qjl_batch[col][row];
                }
                result
            })
            .collect()
    }
}
