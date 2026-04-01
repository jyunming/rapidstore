use nalgebra::DMatrix;
use ndarray::Array1;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

use super::CodeIndex;
use super::mse::MseQuantizer;
use super::qjl::QjlQuantizer;
use crate::linalg::hadamard::srht;

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

        let query_f32: Vec<f32> = query.iter().map(|&v| v as f32).collect();
        let mut y = vec![0.0f32; self.d];
        srht(&query_f32, &self.mse_quantizer.rotation_signs, &mut y);

        // Precompute MSE lookup scores for each dimension/code.
        let centroids = &self.mse_quantizer.centroids;
        let lut_w = centroids.len();
        let mut mse_lut = vec![0.0f32; self.d * lut_w];
        for i in 0..self.d {
            let yi = y[i];
            let row_off = i * lut_w;
            for (k, &c) in centroids.iter().enumerate() {
                mse_lut[row_off + k] = c * yi;
            }
        }

        let mut sq = vec![0.0f32; self.d];
        srht(&query_f32, &self.qjl_quantizer.projection_signs, &mut sq);

        PreparedIpQuery {
            mse_lut,
            mse_lut_width: lut_w,
            sq,
            qjl_scale: (PI / 2.0).sqrt() / self.d as f32,
        }
    }

    pub fn prepare_ip_query_lite(&self, query: &Array1<f64>) -> PreparedIpQueryLite {
        assert_eq!(query.len(), self.d);

        let query_f32: Vec<f32> = query.iter().map(|&v| v as f32).collect();
        let mut y = vec![0.0f32; self.d];
        srht(&query_f32, &self.mse_quantizer.rotation_signs, &mut y);

        let mut sq = vec![0.0f32; self.d];
        srht(&query_f32, &self.qjl_quantizer.projection_signs, &mut sq);

        PreparedIpQueryLite {
            y,
            sq,
            qjl_scale: (PI / 2.0).sqrt() / self.d as f32,
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
            mse_score += c * unsafe { *prep_y.get_unchecked(i) };
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
    #[allow(unsafe_op_in_unsafe_fn)]
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

        let mut mse_sum = 0.0f32;
        let mut res = [0.0f32; 8];
        _mm256_storeu_ps(res.as_mut_ptr(), mse_acc);
        for val in res {
            mse_sum += val;
        }
        while i < self.d {
            unsafe {
                mse_sum += *lut.add(i * lut_width + idx[i] as usize);
            }
            i += 1;
        }

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
                let mut signs = [0.0f32; 8];
                for bit in 0..8 {
                    signs[bit] = if (byte & (1 << bit)) != 0 { 1.0 } else { -1.0 };
                }
                let sign_vec = _mm256_loadu_ps(signs.as_ptr());
                qjl_acc = _mm256_add_ps(qjl_acc, _mm256_mul_ps(s_vec, sign_vec));
            } else {
                let mut qs = 0.0f32;
                for bit in 0..8 {
                    let idx = base_i + bit;
                    if idx >= self.d {
                        break;
                    }
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
        let (idx, qjl, gamma) = (self.mse_quantizer.quantize(x), Vec::<u8>::new(), 0.0);

        let x_tilde_mse = self.mse_quantizer.dequantize(&idx);
        let mut residual = Array1::zeros(self.d);
        let mut gamma_sq = 0.0f64;
        for i in 0..self.d {
            let rv = x[i] - x_tilde_mse[i];
            residual[i] = rv;
            gamma_sq += rv * rv;
        }
        let gamma = gamma_sq.sqrt();
        let qjl = self.qjl_quantizer.quantize(&residual);

        (idx, qjl, gamma)
    }

    pub fn quantize_batch(&self, xs: &[Array1<f64>]) -> Vec<(Vec<CodeIndex>, Vec<u8>, f64)> {
        xs.iter().map(|x| self.quantize(x)).collect()
    }

    pub fn dequantize(&self, idx: &[CodeIndex], qjl: &[u8], gamma: f64) -> Array1<f64> {
        let x_mse = self.mse_quantizer.dequantize(idx);
        let x_qjl = self.qjl_quantizer.dequantize(qjl, gamma);
        x_mse + x_qjl
    }

    pub fn dequantize_single_no_parallel(
        &self,
        idx: &[CodeIndex],
        qjl: &[u8],
        gamma: f64,
    ) -> Array1<f64> {
        self.dequantize(idx, qjl, gamma)
    }

    pub fn dequantize_batch(&self, encoded: &[(Vec<CodeIndex>, Vec<u8>, f64)]) -> Vec<Array1<f64>> {
        encoded
            .iter()
            .map(|(idx, qjl, gamma)| self.dequantize(idx, qjl, *gamma))
            .collect()
    }

    pub fn pack_mse_indices(&self, indices: &[CodeIndex]) -> Vec<u8> {
        let bits_per_idx = self.b - 1;
        if bits_per_idx == 8 {
            return indices.iter().map(|v| *v as u8).collect();
        }

        let total_bits = self.d * bits_per_idx;
        let mut packed = vec![0u8; total_bits.div_ceil(8)];

        for i in 0..self.d {
            let val = indices[i] as u32;
            let bit_start = i * bits_per_idx;
            for b in 0..bits_per_idx {
                if (val >> b) & 1 == 1 {
                    let bit_pos = bit_start + b;
                    packed[bit_pos / 8] |= 1 << (bit_pos % 8);
                }
            }
        }
        packed
    }

    pub fn unpack_mse_indices(&self, packed: &[u8], out: &mut [CodeIndex]) {
        let bits_per_idx = self.b - 1;
        if bits_per_idx == 8 {
            for (dst, src) in out.iter_mut().zip(packed.iter()) {
                *dst = *src as CodeIndex;
            }
            return;
        }
        let mask = (1u32 << bits_per_idx) - 1;
        let mut bit_pos = 0usize;
        for i in 0..self.d {
            let byte_idx = bit_pos >> 3;
            let bit_off = bit_pos & 7;
            let mut val = (packed[byte_idx] as u32) >> bit_off;
            let mut bits = 8 - bit_off;
            let mut byte_offset = 1usize;
            while bits < bits_per_idx {
                val |= (packed[byte_idx + byte_offset] as u32) << bits;
                bits += 8;
                byte_offset += 1;
            }
            out[i] = (val & mask) as CodeIndex;
            bit_pos += bits_per_idx;
        }
    }
}
