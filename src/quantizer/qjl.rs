use nalgebra::DMatrix;
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

use crate::linalg::hadamard::srht;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct QjlQuantizer {
    pub d: usize,
    pub projection_signs: Vec<f32>,
}

impl QjlQuantizer {
    pub fn new(d: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut projection_signs = vec![0.0f32; d];
        for s in &mut projection_signs {
            *s = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
        }
        Self {
            d,
            projection_signs,
        }
    }

    pub fn quantize(&self, r: &Array1<f64>) -> Vec<u8> {
        assert_eq!(r.len(), self.d);
        let r_f32: Vec<f32> = r.iter().map(|&v| v as f32).collect();
        let mut s_r = vec![0.0f32; self.d];
        srht(&r_f32, &self.projection_signs, &mut s_r);

        let packed_len = self.d.div_ceil(8);
        let mut packed = vec![0u8; packed_len];
        for row in 0..self.d {
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
        let n = rs.ncols();
        if n == 0 {
            return Vec::new();
        }

        let packed_len = self.d.div_ceil(8);
        let mut all_qjl = vec![vec![0u8; packed_len]; n];

        all_qjl
            .par_iter_mut()
            .enumerate()
            .for_each(|(col, packed)| {
                let r_col_owned: Vec<f32> = rs.column(col).iter().copied().collect();
                let mut s_r = vec![0.0f32; self.d];
                srht(&r_col_owned, &self.projection_signs, &mut s_r);
                for row in 0..self.d {
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
        assert_eq!(qjl.len(), self.d.div_ceil(8));
        let mut out = self.dequantize_batch(&[(qjl.to_vec(), gamma)]);
        out.pop().unwrap_or_else(|| Array1::zeros(self.d))
    }

    pub fn dequantize_batch(&self, encoded: &[(Vec<u8>, f64)]) -> Vec<Array1<f64>> {
        if encoded.is_empty() {
            return Vec::new();
        }

        let n = encoded.len();
        let scale_base = (PI / (2.0 * self.d as f32)).sqrt();

        let mut out = vec![Array1::zeros(self.d); n];
        out.par_iter_mut().enumerate().for_each(|(col, result)| {
            let (qjl, gamma) = &encoded[col];
            let mut qjl_f32 = vec![0.0f32; self.d];
            for row in 0..self.d {
                let byte_idx = row / 8;
                let bit_idx = row % 8;
                let bit_set = ((qjl[byte_idx] >> bit_idx) & 1u8) == 1u8;
                qjl_f32[row] = if bit_set { 1.0 } else { -1.0 };
            }

            let mut st_qjl = vec![0.0f32; self.d];
            srht(&qjl_f32, &self.projection_signs, &mut st_qjl);

            let multiplier = (scale_base * *gamma as f32) as f64;
            for row in 0..self.d {
                result[row] = multiplier * st_qjl[row] as f64;
            }
        });

        out
    }
}
