use nalgebra::DMatrix;
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use crate::linalg::matmul::gemm;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct QjlQuantizer {
    pub d: usize,
    pub projection: DMatrix<f64>,
}

impl QjlQuantizer {
    pub fn new(d: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let projection = crate::linalg::rotation::generate_projection_matrix(d, &mut rng);
        Self { d, projection }
    }

    pub fn quantize(&self, r: &Array1<f64>) -> Vec<u8> {
        assert_eq!(r.len(), self.d);

        let mut r_mat = DMatrix::zeros(self.d, 1);
        for i in 0..self.d {
            r_mat[(i, 0)] = r[i];
        }

        self.quantize_batch(&r_mat)
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    pub fn quantize_batch(&self, rs: &DMatrix<f64>) -> Vec<Vec<u8>> {
        assert_eq!(rs.nrows(), self.d);

        let n = rs.ncols();
        if n == 0 {
            return Vec::new();
        }

        let s_r_batch = gemm(&self.projection, false, rs, false);
        let packed_len = self.d.div_ceil(8);
        let mut all_qjl = vec![vec![0u8; packed_len]; n];
        all_qjl
            .par_iter_mut()
            .enumerate()
            .for_each(|(col, packed)| {
                for row in 0..self.d {
                    if s_r_batch[(row, col)] >= 0.0 {
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
        let packed_len = self.d.div_ceil(8);
        let mut qjl_mat = DMatrix::zeros(self.d, n);
        for (col, (qjl, _)) in encoded.iter().enumerate() {
            assert_eq!(qjl.len(), packed_len);
            for row in 0..self.d {
                let byte_idx = row / 8;
                let bit_idx = row % 8;
                let bit_set = ((qjl[byte_idx] >> bit_idx) & 1u8) == 1u8;
                qjl_mat[(row, col)] = if bit_set { 1.0 } else { -1.0 };
            }
        }

        let st_qjl_batch = gemm(&self.projection, true, &qjl_mat, false);
        let scale_base = (PI / (2.0 * self.d as f64)).sqrt();

        let mut out = vec![Array1::zeros(self.d); n];
        out.par_iter_mut().enumerate().for_each(|(col, result)| {
            let multiplier = scale_base * encoded[col].1;
            for row in 0..self.d {
                result[row] = multiplier * st_qjl_batch[(row, col)];
            }
        });

        out
    }
}
