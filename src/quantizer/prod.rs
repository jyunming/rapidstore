use nalgebra::DMatrix;
use ndarray::Array1;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

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

    pub fn quantize(&self, x: &Array1<f64>) -> (Vec<CodeIndex>, Vec<i8>, f64) {
        let mut out = self.quantize_batch(&[x.clone()]);
        out.pop().unwrap_or_else(|| (Vec::new(), Vec::new(), 0.0))
    }

    pub fn quantize_batch(&self, xs: &[Array1<f64>]) -> Vec<(Vec<CodeIndex>, Vec<i8>, f64)> {
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

    pub fn dequantize(&self, idx: &[CodeIndex], qjl: &[i8], gamma: f64) -> Array1<f64> {
        let mut out = self.dequantize_batch(&[(idx.to_vec(), qjl.to_vec(), gamma)]);
        out.pop().unwrap_or_else(|| Array1::zeros(self.d))
    }

    pub fn dequantize_batch(&self, encoded: &[(Vec<CodeIndex>, Vec<i8>, f64)]) -> Vec<Array1<f64>> {
        if encoded.is_empty() {
            return Vec::new();
        }

        let idx_batch: Vec<Vec<CodeIndex>> = encoded.iter().map(|(idx, _, _)| idx.clone()).collect();
        let x_tilde_mse_batch = self.mse_quantizer.dequantize_batch(&idx_batch);

        let qjl_batch: Vec<(Vec<i8>, f64)> = encoded
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
