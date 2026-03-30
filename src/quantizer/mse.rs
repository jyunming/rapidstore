use nalgebra::DMatrix;
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::codebook::lloyd_max;
use super::CodeIndex;
use crate::linalg::matmul::gemm;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MseQuantizer {
    pub d: usize,
    pub b: usize,
    pub rotation: DMatrix<f64>,
    pub centroids: Vec<f64>,
}

impl MseQuantizer {
    pub fn new(d: usize, b: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let rotation = crate::linalg::rotation::generate_random_rotation(d, &mut rng);
        let num_points = 20_000;
        let centroids = lloyd_max(b, d, num_points);

        Self {
            d,
            b,
            rotation,
            centroids,
        }
    }

    pub fn quantize(&self, x: &Array1<f64>) -> Vec<CodeIndex> {
        assert_eq!(x.len(), self.d);

        let mut x_mat = DMatrix::zeros(self.d, 1);
        for i in 0..self.d {
            x_mat[(i, 0)] = x[i];
        }

        self.quantize_batch(&x_mat)
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    pub fn quantize_batch(&self, xs: &DMatrix<f64>) -> Vec<Vec<CodeIndex>> {
        assert_eq!(xs.nrows(), self.d);

        let n = xs.ncols();
        if n == 0 {
            return Vec::new();
        }

        let y_batch = gemm(&self.rotation, false, xs, false);
        let mut all_indices = vec![vec![0 as CodeIndex; self.d]; n];

        all_indices
            .par_iter_mut()
            .enumerate()
            .for_each(|(col, idxs)| {
                for row in 0..self.d {
                    let val = y_batch[(row, col)];
                    idxs[row] = self.nearest_centroid_index(val);
                }
            });

        all_indices
    }

    fn nearest_centroid_index(&self, val: f64) -> CodeIndex {
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
        let batch = vec![indices.to_vec()];
        let x_tilde_batch = self.dequantize_batch(&batch);

        let mut x_tilde = Array1::zeros(self.d);
        for i in 0..self.d {
            x_tilde[i] = x_tilde_batch[(i, 0)];
        }
        x_tilde
    }

    pub fn dequantize_batch(&self, indices_batch: &[Vec<CodeIndex>]) -> DMatrix<f64> {
        let n = indices_batch.len();
        if n == 0 {
            return DMatrix::zeros(self.d, 0);
        }

        let mut y_tilde_batch = DMatrix::zeros(self.d, n);
        for (col, indices) in indices_batch.iter().enumerate() {
            assert_eq!(indices.len(), self.d);
            for row in 0..self.d {
                y_tilde_batch[(row, col)] = self.centroids[indices[row] as usize];
            }
        }

        gemm(&self.rotation, true, &y_tilde_batch, false)
    }
}
