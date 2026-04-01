use nalgebra::DMatrix;
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::CodeIndex;
use super::codebook::lloyd_max;
use crate::linalg::hadamard::{fwht, srht};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MseQuantizer {
    pub d: usize,
    pub b: usize,
    pub rotation_signs: Vec<f32>,
    pub centroids: Vec<f32>,
}

impl MseQuantizer {
    pub fn new(d: usize, b: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut rotation_signs = vec![0.0f32; d];
        for s in &mut rotation_signs {
            *s = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
        }

        let num_points = 20_000;
        let centroids: Vec<f32> = lloyd_max(b, d, num_points)
            .into_iter()
            .map(|c| c as f32)
            .collect();
        assert!(
            centroids.len() <= u16::MAX as usize + 1,
            "codebook too large for u16 indices; reduce bits"
        );

        Self {
            d,
            b,
            rotation_signs,
            centroids,
        }
    }

    pub fn quantize(&self, x: &Array1<f64>) -> Vec<CodeIndex> {
        assert_eq!(x.len(), self.d);
        let x_f32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let mut y = vec![0.0f32; self.d];
        srht(&x_f32, &self.rotation_signs, &mut y);

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
        all_indices
            .par_iter_mut()
            .enumerate()
            .for_each(|(col, idxs)| {
                let x_col_owned: Vec<f32> = xs.column(col).iter().copied().collect();
                let mut y = vec![0.0f32; self.d];
                srht(&x_col_owned, &self.rotation_signs, &mut y);
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
        let n = d.next_power_of_two();

        let mut temp = vec![0.0f32; n];
        for i in 0..d {
            temp[i] = self.centroids[indices[i] as usize];
        }
        // Inverse SRHT: forward is y = (1/√n)·H·D·x
        // Inverse: x = (1/√n)·D·H·y — apply H first (no sign multiply), then D
        fwht(&mut temp);
        let norm = 1.0 / (n as f32).sqrt();
        let mut out = Array1::zeros(d);
        for i in 0..d {
            out[i] = (temp[i] * self.rotation_signs[i] * norm) as f64;
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
        let signs = &self.rotation_signs;
        let centroids = &self.centroids;

        let pad = d.next_power_of_two();
        let norm = 1.0 / (pad as f32).sqrt();
        for (col, indices) in indices_batch.iter().enumerate() {
            let mut temp = vec![0.0f32; pad];
            for row in 0..d {
                temp[row] = centroids[indices[row] as usize];
            }
            // Inverse SRHT: apply H first (no sign multiply), then D·(1/√n)
            fwht(&mut temp);
            for row in 0..d {
                out[(row, col)] = temp[row] * signs[row] * norm;
            }
        }
        out
    }
}
