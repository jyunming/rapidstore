// mse.rs
use ndarray::{Array1, Array2, Axis};
use nalgebra::DMatrix;
use rand::{SeedableRng};
use rand::rngs::StdRng;
use std::collections::HashMap;

use super::codebook::{beta_pdf, lloyd_max, expected_mse};

/// TurboQuant_mse Quantizer structure
pub struct MseQuantizer {
    pub d: usize,
    pub b: usize,
    pub rotation: DMatrix<f64>,
    pub centroids: Vec<f64>,
}

impl MseQuantizer {
    /// Creates a new MseQuantizer with the specified dimension and bit-width.
    pub fn new(d: usize, b: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        
        // 1. Generate Rotation Matrix using Linalg module
        let rotation = crate::linalg::rotation::generate_random_rotation(d, &mut rng);
        
        // 2. Compute Lloyd-Max Codebook for beta distribution
        let num_points = 20_000;
        let centroids = lloyd_max(b, d, num_points);
        
        Self { d, b, rotation, centroids }
    }

    /// Quantizes a single input vector.
    /// Input x: D-dimensional vector (should be unit length for optimal theoretical match).
    /// Output: b-bit indices for each coordinate.
    pub fn quantize(&self, x: &Array1<f64>) -> Vec<usize> {
        assert_eq!(x.len(), self.d);
        
        // Convert to nalgebra for matrix multiplication
        let mut nalgebra_x = DMatrix::zeros(self.d, 1);
        for i in 0..self.d {
            nalgebra_x[(i, 0)] = x[i];
        }
        
        // y = Π * x
        let y = &self.rotation * nalgebra_x;
        
        // Quantize each coordinate using the codebook
        let mut indices = Vec::with_capacity(self.d);
        for i in 0..self.d {
            let val = y[(i, 0)];
            // Find nearest centroid
            let mut min_dist = f64::MAX;
            let mut min_idx = 0;
            for (idx, &c) in self.centroids.iter().enumerate() {
                let dist = (val - c).abs();
                if dist < min_dist {
                    min_dist = dist;
                    min_idx = idx;
                }
            }
            indices.push(min_idx);
        }
        
        indices
    }

    /// Dequantizes the index vector back to a real vector.
    pub fn dequantize(&self, indices: &[usize]) -> Array1<f64> {
        assert_eq!(indices.len(), self.d);
        
        // Construct y_tilde from codebook
        let mut y_tilde = DMatrix::zeros(self.d, 1);
        for i in 0..self.d {
            y_tilde[(i, 0)] = self.centroids[indices[i]];
        }
        
        // x_tilde = Π^T * y_tilde
        let x_tilde_nalg = self.rotation.transpose() * y_tilde;
        
        // Convert back to ndarray
        let mut x_tilde = Array1::zeros(self.d);
        for i in 0..self.d {
            x_tilde[i] = x_tilde_nalg[(i, 0)];
        }
        
        x_tilde
    }
}
