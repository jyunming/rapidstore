use ndarray::{Array1, Array2, Axis};
use nalgebra::DMatrix;
use rand::{SeedableRng};
use rand::rngs::StdRng;
use serde::{Serialize, Deserialize};
use std::f64::consts::PI;

/// QJL Quantizer structure (1-bit inner product quantization on residual)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct QjlQuantizer {
    pub d: usize,
    pub projection: DMatrix<f64>,
}

impl QjlQuantizer {
    pub fn new(d: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        // S matrix is d x d, i.i.d N(0,1)
        let projection = crate::linalg::rotation::generate_projection_matrix(d, &mut rng);
        Self { d, projection }
    }

    /// Quantize residual vector into a bit vector
    /// Return is represented as Vec<i8> containing +1 or -1
    pub fn quantize(&self, r: &Array1<f64>) -> Vec<i8> {
        assert_eq!(r.len(), self.d);
        
        // Convert to nalgebra
        let mut r_nalg = DMatrix::zeros(self.d, 1);
        for i in 0..self.d {
            r_nalg[(i, 0)] = r[i];
        }
        
        // S * r
        let s_r = &self.projection * r_nalg;
        
        // Sign function entry-wise
        let mut qjl = Vec::with_capacity(self.d);
        for i in 0..self.d {
            if s_r[(i, 0)] >= 0.0 {
                qjl.push(1);
            } else {
                qjl.push(-1);
            }
        }
        
        qjl
    }

    /// Dequantize QJL string
    /// x_tilde = sqrt(pi / 2d) * gamma * S^T * qjl
    pub fn dequantize(&self, qjl: &[i8], gamma: f64) -> Array1<f64> {
        assert_eq!(qjl.len(), self.d);
        
        // Convert qjl to f64 nalgebra vector
        let mut qjl_nalg = DMatrix::zeros(self.d, 1);
        for i in 0..self.d {
            qjl_nalg[(i, 0)] = qjl[i] as f64;
        }
        
        let multiplier = (PI / (2.0 * self.d as f64)).sqrt() * gamma;
        
        // S^T * qjl
        let st_qjl = self.projection.transpose() * qjl_nalg;
        
        // Construct final array
        let mut result = Array1::zeros(self.d);
        for i in 0..self.d {
            result[i] = multiplier * st_qjl[(i, 0)];
        }
        
        result
    }
}
