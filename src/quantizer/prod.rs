use ndarray::Array1;

use super::mse::MseQuantizer;
use super::qjl::QjlQuantizer;

/// TurboQuant_prod Quantizer structure
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
        // Use a different seed for the projection matrix to ensure independence
        let qjl_quantizer = QjlQuantizer::new(d, seed ^ 0xdeadbeef);
        
        Self {
            d,
            b,
            mse_quantizer,
            qjl_quantizer,
        }
    }

    /// Quantize a vector into (MSE indices, QJL bits, residual norm gamma)
    pub fn quantize(&self, x: &Array1<f64>) -> (Vec<usize>, Vec<i8>, f64) {
        // 1. Quantize with MSE (b-1 bits)
        let idx = self.mse_quantizer.quantize(x);
        
        // 2. Residual vector r = x - dequant_mse(idx)
        let x_tilde_mse = self.mse_quantizer.dequantize(&idx);
        let mut r = Array1::zeros(self.d);
        for i in 0..self.d {
            r[i] = x[i] - x_tilde_mse[i];
        }
        
        // 3. Compute gamma = L2 norm of residual vector
        let mut gamma = 0.0;
        for i in 0..self.d {
            gamma += r[i] * r[i];
        }
        gamma = gamma.sqrt();
        
        // 4. QJL on residual
        let qjl = self.qjl_quantizer.quantize(&r);
        
        (idx, qjl, gamma)
    }

    /// Dequantizes the components back into a real vector.
    pub fn dequantize(&self, idx: &[usize], qjl: &[i8], gamma: f64) -> Array1<f64> {
        // 1. MSE dequantization
        let x_tilde_mse = self.mse_quantizer.dequantize(idx);
        
        // 2. QJL dequantization
        let x_tilde_qjl = self.qjl_quantizer.dequantize(qjl, gamma);
        
        // 3. Combine: x_tilde = x_tilde_mse + x_tilde_qjl
        let mut result = Array1::zeros(self.d);
        for i in 0..self.d {
            result[i] = x_tilde_mse[i] + x_tilde_qjl[i];
        }
        
        result
    }
}
