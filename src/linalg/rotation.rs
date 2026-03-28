use nalgebra::{DMatrix, QR};
use rand::{CryptoRng, Rng};
use rand_distr::{StandardNormal, Distribution};

/// Generates a random rotation matrix Π (d x d)
/// This is done by:
/// 1. Creating a d x d matrix with i.i.d N(0, 1) entries
/// 2. Computing its QR decomposition
/// 3. Extracting Q, which is uniformly distributed over the orthogonal group O(d)
pub fn generate_random_rotation<R: Rng + CryptoRng>(d: usize, rng: &mut R) -> DMatrix<f64> {
    let mut mat = DMatrix::zeros(d, d);
    let dist = StandardNormal;
    
    // Fill with N(0,1)
    for i in 0..d {
        for j in 0..d {
            mat[(i, j)] = dist.sample(rng);
        }
    }
    
    // Compute QR decomposition
    let qr = QR::new(mat);
    
    // Extract Q matrix
    let q = qr.q();
    
    // Q is an orthogonal matrix, so it's a valid rotation matrix (or reflection)
    q
}

/// Generates a random projection matrix S (d x d) with iid N(0, 1) entries
pub fn generate_projection_matrix<R: Rng + CryptoRng>(d: usize, rng: &mut R) -> DMatrix<f64> {
    let mut mat = DMatrix::zeros(d, d);
    let dist = StandardNormal;
    
    // Fill with N(0,1)
    for i in 0..d {
        for j in 0..d {
            mat[(i, j)] = dist.sample(rng);
        }
    }
    
    mat
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_rotation_matrix_properties() {
        let d = 10;
        let mut rng = StdRng::seed_from_u64(42);
        let q = generate_random_rotation(d, &mut rng);
        
        // Orthogonality: Q^T * Q should be Identity
        let qt_q = q.transpose() * &q;
        let identity: DMatrix<f64> = DMatrix::identity(d, d);
        
        for i in 0..d {
            for j in 0..d {
                let val: f64 = qt_q[(i, j)] - identity[(i, j)];
                assert!(val.abs() < 1e-10, "Failed orthogonality at ({},{})", i, j);
            }
        }
    }
}
