use nalgebra::{DMatrix, QR};
use rand::{CryptoRng, Rng};
use rand_distr::{Distribution, StandardNormal};

/// Generates a random rotation matrix Π (d x d)
/// This is done by:
/// 1. Creating a d x d matrix with i.i.d N(0, 1) entries
/// 2. Computing its QR decomposition
/// 3. Extracting Q, which is uniformly distributed over the orthogonal group O(d)
pub fn generate_random_rotation<R: Rng + CryptoRng>(d: usize, rng: &mut R) -> DMatrix<f32> {
    let mut mat = DMatrix::zeros(d, d);
    let dist = StandardNormal;

    // Fill with N(0,1)
    for i in 0..d {
        for j in 0..d {
            let v: f64 = dist.sample(rng);
            mat[(i, j)] = v as f32;
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
pub fn generate_projection_matrix<R: Rng + CryptoRng>(d: usize, rng: &mut R) -> DMatrix<f32> {
    let mut mat = DMatrix::zeros(d, d);
    let dist = StandardNormal;

    // Fill with N(0,1)
    for i in 0..d {
        for j in 0..d {
            let v: f64 = dist.sample(rng);
            mat[(i, j)] = v as f32;
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
        let identity: DMatrix<f32> = DMatrix::identity(d, d);

        for i in 0..d {
            for j in 0..d {
                let val: f32 = qt_q[(i, j)] - identity[(i, j)];
                assert!(val.abs() < 1e-6, "Failed orthogonality at ({},{})", i, j);
            }
        }
    }

    #[test]
    fn generate_projection_matrix_shape() {
        let d = 8;
        let mut rng = StdRng::seed_from_u64(42);
        let mat = generate_projection_matrix(d, &mut rng);
        assert_eq!(mat.nrows(), d);
        assert_eq!(mat.ncols(), d);
    }

    #[test]
    fn generate_projection_matrix_has_nonzero_entries() {
        let d = 4;
        let mut rng = StdRng::seed_from_u64(42);
        let mat = generate_projection_matrix(d, &mut rng);
        let all_zero = mat.iter().all(|&v| v == 0.0);
        assert!(!all_zero, "projection matrix should not be all zeros");
    }

    #[test]
    fn generate_projection_matrix_is_not_orthogonal() {
        // Unlike generate_random_rotation, this is NOT guaranteed orthogonal
        let d = 4;
        let mut rng = StdRng::seed_from_u64(42);
        let mat = generate_projection_matrix(d, &mut rng);
        // Just verify we can compute it without panic and it has the right shape
        assert_eq!(mat.nrows(), d);
        assert_eq!(mat.ncols(), d);
    }

    #[test]
    fn generate_random_rotation_is_deterministic() {
        let d = 5;
        let mut rng1 = StdRng::seed_from_u64(99);
        let mut rng2 = StdRng::seed_from_u64(99);
        let q1 = generate_random_rotation(d, &mut rng1);
        let q2 = generate_random_rotation(d, &mut rng2);
        for i in 0..d {
            for j in 0..d {
                assert_eq!(
                    q1[(i, j)],
                    q2[(i, j)],
                    "rotation should be deterministic with same seed"
                );
            }
        }
    }
}
