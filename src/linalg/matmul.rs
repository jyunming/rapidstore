use nalgebra::DMatrix;

pub fn gemm(a: &DMatrix<f64>, trans_a: bool, b: &DMatrix<f64>, trans_b: bool) -> DMatrix<f64> {
    let (m, k_a, rsa, csa) = if trans_a {
        (a.ncols(), a.nrows(), a.nrows() as isize, 1isize)
    } else {
        (a.nrows(), a.ncols(), 1isize, a.nrows() as isize)
    };

    let (k_b, n, rsb, csb) = if trans_b {
        (b.ncols(), b.nrows(), b.nrows() as isize, 1isize)
    } else {
        (b.nrows(), b.ncols(), 1isize, b.nrows() as isize)
    };

    assert_eq!(
        k_a, k_b,
        "gemm dimension mismatch: {}x{} times {}x{}",
        m, k_a, k_b, n
    );

    let mut c = DMatrix::<f64>::zeros(m, n);
    unsafe {
        matrixmultiply::dgemm(
            m,
            k_a,
            n,
            1.0,
            a.as_ptr(),
            rsa,
            csa,
            b.as_ptr(),
            rsb,
            csb,
            0.0,
            c.as_mut_ptr(),
            1isize,
            m as isize,
        );
    }
    c
}

pub fn sgemm(a: &DMatrix<f32>, trans_a: bool, b: &DMatrix<f32>, trans_b: bool) -> DMatrix<f32> {
    let (m, k_a, rsa, csa) = if trans_a {
        (a.ncols(), a.nrows(), a.nrows() as isize, 1isize)
    } else {
        (a.nrows(), a.ncols(), 1isize, a.nrows() as isize)
    };

    let (k_b, n, rsb, csb) = if trans_b {
        (b.ncols(), b.nrows(), b.nrows() as isize, 1isize)
    } else {
        (b.nrows(), b.ncols(), 1isize, b.nrows() as isize)
    };

    assert_eq!(
        k_a, k_b,
        "sgemm dimension mismatch: {}x{} times {}x{}",
        m, k_a, k_b, n
    );

    let mut c = DMatrix::<f32>::zeros(m, n);
    unsafe {
        matrixmultiply::sgemm(
            m,
            k_a,
            n,
            1.0,
            a.as_ptr(),
            rsa,
            csa,
            b.as_ptr(),
            rsb,
            csb,
            0.0,
            c.as_mut_ptr(),
            1isize,
            m as isize,
        );
    }
    c
}
#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    fn make_f64(rows: usize, cols: usize, values: &[f64]) -> DMatrix<f64> {
        DMatrix::from_row_slice(rows, cols, values)
    }

    fn make_f32(rows: usize, cols: usize, values: &[f32]) -> DMatrix<f32> {
        DMatrix::from_row_slice(rows, cols, values)
    }

    #[test]
    fn gemm_no_transpose_identity() {
        let a = DMatrix::<f64>::identity(3, 3);
        let b = make_f64(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let c = gemm(&a, false, &b, false);
        assert_eq!(c.nrows(), 3);
        assert_eq!(c.ncols(), 2);
        for i in 0..3 {
            for j in 0..2 {
                assert!((c[(i, j)] - b[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn gemm_transpose_a() {
        let a = make_f64(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = make_f64(2, 1, &[1.0, 1.0]);
        let c = gemm(&a, true, &b, false);
        assert_eq!(c.nrows(), 3);
        assert_eq!(c.ncols(), 1);
        assert!((c[(0, 0)] - 5.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 7.0).abs() < 1e-10);
        assert!((c[(2, 0)] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn gemm_transpose_b() {
        let a = make_f64(1, 3, &[1.0, 2.0, 3.0]);
        let b = make_f64(1, 3, &[4.0, 5.0, 6.0]);
        let c = gemm(&a, false, &b, true);
        assert!((c[(0, 0)] - 32.0).abs() < 1e-10);
    }

    #[test]
    fn gemm_both_transpose() {
        let a = make_f64(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = make_f64(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let c = gemm(&a, true, &b, true);
        assert_eq!(c.nrows(), 2);
        assert_eq!(c.ncols(), 2);
        assert!(c.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sgemm_no_transpose_identity() {
        let a = DMatrix::<f32>::identity(3, 3);
        let b = make_f32(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let c = sgemm(&a, false, &b, false);
        for i in 0..3 {
            for j in 0..2 {
                assert!((c[(i, j)] - b[(i, j)]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn sgemm_transpose_a() {
        let a = make_f32(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = make_f32(2, 1, &[1.0, 1.0]);
        let c = sgemm(&a, true, &b, false);
        assert_eq!(c.nrows(), 3);
        assert_eq!(c.ncols(), 1);
        assert!((c[(0, 0)] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn sgemm_transpose_b() {
        let a = make_f32(1, 3, &[1.0, 2.0, 3.0]);
        let b = make_f32(1, 3, &[4.0, 5.0, 6.0]);
        let c = sgemm(&a, false, &b, true);
        assert!((c[(0, 0)] - 32.0).abs() < 1e-5);
    }

    #[test]
    fn sgemm_both_transpose() {
        let a = make_f32(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let b = make_f32(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let c = sgemm(&a, true, &b, true);
        assert!(c.iter().all(|v| v.is_finite()));
    }
}
