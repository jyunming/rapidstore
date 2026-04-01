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
