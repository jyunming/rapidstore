/// In-place Fast Walsh-Hadamard Transform
/// Input length must be a power of 2.
pub fn fwht(a: &mut [f32]) {
    let n = a.len();
    assert!(n.is_power_of_two(), "FWHT requires power-of-2 length");

    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let x = a[j];
                let y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
        }
        h *= 2;
    }
}

/// SRHT (Structured Random Hadamard Transform)
/// Forward: H * D * x
pub fn srht(x: &[f32], signs: &[f32], out: &mut [f32]) {
    let d = x.len();
    let n = d.next_power_of_two();
    
    let mut temp = vec![0.0f32; n];
    for i in 0..d {
        temp[i] = x[i] * signs[i];
    }
    
    fwht(&mut temp);
    
    let norm = 1.0 / (n as f32).sqrt();
    for i in 0..d {
        out[i] = temp[i] * norm;
    }
}

/// Inverse SRHT — exact when `d` is a power of 2 (`d == d.next_power_of_two()`).
/// For non-power-of-2 `d`, the forward SRHT takes only d of n Hadamard rows,
/// making the map non-orthogonal; this function gives only an approximation in that case.
pub fn inverse_srht(y: &[f32], signs: &[f32], out: &mut [f32]) {
    let d = y.len();
    let n = d.next_power_of_two();
    
    let mut temp = vec![0.0f32; n];
    for i in 0..d {
        temp[i] = y[i];
    }
    
    fwht(&mut temp);
    
    let norm = 1.0 / (n as f32).sqrt();
    for i in 0..d {
        out[i] = temp[i] * norm * signs[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fwht_power_of_2() {
        let mut a = vec![1.0, 1.0, 1.0, 1.0];
        fwht(&mut a);
        assert_eq!(a, vec![4.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_srht_roundtrip() {
        // d must be a power of 2 for the SRHT to be exactly orthogonal.
        let d = 8;
        let x: Vec<f32> = (1..=d as i32).map(|v| v as f32).collect();
        let signs = vec![1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
        let mut y = vec![0.0; d];
        let mut x_hat = vec![0.0; d];

        srht(&x, &signs, &mut y);
        inverse_srht(&y, &signs, &mut x_hat);

        for i in 0..d {
            assert!((x[i] - x_hat[i]).abs() < 1e-5, "mismatch at i={}: {} vs {}", i, x[i], x_hat[i]);
        }
    }
}
