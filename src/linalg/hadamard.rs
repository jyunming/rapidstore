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
/// Forward: H * D * (x, 0)
/// out must be length n = next_power_of_two(x.len())
pub fn srht(x: &[f32], signs: &[f32], out: &mut [f32]) {
    let d = x.len();
    let n = d.next_power_of_two();
    assert_eq!(out.len(), n);
    assert!(signs.len() >= n);
    
    for i in 0..d {
        out[i] = x[i] * signs[i];
    }
    for i in d..n {
        out[i] = 0.0;
    }
    
    fwht(out);
    
    let norm = 1.0 / (n as f32).sqrt();
    for i in 0..n {
        out[i] *= norm;
    }
}

/// Inverse SRHT
/// Inverse of (H * D): D * H
/// out must be length d, y must be length n = next_power_of_two(d)
pub fn inverse_srht(y: &[f32], signs: &[f32], out: &mut [f32]) {
    let d = out.len();
    let n = d.next_power_of_two();
    assert_eq!(y.len(), n);
    assert!(signs.len() >= n);
    
    let mut temp = y.to_vec();
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
        let d: usize = 5;
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let n = d.next_power_of_two(); // 8
        let signs = vec![1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0];
        let mut y = vec![0.0; n];
        let mut x_hat = vec![0.0; d];
        
        srht(&x, &signs, &mut y);
        inverse_srht(&y, &signs, &mut x_hat);
        
        for i in 0..d {
            assert!((x[i] - x_hat[i]).abs() < 1e-5);
        }
    }
}
