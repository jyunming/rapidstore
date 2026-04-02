/// In-place Fast Walsh-Hadamard Transform
/// Input length must be a power of 2.
pub fn fwht(a: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: we just checked the feature flag above
            return unsafe { fwht_avx2(a) };
        }
    }
    fwht_scalar(a);
}

fn fwht_scalar(a: &mut [f32]) {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fwht_avx2(a: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = a.len();
    assert!(n.is_power_of_two(), "FWHT requires power-of-2 length");

    let ptr = a.as_mut_ptr();
    let mut h = 1usize;

    while h < n {
        if h >= 8 {
            // AVX2 path: process 8 butterfly pairs per iteration
            let full_blocks = h / 8;
            let rem = h % 8;
            for i in (0..n).step_by(h * 2) {
                // Full 8-wide blocks
                for b in 0..full_blocks {
                    let lo = ptr.add(i + b * 8);
                    let hi = ptr.add(i + h + b * 8);
                    let x = _mm256_loadu_ps(lo);
                    let y = _mm256_loadu_ps(hi);
                    _mm256_storeu_ps(lo, _mm256_add_ps(x, y));
                    _mm256_storeu_ps(hi, _mm256_sub_ps(x, y));
                }
                // Scalar remainder
                let base = full_blocks * 8;
                for j in base..base + rem {
                    let lo = i + j;
                    let hi = lo + h;
                    let x = *ptr.add(lo);
                    let y = *ptr.add(hi);
                    *ptr.add(lo) = x + y;
                    *ptr.add(hi) = x - y;
                }
            }
        } else {
            // Scalar path for h < 8
            for i in (0..n).step_by(h * 2) {
                for j in i..i + h {
                    let x = a[j];
                    let y = a[j + h];
                    a[j] = x + y;
                    a[j + h] = x - y;
                }
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

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: feature checked above
        unsafe { scale_avx2(out, norm) };
        return;
    }
    for v in out.iter_mut() {
        *v *= norm;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn scale_avx2(a: &mut [f32], scale: f32) {
    use std::arch::x86_64::*;
    let n = a.len();
    let sv = _mm256_set1_ps(scale);
    let ptr = a.as_mut_ptr();
    let full = n / 8;
    for i in 0..full {
        let v = _mm256_loadu_ps(ptr.add(i * 8));
        _mm256_storeu_ps(ptr.add(i * 8), _mm256_mul_ps(v, sv));
    }
    for i in full * 8..n {
        *ptr.add(i) *= scale;
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
