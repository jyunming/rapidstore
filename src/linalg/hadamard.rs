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

    // Thread-local temp buffer: avoids one heap allocation per dequantized vector,
    // which matters when dequantize_batch is called for reranking candidates.
    std::thread_local! {
        static SRHT_TEMP: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    SRHT_TEMP.with(|cell| {
        let mut temp = cell.borrow_mut();
        if temp.len() < n {
            temp.resize(n, 0.0);
        }
        temp[..n].copy_from_slice(y);
        fwht(&mut temp[..n]);
        let norm = 1.0 / (n as f32).sqrt();
        for i in 0..d {
            out[i] = temp[i] * norm * signs[i];
        }
    });
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

    /// Directly tests the scalar FWHT path (avoids AVX2 dispatch).
    #[test]
    fn fwht_scalar_all_ones_size4() {
        let mut a = vec![1.0f32, 1.0, 1.0, 1.0];
        fwht_scalar(&mut a);
        assert_eq!(a, vec![4.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn fwht_scalar_size_2() {
        let mut a = vec![3.0f32, 1.0];
        fwht_scalar(&mut a);
        assert_eq!(a, vec![4.0, 2.0]);
    }

    #[test]
    fn fwht_scalar_size_8_all_ones() {
        let mut a = vec![1.0f32; 8];
        fwht_scalar(&mut a);
        assert_eq!(a[0], 8.0);
        for i in 1..8 {
            assert_eq!(a[i], 0.0, "element {} should be 0", i);
        }
    }

    #[test]
    fn fwht_scalar_involution() {
        // Applying FWHT twice gives n * original (it is its own inverse up to scale)
        let orig = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut a = orig.clone();
        fwht_scalar(&mut a);
        fwht_scalar(&mut a);
        let n = orig.len() as f32;
        for i in 0..orig.len() {
            let diff = (a[i] - n * orig[i]).abs();
            assert!(
                diff < 1e-5,
                "FWHT^2 mismatch at {}: {} vs {}",
                i,
                a[i],
                n * orig[i]
            );
        }
    }

    #[test]
    fn srht_preserves_norm() {
        let d = 4;
        let x = vec![1.0f32, 0.0, 0.0, 0.0];
        let signs = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 4];
        srht(&x, &signs, &mut out);
        let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0f32).abs() < 0.01,
            "SRHT should preserve norm: {}",
            norm
        );
    }

    #[test]
    fn srht_padding_with_non_power_of_two_d() {
        let d: usize = 3;
        let x = vec![1.0f32, 2.0, 3.0];
        let signs = vec![1.0f32, -1.0, 1.0, 1.0];
        let n = d.next_power_of_two();
        let mut out = vec![0.0f32; n];
        srht(&x, &signs, &mut out);
        assert_eq!(out.len(), n);
        assert!(out.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn inverse_srht_roundtrip_non_power_of_two() {
        let d: usize = 5;
        let x = vec![2.0f32, -1.0, 0.5, 3.0, -0.5];
        let n = d.next_power_of_two();
        let signs: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let mut y = vec![0.0f32; n];
        srht(&x, &signs, &mut y);
        let mut x_hat = vec![0.0f32; d];
        inverse_srht(&y, &signs, &mut x_hat);
        for i in 0..d {
            assert!(
                (x[i] - x_hat[i]).abs() < 1e-5,
                "mismatch at {}: {} vs {}",
                i,
                x[i],
                x_hat[i]
            );
        }
    }
}
