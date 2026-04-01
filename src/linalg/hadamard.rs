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
/// 1. Multiply by random signs (Diagonal matrix D)
/// 2. Apply FWHT (Hadamard matrix H)
/// 3. Shuffle (Permutation matrix P)
/// Output is normalized by 1/sqrt(d)
pub fn srht(x: &[f32], signs: &[f32], out: &mut [f32]) {
    let d = x.len();
    let n = d.next_power_of_two();

    // Copy and pad to power of 2
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fwht_power_of_2() {
        let mut a = vec![1.0, 1.0, 1.0, 1.0];
        fwht(&mut a);
        // H4 = [[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]
        // [1,1,1,1] * H4 = [4, 0, 0, 0]
        assert_eq!(a, vec![4.0, 0.0, 0.0, 0.0]);
    }
}
