use std::f64::consts::PI;

/// Numerically stable log-gamma using Lanczos approximation (g=7, n=8 coefficients).
///
/// This is equivalent to the standard Wikipedia/Numerical Recipes formulation with a
/// `z -= 1` pre-shift: using `c[i] / (z + i)` with i starting at 0 is algebraically
/// identical to `c[i] / ((z-1) + i + 1)` with i starting at 0. The final formula
/// `(z - 0.5) * t.ln()` and `t = z + 6.5` are derived from that same substitution.
/// Verified: log_gamma(1) ≈ 0, log_gamma(2) ≈ 0, log_gamma(5) ≈ ln(24) = 3.178.
///
/// **Precondition:** `z > 0`. Negative or zero inputs produce NaN via the reflection
/// branch (sin can be zero or negative). Callers must ensure `z > 0`.
fn log_gamma(z: f64) -> f64 {
    debug_assert!(z > 0.0, "log_gamma undefined for z <= 0, got {z}");
    // Reflection for z in (0, 0.5) keeps the Lanczos series in its accurate region.
    if z < 0.5 {
        return (PI / (PI * z).sin()).ln() - log_gamma(1.0 - z);
    }
    let p = [
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278226905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let mut y = 0.99999999999980993;
    for (i, &val) in p.iter().enumerate() {
        y += val / (z + i as f64);
    }
    let t = z + 6.5;
    0.5 * (2.0 * PI).ln() + (z - 0.5) * t.ln() - t + y.ln()
}

/// Evaluates the Beta distribution PDF f_X(x) for the paper's coordinate distribution.
/// The distribution is scaled to [-1, 1].
/// f_X(x) = (\Gamma(d/2) / (\sqrt{\pi} \Gamma((d-1)/2))) * (1 - x^2)^((d-3)/2)
///
/// Returns 0.0 for `d < 2` (the gamma argument `(d-1)/2` would be ≤ 0, making
/// log_gamma undefined) and for `|x| >= 1.0`.
pub fn beta_pdf(x: f64, d: usize) -> f64 {
    if d < 2 || x.abs() >= 1.0 {
        return 0.0;
    }
    let df = d as f64;
    let log_coeff = log_gamma(df / 2.0) - 0.5 * PI.ln() - log_gamma((df - 1.0) / 2.0);
    let base = (1.0 - x * x).max(0.0);
    if base == 0.0 {
        return 0.0;
    }
    let log_term = ((df - 3.0) / 2.0) * base.ln();
    (log_coeff + log_term).exp()
}

/// Computes the optimal Lloyd-Max quantizer centroids for a given distribution and bit-width.
/// Approximates the continuous k-means using a fine grid.
pub fn lloyd_max(b: usize, d: usize, num_points: usize) -> Vec<f64> {
    let num_centroids = 1 << b; // 2^b

    // Create a fine grid points in [-1, 1]
    let step = 2.0 / (num_points as f64 - 1.0);
    let grid: Vec<(f64, f64)> = (0..num_points)
        .map(|i| {
            let x = -1.0 + i as f64 * step;
            let prob = beta_pdf(x, d) * step;
            (x, prob)
        })
        // Filter out effectively zero probabilities for stability
        .filter(|(_, p)| *p > 1e-10)
        .collect();

    // 1. Initial guess: uniformly distribute quantiles or roughly uniform space
    // Let's use uniform spacing within inner range based on variance (~1/d for large d)
    let bound = (3.0 / (d as f64).sqrt()).min(0.99); // roughly +/- 3 std deviations
    let mut centroids: Vec<f64> = (0..num_centroids)
        .map(|i| -bound + (2.0 * bound) * (i as f64 + 0.5) / (num_centroids as f64))
        .collect();

    let mut boundaries = vec![0.0; num_centroids - 1];

    // Lloyd-Max iterations
    let max_iter = 1000;
    let tol = 1e-7;
    for _iter in 0..max_iter {
        // Compute boundaries: midpoints between centroids
        for i in 0..(num_centroids - 1) {
            boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
        }

        let mut next_centroids = vec![0.0; num_centroids];
        let mut masses = vec![0.0; num_centroids];

        // Assign points to clusters and compute updated centroids
        for &(x, p) in &grid {
            // Find cluster
            let mut cluster_idx = 0;
            for i in 0..(num_centroids - 1) {
                if x > boundaries[i] {
                    cluster_idx = i + 1;
                }
            }

            next_centroids[cluster_idx] += x * p;
            masses[cluster_idx] += p;
        }

        let mut max_diff = 0.0_f64;
        for i in 0..num_centroids {
            if masses[i] > 0.0 {
                let new_c = next_centroids[i] / masses[i];
                max_diff = max_diff.max((new_c - centroids[i]).abs());
                centroids[i] = new_c;
            }
        }

        if max_diff < tol {
            break;
        }
    }

    centroids
}

/// Calculates the expected MSE configuration for given centroids under the distribution.
pub fn expected_mse(centroids: &[f64], d: usize, num_points: usize) -> f64 {
    let step = 2.0 / (num_points as f64 - 1.0);
    let mut var_1d = 0.0;

    for i in 0..num_points {
        let x = -1.0 + i as f64 * step;
        let p = beta_pdf(x, d) * step;

        let nearest = centroids
            .iter()
            .min_by(|a, b| (x - **a).abs().partial_cmp(&(x - **b).abs()).unwrap())
            .unwrap();

        var_1d += (x - *nearest).powi(2) * p;
    }

    var_1d * d as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_gamma_known_values() {
        // Γ(1) = 0! = 1  →  log_gamma(1) = 0
        assert!(
            (log_gamma(1.0)).abs() < 1e-9,
            "log_gamma(1) = {}",
            log_gamma(1.0)
        );
        // Γ(2) = 1! = 1  →  log_gamma(2) = 0
        assert!(
            (log_gamma(2.0)).abs() < 1e-9,
            "log_gamma(2) = {}",
            log_gamma(2.0)
        );
        // Γ(5) = 4! = 24  →  log_gamma(5) = ln(24) ≈ 3.17805...
        let expected = 24.0_f64.ln();
        assert!(
            (log_gamma(5.0) - expected).abs() < 1e-7,
            "log_gamma(5) = {}, expected {}",
            log_gamma(5.0),
            expected
        );
        // Γ(0.5) = sqrt(π)  →  log_gamma(0.5) ≈ 0.57236...
        let expected_half = std::f64::consts::PI.sqrt().ln();
        assert!(
            (log_gamma(0.5) - expected_half).abs() < 1e-7,
            "log_gamma(0.5) = {}, expected {}",
            log_gamma(0.5),
            expected_half
        );
    }

    #[test]
    fn log_gamma_reflection_formula_z_less_than_half() {
        // Exercises the z < 0.5 reflection branch.
        // By reflection: Γ(z)·Γ(1-z) = π/sin(πz)
        // → log_gamma(z) + log_gamma(1-z) = ln(π) - ln(sin(πz))
        use std::f64::consts::PI;
        for &z in &[0.1_f64, 0.2, 0.3, 0.4, 0.49] {
            let lg_z = log_gamma(z); // triggers reflection branch
            let lg_1mz = log_gamma(1.0 - z); // normal branch
            let expected = (PI / (PI * z).sin()).ln();
            let actual = lg_z + lg_1mz;
            assert!(
                (actual - expected).abs() < 1e-7,
                "reflection identity failed for z={z}: got {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_beta_pdf_integral() {
        let d = 1536;
        let num_points = 100_000;
        let step = 2.0 / (num_points as f64 - 1.0);
        let mut sum = 0.0;
        for i in 0..num_points {
            let x = -1.0 + i as f64 * step;
            sum += beta_pdf(x, d) * step;
        }
        // Integral under PDF should be 1
        assert!((sum - 1.0).abs() < 1e-3, "Integral is {}", sum);
    }

    #[test]
    fn test_lloyd_max_b1() {
        let d = 1536;
        let centroids = lloyd_max(1, d, 20_000);
        // For b=1, large d, should be around +/- sqrt(2 / (pi * d))
        let expected = (2.0 / (PI * d as f64)).sqrt();
        assert!(
            (centroids[1] - expected).abs() < 1e-3,
            "Expected ~{}, got {}",
            expected,
            centroids[1]
        );
        assert!((centroids[0] - (-expected)).abs() < 1e-3);
    }

    #[test]
    fn test_expected_mse_bounds() {
        // Test paper's MSE bounds for low bit-widths
        let d = 1536;
        let n_pts = 100_000;

        let cents_b1 = lloyd_max(1, d, n_pts);
        let mse_b1 = expected_mse(&cents_b1, d, n_pts);

        // Let's print out what we found
        println!("b=1 MSE: {}", mse_b1);

        // Assert within ~5% of paper's 0.36
        assert!((mse_b1 - 0.36).abs() < 0.05);
    }

    #[test]
    fn beta_pdf_out_of_range_returns_zero() {
        // Exercises the early-return branch: |x| >= 1.0
        assert_eq!(beta_pdf(1.0, 16), 0.0);
        assert_eq!(beta_pdf(-1.0, 16), 0.0);
        assert_eq!(beta_pdf(1.5, 16), 0.0);
        assert_eq!(beta_pdf(-2.0, 16), 0.0);
    }
}
