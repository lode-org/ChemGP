//! Configuration distance metrics.
//!
//! Ports `distances.jl`: interatomic_distances, max_1d_log_distance, rmsd_distance.

/// Pairwise interatomic distances from a flat coordinate vector.
/// N atoms -> N*(N-1)/2 distances in canonical order (j < i).
pub fn interatomic_distances(x: &[f64]) -> Vec<f64> {
    let n = x.len() / 3;
    let n_pairs = n * (n - 1) / 2;
    let mut dists = Vec::with_capacity(n_pairs);

    for j in 0..n {
        let xj = &x[3 * j..3 * j + 3];
        for i in (j + 1)..n {
            let xi = &x[3 * i..3 * i + 3];
            let d2 = (xi[0] - xj[0]).powi(2) + (xi[1] - xj[1]).powi(2) + (xi[2] - xj[2]).powi(2);
            dists.push(d2.sqrt());
        }
    }

    dists
}

/// MAX_1D_LOG distance: max |log(r1_k / r2_k)| over all interatomic distance pairs.
///
/// Rotationally and translationally invariant. Primary metric in gpr_optim.
pub fn max_1d_log_distance(x1: &[f64], x2: &[f64]) -> f64 {
    let d1 = interatomic_distances(x1);
    let d2 = interatomic_distances(x2);

    d1.iter()
        .zip(d2.iter())
        .map(|(a, b)| (a / (b + 1e-18)).ln().abs())
        .fold(0.0f64, f64::max)
}

/// RMSD between two flat coordinate vectors (no alignment).
pub fn rmsd_distance(x1: &[f64], x2: &[f64]) -> f64 {
    let n = x1.len() / 3;
    let sum_sq: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    (sum_sq / n as f64).sqrt()
}

/// Euclidean norm of the difference.
pub fn euclidean_distance(x1: &[f64], x2: &[f64]) -> f64 {
    x1.iter()
        .zip(x2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_1d_log_self() {
        let x = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        assert!(max_1d_log_distance(&x, &x) < 1e-15);
    }

    #[test]
    fn test_rmsd() {
        let x1 = vec![0.0, 0.0, 0.0];
        let x2 = vec![1.0, 0.0, 0.0];
        assert!((rmsd_distance(&x1, &x2) - 1.0).abs() < 1e-12);
    }
}
