//! Trust region utilities.
//!
//! Ports `trust_region.jl` and `distances_trust.jl`.

use crate::distances::{euclidean_distance, max_1d_log_distance};
use crate::emd::emd_distance;

/// Distance metric type for trust region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TrustMetric {
    Emd,
    Max1dLog,
    Euclidean,
}

/// Return a distance function for the given metric.
pub fn trust_distance(metric: TrustMetric, atom_types: &[i32], x1: &[f64], x2: &[f64]) -> f64 {
    match metric {
        TrustMetric::Emd => {
            if x1.len() % 3 == 0 {
                emd_distance(x1, x2, atom_types)
            } else {
                euclidean_distance(x1, x2)
            }
        }
        TrustMetric::Max1dLog => max_1d_log_distance(x1, x2),
        TrustMetric::Euclidean => euclidean_distance(x1, x2),
    }
}

/// Minimum distance from x to any column of x_train.
pub fn trust_min_distance(
    x: &[f64],
    x_train: &[f64],
    dim: usize,
    n: usize,
    metric: TrustMetric,
    atom_types: &[i32],
) -> f64 {
    if n == 0 {
        return f64::INFINITY;
    }
    let mut min_d = f64::INFINITY;
    for i in 0..n {
        let xi = &x_train[i * dim..(i + 1) * dim];
        let d = trust_distance(metric, atom_types, x, xi);
        min_d = min_d.min(d);
    }
    min_d
}

/// Adaptive trust threshold with sigmoidal decay.
pub fn adaptive_trust_threshold(
    trust_radius: f64,
    n_data: usize,
    n_atoms: usize,
    use_adaptive: bool,
    t_min: f64,
    delta_t: f64,
    n_half: usize,
    a: f64,
    floor: f64,
) -> f64 {
    if !use_adaptive {
        return trust_radius;
    }
    let n_eff = n_data as f64 / n_atoms.max(1) as f64;
    let t = t_min + delta_t / (1.0 + a * (n_eff / n_half as f64).exp());
    t.max(floor)
}

/// Minimum distance from x to any training point (Euclidean).
pub fn min_distance_to_data(x: &[f64], x_train: &[f64], dim: usize, n: usize) -> f64 {
    if n == 0 {
        return f64::INFINITY;
    }
    let mut min_d = f64::INFINITY;
    for i in 0..n {
        let xi = &x_train[i * dim..(i + 1) * dim];
        let d = euclidean_distance(x, xi);
        min_d = min_d.min(d);
    }
    min_d
}

/// Check interatomic distance ratio constraint.
pub fn check_interatomic_ratio(
    x_new: &[f64],
    x_train: &[f64],
    dim: usize,
    n: usize,
    ratio_limit: f64,
) -> bool {
    if x_new.len() < 6 {
        return true;
    }
    let threshold = ratio_limit.ln().abs();
    for k in 0..n {
        let xk = &x_train[k * dim..(k + 1) * dim];
        let d = max_1d_log_distance(x_new, xk);
        if d < threshold {
            return true;
        }
    }
    false
}

/// Remove rigid body modes from a step vector.
/// Projects out 3 translational + 3 rotational modes.
pub fn remove_rigid_body_modes(step: &mut [f64], x: &[f64], n_atoms: usize) -> f64 {
    let d = 3 * n_atoms;
    assert_eq!(step.len(), d);
    assert_eq!(x.len(), d);

    // Center of mass
    let mut com = [0.0f64; 3];
    for i in 0..n_atoms {
        for k in 0..3 {
            com[k] += x[3 * i + k];
        }
    }
    for k in 0..3 {
        com[k] /= n_atoms as f64;
    }

    // Build 6 basis vectors
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(6);

    // Translational modes
    for dd in 0..3 {
        let mut t = vec![0.0; d];
        for i in 0..n_atoms {
            t[3 * i + dd] = 1.0;
        }
        basis.push(t);
    }

    // Rotational modes
    for ax in 0..3usize {
        let mut r = vec![0.0; d];
        for i in 0..n_atoms {
            let pos = [
                x[3 * i] - com[0],
                x[3 * i + 1] - com[1],
                x[3 * i + 2] - com[2],
            ];
            match ax {
                0 => {
                    r[3 * i + 1] = -pos[2];
                    r[3 * i + 2] = pos[1];
                }
                1 => {
                    r[3 * i] = pos[2];
                    r[3 * i + 2] = -pos[0];
                }
                2 => {
                    r[3 * i] = -pos[1];
                    r[3 * i + 1] = pos[0];
                }
                _ => unreachable!(),
            }
        }
        basis.push(r);
    }

    // Gram-Schmidt orthonormalization
    let mut ortho: Vec<Vec<f64>> = Vec::new();
    for v in &basis {
        let mut u = v.clone();
        for ou in &ortho {
            let proj: f64 = v.iter().zip(ou.iter()).map(|(a, b)| a * b).sum();
            for j in 0..u.len() {
                u[j] -= proj * ou[j];
            }
        }
        let un: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        if un > 1e-9 {
            for j in 0..u.len() {
                u[j] /= un;
            }
            ortho.push(u);
        }
    }

    // Project out
    let mut removed = vec![0.0; d];
    for u in &ortho {
        let proj: f64 = step.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
        for j in 0..d {
            removed[j] += proj * u[j];
        }
    }
    for j in 0..d {
        step[j] -= removed[j];
    }

    removed.iter().map(|x| x * x).sum::<f64>().sqrt()
}
