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
            if x1.len().is_multiple_of(3) {
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

/// Adaptive trust threshold with exponential saturation (C++ AtomicDimer.cpp:429).
///
/// More data -> higher threshold -> more permissive exploration.
/// Capped by `physical_limit = max(floor, a / sqrt(n_atoms))`.
/// Parameters controlling adaptive trust threshold growth.
pub struct AdaptiveTrustParams {
    pub t_min: f64,
    pub delta_t: f64,
    pub n_half: usize,
    pub a: f64,
    pub floor: f64,
}

pub fn adaptive_trust_threshold(
    trust_radius: f64,
    n_data: usize,
    n_atoms: usize,
    use_adaptive: bool,
    params: &AdaptiveTrustParams,
) -> f64 {
    if !use_adaptive {
        return trust_radius;
    }
    let k = 2.0_f64.ln() / params.n_half.max(1) as f64;
    let earned = params.t_min + params.delta_t * (1.0 - (-k * n_data as f64).exp());
    let physical = params.floor.max(params.a / (n_atoms.max(1) as f64).sqrt());
    earned.min(physical)
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

/// Parameters needed for trust-region clipping, shared across all methods.
pub struct TrustClipParams<'a> {
    pub trust_radius: f64,
    pub trust_metric: TrustMetric,
    pub atom_types: &'a [i32],
    pub use_adaptive: bool,
    pub adaptive_t_min: f64,
    pub adaptive_delta_t: f64,
    pub adaptive_n_half: usize,
    pub adaptive_a: f64,
    pub adaptive_floor: f64,
}

impl<'a> TrustClipParams<'a> {
    fn threshold(&self, n_data: usize, n_atoms: usize) -> f64 {
        let params = AdaptiveTrustParams {
            t_min: self.adaptive_t_min,
            delta_t: self.adaptive_delta_t,
            n_half: self.adaptive_n_half,
            a: self.adaptive_a,
            floor: self.adaptive_floor,
        };
        adaptive_trust_threshold(
            self.trust_radius,
            n_data,
            n_atoms,
            self.use_adaptive,
            &params,
        )
    }
}

/// Clip a single position to the trust region around training data.
///
/// If `x` is farther than the adaptive threshold from any training point,
/// snaps it to `nearest + 0.95 * (threshold / distance) * displacement`.
/// Returns true if clipping occurred.
pub fn clip_point_to_trust(
    x: &mut Vec<f64>,
    td: &crate::types::TrainingData,
    params: &TrustClipParams,
) -> bool {
    let d = x.len();
    let n_atoms = d / 3;
    let thresh = params.threshold(td.npoints(), n_atoms);
    let dist = trust_min_distance(x, &td.data, d, td.npoints(), params.trust_metric, params.atom_types);
    if dist > thresh {
        let nearest_idx = (0..td.npoints())
            .min_by(|&i, &j| {
                let di = trust_distance(params.trust_metric, params.atom_types, x, td.col(i));
                let dj = trust_distance(params.trust_metric, params.atom_types, x, td.col(j));
                di.partial_cmp(&dj).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        let nearest = td.col(nearest_idx).to_vec();
        let scale = thresh / dist * 0.95;
        *x = nearest
            .iter()
            .zip(x.iter())
            .map(|(n, xi)| n + (xi - n) * scale)
            .collect();
        true
    } else {
        false
    }
}

/// Clip intermediate NEB images to the trust region.
///
/// Iterates images `1..n-1` (skipping fixed endpoints) and clips each
/// to the nearest training point if outside the trust radius.
pub fn clip_images_to_trust(
    images: &mut [Vec<f64>],
    td: &crate::types::TrainingData,
    params: &TrustClipParams,
) {
    let n = images.len();
    let d = images[0].len();
    let n_atoms = d / 3;
    let thresh = params.threshold(td.npoints(), n_atoms);

    for image in images.iter_mut().take(n - 1).skip(1) {
        let dist = trust_min_distance(
            image, &td.data, d, td.npoints(), params.trust_metric, params.atom_types,
        );
        if dist > thresh {
            let nearest_idx = (0..td.npoints())
                .min_by(|&a, &b| {
                    let da = trust_distance(params.trust_metric, params.atom_types, image, td.col(a));
                    let db = trust_distance(params.trust_metric, params.atom_types, image, td.col(b));
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            let nearest = td.col(nearest_idx).to_vec();
            let scale = thresh / dist * 0.95;
            *image = nearest
                .iter()
                .zip(image.iter())
                .map(|(n, xi)| n + (xi - n) * scale)
                .collect();
        }
    }
}

/// Check whether a position exceeds the trust radius (without clipping).
///
/// Used by OTGPD which rejects steps rather than clipping.
pub fn exceeds_trust_radius(
    x: &[f64],
    td: &crate::types::TrainingData,
    params: &TrustClipParams,
) -> bool {
    let d = x.len();
    let n_atoms = d / 3;
    let thresh = params.threshold(td.npoints(), n_atoms);
    let dist = trust_min_distance(x, &td.data, d, td.npoints(), params.trust_metric, params.atom_types);
    dist > thresh
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
    for c in &mut com {
        *c /= n_atoms as f64;
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
            for (u_j, &ou_j) in u.iter_mut().zip(ou.iter()) {
                *u_j -= proj * ou_j;
            }
        }
        let un: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        if un > 1e-9 {
            for u_j in u.iter_mut() {
                *u_j /= un;
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
