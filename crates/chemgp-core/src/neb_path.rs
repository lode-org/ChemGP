//! NEB path utilities: tangent, spring force, NEB force, force computation.
//!
//! Ports `neb_path.jl`: improved tangent (Henkelman & Jonsson 2000),
//! energy-weighted springs, climbing image force.

use crate::prior_mean::PriorMeanConfig;
use serde::{Deserialize, Serialize};

/// Mutable NEB path state.
#[derive(Debug, Clone)]
pub struct NEBPath {
    pub images: Vec<Vec<f64>>,
    pub energies: Vec<f64>,
    pub gradients: Vec<Vec<f64>>,
    pub spring_constant: f64,
}

/// NEB force computation results.
pub struct NEBForces {
    pub forces: Vec<Vec<f64>>,
    pub max_f: f64,
    pub ci_f: f64,
    pub i_max: usize,
}

/// Linear interpolation between two endpoints.
pub fn linear_interpolation(x_start: &[f64], x_end: &[f64], n_images: usize) -> Vec<Vec<f64>> {
    let mut images = Vec::with_capacity(n_images);
    for i in 0..n_images {
        let t = i as f64 / (n_images - 1) as f64;
        let img: Vec<f64> = x_start
            .iter()
            .zip(x_end.iter())
            .map(|(&a, &b)| (1.0 - t) * a + t * b)
            .collect();
        images.push(img);
    }
    images
}

/// Improved tangent estimate (Henkelman & Jonsson 2000).
pub fn path_tangent(images: &[Vec<f64>], energies: &[f64], i: usize) -> Vec<f64> {
    let n = images.len();
    assert!(i >= 1 && i <= n - 2, "Tangent only for intermediate images");

    let tau_plus: Vec<f64> = images[i + 1]
        .iter()
        .zip(images[i].iter())
        .map(|(a, b)| a - b)
        .collect();
    let tau_minus: Vec<f64> = images[i]
        .iter()
        .zip(images[i - 1].iter())
        .map(|(a, b)| a - b)
        .collect();

    let e_prev = energies[i - 1];
    let e_curr = energies[i];
    let e_next = energies[i + 1];

    let tau = if e_prev < e_curr && e_curr < e_next {
        tau_plus.clone()
    } else if e_prev > e_curr && e_curr > e_next {
        tau_minus.clone()
    } else {
        let de_max = (e_next - e_curr).abs().max((e_prev - e_curr).abs());
        let de_min = (e_next - e_curr).abs().min((e_prev - e_curr).abs());

        if e_prev < e_next {
            tau_plus
                .iter()
                .zip(tau_minus.iter())
                .map(|(p, m)| de_max * p + de_min * m)
                .collect()
        } else {
            tau_plus
                .iter()
                .zip(tau_minus.iter())
                .map(|(p, m)| de_min * p + de_max * m)
                .collect()
        }
    };

    let tn: f64 = tau.iter().map(|x| x * x).sum::<f64>().sqrt();
    if tn > 1e-18 {
        tau.iter().map(|x| x / tn).collect()
    } else {
        let tp_norm: f64 = tau_plus.iter().map(|x| x * x).sum::<f64>().sqrt() + 1e-18;
        tau_plus.iter().map(|x| x / tp_norm).collect()
    }
}

/// Spring force parallel to tangent.
pub fn spring_force(images: &[Vec<f64>], i: usize, k_spring: f64, tangent: &[f64]) -> Vec<f64> {
    let d_next: f64 = images[i + 1]
        .iter()
        .zip(images[i].iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let d_prev: f64 = images[i]
        .iter()
        .zip(images[i - 1].iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    tangent
        .iter()
        .map(|&t| k_spring * (d_next - d_prev) * t)
        .collect()
}

/// Energy-weighted spring constant (Asgeirsson et al. 2021).
pub fn energy_weighted_k(
    energies: &[f64],
    i_lo: usize,
    i_hi: usize,
    k_min: f64,
    k_max: f64,
) -> f64 {
    let e_ref = energies[0].max(energies[energies.len() - 1]);
    let e_max = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let de = e_max - e_ref;
    if de < 1e-18 {
        return k_max;
    }
    let e_spring = energies[i_lo].max(energies[i_hi]);
    if e_spring > e_ref {
        k_max - (k_max - k_min) * (e_max - e_spring) / de
    } else {
        k_min
    }
}

/// NEB force at a single image.
pub fn neb_force(
    gradient: &[f64],
    spring_f: &[f64],
    tangent: &[f64],
    climbing: bool,
    is_highest: bool,
) -> Vec<f64> {
    if climbing && is_highest {
        // CI: F = -G + 2*(G.tau)*tau
        let g_dot_t: f64 = gradient.iter().zip(tangent.iter()).map(|(g, t)| g * t).sum();
        gradient
            .iter()
            .zip(tangent.iter())
            .map(|(&g, &t)| -g + 2.0 * g_dot_t * t)
            .collect()
    } else {
        // Standard: F = -G_perp + F_spring
        let g_dot_t: f64 = gradient.iter().zip(tangent.iter()).map(|(g, t)| g * t).sum();
        gradient
            .iter()
            .zip(tangent.iter())
            .zip(spring_f.iter())
            .map(|((&g, &t), &s)| -(g - g_dot_t * t) + s)
            .collect()
    }
}

/// Hessian perturbation points around endpoints (Koistinen et al. 2017).
pub fn get_hessian_points(x_start: &[f64], x_end: &[f64], epsilon: f64) -> Vec<Vec<f64>> {
    let d = x_start.len();
    let mut points = Vec::with_capacity(2 * d);
    for dd in 0..d {
        let mut p1 = x_start.to_vec();
        p1[dd] += epsilon;
        points.push(p1);

        let mut p2 = x_end.to_vec();
        p2[dd] += epsilon;
        points.push(p2);
    }
    points
}

/// Per-atom max force norm.
pub fn max_atom_force(force: &[f64], n_atoms: usize, n_coords: usize) -> f64 {
    let mut max_f = 0.0f64;
    for a in 0..n_atoms {
        let off = a * n_coords;
        let f: f64 = (0..n_coords).map(|d| force[off + d].powi(2)).sum::<f64>().sqrt();
        max_f = max_f.max(f);
    }
    max_f
}

/// Acquisition strategy for OIE image selection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AcquisitionStrategy {
    /// Max energy variance among unevaluated images.
    MaxVariance,
    /// Max GP-predicted NEB force among unevaluated images.
    MaxForce,
    /// Upper confidence bound: |F| + kappa * sigma_perp.
    Ucb,
    /// Expected improvement: E[max(F_i - F_max, 0)] over force.
    ExpectedImprovement,
    /// Thompson sampling: draw from GP posterior, pick highest sampled force.
    ThompsonSampling,
}

/// NEB configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NEBConfig {
    pub images: usize,
    pub spring_constant: f64,
    pub climbing_image: bool,
    pub ci_activation_tol: f64,
    pub ci_trigger_rel: f64,
    pub ci_converged_only: bool,
    pub max_iter: usize,
    pub conv_tol: f64,
    pub step_size: f64,
    pub use_lbfgs: bool,
    pub max_move: f64,
    pub lbfgs_memory: usize,
    pub energy_weighted: bool,
    pub ew_k_min: f64,
    pub ew_k_max: f64,
    pub gp_train_iter: usize,
    pub max_outer_iter: usize,
    pub max_gp_points: usize,
    pub rff_features: usize,
    pub trust_radius: f64,
    pub trust_metric: crate::trust::TrustMetric,
    pub atom_types: Vec<i32>,
    pub use_adaptive_threshold: bool,
    pub adaptive_t_min: f64,
    pub adaptive_delta_t: f64,
    pub adaptive_n_half: usize,
    pub adaptive_a: f64,
    pub adaptive_floor: f64,
    pub num_hess_iter: usize,
    pub eps_hess: f64,
    /// Path initializer: "linear", "idpp", or "sidpp".
    pub initializer: String,
    // OIE-specific fields
    /// Separate CI convergence threshold; -1 means use conv_tol.
    pub ci_force_tol: f64,
    /// GP force level to activate CI during inner relaxation; 0 disables.
    pub inner_ci_threshold: f64,
    /// Adaptive inner GP tol = smallest_accurate_force / divisor; 0 = fixed.
    pub gp_tol_divisor: usize,
    /// Max displacement from nearest training point as fraction of initial path length.
    pub max_step_frac: f64,
    /// Bond ratio limit; reject if any ratio < limit or > 1/limit.
    pub bond_stretch_limit: f64,
    /// LCB exploration weight: score = |F| + kappa * sigma_perp.
    pub lcb_kappa: f64,
    /// Acquisition strategy for image selection (OIE only).
    pub acquisition: AcquisitionStrategy,
    /// FPS subset size for hyperparameter training.
    pub fps_history: usize,
    /// Most recent points always included in FPS.
    pub fps_latest_points: usize,
    /// Use Quick-min Velocity Verlet for inner GP relaxation instead of L-BFGS.
    /// Naturally conservative steps for OIE inner relaxation.
    pub use_quickmin: bool,
    /// QM-VV time step (dt).
    pub qm_dt: f64,
    /// Two-phase acquisition: when max GP energy uncertainty across
    /// unevaluated images exceeds this, select highest-uncertainty image
    /// (pure exploration). Below this threshold, use configured strategy.
    /// Default: 0.05. Set 0 to disable (always use strategy).
    pub unc_convergence: f64,
    /// Uncertainty gate for convergence: require max GP uncertainty < this
    /// in addition to force convergence. 0 disables the check.
    pub unc_conv_tol: f64,
    /// Uncertainty gate for inner relaxation revert: if max GP uncertainty
    /// at relaxed positions exceeds this, revert to pre-relaxation path.
    /// Revert relaxed path when max uncertainty exceeds this. 0 disables.
    pub unc_revert_tol: f64,
    /// Images evaluated per outer iteration: 1 (classic OIE) or 3 (triplet).
    /// Triplet mode evaluates {i-1, i, i+1} so the Henkelman-Jonsson improved
    /// tangent at image i uses ground-truth data on both sides.
    pub evals_per_iter: usize,
    /// Hard oracle call budget for NEB OIE (0 = unlimited).
    pub max_neb_oracle_calls: usize,
    /// Enable Hyperparameter Oscillation Detection (HOD).
    /// When hyperparameters oscillate, grows FPS subset to stabilize training.
    pub use_hod: bool,
    /// HOD sliding window size.
    pub hod_monitoring_window: usize,
    /// HOD sign-flip threshold (0.0--1.0).
    pub hod_flip_threshold: f64,
    /// HOD FPS growth increment.
    pub hod_history_increment: usize,
    /// HOD maximum FPS subset size.
    pub hod_max_history: usize,
    /// Max training points for GP prediction subset (KNN per bead).
    /// When > 0, selects nearest neighbors around each NEB image and
    /// uses exact GP instead of RFF. 0 = use FPS training subset
    /// (when rff_features=0) or full data (when rff_features > 0).
    pub max_pred_points: usize,
    /// Constant kernel variance added to energy-energy block.
    /// Set to 1.0 for molecular systems (matches C++ ConstantCF).
    /// Default 0.0 (disabled, backward compatible for 2D surfaces).
    pub const_sigma2: f64,
    pub prior_mean: PriorMeanConfig,
    pub verbose: bool,
}

impl Default for NEBConfig {
    fn default() -> Self {
        Self {
            images: 5,
            spring_constant: 5.0,
            climbing_image: true,
            ci_activation_tol: 0.5,
            ci_trigger_rel: 0.8,
            ci_converged_only: true,
            max_iter: 1000,
            conv_tol: 0.05,
            step_size: 0.01,
            use_lbfgs: true,
            max_move: 0.1,
            lbfgs_memory: 20,
            energy_weighted: false,
            ew_k_min: 1.0,
            ew_k_max: 10.0,
            gp_train_iter: 300,
            max_outer_iter: 50,
            max_gp_points: 0,
            rff_features: 0,
            trust_radius: 0.1,
            trust_metric: crate::trust::TrustMetric::Emd,
            atom_types: Vec::new(),
            use_adaptive_threshold: false,
            adaptive_t_min: 0.15,
            adaptive_delta_t: 0.35,
            adaptive_n_half: 50,
            adaptive_a: 1.3,
            adaptive_floor: 0.2,
            num_hess_iter: 0,
            eps_hess: 0.01,
            initializer: "linear".to_string(),
            ci_force_tol: -1.0,
            inner_ci_threshold: 0.5,
            gp_tol_divisor: 10,
            max_step_frac: 0.1,
            bond_stretch_limit: 2.0 / 3.0,
            lcb_kappa: 2.0,
            acquisition: AcquisitionStrategy::Ucb,
            fps_history: 0,
            fps_latest_points: 2,
            use_quickmin: false,
            qm_dt: 0.1,
            unc_convergence: 0.0,
            unc_conv_tol: 0.0,
            unc_revert_tol: 0.0,
            evals_per_iter: 1,
            max_neb_oracle_calls: 0,
            use_hod: true,
            hod_monitoring_window: 5,
            hod_flip_threshold: 0.8,
            hod_history_increment: 2,
            hod_max_history: 30,
            max_pred_points: 0,
            const_sigma2: 0.0,
            prior_mean: PriorMeanConfig::Reference,
            verbose: true,
        }
    }
}

/// Compute NEB forces at all intermediate images.
pub fn compute_all_neb_forces(path: &NEBPath, cfg: &NEBConfig, ci_on: bool) -> NEBForces {
    let n = path.images.len();
    let d = path.images[0].len();
    let mut forces: Vec<Vec<f64>> = (0..n).map(|_| vec![0.0; d]).collect();

    // Highest energy intermediate image
    let i_max = (1..n - 1)
        .max_by(|&a, &b| path.energies[a].partial_cmp(&path.energies[b]).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(1);  // Fallback to first intermediate image

    let mut max_f_norm = 0.0f64;
    let mut ci_f_norm = 0.0f64;

    for (i, force_i) in forces.iter_mut().enumerate().take(n - 1).skip(1) {
        let tau = path_tangent(&path.images, &path.energies, i);

        let f_spring = if cfg.energy_weighted {
            let k_prev = energy_weighted_k(&path.energies, i - 1, i, cfg.ew_k_min, cfg.ew_k_max);
            let k_next = energy_weighted_k(&path.energies, i, i + 1, cfg.ew_k_min, cfg.ew_k_max);
            let d_next: f64 = path.images[i + 1]
                .iter()
                .zip(path.images[i].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let d_prev: f64 = path.images[i]
                .iter()
                .zip(path.images[i - 1].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            tau.iter()
                .map(|&t| (k_next * d_next - k_prev * d_prev) * t)
                .collect()
        } else {
            spring_force(&path.images, i, path.spring_constant, &tau)
        };

        let is_highest = i == i_max;
        let f = neb_force(
            &path.gradients[i],
            &f_spring,
            &tau,
            ci_on && cfg.climbing_image,
            is_highest,
        );

        // Per-atom max force for molecular systems
        let n_atoms = d / 3;
        let fn_val = if n_atoms >= 1 && d == 3 * n_atoms {
            max_atom_force(&f, n_atoms, 3)
        } else {
            f.iter().map(|x| x * x).sum::<f64>().sqrt()
        };

        max_f_norm = max_f_norm.max(fn_val);
        if is_highest {
            ci_f_norm = fn_val;
        }
        *force_i = f;
    }

    NEBForces {
        forces,
        max_f: max_f_norm,
        ci_f: ci_f_norm,
        i_max,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_interpolation() {
        let start = vec![0.0, 0.0];
        let end = vec![1.0, 2.0];
        let images = linear_interpolation(&start, &end, 3);
        assert_eq!(images.len(), 3);
        assert!((images[1][0] - 0.5).abs() < 1e-12);
        assert!((images[1][1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_path_tangent_monotonic() {
        let images = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![2.0, 0.0],
        ];
        let energies = vec![0.0, 1.0, 2.0];
        let tau = path_tangent(&images, &energies, 1);
        assert!((tau[0] - 1.0).abs() < 1e-12);
        assert!(tau[1].abs() < 1e-12);
    }
}
