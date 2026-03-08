//! NEB optimization methods.
//!
//! Ports `neb.jl`: Standard NEB and GP-NEB AIE.
//!
//! Reference: Goswami et al., J. Chem. Theory Comput. (2025).

use crate::idpp::{idpp_interpolation, sidpp_interpolation};
use crate::kernel::Kernel;
use crate::neb_path::{
    compute_all_neb_forces, get_hessian_points, linear_interpolation, NEBConfig, NEBPath,
};
use crate::optim_step::OptimState;
use crate::predict::{build_pred_model, PredModel};
use crate::train::train_model;
use crate::trust::{
    adaptive_trust_threshold, trust_distance, trust_min_distance, TrustMetric,
};
use crate::types::{init_kernel, GPModel, TrainingData};
use crate::StopReason;

/// Result of NEB optimization.
#[derive(Debug, Clone)]
pub struct NEBResult {
    pub path: NEBPath,
    pub converged: bool,
    pub stop_reason: StopReason,
    pub oracle_calls: usize,
    pub max_energy_image: usize,
    pub history: NEBHistory,
}

/// Convergence history for NEB.
#[derive(Debug, Clone, Default)]
pub struct NEBHistory {
    pub max_force: Vec<f64>,
    pub ci_force: Vec<f64>,
    pub oracle_calls: Vec<usize>,
    pub max_energy: Vec<f64>,
}

/// Oracle function type.
pub type OracleFn = dyn Fn(&[f64]) -> (f64, Vec<f64>);

/// Initialize NEB path images.
fn init_neb_images(cfg: &NEBConfig, x_start: &[f64], x_end: &[f64]) -> Vec<Vec<f64>> {
    let n_total = cfg.images + 2;
    match cfg.initializer.as_str() {
        "sidpp" => sidpp_interpolation(
            x_start,
            x_end,
            n_total,
            3,
            200,
            0.1,
            0.01,
            10,
            cfg.spring_constant,
            0.3,
        ),
        "idpp" => idpp_interpolation(x_start, x_end, n_total, 3, 200, 0.1, 0.01, 10),
        _ => linear_interpolation(x_start, x_end, n_total),
    }
}

/// Check climbing image activation (eOn-style dynamic thresholding).
fn check_ci(
    cfg: &NEBConfig,
    ci_on: bool,
    max_f: f64,
    ci_f: f64,
    baseline_force: f64,
    iter: usize,
) -> (bool, f64, bool) {
    let mut new_ci = ci_on;
    let mut activated = false;

    if cfg.climbing_image && iter > 1 {
        let should_ci =
            max_f < cfg.ci_trigger_rel * baseline_force || max_f < cfg.ci_activation_tol;
        activated = should_ci && !ci_on;
        new_ci = should_ci;
    }

    let conv_metric = if new_ci && cfg.ci_converged_only {
        ci_f
    } else {
        max_f
    };
    (new_ci, conv_metric, activated)
}

/// Standard NEB optimization (oracle at every step).
pub fn neb_optimize(
    oracle: &OracleFn,
    x_start: &[f64],
    x_end: &[f64],
    config: &NEBConfig,
) -> NEBResult {
    let cfg = config;
    let n = cfg.images + 2;
    let d = x_start.len();

    let mut images = init_neb_images(cfg, x_start, x_end);

    // Evaluate endpoints
    let (e_start, g_start) = oracle(x_start);
    let (e_end, g_end) = oracle(x_end);
    let mut energies = vec![0.0; n];
    let mut gradients: Vec<Vec<f64>> = (0..n).map(|_| vec![0.0; d]).collect();
    energies[0] = e_start;
    energies[n - 1] = e_end;
    gradients[0] = g_start;
    gradients[n - 1] = g_end;

    let mut oracle_calls = 2;

    // Evaluate intermediate images
    for i in 1..n - 1 {
        let (e, g) = oracle(&images[i]);
        energies[i] = e;
        gradients[i] = g;
        oracle_calls += 1;
    }

    let mut path = NEBPath {
        images: images.clone(),
        energies: energies.clone(),
        gradients: gradients.clone(),
        spring_constant: cfg.spring_constant,
    };

    let mut history = NEBHistory::default();
    let mut optim = OptimState::new(cfg.lbfgs_memory);

    let mut ci_on = false;
    let mut stop_reason = StopReason::MaxIterations;
    let mut baseline_force = 0.0;
    let mut stagnation_count = 0;
    let mut prev_max_f = f64::NEG_INFINITY;

    for iter in 0..cfg.max_iter {
        let neb_forces = compute_all_neb_forces(&path, cfg, ci_on);

        // Stagnation check
        let stag_tol = 1e-6f64.max(1e-4 * neb_forces.max_f);
        if (neb_forces.max_f - prev_max_f).abs() < stag_tol {
            stagnation_count += 1;
        } else {
            stagnation_count = 0;
        }
        prev_max_f = neb_forces.max_f;

        if stagnation_count >= 3 {
            stop_reason = StopReason::ForceStagnation;
            break;
        }

        history.max_force.push(neb_forces.max_f);
        history.ci_force.push(neb_forces.ci_f);
        history.oracle_calls.push(oracle_calls);
        history
            .max_energy
            .push(energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

        if iter == 0 {
            baseline_force = neb_forces.max_f;
        }

        // CI activation
        let (new_ci, conv_metric, _ci_activated) = check_ci(
            cfg,
            ci_on,
            neb_forces.max_f,
            neb_forces.ci_f,
            baseline_force,
            iter,
        );
        ci_on = new_ci;

        let conv_check = ci_on || !cfg.climbing_image;
        if conv_check && conv_metric < cfg.conv_tol {
            stop_reason = StopReason::Converged;
            break;
        }

        if cfg.verbose {
            eprintln!(
                "  Iter {:3}: max|F| = {:.4e} | CI|F| = {:.4e} | CI={} | E_max = {:.4}",
                iter,
                neb_forces.max_f,
                neb_forces.ci_f,
                ci_on,
                path.energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            );
        }

        // Concatenate movable images
        let n_mov = n - 2;
        let mut cur_x = Vec::with_capacity(n_mov * d);
        let mut cur_force = Vec::with_capacity(n_mov * d);
        for i in 1..=n_mov {
            cur_x.extend_from_slice(&path.images[i]);
            cur_force.extend_from_slice(&neb_forces.forces[i]);
        }

        let disp = optim.step(&cur_x, &cur_force, cfg.max_move, 3);
        let new_x: Vec<f64> = cur_x.iter().zip(disp.iter()).map(|(a, b)| a + b).collect();

        for i in 0..n_mov {
            let off = i * d;
            images[i + 1] = new_x[off..off + d].to_vec();
        }

        // Re-evaluate oracle
        for i in 1..n - 1 {
            let (e, g) = oracle(&images[i]);
            energies[i] = e;
            gradients[i] = g;
            oracle_calls += 1;
        }

        path.images = images.clone();
        path.energies = energies.clone();
        path.gradients = gradients.clone();
    }

    let i_max = (1..n - 1)
        .max_by(|&a, &b| energies[a].partial_cmp(&energies[b]).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(1);  // Must be intermediate image (not endpoint)

    NEBResult {
        path,
        converged: stop_reason == StopReason::Converged,
        stop_reason,
        oracle_calls,
        max_energy_image: i_max,
        history,
    }
}

/// GP-NEB with All Images Evaluated per outer iteration.
pub fn gp_neb_aie(
    oracle: &OracleFn,
    x_start: &[f64],
    x_end: &[f64],
    kernel: &Kernel,
    config: &NEBConfig,
) -> NEBResult {
    let cfg = config;
    let n = cfg.images + 2;
    let d = x_start.len();
    let mut images = init_neb_images(cfg, x_start, x_end);

    // Evaluate endpoints
    let (e_start, g_start) = oracle(x_start);
    let (e_end, g_end) = oracle(x_end);
    let mut oracle_calls = 2;

    let mut td = TrainingData::new(d);
    td.add_point(x_start, e_start, &g_start).expect("add_point failed: invalid data");
    td.add_point(x_end, e_end, &g_end).expect("add_point failed: invalid data");

    // Virtual Hessian points
    let mut hess_calls = 0;
    let mut hess_x: Vec<Vec<f64>> = Vec::new();
    let mut hess_e: Vec<f64> = Vec::new();
    let mut hess_g: Vec<f64> = Vec::new();
    if cfg.num_hess_iter > 0 {
        let hpts = get_hessian_points(x_start, x_end, cfg.eps_hess);
        for pt in &hpts {
            let (e, g) = oracle(pt);
            hess_x.push(pt.clone());
            hess_e.push(e);
            hess_g.extend_from_slice(&g);
            hess_calls += 1;
        }
    }
    oracle_calls += hess_calls;

    // Evaluate all intermediate images
    let mut energies = vec![0.0; n];
    let mut gradients: Vec<Vec<f64>> = (0..n).map(|_| vec![0.0; d]).collect();
    energies[0] = e_start;
    energies[n - 1] = e_end;
    gradients[0] = g_start.clone();
    gradients[n - 1] = g_end.clone();

    for i in 1..n - 1 {
        let (e, g) = oracle(&images[i]);
        energies[i] = e;
        gradients[i] = g.clone();
        td.add_point(&images[i], e, &g).expect("add_point failed: invalid data");
        oracle_calls += 1;
    }

    let mut path = NEBPath {
        images: images.clone(),
        energies: energies.clone(),
        gradients: gradients.clone(),
        spring_constant: cfg.spring_constant,
    };

    // Path scale for early stopping displacement check
    let path_scale: f64 = (0..n - 1)
        .map(|i| {
            images[i]
                .iter()
                .zip(images[i + 1].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .sum();

    let mut history = NEBHistory::default();
    let mut ci_on = false;
    let mut stop_reason = StopReason::MaxIterations;
    let mut prev_kern: Option<Kernel> = None;
    let mut baseline_force = 0.0;
    let mut stagnation_count = 0;
    let mut prev_max_f = f64::NEG_INFINITY;

    for outer_iter in 0..cfg.max_outer_iter {
        // Compute true forces
        let neb_forces = compute_all_neb_forces(&path, cfg, ci_on);

        // Stagnation check
        let stag_tol = 1e-6f64.max(1e-4 * neb_forces.max_f);
        if (neb_forces.max_f - prev_max_f).abs() < stag_tol {
            stagnation_count += 1;
        } else {
            stagnation_count = 0;
        }
        prev_max_f = neb_forces.max_f;

        if stagnation_count >= 3 {
            stop_reason = StopReason::ForceStagnation;
            break;
        }

        history.max_force.push(neb_forces.max_f);
        history.ci_force.push(neb_forces.ci_f);
        history.oracle_calls.push(oracle_calls);
        history
            .max_energy
            .push(energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

        if outer_iter == 0 {
            baseline_force = neb_forces.max_f;
        }

        if cfg.verbose {
            eprintln!(
                "GP-NEB-AIE outer {}: max|F| = {:.5} | CI|F| = {:.5} | N_train = {} | calls = {}",
                outer_iter,
                neb_forces.max_f,
                neb_forces.ci_f,
                td.npoints(),
                oracle_calls
            );
        }

        // CI activation
        let (new_ci, conv_metric, _) = check_ci(
            cfg,
            ci_on,
            neb_forces.max_f,
            neb_forces.ci_f,
            baseline_force,
            outer_iter,
        );
        ci_on = new_ci;

        let conv_check = ci_on || !cfg.climbing_image;
        if conv_check && conv_metric < cfg.conv_tol {
            stop_reason = StopReason::Converged;
            break;
        }

        // Train GP (per-bead subset when max_gp_points > 0)
        let td_use = if cfg.max_gp_points > 0 && td.npoints() > cfg.max_gp_points {
            // Per-bead nearest-neighbor subset
            bead_local_subset(&td, cfg.max_gp_points, &images, cfg.trust_metric, &cfg.atom_types)
        } else {
            td.clone()
        };

        let train_iters = if prev_kern.is_none() {
            cfg.gp_train_iter
        } else {
            (cfg.gp_train_iter / 3).max(50)
        };

        let e_ref = td_use.energies[0];
        let mut y_sub: Vec<f64> = td_use.energies.iter().map(|e| e - e_ref).collect();

        // Optionally include Hessian data
        let use_hess = !hess_x.is_empty() && outer_iter < cfg.num_hess_iter;
        if use_hess {
            let hess_y: Vec<f64> = hess_e.iter().map(|e| e - e_ref).collect();
            y_sub.splice(0..0, hess_y);
            y_sub.extend_from_slice(&hess_g);
        }
        y_sub.extend_from_slice(&td_use.gradients);

        let kern = match &prev_kern {
            None => init_kernel(&td_use, kernel),
            Some(k) => k.clone(),
        };

        let mut gp_sub = GPModel::new(kern, &td_use, y_sub, 1e-6, 1e-4, 1e-6)
            .expect("GPModel::new failed: invalid training data or kernel params");
        gp_sub.const_sigma2 = cfg.const_sigma2;
        train_model(&mut gp_sub, train_iters, cfg.verbose);
        prev_kern = Some(gp_sub.kernel.clone());

        // Build prediction model: RFF (fast) or exact GP (small data)
        let e_ref_full = td.energies[0];
        let pred_model = build_pred_model(&gp_sub.kernel, &td, cfg.rff_features, 42, cfg.const_sigma2);

        // Inner loop: relax on GP/RFF surface
        let gp_tol = (neb_forces.max_f / 10.0)
            .min(cfg.conv_tol)
            .max(cfg.conv_tol / 10.0);
        let (new_images, _early_stopped) = gp_inner_relax(
            &pred_model,
            &images,
            &energies,
            &gradients,
            &td,
            cfg,
            ci_on,
            e_ref_full,
            gp_tol,
            path_scale,
        );
        images = new_images;

        // EMD trust clip
        emd_trust_clip(&mut images, &td, cfg);

        // Re-evaluate oracle at new positions
        for i in 1..n - 1 {
            let (e, g) = oracle(&images[i]);
            energies[i] = e;
            gradients[i] = g.clone();
            oracle_calls += 1;
            td.add_point(&images[i], e, &g).expect("add_point failed: invalid data");
        }

        path.images = images.clone();
        path.energies = energies.clone();
        path.gradients = gradients.clone();
    }

    let i_max = (1..n - 1)
        .max_by(|&a, &b| energies[a].partial_cmp(&energies[b]).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(1);  // Must be intermediate image (not endpoint)

    NEBResult {
        path,
        converged: stop_reason == StopReason::Converged,
        stop_reason,
        oracle_calls,
        max_energy_image: i_max,
        history,
    }
}

/// Inner GP/RFF surface relaxation for NEB images.
///
/// Returns (relaxed_images, early_stopped) where early_stopped is true
/// if bond stretch or displacement guards triggered.
fn gp_inner_relax(
    model: &PredModel,
    images: &[Vec<f64>],
    energies: &[f64],
    gradients: &[Vec<f64>],
    td: &TrainingData,
    cfg: &NEBConfig,
    ci_on: bool,
    e_ref: f64,
    gp_tol: f64,
    path_scale: f64,
) -> (Vec<Vec<f64>>, bool) {
    let n = images.len();
    let d = images[0].len();
    let n_mov = n - 2;

    let mut gp_images = images.to_vec();
    let start_images = images.to_vec();
    let mut optim = OptimState::new(cfg.lbfgs_memory);
    let mut early_stopped = false;

    for inner in 0..cfg.max_iter {
        let mut gp_energies = energies.to_vec();
        let mut gp_gradients = gradients.to_vec();

        for i in 1..n - 1 {
            let preds = model.predict(&gp_images[i]);
            gp_energies[i] = preds[0] + e_ref;
            gp_gradients[i] = preds[1..].to_vec();
        }

        let gp_path = NEBPath {
            images: gp_images.clone(),
            energies: gp_energies,
            gradients: gp_gradients,
            spring_constant: cfg.spring_constant,
        };
        let gp_forces = compute_all_neb_forces(&gp_path, cfg, ci_on);

        if gp_forces.max_f < gp_tol {
            break;
        }

        // Concatenate movable images
        let mut cur_x = Vec::with_capacity(n_mov * d);
        let mut cur_force = Vec::with_capacity(n_mov * d);
        for i in 1..=n_mov {
            cur_x.extend_from_slice(&gp_images[i]);
            cur_force.extend_from_slice(&gp_forces.forces[i]);
        }

        let disp = optim.step(&cur_x, &cur_force, cfg.max_move, 3);
        let new_x: Vec<f64> = cur_x.iter().zip(disp.iter()).map(|(a, b)| a + b).collect();

        let pre_step = gp_images.clone();

        for img_idx in 0..n_mov {
            let off = img_idx * d;
            let candidate = &new_x[off..off + d];
            // Euclidean trust clip from oracle-evaluated anchor
            let disp_vec: Vec<f64> = candidate
                .iter()
                .zip(start_images[img_idx + 1].iter())
                .map(|(a, b)| a - b)
                .collect();
            let dn: f64 = disp_vec.iter().map(|x| x * x).sum::<f64>().sqrt();
            if dn > cfg.trust_radius {
                gp_images[img_idx + 1] = start_images[img_idx + 1]
                    .iter()
                    .zip(disp_vec.iter())
                    .map(|(a, b)| a + b * (cfg.trust_radius / dn))
                    .collect();
            } else {
                gp_images[img_idx + 1] = candidate.to_vec();
            }
        }

        // Early stopping guard (from step 2 onward)
        if inner >= 1 {
            let (stop, _offending) =
                crate::neb_oie::oie_check_early_stop(&gp_images, td, cfg, path_scale);
            if stop {
                gp_images = pre_step;
                early_stopped = true;
                break;
            }
        }
    }

    (gp_images, early_stopped)
}

/// Post-inner-loop EMD trust region clip.
fn emd_trust_clip(images: &mut [Vec<f64>], td: &TrainingData, cfg: &NEBConfig) {
    let n = images.len();
    let d = images[0].len();
    let n_atoms = d / 3;

    let thresh = adaptive_trust_threshold(
        cfg.trust_radius,
        td.npoints(),
        n_atoms,
        cfg.use_adaptive_threshold,
        cfg.adaptive_t_min,
        cfg.adaptive_delta_t,
        cfg.adaptive_n_half,
        cfg.adaptive_a,
        cfg.adaptive_floor,
    );

    for i in 1..n - 1 {
        let dist = trust_min_distance(
            &images[i],
            &td.data,
            d,
            td.npoints(),
            cfg.trust_metric,
            &cfg.atom_types,
        );
        if dist > thresh {
            let nearest_idx = (0..td.npoints())
                .min_by(|&a, &b| {
                    let da = trust_distance(cfg.trust_metric, &cfg.atom_types, &images[i], td.col(a));
                    let db = trust_distance(cfg.trust_metric, &cfg.atom_types, &images[i], td.col(b));
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);  // Safe fallback
            let nearest = td.col(nearest_idx).to_vec();
            let disp: Vec<f64> = images[i]
                .iter()
                .zip(nearest.iter())
                .map(|(a, b)| a - b)
                .collect();
            images[i] = nearest
                .iter()
                .zip(disp.iter())
                .map(|(a, b)| a + b * (thresh / dist * 0.95))
                .collect();
        }
    }
}

/// Per-bead nearest-neighbor subset selection for NEB.
pub(crate) fn bead_local_subset(
    td: &TrainingData,
    max_points: usize,
    images: &[Vec<f64>],
    metric: TrustMetric,
    atom_types: &[i32],
) -> TrainingData {
    let n = td.npoints();
    if max_points == 0 || n <= max_points || images.is_empty() {
        return td.clone();
    }

    let _d = td.dim;
    let n_images = images.len();
    let k_per_bead = (max_points / n_images).max(3);

    let mut keep = std::collections::BTreeSet::new();
    keep.insert(0);
    if n >= 2 {
        keep.insert(1);
    }

    for img in images {
        let mut dists: Vec<(usize, f64)> = (0..n)
            .map(|j| (j, trust_distance(metric, atom_types, img, td.col(j))))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for &(idx, _) in dists.iter().take(k_per_bead) {
            keep.insert(idx);
        }
    }

    let keep_vec: Vec<usize> = keep.into_iter().collect();
    td.extract_subset(&keep_vec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::potentials::muller_brown_energy_gradient;

    #[test]
    fn test_neb_optimize_muller_brown() {
        // Simple 2D test on Muller-Brown surface
        let oracle = |x: &[f64]| -> (f64, Vec<f64>) {
            let (e, g) = muller_brown_energy_gradient(x);
            (e, g)
        };

        // Two minima of Muller-Brown
        let x_start = vec![-0.558, 1.442];
        let x_end = vec![0.623, 0.028];

        let mut cfg = NEBConfig::default();
        cfg.images = 5;
        cfg.max_iter = 200;
        cfg.conv_tol = 0.5;
        cfg.climbing_image = false;
        cfg.verbose = false;

        let result = neb_optimize(&oracle, &x_start, &x_end, &cfg);
        assert_eq!(result.path.images.len(), 7);
        assert!(result.oracle_calls > 2);
        // Path should have endpoints preserved
        assert!((result.path.images[0][0] - x_start[0]).abs() < 1e-10);
        assert!((result.path.images[6][0] - x_end[0]).abs() < 1e-10);
    }
}
