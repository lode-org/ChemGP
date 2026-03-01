//! GP-NEB with Outer Iteration Evaluation (OIE).
//!
//! Ports `neb_oie.jl`: one oracle call per outer iteration with
//! LCB-guided image selection and adaptive inner relaxation.
//!
//! Reference: Goswami et al., J. Chem. Theory Comput. (2025).

use crate::distances::euclidean_distance;
use crate::kernel::Kernel;
use crate::neb_path::{
    compute_all_neb_forces, get_hessian_points, linear_interpolation, max_atom_force, path_tangent,
    NEBConfig, NEBPath,
};
use crate::neb::{NEBHistory, NEBResult, OracleFn};
use crate::optim_step::OptimState;
use crate::predict::{build_pred_model, PredModel};
use crate::sampling::select_optim_subset;
use crate::train::train_model;
use crate::trust::{
    adaptive_trust_threshold, min_distance_to_data, trust_distance, trust_min_distance,
};
use crate::types::{init_kernel, GPModel, TrainingData};
use crate::StopReason;

use crate::idpp::{idpp_interpolation, sidpp_interpolation};

/// Initialize NEB path images (same as neb.rs).
fn init_neb_images(cfg: &NEBConfig, x_start: &[f64], x_end: &[f64]) -> Vec<Vec<f64>> {
    let n_total = cfg.images + 2;
    match cfg.initializer.as_str() {
        "sidpp" => sidpp_interpolation(
            x_start,
            x_end,
            n_total,
            3, 200, 0.1, 0.01, 10,
            cfg.spring_constant, 0.3,
        ),
        "idpp" => idpp_interpolation(x_start, x_end, n_total, 3, 200, 0.1, 0.01, 10),
        _ => linear_interpolation(x_start, x_end, n_total),
    }
}

/// Effective CI tolerance.
fn oie_effective_ci_tol(cfg: &NEBConfig) -> f64 {
    if cfg.ci_force_tol > 0.0 {
        cfg.ci_force_tol
    } else {
        cfg.conv_tol
    }
}

/// Adaptive inner GP convergence threshold.
fn oie_gp_tol(cfg: &NEBConfig, smallest_acc_force: f64) -> f64 {
    let ci_tol = oie_effective_ci_tol(cfg);
    let floor_tol = cfg.conv_tol.min(ci_tol) / 10.0;
    if cfg.gp_tol_divisor > 0 && smallest_acc_force.is_finite() {
        (smallest_acc_force / cfg.gp_tol_divisor as f64).max(floor_tol)
    } else {
        floor_tol
    }
}

/// Sum of consecutive Euclidean distances along the path.
fn oie_path_scale(images: &[Vec<f64>]) -> f64 {
    let mut total = 0.0;
    for i in 1..images.len() {
        total += euclidean_distance(&images[i], &images[i - 1]);
    }
    total
}

/// Check if any intermediate image has violated bond stretch or displacement limits.
/// Returns (should_stop, offending_image_index).
fn oie_check_early_stop(
    images: &[Vec<f64>],
    td: &TrainingData,
    cfg: &NEBConfig,
    path_scale: f64,
) -> (bool, usize) {
    let n = images.len();
    let d = images[0].len();
    let log_limit = cfg.bond_stretch_limit.ln().abs();
    let n_atoms = (d / 3).max(1);
    let disp_limit = cfg.max_step_frac * path_scale * (n_atoms as f64).sqrt();

    for i in 1..n - 1 {
        // Bond stretch check (only for systems with >= 2 atoms)
        if d >= 6 {
            let min_log_d = (0..td.npoints())
                .map(|j| crate::distances::max_1d_log_distance(&images[i], td.col(j)))
                .fold(f64::INFINITY, f64::min);
            if min_log_d > log_limit {
                return (true, i);
            }
        }

        // Displacement check
        let min_disp = (0..td.npoints())
            .map(|j| euclidean_distance(&images[i], td.col(j)))
            .fold(f64::INFINITY, f64::min);
        if min_disp > disp_limit {
            return (true, i);
        }
    }

    (false, 0)
}

/// Inner GP relaxation for OIE with CI mid-activation and early stopping.
///
/// Returns (relaxed_images, ci_index, early_stop_image).
fn oie_inner_relax(
    model: &PredModel,
    images: &[Vec<f64>],
    energies: &[f64],
    gradients: &[Vec<f64>],
    td: &TrainingData,
    cfg: &NEBConfig,
    ci_on_outer: bool,
    e_ref: f64,
    gp_tol: f64,
    path_scale: f64,
) -> (Vec<Vec<f64>>, usize, usize) {
    let n = images.len();
    let d = images[0].len();
    let n_mov = n - 2;

    let mut gp_images = images.to_vec();
    let start_images = images.to_vec();
    let mut optim = OptimState::new(cfg.lbfgs_memory);
    let mut ci_on = false;
    let mut early_stop_image = 0;

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
            energies: gp_energies.clone(),
            gradients: gp_gradients,
            spring_constant: cfg.spring_constant,
        };
        let gp_forces = compute_all_neb_forces(&gp_path, cfg, ci_on);

        // CI activation mid-relaxation
        if !ci_on && ci_on_outer && cfg.inner_ci_threshold > 0.0
            && gp_forces.max_f < cfg.inner_ci_threshold
        {
            ci_on = true;
            optim = OptimState::new(cfg.lbfgs_memory);
            // Recompute forces with CI
            let gp_forces_ci = compute_all_neb_forces(&gp_path, cfg, true);
            if inner > 0 && gp_forces_ci.max_f < gp_tol {
                let ci_idx = gp_forces_ci.i_max;
                return (gp_images, ci_idx, 0);
            }
            // Continue with CI forces below
        }

        // Check convergence (skip first iteration)
        if inner > 0 && gp_forces.max_f < gp_tol {
            let ci_idx = gp_forces.i_max;
            return (gp_images, ci_idx, 0);
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

        // Save pre-step images for potential revert
        let pre_step_images = gp_images.clone();

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

        // Early stopping guard (from iteration 2 onward)
        if inner >= 1 {
            let (stop, offending) = oie_check_early_stop(&gp_images, td, cfg, path_scale);
            if stop {
                // Revert to pre-step images
                gp_images = pre_step_images;
                early_stop_image = offending;
                break;
            }
        }
    }

    let ci_idx = (1..n - 1)
        .max_by(|&a, &b| {
            let ea = {
                let p = model.predict(&gp_images[a]);
                p[0] + e_ref
            };
            let eb = {
                let p = model.predict(&gp_images[b]);
                p[0] + e_ref
            };
            ea.partial_cmp(&eb).unwrap()
        })
        .unwrap_or(1);

    (gp_images, ci_idx, early_stop_image)
}

/// EMD trust clip (shared with AIE, but kept as local fn for clarity).
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
            &images[i], &td.data, d, td.npoints(), cfg.trust_metric, &cfg.atom_types,
        );
        if dist > thresh {
            let nearest_idx = (0..td.npoints())
                .min_by(|&a, &b| {
                    let da = trust_distance(cfg.trust_metric, &cfg.atom_types, &images[i], td.col(a));
                    let db = trust_distance(cfg.trust_metric, &cfg.atom_types, &images[i], td.col(b));
                    da.partial_cmp(&db).unwrap()
                })
                .unwrap();
            let nearest = td.col(nearest_idx).to_vec();
            let disp: Vec<f64> = images[i].iter().zip(nearest.iter()).map(|(a, b)| a - b).collect();
            images[i] = nearest
                .iter()
                .zip(disp.iter())
                .map(|(a, b)| a + b * (thresh / dist * 0.95))
                .collect();
        }
    }
}

/// GP-NEB with Outer Iteration Evaluation.
///
/// Each outer iteration evaluates the oracle at exactly one image, selected
/// by a priority cascade: early-stop image > climbing image > LCB selection.
pub fn gp_neb_oie(
    oracle: &OracleFn,
    x_start: &[f64],
    x_end: &[f64],
    kernel: &Kernel,
    config: &NEBConfig,
) -> NEBResult {
    let cfg = config;
    let n = cfg.images + 2;
    let d = x_start.len();
    let dedup_tol = cfg.conv_tol * 0.1;
    let ci_tol = oie_effective_ci_tol(cfg);

    let mut images = init_neb_images(cfg, x_start, x_end);
    let path_scale = oie_path_scale(&images);

    // Evaluate endpoints
    let (e_start, g_start) = oracle(x_start);
    let (e_end, g_end) = oracle(x_end);
    let mut oracle_calls = 2;

    let mut td = TrainingData::new(d);
    td.add_point(x_start, e_start, &g_start);
    td.add_point(x_end, e_end, &g_end);

    // Virtual Hessian points
    if cfg.num_hess_iter > 0 {
        let hpts = get_hessian_points(x_start, x_end, cfg.eps_hess);
        for pt in &hpts {
            let (e, g) = oracle(pt);
            td.add_point(pt, e, &g);
            oracle_calls += 1;
        }
    }

    // Initialize energies/gradients
    let mut energies = vec![0.0; n];
    let mut gradients: Vec<Vec<f64>> = (0..n).map(|_| vec![0.0; d]).collect();
    energies[0] = e_start;
    energies[n - 1] = e_end;
    gradients[0] = g_start.clone();
    gradients[n - 1] = g_end.clone();

    // Mark which images are unevaluated
    let mut uneval = vec![true; n];
    uneval[0] = false;
    uneval[n - 1] = false;

    // Train initial GP and predict at all unevaluated images
    let e_ref_init = td.energies[0];
    let mut y_init: Vec<f64> = td.energies.iter().map(|e| e - e_ref_init).collect();
    y_init.extend_from_slice(&td.gradients);
    let kern_init = init_kernel(&td, kernel);
    let mut gp_init = GPModel::new(kern_init, &td, y_init, 1e-6, 1e-4, 1e-6);
    train_model(&mut gp_init, cfg.gp_train_iter, cfg.verbose);

    // Build initial prediction model
    let init_pred_model = build_pred_model(&gp_init.kernel, &td, cfg.rff_features, 42);

    for i in 1..n - 1 {
        let preds = init_pred_model.predict(&images[i]);
        energies[i] = preds[0] + e_ref_init;
        gradients[i] = preds[1..].to_vec();
    }

    let mut path = NEBPath {
        images: images.clone(),
        energies: energies.clone(),
        gradients: gradients.clone(),
        spring_constant: cfg.spring_constant,
    };

    // Compute initial NEB forces for LCB selection
    let init_forces = compute_all_neb_forces(&path, cfg, false);

    let mut history = NEBHistory::default();
    let mut prev_kern: Option<Kernel> = Some(gp_init.kernel.clone());
    let mut stop_reason = StopReason::MaxIterations;
    let mut eval_next_early: usize = 0;
    let mut eval_next_ci = false;
    let mut smallest_acc_force = f64::INFINITY;
    let mut cached_model = init_pred_model;
    let mut cached_forces = init_forces;
    let mut stagnation_count = 0;
    let mut prev_max_f = f64::NEG_INFINITY;

    for outer_iter in 0..cfg.max_outer_iter {
        // ---- STEP 1: Select image to evaluate ----
        let i_eval;

        if eval_next_early > 0 {
            // Priority 1: early-stop image
            i_eval = eval_next_early;
            eval_next_early = 0;
        } else if eval_next_ci {
            // Priority 2: climbing image (highest energy)
            i_eval = (1..n - 1)
                .max_by(|&a, &b| energies[a].partial_cmp(&energies[b]).unwrap())
                .unwrap_or(1);
            eval_next_ci = false;
        } else {
            // Priority 3: LCB image selection
            let mut best_score = f64::NEG_INFINITY;
            let mut best_i = 1;

            for i in 1..n - 1 {
                if !uneval[i] {
                    // Use cached force magnitude for evaluated images
                    let fn_val = {
                        let f = &cached_forces.forces[i];
                        let n_atoms = d / 3;
                        if n_atoms >= 1 && d == 3 * n_atoms {
                            max_atom_force(f, n_atoms, 3)
                        } else {
                            f.iter().map(|x| x * x).sum::<f64>().sqrt()
                        }
                    };
                    if fn_val > best_score {
                        best_score = fn_val;
                        best_i = i;
                    }
                    continue;
                }

                // For unevaluated images: compute LCB score
                let tau = path_tangent(&images, &energies, i);
                let (_mu, var) = cached_model.predict_with_variance(&images[i]);

                // Perpendicular variance: project gradient variance into perpendicular subspace
                let mut var_perp = 0.0;
                for dd in 0..d {
                    let v = var[1 + dd].max(0.0);
                    var_perp += v * (1.0 - tau[dd] * tau[dd]);
                }
                let sigma_perp = var_perp.max(0.0).sqrt();

                // NEB force magnitude at this image
                let fn_val = {
                    let f = &cached_forces.forces[i];
                    let n_atoms = d / 3;
                    if n_atoms >= 1 && d == 3 * n_atoms {
                        max_atom_force(f, n_atoms, 3)
                    } else {
                        f.iter().map(|x| x * x).sum::<f64>().sqrt()
                    }
                };

                let score = if sigma_perp > 1e-4 {
                    fn_val + cfg.lcb_kappa * sigma_perp
                } else {
                    fn_val
                };

                if score > best_score {
                    best_score = score;
                    best_i = i;
                }
            }
            i_eval = best_i;
        }

        // ---- STEP 2: Evaluate oracle at selected image ----
        let (e_eval, g_eval) = oracle(&images[i_eval]);
        oracle_calls += 1;
        energies[i_eval] = e_eval;
        gradients[i_eval] = g_eval.clone();
        uneval[i_eval] = false;

        if min_distance_to_data(&images[i_eval], &td.data, d, td.npoints()) > dedup_tol {
            td.add_point(&images[i_eval], e_eval, &g_eval);
        }

        // Update smallest accurate force
        let f_eval_norm = {
            let n_atoms = d / 3;
            if n_atoms >= 1 && d == 3 * n_atoms {
                max_atom_force(&g_eval, n_atoms, 3)
            } else {
                g_eval.iter().map(|x| x * x).sum::<f64>().sqrt()
            }
        };
        smallest_acc_force = smallest_acc_force.min(f_eval_norm);

        // ---- Convergence check ----
        let i_ci = (1..n - 1)
            .max_by(|&a, &b| energies[a].partial_cmp(&energies[b]).unwrap())
            .unwrap_or(1);
        if !uneval[i_ci] {
            // CI has been evaluated: check its NEB force
            path.images = images.clone();
            path.energies = energies.clone();
            path.gradients = gradients.clone();
            let ci_forces = compute_all_neb_forces(&path, cfg, true);
            if ci_forces.ci_f < ci_tol {
                stop_reason = StopReason::Converged;
                history.max_force.push(ci_forces.max_f);
                history.ci_force.push(ci_forces.ci_f);
                history.oracle_calls.push(oracle_calls);
                history.max_energy.push(energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
                break;
            }
        }

        // ---- STEP 3: Train GP and update ----
        let td_use = if cfg.fps_history > 0 && td.npoints() > cfg.fps_history {
            let dist_fn = |a: &[f64], b: &[f64]| -> f64 {
                trust_distance(cfg.trust_metric, &cfg.atom_types, a, b)
            };
            let center: Vec<f64> = images[1..n-1].iter()
                .flat_map(|img| img.iter())
                .copied()
                .collect::<Vec<_>>()
                .chunks(d)
                .next()
                .unwrap_or(&images[1])
                .to_vec();
            let sub_idx = select_optim_subset(
                &td, &center, cfg.fps_history, cfg.fps_latest_points, &dist_fn,
            );
            td.extract_subset(&sub_idx)
        } else {
            td.clone()
        };

        let train_iters = if prev_kern.is_none() {
            cfg.gp_train_iter
        } else {
            (cfg.gp_train_iter / 3).max(50)
        };

        let e_ref_sub = td_use.energies[0];
        let mut y_sub: Vec<f64> = td_use.energies.iter().map(|e| e - e_ref_sub).collect();
        y_sub.extend_from_slice(&td_use.gradients);

        let kern = match &prev_kern {
            None => init_kernel(&td_use, kernel),
            Some(k) => k.clone(),
        };

        let mut gp_sub = GPModel::new(kern, &td_use, y_sub, 1e-6, 1e-4, 1e-6);
        train_model(&mut gp_sub, train_iters, cfg.verbose);
        prev_kern = Some(gp_sub.kernel.clone());

        // Build prediction model on full data
        let e_ref_full = td.energies[0];
        let pred_model = build_pred_model(&gp_sub.kernel, &td, cfg.rff_features, 42);

        // Predict at unevaluated images
        for i in 1..n - 1 {
            if uneval[i] {
                let preds = pred_model.predict(&images[i]);
                energies[i] = preds[0] + e_ref_full;
                gradients[i] = preds[1..].to_vec();
            }
        }

        path.images = images.clone();
        path.energies = energies.clone();
        path.gradients = gradients.clone();
        let neb_forces = compute_all_neb_forces(&path, cfg, cfg.climbing_image);

        // Cache model and forces for next iteration's LCB
        cached_model = pred_model;
        cached_forces = neb_forces;

        // History
        history.max_force.push(cached_forces.max_f);
        history.ci_force.push(cached_forces.ci_f);
        history.oracle_calls.push(oracle_calls);
        history.max_energy.push(energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

        // Stagnation check (5-step window for OIE)
        let stag_tol = 1e-6f64.max(1e-4 * cached_forces.max_f);
        if (cached_forces.max_f - prev_max_f).abs() < stag_tol {
            stagnation_count += 1;
        } else {
            stagnation_count = 0;
        }
        prev_max_f = cached_forces.max_f;

        if stagnation_count >= 5 {
            stop_reason = StopReason::ForceStagnation;
            break;
        }

        if cfg.verbose {
            eprintln!(
                "GP-NEB-OIE outer {}: max|F| = {:.5} | CI|F| = {:.5} | eval={} | calls = {}",
                outer_iter, cached_forces.max_f, cached_forces.ci_f, i_eval, oracle_calls,
            );
        }

        // ---- STEP 4: Decide whether to relax ----
        let mut start_relax = false;

        if cached_forces.max_f >= cfg.conv_tol {
            start_relax = true;
        } else if uneval[i_ci] {
            eval_next_ci = true;
        } else if cached_forces.ci_f >= ci_tol {
            start_relax = true;
            eval_next_ci = true;
        }
        // else: ci_f < ci_tol, convergence will be checked next iteration

        // ---- STEP 5: Relax on GP surface ----
        if start_relax {
            let gp_tol_val = oie_gp_tol(cfg, smallest_acc_force);
            let (new_images, _ci_idx, early_img) = oie_inner_relax(
                &cached_model,
                &images,
                &energies,
                &gradients,
                &td,
                cfg,
                cfg.climbing_image,
                e_ref_full,
                gp_tol_val,
                path_scale,
            );

            images = new_images;

            // Post-inner EMD trust clip
            emd_trust_clip(&mut images, &td, cfg);

            // Reset all intermediate images to unevaluated
            for i in 1..n - 1 {
                uneval[i] = true;
            }

            // Record early stop image for priority evaluation
            if early_img > 0 {
                eval_next_early = early_img;
            }

            // Update energies/gradients with GP predictions at new positions
            for i in 1..n - 1 {
                let preds = cached_model.predict(&images[i]);
                energies[i] = preds[0] + e_ref_full;
                gradients[i] = preds[1..].to_vec();
            }

            path.images = images.clone();
            path.energies = energies.clone();
            path.gradients = gradients.clone();
        }
    }

    let i_max = (1..n - 1)
        .max_by(|&a, &b| energies[a].partial_cmp(&energies[b]).unwrap())
        .unwrap_or(1);

    NEBResult {
        path,
        converged: stop_reason == StopReason::Converged,
        stop_reason,
        oracle_calls,
        max_energy_image: i_max,
        history,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::potentials::{leps_energy_gradient, LEPS_REACTANT, LEPS_PRODUCT};

    #[test]
    fn test_gp_neb_oie_leps() {
        let oracle = |x: &[f64]| -> (f64, Vec<f64>) { leps_energy_gradient(x) };

        let x_start = LEPS_REACTANT.to_vec();
        let x_end = LEPS_PRODUCT.to_vec();

        let mut cfg = NEBConfig::default();
        cfg.images = 3;
        cfg.max_outer_iter = 10;
        cfg.max_iter = 50;
        cfg.conv_tol = 1.0;
        cfg.climbing_image = false;
        cfg.verbose = false;

        let kernel = Kernel::MolInvDist(crate::kernel::MolInvDistSE::isotropic(1.0, 1.0, vec![]));

        let result = gp_neb_oie(&oracle, &x_start, &x_end, &kernel, &cfg);
        assert_eq!(result.path.images.len(), 5);
        assert!(result.oracle_calls > 2);
        // Endpoints preserved
        assert!((result.path.images[0][0] - x_start[0]).abs() < 1e-10);
        assert!((result.path.images[4][0] - x_end[0]).abs() < 1e-10);
    }
}
