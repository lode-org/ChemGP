//! GP-NEB with Outer Iteration Evaluation (OIE).
//!
//! Ports `neb_oie.jl`: one oracle call per outer iteration with
//! LCB-guided image selection and adaptive inner relaxation.
//!
//! Reference: Goswami et al., J. Chem. Theory Comput. (2025).

use crate::distances::euclidean_distance;
use crate::hod::{HodConfig, HodState};
use crate::kernel::Kernel;
use crate::neb_path::{
    compute_all_neb_forces, get_hessian_points, linear_interpolation, max_atom_force, path_tangent,
    AcquisitionStrategy, NEBConfig, NEBPath,
};
use crate::neb::{NEBHistory, NEBResult, OracleFn};
use crate::optim_step::OptimState;
use crate::predict::{build_pred_model, PredModel};
use crate::train::{adaptive_train_iters, train_model};
use crate::trust::{clip_images_to_trust, trust_distance, TrustClipParams};
use crate::types::{init_kernel, GPModel, TrainingData};
use crate::StopReason;

use crate::idpp::{idpp_interpolation, sidpp_interpolation};
use rand::SeedableRng;

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
pub fn oie_check_early_stop(
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

/// Per-atom or total force norm for one image.
fn image_force_norm(force: &[f64], d: usize) -> f64 {
    let n_atoms = d / 3;
    if n_atoms >= 1 && d == 3 * n_atoms {
        max_atom_force(force, n_atoms, 3)
    } else {
        force.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

/// Abramowitz & Stegun erf approximation (max error 1.5e-7).
fn erf_approx(x: f64) -> f64 {
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

fn normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))
}

/// Fallback selection: highest NEB force among all intermediate images.
fn fallback_max_force(forces: &crate::neb_path::NEBForces, d: usize, n: usize) -> usize {
    let mut best_f = f64::NEG_INFINITY;
    let mut best_i = 1;
    for i in 1..n - 1 {
        let fn_val = image_force_norm(&forces.forces[i], d);
        if fn_val > best_f {
            best_f = fn_val;
            best_i = i;
        }
    }
    best_i
}

/// Expand a selected center image into a triplet {i-1, i, i+1}.
///
/// Clamps to intermediate images [1, n-2]. At boundaries: center=1
/// gives {1,2}, center=n-2 gives {n-3,n-2}.
fn expand_to_triplet(center: usize, n: usize) -> Vec<usize> {
    let lo = center.saturating_sub(1).max(1);
    let hi = (center + 1).min(n - 2);
    (lo..=hi).collect()
}

/// Select which image to evaluate next using the given acquisition strategy.
fn select_image(
    strategy: &AcquisitionStrategy,
    images: &[Vec<f64>],
    energies: &[f64],
    uneval: &[bool],
    cached_model: &PredModel,
    cached_forces: &crate::neb_path::NEBForces,
    cfg: &NEBConfig,
    d: usize,
    rng: &mut impl rand::Rng,
) -> usize {
    let n = images.len();

    // Only consider unevaluated intermediate images.
    let candidates: Vec<usize> = (1..n - 1).filter(|&i| uneval[i]).collect();
    if candidates.is_empty() {
        return fallback_max_force(cached_forces, d, n);
    }

    // Two-phase gate (CatLearn AcqUUCB pattern): when GP gradient
    // uncertainty is high, select the most uncertain image (pure
    // exploration). Once gradient uncertainty drops below threshold,
    // switch to configured strategy.
    if cfg.unc_convergence > 0.0 {
        let grad_unc = |i: usize| -> f64 {
            let (_, var) = cached_model.predict_with_variance(&images[i]);
            var[1..].iter().map(|v| v.max(0.0).sqrt()).fold(0.0f64, f64::max)
        };
        let max_unc = candidates.iter().map(|&i| grad_unc(i)).fold(0.0f64, f64::max);

        if max_unc > cfg.unc_convergence {
            return candidates
                .iter()
                .copied()
                .max_by(|&a, &b| {
                    grad_unc(a).partial_cmp(&grad_unc(b)).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(1);  // Must be intermediate image (not endpoint)
        }
    }

    match strategy {
        AcquisitionStrategy::MaxVariance => {
            candidates
                .iter()
                .copied()
                .max_by(|&a, &b| {
                    let (_, va) = cached_model.predict_with_variance(&images[a]);
                    let (_, vb) = cached_model.predict_with_variance(&images[b]);
                    va[0].partial_cmp(&vb[0]).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(1)
        }
        AcquisitionStrategy::MaxForce => {
            candidates
                .iter()
                .copied()
                .max_by(|&a, &b| {
                    let fa = image_force_norm(&cached_forces.forces[a], d);
                    let fb = image_force_norm(&cached_forces.forces[b], d);
                    fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(1)
        }
        AcquisitionStrategy::Ucb => {
            let mut best_score = f64::NEG_INFINITY;
            let mut best_i = candidates[0];
            for &i in &candidates {
                let tau = path_tangent(images, energies, i);
                let (_mu, var) = cached_model.predict_with_variance(&images[i]);
                let var_perp: f64 = (0..d)
                    .map(|dd| var[1 + dd].max(0.0) * (1.0 - tau[dd] * tau[dd]))
                    .sum();
                let sigma_perp = var_perp.max(0.0).sqrt();
                let fn_val = image_force_norm(&cached_forces.forces[i], d);
                let score = fn_val + cfg.lcb_kappa * sigma_perp;
                if score > best_score {
                    best_score = score;
                    best_i = i;
                }
            }
            best_i
        }
        AcquisitionStrategy::ExpectedImprovement => {
            // EI over force: E[max(F_i - F_max, 0)]
            // Targets images whose true force may exceed the current worst.
            let f_max = (1..n - 1)
                .map(|i| image_force_norm(&cached_forces.forces[i], d))
                .fold(f64::NEG_INFINITY, f64::max);

            let mut best_ei = f64::NEG_INFINITY;
            let mut best_i = candidates[0];
            for &i in &candidates {
                let tau = path_tangent(images, energies, i);
                let (_mu, var) = cached_model.predict_with_variance(&images[i]);
                let var_perp: f64 = (0..d)
                    .map(|dd| var[1 + dd].max(0.0) * (1.0 - tau[dd] * tau[dd]))
                    .sum();
                let sigma_perp = var_perp.max(0.0).sqrt();
                let fn_val = image_force_norm(&cached_forces.forces[i], d);

                let ei = if sigma_perp < 1e-10 {
                    (fn_val - f_max).max(0.0)
                } else {
                    let z = (fn_val - f_max) / sigma_perp;
                    (fn_val - f_max) * normal_cdf(z) + sigma_perp * normal_pdf(z)
                };
                if ei > best_ei {
                    best_ei = ei;
                    best_i = i;
                }
            }
            best_i
        }
        AcquisitionStrategy::ThompsonSampling => {
            // Draw force samples from GP posterior, pick highest.
            let mut best_sample = f64::NEG_INFINITY;
            let mut best_i = candidates[0];
            for &i in &candidates {
                let tau = path_tangent(images, energies, i);
                let (_mu, var) = cached_model.predict_with_variance(&images[i]);
                let var_perp: f64 = (0..d)
                    .map(|dd| var[1 + dd].max(0.0) * (1.0 - tau[dd] * tau[dd]))
                    .sum();
                let sigma_perp = var_perp.max(0.0).sqrt();
                let fn_val = image_force_norm(&cached_forces.forces[i], d);

                // Box-Muller sample from N(fn_val, sigma_perp^2)
                let u1: f64 = rng.random::<f64>().max(1e-300);
                let u2: f64 = rng.random::<f64>();
                let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                let sample = fn_val + sigma_perp * z;
                if sample > best_sample {
                    best_sample = sample;
                    best_i = i;
                }
            }
            best_i
        }
    }
}

/// Quick-min Velocity Verlet step for one image (Koistinen 2018).
///
/// Projects velocity onto force direction; zeros if antiparallel.
/// Returns updated velocity.
fn qm_vv_update_velocity(
    v: &mut [f64],
    force: &[f64],
    force_old: &[f64],
    dt: f64,
    zero_v: bool,
) {
    let d = v.len();
    if zero_v {
        for i in 0..d { v[i] = 0.0; }
    } else {
        // Velocity Verlet half-step: v += dt/2 * (F_old + F_new)
        for i in 0..d {
            v[i] += dt * 0.5 * (force_old[i] + force[i]);
        }
    }

    // Quick-min: project v onto F direction
    let f_norm_sq: f64 = force.iter().map(|x| x * x).sum();
    if f_norm_sq > 1e-30 {
        let p: f64 = v.iter().zip(force.iter()).map(|(vi, fi)| vi * fi).sum::<f64>()
            / f_norm_sq.sqrt();
        if p < 0.0 {
            // Velocity antiparallel to force: zero it
            for i in 0..d { v[i] = 0.0; }
        } else {
            // Project velocity onto force direction
            let inv_fn = 1.0 / f_norm_sq.sqrt();
            for i in 0..d {
                v[i] = p * force[i] * inv_fn;
            }
        }
    }
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

    // Inner displacement limit for L-BFGS mode
    let inner_trust = if cfg.trust_radius > 0.0 {
        cfg.trust_radius
    } else {
        cfg.max_step_frac * path_scale / (n - 2) as f64
    };

    // QM-VV state: per-image velocity and previous forces
    let dt = cfg.qm_dt;
    let mut velocities: Vec<Vec<f64>> = (0..n_mov).map(|_| vec![0.0; d]).collect();
    let mut prev_forces: Vec<Vec<f64>> = (0..n_mov).map(|_| vec![0.0; d]).collect();
    let mut zero_v = true;

    // Per-image uncertainty scaling (computed once; model is fixed during
    // inner relaxation). Images much more uncertain than the best-known
    // image get smaller steps. Uses excess uncertainty above the minimum
    // so that uniformly-confident GPs (2D surfaces) get scale=1.0
    // everywhere while high-D systems with uneven coverage get selective
    // damping.
    let unc_scales: Vec<f64> = (1..n - 1)
        .map(|i| {
            let (_, var) = model.predict_with_variance(&gp_images[i]);
            // Max gradient uncertainty (not energy) -- NEB forces need
            // accurate gradients; energy sigma can be misleadingly low.
            var[1..].iter().map(|v| v.max(0.0).sqrt()).fold(0.0f64, f64::max)
        })
        .collect();
    let sigma_min = unc_scales.iter().cloned().fold(f64::INFINITY, f64::min);
    let sigma_ref = {
        let sum: f64 = unc_scales.iter().sum();
        (sum / unc_scales.len() as f64).max(1e-10)
    };

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
        let mut gp_forces = compute_all_neb_forces(&gp_path, cfg, ci_on);

        // CI activation mid-relaxation
        if !ci_on && ci_on_outer && cfg.inner_ci_threshold > 0.0
            && gp_forces.max_f < cfg.inner_ci_threshold
        {
            ci_on = true;
            if !cfg.use_quickmin {
                optim = OptimState::new(cfg.lbfgs_memory);
            }
            zero_v = true;
            gp_forces = compute_all_neb_forces(&gp_path, cfg, true);
            if inner > 0 && gp_forces.max_f < gp_tol {
                let ci_idx = gp_forces.i_max;
                return (gp_images, ci_idx, 0);
            }
        }

        // Check convergence (skip first iteration)
        if inner > 0 && gp_forces.max_f < gp_tol {
            let ci_idx = gp_forces.i_max;
            return (gp_images, ci_idx, 0);
        }

        // Save pre-step images for potential revert
        let pre_step_images = gp_images.clone();

        if cfg.use_quickmin {
            // Quick-min Velocity Verlet (MATLAB baseline)
            for im in 0..n_mov {
                let excess = (unc_scales[im] - sigma_min).max(0.0);
                let scale = 1.0 / (1.0 + excess / sigma_ref);
                let force = &gp_forces.forces[im + 1];
                qm_vv_update_velocity(
                    &mut velocities[im], force, &prev_forces[im], dt, zero_v,
                );
                // R_new = R + dt * V + dt^2/2 * F
                for dd in 0..d {
                    gp_images[im + 1][dd] += scale * (dt * velocities[im][dd]
                        + 0.5 * dt * dt * force[dd]);
                }
                // Clip max per-image displacement
                let disp: Vec<f64> = gp_images[im + 1].iter()
                    .zip(pre_step_images[im + 1].iter())
                    .map(|(a, b)| a - b).collect();
                let dn: f64 = disp.iter().map(|x| x * x).sum::<f64>().sqrt();
                if dn > cfg.max_move {
                    for dd in 0..d {
                        gp_images[im + 1][dd] = pre_step_images[im + 1][dd]
                            + disp[dd] * (cfg.max_move / dn);
                    }
                }
                prev_forces[im] = force.clone();
            }
            zero_v = false;
        } else {
            // L-BFGS with displacement trust clip
            let mut cur_x = Vec::with_capacity(n_mov * d);
            let mut cur_force = Vec::with_capacity(n_mov * d);
            for i in 1..=n_mov {
                cur_x.extend_from_slice(&gp_images[i]);
                cur_force.extend_from_slice(&gp_forces.forces[i]);
            }

            let disp = optim.step(&cur_x, &cur_force, cfg.max_move, 3);
            let new_x: Vec<f64> = cur_x.iter().zip(disp.iter()).map(|(a, b)| a + b).collect();

            for img_idx in 0..n_mov {
                let off = img_idx * d;
                let candidate = &new_x[off..off + d];
                let excess = (unc_scales[img_idx] - sigma_min).max(0.0);
                let scale = 1.0 / (1.0 + excess / sigma_ref);
                // Scale displacement from start position by uncertainty
                let disp_vec: Vec<f64> = candidate
                    .iter()
                    .zip(start_images[img_idx + 1].iter())
                    .map(|(a, b)| (a - b) * scale)
                    .collect();
                let dn: f64 = disp_vec.iter().map(|x| x * x).sum::<f64>().sqrt();
                if dn > inner_trust {
                    gp_images[img_idx + 1] = start_images[img_idx + 1]
                        .iter()
                        .zip(disp_vec.iter())
                        .map(|(a, b)| a + b * (inner_trust / dn))
                        .collect();
                } else {
                    gp_images[img_idx + 1] = start_images[img_idx + 1]
                        .iter()
                        .zip(disp_vec.iter())
                        .map(|(a, b)| a + b)
                        .collect();
                }
            }
        }

        // Early stopping guard (from iteration 2 onward)
        if inner >= 1 {
            let (stop, offending) = oie_check_early_stop(&gp_images, td, cfg, path_scale);
            if stop {
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
            ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(1);  // Must be intermediate image (not endpoint)

    (gp_images, ci_idx, early_stop_image)
}

/// GP-NEB with Outer Iteration Evaluation.
///
/// Follows the Koistinen et al. (2019) MATLAB reference:
///   - Image selection by max energy variance among unevaluated images
///   - Path reset to initial/last-converged at start of each relaxation
///   - Exact GP with cached Cholesky (CachedGpModel)
///   - Displacement-based early stopping (disp_max * path_scale)
///   - L-BFGS inner optimizer (replaces MATLAB Quick-min VV)
///
/// When `lcb_kappa > 0`, uses LCB selection (|F| + kappa * sigma_perp)
/// instead of pure variance. When `rff_features > 0`, uses RFF approximation.
/// These extensions can be enabled for larger systems where exact GP is too
/// expensive, but the baseline (lcb_kappa=0, rff_features=0) should be tried
/// first.
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
    let ci_tol = oie_effective_ci_tol(cfg);

    let init_images = init_neb_images(cfg, x_start, x_end);
    let path_scale = oie_path_scale(&init_images);
    let mut images = init_images.clone();

    // Evaluate endpoints
    let (e_start, g_start) = oracle(x_start);
    let (e_end, g_end) = oracle(x_end);
    let mut oracle_calls = 2;

    let mut td = TrainingData::new(d);
    td.add_point(x_start, e_start, &g_start).expect("add_point failed: invalid data");
    td.add_point(x_end, e_end, &g_end).expect("add_point failed: invalid data");

    // Virtual Hessian points
    if cfg.num_hess_iter > 0 {
        let hpts = get_hessian_points(x_start, x_end, cfg.eps_hess);
        for pt in &hpts {
            let (e, g) = oracle(pt);
            td.add_point(pt, e, &g).expect("add_point failed: invalid data");
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

    // Seed GP with initial path evaluation (like standard NEB iteration 0).
    // For molecular systems in high-D, the GP needs path coverage to produce
    // meaningful forces. Cost: n_images oracle calls; benefit: GP starts with
    // full path data instead of just endpoints.
    if cfg.num_hess_iter == 0 {
        // Evaluate all initial intermediate images
        for i in 1..n - 1 {
            let (e, g) = oracle(&images[i]);
            oracle_calls += 1;
            energies[i] = e;
            gradients[i] = g.clone();
            uneval[i] = false;
            td.add_point(&images[i], e, &g).expect("add_point failed: invalid data");
        }
        if cfg.verbose {
            eprintln!(
                "  Seeded GP with {} path evaluations ({} total calls)",
                n - 2, oracle_calls,
            );
        }
    }

    // Train initial GP and predict at all unevaluated images
    let e_ref_init = td.energies[0];
    let mut y_init: Vec<f64> = td.energies.iter().map(|e| e - e_ref_init).collect();
    y_init.extend_from_slice(&td.gradients);
    let kern_init = init_kernel(&td, kernel);
    let mut gp_init = GPModel::new(kern_init, &td, y_init, 1e-6, 1e-4, 1e-6)
        .expect("GPModel::new failed: invalid training data or kernel params");
    gp_init.const_sigma2 = cfg.const_sigma2;
    train_model(&mut gp_init, cfg.gp_train_iter, cfg.verbose);

    // Build initial prediction model
    let init_pred_model = build_pred_model(&gp_init.kernel, &td, cfg.rff_features, 42, cfg.const_sigma2);

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

    // Compute initial NEB forces
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
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut hod_state = HodState::new(cfg.fps_history);
    let hod_cfg = HodConfig {
        monitoring_window: cfg.hod_monitoring_window,
        flip_threshold: cfg.hod_flip_threshold,
        history_increment: cfg.hod_history_increment,
        max_history: cfg.hod_max_history,
    };

    for outer_iter in 0..cfg.max_outer_iter {
        // ---- STEP 1: Select image to evaluate ----
        let i_eval;
        let mut i_extra: Option<usize> = None;

        if eval_next_early > 0 {
            // Priority 1: early-stop image (too far from training data)
            i_eval = eval_next_early;
            eval_next_early = 0;
        } else if eval_next_ci {
            // Priority 2: climbing image (highest energy)
            i_eval = (1..n - 1)
                .max_by(|&a, &b| energies[a].partial_cmp(&energies[b]).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(1);  // Must be intermediate image (not endpoint)
            eval_next_ci = false;
            // Also select an acquisition image so we always evaluate
            // CI + one other, preventing CI drift from stale neighbors.
            let i_acq = select_image(
                &cfg.acquisition, &images, &energies, &uneval,
                &cached_model, &cached_forces, cfg, d, &mut rng,
            );
            if i_acq != i_eval {
                i_extra = Some(i_acq);
            }
        } else {
            // Priority 3: acquisition strategy
            i_eval = select_image(
                &cfg.acquisition, &images, &energies, &uneval,
                &cached_model, &cached_forces, cfg, d, &mut rng,
            );
        }

        // ---- STEP 2: Evaluate images with high GP uncertainty ----
        // Adaptive evaluation: compute uncertainty at all intermediate images
        // and evaluate any whose sigma exceeds the unc_convergence threshold.
        // Early iterations: most images are uncertain -> evaluate many (like std NEB).
        // Later: GP improves -> fewer images need evaluation -> efficient.
        // Always includes the acquisition-selected image.
        // Fallback: when unc_convergence=0, uses evals_per_iter (triplet or single).
        let mut eval_set = if cfg.unc_convergence > 0.0 {
            // Use max gradient uncertainty (not energy) -- NEB forces
            // depend on gradient accuracy. GP can have low energy sigma
            // but high gradient sigma, especially after images move.
            let mut candidates: Vec<(usize, f64)> = (1..n - 1)
                .map(|i| {
                    let (_, var) = cached_model.predict_with_variance(&images[i]);
                    let sigma_g = var[1..].iter()
                        .map(|v| v.max(0.0).sqrt())
                        .fold(0.0f64, f64::max);
                    (i, sigma_g)
                })
                .collect();
            // Sort by uncertainty descending (evaluate most uncertain first)
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut set: Vec<usize> = candidates
                .iter()
                .filter(|&&(_, sigma)| sigma > cfg.unc_convergence)
                .map(|&(i, _)| i)
                .collect();

            // Always include acquisition-selected image
            if !set.contains(&i_eval) {
                set.push(i_eval);
            }
            // When CI is configured, also include the CI image so it stays
            // fresh alongside the acquisition-selected image.  Without
            // this the CI drifts on stale GP forces from neighbors while
            // the acquisition function chases other images.
            if cfg.climbing_image && cached_forces.ci_f >= ci_tol {
                let i_ci = (1..n - 1)
                    .max_by(|&a, &b| {
                        energies[a]
                            .partial_cmp(&energies[b])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(1);
                if !set.contains(&i_ci) {
                    set.push(i_ci);
                }
            }
            set
        } else if cfg.evals_per_iter >= 3 {
            expand_to_triplet(i_eval, n)
        } else {
            vec![i_eval]
        };

        // Include acquisition-selected extra image when CI was forced
        if let Some(extra) = i_extra {
            if !eval_set.contains(&extra) {
                eval_set.push(extra);
            }
        }

        let n_eval_this_iter = eval_set.len();
        for &idx in &eval_set {
            if cfg.max_neb_oracle_calls > 0 && oracle_calls >= cfg.max_neb_oracle_calls {
                break;
            }
            let (e, g) = oracle(&images[idx]);
            oracle_calls += 1;
            energies[idx] = e;
            gradients[idx] = g.clone();
            uneval[idx] = false;

            td.add_point(&images[idx], e, &g).expect("add_point failed: invalid data");
        }

        // Budget exhaustion check
        if cfg.max_neb_oracle_calls > 0 && oracle_calls >= cfg.max_neb_oracle_calls {
            stop_reason = StopReason::OracleCap;
            break;
        }

        // ---- Convergence check (all images evaluated) ----
        let n_uneval: usize = uneval[1..n-1].iter().filter(|&&u| u).count();
        if n_uneval == 0 {
            path.images = images.clone();
            path.energies = energies.clone();
            path.gradients = gradients.clone();
            let all_forces = compute_all_neb_forces(&path, cfg, true);
            // eOn-style convergence: only the climbing image force matters.
            // Non-CI images are along for the ride.
            let conv_f = if cfg.climbing_image { all_forces.ci_f } else { all_forces.max_f };
            if conv_f < ci_tol {
                // Uncertainty gate: require GP uncertainty below threshold.
                let unc_ok = if cfg.unc_conv_tol > 0.0 {
                    let max_unc = (1..n - 1)
                        .map(|i| {
                            let (_, var) = cached_model.predict_with_variance(&images[i]);
                            var[0].max(0.0).sqrt()
                        })
                        .fold(0.0f64, f64::max);
                    max_unc <= cfg.unc_conv_tol
                } else {
                    true
                };
                if unc_ok {
                    stop_reason = StopReason::Converged;
                    history.max_force.push(all_forces.max_f);
                    history.ci_force.push(all_forces.ci_f);
                    history.oracle_calls.push(oracle_calls);
                    history.max_energy.push(energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
                    break;
                }
            }
        }

        // ---- STEP 3: Train GP and update ----
        // FPS subset for hyperparameter training (HOD may grow this).
        // Uses select_optim_subset (FPS seeded from latest points) instead of
        // bead_local_subset for stability: subset changes incrementally as new
        // data arrives rather than jumping when images move.
        let fps_size = if cfg.use_hod {
            hod_state.current_fps_history.max(cfg.max_gp_points)
        } else {
            cfg.max_gp_points
        };
        let td_use = if fps_size > 0 && td.npoints() > fps_size {
            let dist_fn = |a: &[f64], b: &[f64]| -> f64 {
                trust_distance(cfg.trust_metric, &cfg.atom_types, a, b)
            };
            let sub_idx = crate::sampling::select_optim_subset(
                &td, &images[n / 2], fps_size, cfg.fps_latest_points, &dist_fn,
            );
            td.extract_subset(&sub_idx)
        } else {
            td.clone()
        };

        let train_iters = adaptive_train_iters(cfg.gp_train_iter, prev_kern.is_none());

        let e_ref_sub = td_use.energies[0];
        let mut y_sub: Vec<f64> = td_use.energies.iter().map(|e| e - e_ref_sub).collect();
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

        // HOD: detect hyperparameter oscillation and grow FPS subset
        if cfg.use_hod {
            let grew = hod_state.check(&gp_sub, &hod_cfg);
            if grew && cfg.verbose {
                eprintln!(
                    "  HOD: oscillation detected, FPS subset grown to {}",
                    hod_state.current_fps_history,
                );
            }
        }

        // Build prediction model on full data
        let e_ref_full = td.energies[0];
        let pred_model = build_pred_model(&gp_sub.kernel, &td, cfg.rff_features, 42, cfg.const_sigma2);

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

        // Cache model and forces for next iteration
        cached_model = pred_model;
        cached_forces = neb_forces;

        // Update smallest accurate force from NEB force at evaluated image
        {
            let f_neb_norm = image_force_norm(&cached_forces.forces[i_eval], d);
            smallest_acc_force = smallest_acc_force.min(f_neb_norm);
        }

        // History
        history.max_force.push(cached_forces.max_f);
        history.ci_force.push(cached_forces.ci_f);
        history.oracle_calls.push(oracle_calls);
        history.max_energy.push(energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

        // eOn-style GP-based convergence: only the CI force matters.
        // Accept convergence even if some non-CI images have larger forces
        // or are still unevaluated.
        let gp_conv_f = if cfg.climbing_image { cached_forces.ci_f } else { cached_forces.max_f };
        if gp_conv_f < ci_tol {
            stop_reason = StopReason::Converged;
            if cfg.verbose {
                eprintln!(
                    "  GP-convergence: CI|F| = {:.5} < ci_tol = {:.3}",
                    gp_conv_f, ci_tol,
                );
            }
            break;
        }

        // Stagnation check (longer window to allow path evolution)
        let stag_tol = 1e-6f64.max(1e-4 * cached_forces.max_f);
        if (cached_forces.max_f - prev_max_f).abs() < stag_tol {
            stagnation_count += 1;
        } else {
            stagnation_count = 0;
        }
        prev_max_f = cached_forces.max_f;

        if stagnation_count >= 15 {
            stop_reason = StopReason::ForceStagnation;
            break;
        }

        if cfg.verbose {
            eprintln!(
                "GP-NEB-OIE outer {}: max|F| = {:.5} | CI|F| = {:.5} | eval={} ({} imgs) | calls = {}",
                outer_iter, cached_forces.max_f, cached_forces.ci_f, i_eval, n_eval_this_iter, oracle_calls,
            );
        }

        // ---- STEP 4: Decide whether to relax (MATLAB lines 282-298) ----
        let i_ci = (1..n - 1)
            .max_by(|&a, &b| energies[a].partial_cmp(&energies[b]).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(1);  // Must be intermediate image (not endpoint)

        let mut start_relax = false;

        if cached_forces.max_f >= cfg.conv_tol {
            start_relax = true;
            eval_next_ci = false;
        } else if uneval[i_ci] {
            eval_next_ci = true;
        } else if cached_forces.ci_f >= ci_tol {
            start_relax = true;
            eval_next_ci = true;
        }

        // ---- STEP 5: Relax on GP surface ----
        if start_relax {
            let pre_relax_images = images.clone();

            let gp_tol_val = oie_gp_tol(cfg, smallest_acc_force);

            // GP-predict at current positions
            let mut relax_energies = energies.clone();
            let mut relax_gradients = gradients.clone();
            for i in 1..n - 1 {
                let preds = cached_model.predict(&images[i]);
                relax_energies[i] = preds[0] + e_ref_full;
                relax_gradients[i] = preds[1..].to_vec();
            }

            // Pre-relaxation GP-only forces (for revert comparison)
            let pre_relax_max_f = if cfg.unc_revert_tol > 0.0 {
                let pre_path = NEBPath {
                    images: images.clone(),
                    energies: relax_energies.clone(),
                    gradients: relax_gradients.clone(),
                    spring_constant: cfg.spring_constant,
                };
                compute_all_neb_forces(&pre_path, cfg, cfg.climbing_image).max_f
            } else {
                0.0 // unused
            };

            let (new_images, _ci_idx, early_img) = oie_inner_relax(
                &cached_model,
                &images,
                &relax_energies,
                &relax_gradients,
                &td,
                cfg,
                cfg.climbing_image,
                e_ref_full,
                gp_tol_val,
                path_scale,
            );

            images = new_images;

            // Optional EMD trust clip
            if cfg.trust_radius > 0.0 {
                let trust_params = TrustClipParams {
                    trust_radius: cfg.trust_radius,
                    trust_metric: cfg.trust_metric,
                    atom_types: &cfg.atom_types,
                    use_adaptive: cfg.use_adaptive_threshold,
                    adaptive_t_min: cfg.adaptive_t_min,
                    adaptive_delta_t: cfg.adaptive_delta_t,
                    adaptive_n_half: cfg.adaptive_n_half,
                    adaptive_a: cfg.adaptive_a,
                    adaptive_floor: cfg.adaptive_floor,
                };
                clip_images_to_trust(&mut images, &td, &trust_params);
            }

            // Force-improvement + uncertainty gate (opt-in via unc_revert_tol > 0).
            // For molecular systems: re-predict NEB forces and GP variance at
            // relaxed positions. Revert if forces worsened OR GP too uncertain.
            // For 2D surfaces (unc_revert_tol=0): always accept relaxation.
            let mut should_revert = false;
            if cfg.unc_revert_tol > 0.0 {
                let mut post_energies = energies.clone();
                let mut post_gradients = gradients.clone();
                let mut max_unc: f64 = 0.0;
                for i in 1..n - 1 {
                    let preds = cached_model.predict(&images[i]);
                    post_energies[i] = preds[0] + e_ref_full;
                    post_gradients[i] = preds[1..].to_vec();
                    let (_, var) = cached_model.predict_with_variance(&images[i]);
                    max_unc = max_unc.max(var[0].max(0.0).sqrt());
                }
                let post_path = NEBPath {
                    images: images.clone(),
                    energies: post_energies,
                    gradients: post_gradients,
                    spring_constant: cfg.spring_constant,
                };
                let post_forces = compute_all_neb_forces(&post_path, cfg, cfg.climbing_image);

                let force_worse = post_forces.max_f > pre_relax_max_f;
                let unc_too_high = max_unc > cfg.unc_revert_tol;

                if force_worse || unc_too_high {
                    should_revert = true;
                    if cfg.verbose {
                        eprintln!(
                            "  Revert relaxation: force_worse={} (post={:.4} > pre={:.4}), unc_high={} (max_unc={:.4})",
                            force_worse, post_forces.max_f, pre_relax_max_f, unc_too_high, max_unc,
                        );
                    }
                }
            }

            if should_revert {
                // Partial revert: accept 10% of relaxation displacement.
                // Full revert creates a deadlock (no new positions -> no new
                // training data -> no GP improvement). Partial acceptance
                // ensures images always move, providing fresh data while
                // limiting damage from bad GP predictions.
                let scale = 0.1;
                for i in 1..n - 1 {
                    for dd in 0..d {
                        images[i][dd] = pre_relax_images[i][dd]
                            + scale * (images[i][dd] - pre_relax_images[i][dd]);
                    }
                }
            }
            // Always mark unevaluated: images moved (fully or partially)
            for i in 1..n - 1 {
                uneval[i] = true;
            }

            // Record early stop image for priority evaluation
            if early_img > 0 {
                eval_next_early = early_img;
            }

            // Reset stagnation counter
            stagnation_count = 0;
        }
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
