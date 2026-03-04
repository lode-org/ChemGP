//! GP-guided minimization.
//!
//! Ports `minimize.jl`: the main outer loop.

use crate::distances::euclidean_distance;
use crate::kernel::Kernel;
use crate::optim_step::clip_to_max_move;
use crate::predict::build_pred_model;
use crate::sampling::{prune_training_data, select_optim_subset};
use crate::train::train_model;
use crate::trust::{
    adaptive_trust_threshold, remove_rigid_body_modes, trust_distance, trust_min_distance,
    TrustMetric,
};
use crate::types::{init_kernel, GPModel, TrainingData};
use crate::StopReason;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for GP-guided minimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimizationConfig {
    pub trust_radius: f64,
    pub conv_tol: f64,
    pub max_iter: usize,
    pub max_oracle_calls: usize,
    pub gp_train_iter: usize,
    pub n_initial_perturb: usize,
    pub perturb_scale: f64,
    pub penalty_coeff: f64,
    pub max_move: f64,
    pub energy_regression_tol: f64,
    pub max_training_points: usize,
    pub rff_features: usize,
    pub fps_history: usize,
    pub fps_latest_points: usize,
    pub trust_metric: TrustMetric,
    pub atom_types: Vec<i32>,
    pub use_adaptive_threshold: bool,
    pub adaptive_t_min: f64,
    pub adaptive_delta_t: f64,
    pub adaptive_n_half: usize,
    pub adaptive_a: f64,
    pub adaptive_floor: f64,
    /// Constant kernel variance for energy-energy block.
    /// Set to 1.0 for molecular systems; 0.0 disables (default).
    pub const_sigma2: f64,
    /// LCB kappa for inner loop convergence. When > 0, uses
    /// |G| + kappa * sigma_g < threshold, where sigma_g is the total
    /// gradient uncertainty (no orient direction for minimize).
    /// 0.0 = disabled (default, backward compatible).
    pub lcb_kappa: f64,
    pub verbose: bool,
}

impl Default for MinimizationConfig {
    fn default() -> Self {
        Self {
            trust_radius: 0.1,
            conv_tol: 5e-3,
            max_iter: 500,
            max_oracle_calls: 0,
            gp_train_iter: 300,
            n_initial_perturb: 4,
            perturb_scale: 0.1,
            penalty_coeff: 1e3,
            max_move: 0.1,
            energy_regression_tol: 0.0,
            max_training_points: 0,
            rff_features: 0,
            fps_history: 0,
            fps_latest_points: 2,
            trust_metric: TrustMetric::Emd,
            atom_types: Vec::new(),
            use_adaptive_threshold: false,
            adaptive_t_min: 0.15,
            adaptive_delta_t: 0.35,
            adaptive_n_half: 50,
            adaptive_a: 1.3,
            adaptive_floor: 0.2,
            const_sigma2: 0.0,
            lcb_kappa: 0.0,
            verbose: true,
        }
    }
}

/// Result of GP-guided minimization.
#[derive(Debug, Clone)]
pub struct MinimizationResult {
    pub x_final: Vec<f64>,
    pub e_final: f64,
    pub g_final: Vec<f64>,
    pub converged: bool,
    pub stop_reason: StopReason,
    pub oracle_calls: usize,
    pub trajectory: Vec<Vec<f64>>,
    pub energies: Vec<f64>,
}

/// Oracle function type: x -> (energy, gradient).
pub type OracleFn = dyn Fn(&[f64]) -> (f64, Vec<f64>);

/// GP-guided minimization of an oracle function.
pub fn gp_minimize(
    oracle: &OracleFn,
    x_init: &[f64],
    kernel: &Kernel,
    config: &MinimizationConfig,
    training_data: Option<TrainingData>,
) -> MinimizationResult {
    let d = x_init.len();
    let cfg = config;

    let mut td = training_data.unwrap_or_else(|| TrainingData::new(d));
    let mut trajectory: Vec<Vec<f64>> = Vec::new();
    let mut all_energies: Vec<f64> = Vec::new();

    // Step 1: Generate initial training data
    if td.npoints() == 0 {
        if cfg.verbose {
            eprintln!("Generating initial training data...");
        }

        let (e, g) = oracle(x_init);
        td.add_point(x_init, e, &g);
        trajectory.push(x_init.to_vec());
        all_energies.push(e);

        let mut rng = rand::rng();
        for _k in 0..cfg.n_initial_perturb {
            let perturb: Vec<f64> = (0..d)
                .map(|_| (rng.random::<f64>() - 0.5) * cfg.perturb_scale)
                .collect();
            let x_p: Vec<f64> = x_init.iter().zip(perturb.iter()).map(|(a, b)| a + b).collect();
            let (e_p, g_p) = oracle(&x_p);
            if e_p.is_finite() && e_p < 1e6 {
                td.add_point(&x_p, e_p, &g_p);
                trajectory.push(x_p);
                all_energies.push(e_p);
            }
        }
    }

    let mut x_curr = x_init.to_vec();
    let mut oracle_calls = td.npoints();
    let mut prev_kern: Option<Kernel> = None;
    let mut stagnation_count = 0;
    let mut prev_force = f64::NEG_INFINITY;
    let mut stop_reason = StopReason::MaxIterations;

    for outer_step in 0..cfg.max_iter {
        if cfg.max_oracle_calls > 0 && oracle_calls >= cfg.max_oracle_calls {
            stop_reason = StopReason::OracleCap;
            break;
        }

        // FPS subset selection
        let dist_fn = |a: &[f64], b: &[f64]| -> f64 {
            trust_distance(cfg.trust_metric, &cfg.atom_types, a, b)
        };

        let td_sub = if cfg.fps_history > 0 && td.npoints() > cfg.fps_history {
            let sub_idx =
                select_optim_subset(&td, &x_curr, cfg.fps_history, cfg.fps_latest_points, &dist_fn);
            td.extract_subset(&sub_idx)
        } else {
            td.clone()
        };

        let train_iters = if prev_kern.is_none() {
            cfg.gp_train_iter
        } else {
            (cfg.gp_train_iter / 3).max(50)
        };

        // Train on subset
        let e_ref_sub = td_sub.energies[0];
        let mut y_sub: Vec<f64> = td_sub.energies.iter().map(|e| e - e_ref_sub).collect();
        y_sub.extend_from_slice(&td_sub.gradients);

        let kern = match &prev_kern {
            None => init_kernel(&td_sub, kernel),
            Some(k) => k.clone(),
        };

        // Clamp hyperparams
        let clamped_ls: Vec<f64> = kern
            .inv_lengthscales()
            .iter()
            .map(|&x| {
                if x.is_finite() {
                    x.clamp(1e-6, 1e10)
                } else {
                    1e10
                }
            })
            .collect();
        let clamped_sv = if kern.signal_variance().is_finite() {
            kern.signal_variance().clamp(1e-6, 1e10)
        } else {
            1e10
        };
        let kern = kern.with_params(clamped_sv, clamped_ls);

        let mut gp_sub = GPModel::new(kern, &td_sub, y_sub, 1e-6, 1e-4, 1e-6);
        gp_sub.const_sigma2 = cfg.const_sigma2;
        train_model(&mut gp_sub, train_iters, cfg.verbose);
        prev_kern = Some(gp_sub.kernel.clone());

        // Build prediction model on full data (RFF if configured, else exact GP)
        let pred_model = build_pred_model(&gp_sub.kernel, &td, cfg.rff_features, 42, cfg.const_sigma2);

        // Step 3: Optimize on GP surface via L-BFGS
        let best_idx = td
            .energies
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let x_start = if td.energies[best_idx] < *all_energies.last().unwrap_or(&f64::INFINITY) {
            td.col(best_idx).to_vec()
        } else {
            x_curr.clone()
        };

        let x_prev = x_curr.clone();

        // L-BFGS on GP/RFF surface
        let mut x_opt = x_start;
        let mut lbfgs = crate::lbfgs::LbfgsHistory::new(10);
        let mut prev_grad: Option<Vec<f64>> = None;
        let mut x_inner_prev = x_opt.clone();

        let e_ref = td.energies[0];

        for inner in 0..100 {
            let preds = pred_model.predict(&x_opt);
            let e_pred = preds[0] + e_ref;
            let g_pred: Vec<f64> = preds[1..].to_vec();

            // Trust penalty gradient
            let (min_dist, nearest_idx) = {
                let mut md = f64::INFINITY;
                let mut ni = 0;
                for i in 0..td.npoints() {
                    let d = euclidean_distance(&x_opt, td.col(i));
                    if d < md {
                        md = d;
                        ni = i;
                    }
                }
                (md, ni)
            };

            let grad: Vec<f64> = if min_dist > cfg.trust_radius {
                let nearest = td.col(nearest_idx);
                let direction: Vec<f64> = x_opt
                    .iter()
                    .zip(nearest.iter())
                    .map(|(a, b)| (a - b) / (min_dist + 1e-10))
                    .collect();
                let penalty_scale = 2.0 * cfg.penalty_coeff * (min_dist - cfg.trust_radius);
                g_pred
                    .iter()
                    .zip(direction.iter())
                    .map(|(g, d)| g + penalty_scale * d)
                    .collect()
            } else {
                g_pred
            };

            let g_norm: f64 = grad.iter().map(|x| x * x).sum::<f64>().sqrt();
            // LCB-augmented convergence: total gradient sigma (no orient for minimize)
            let g_eff = if cfg.lcb_kappa > 0.0 {
                let (_, var) = pred_model.predict_with_variance(&x_opt);
                let sigma_g = var[1..].iter().map(|v| v.max(0.0)).sum::<f64>().sqrt();
                g_norm + cfg.lcb_kappa * sigma_g
            } else {
                g_norm
            };
            if g_eff < 1e-4 {
                break;
            }

            // L-BFGS direction (track inner iterates, not outer x_prev)
            if let Some(ref pg) = prev_grad {
                let s: Vec<f64> = x_opt
                    .iter()
                    .zip(x_inner_prev.iter())
                    .map(|(a, b)| a - b)
                    .collect();
                let y: Vec<f64> = grad.iter().zip(pg.iter()).map(|(a, b)| a - b).collect();
                lbfgs.push_pair(s, y);
            }
            prev_grad = Some(grad.clone());
            x_inner_prev = x_opt.clone();

            let dir = lbfgs.compute_direction(&grad);

            // Step size: L-BFGS direction is curvature-scaled, so use
            // alpha=1.0 by default, clamp displacement to trust_radius.
            // For steepest descent (no pairs), cap step to avoid overshoot.
            let dir_norm: f64 = dir.iter().map(|x| x * x).sum::<f64>().sqrt();
            let step_size = if lbfgs.count > 0 {
                // L-BFGS: trust the direction, clip by trust radius
                (1.0f64).min(cfg.trust_radius / (dir_norm + 1e-30))
            } else {
                // Steepest descent: step = trust_radius / (2 * |dir|)
                (cfg.trust_radius * 0.5 / (dir_norm + 1e-30)).min(0.1 / g_norm)
            };
            for j in 0..d {
                x_opt[j] += step_size * dir[j];
            }

            let _ = (inner, e_pred); // suppress warnings
        }

        x_curr = x_opt;

        // 6-DOF rigid body projection (molecular systems only)
        let n_at = d / 3;
        if n_at >= 2 && d == 3 * n_at {
            let mut step: Vec<f64> = x_curr
                .iter()
                .zip(x_prev.iter())
                .map(|(a, b)| a - b)
                .collect();
            remove_rigid_body_modes(&mut step, &x_prev, n_at);
            x_curr = x_prev
                .iter()
                .zip(step.iter())
                .map(|(a, b)| a + b)
                .collect();
        }

        // Per-atom max-move clip (only for 3D molecular coordinates)
        if n_at >= 2 && d == 3 * n_at {
            let disp: Vec<f64> = x_curr
                .iter()
                .zip(x_prev.iter())
                .map(|(a, b)| a - b)
                .collect();
            let clipped = clip_to_max_move(&disp, cfg.max_move, 3);
            x_curr = x_prev
                .iter()
                .zip(clipped.iter())
                .map(|(a, b)| a + b)
                .collect();
        }

        // Trust clip
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

        let d_trust = trust_min_distance(
            &x_curr,
            &td.data,
            d,
            td.npoints(),
            cfg.trust_metric,
            &cfg.atom_types,
        );

        if d_trust > thresh {
            let nearest_idx = (0..td.npoints())
                .min_by(|&i, &j| {
                    let di = trust_distance(cfg.trust_metric, &cfg.atom_types, &x_curr, td.col(i));
                    let dj = trust_distance(cfg.trust_metric, &cfg.atom_types, &x_curr, td.col(j));
                    di.partial_cmp(&dj).unwrap()
                })
                .unwrap();
            let nearest = td.col(nearest_idx).to_vec();
            let disp: Vec<f64> = x_curr
                .iter()
                .zip(nearest.iter())
                .map(|(a, b)| a - b)
                .collect();
            let scale = thresh / d_trust * 0.95;
            x_curr = nearest
                .iter()
                .zip(disp.iter())
                .map(|(a, b)| a + b * scale)
                .collect();
        }

        // Step 4: Call oracle
        let (e_true, g_true) = oracle(&x_curr);
        oracle_calls += 1;

        // Per-atom max force (only for 3D molecules, otherwise L2 norm)
        let g_norm = if n_atoms >= 1 && d == 3 * n_atoms {
            (0..n_atoms)
                .map(|a| {
                    let off = 3 * a;
                    (g_true[off].powi(2) + g_true[off + 1].powi(2) + g_true[off + 2].powi(2))
                        .sqrt()
                })
                .fold(0.0f64, f64::max)
        } else {
            g_true.iter().map(|x| x * x).sum::<f64>().sqrt()
        };

        // Stagnation check
        if (g_norm - prev_force).abs() < 1e-10 {
            stagnation_count += 1;
        } else {
            stagnation_count = 0;
        }
        prev_force = g_norm;

        if stagnation_count >= 3 {
            stop_reason = StopReason::ForceStagnation;
            break;
        }

        // Explosion recovery
        if !e_true.is_finite() || e_true > 1e6 {
            let best_idx = td
                .energies
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            let mut rng = rand::rng();
            x_curr = td.col(best_idx).to_vec();
            for j in 0..d {
                x_curr[j] += (rng.random::<f64>() - 0.5) * cfg.perturb_scale * 0.5;
            }
            continue;
        }

        // Energy regression gate
        let best_idx = td
            .energies
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let e_best = td.energies[best_idx];
        let regress_tol = if cfg.energy_regression_tol > 0.0 {
            cfg.energy_regression_tol
        } else if td.npoints() >= 3 {
            let mean_e: f64 = td.energies.iter().sum::<f64>() / td.npoints() as f64;
            let var_e: f64 = td
                .energies
                .iter()
                .map(|e| (e - mean_e).powi(2))
                .sum::<f64>()
                / (td.npoints() as f64 - 1.0);
            (var_e.sqrt() * 3.0).max(1.0)
        } else {
            f64::INFINITY
        };

        if e_true > e_best + regress_tol && g_norm > cfg.conv_tol * 10.0 {
            trajectory.push(x_curr.clone());
            all_energies.push(e_true);
            td.add_point(&x_curr, e_true, &g_true);
            x_curr = td.col(best_idx).to_vec();
            continue;
        }

        trajectory.push(x_curr.clone());
        all_energies.push(e_true);
        td.add_point(&x_curr, e_true, &g_true);

        if cfg.max_training_points > 0 {
            let dist_fn = |a: &[f64], b: &[f64]| euclidean_distance(a, b);
            prune_training_data(&mut td, &x_curr, cfg.max_training_points, &dist_fn);
        }

        // Convergence check
        if g_norm < cfg.conv_tol {
            stop_reason = StopReason::Converged;
            break;
        }

        if cfg.verbose && outer_step % 10 == 0 {
            eprintln!(
                "Step {}: E={:.4}, |F|={:.5}, oc={}",
                outer_step, e_true, g_norm, oracle_calls
            );
        }
    }

    let (e_final, g_final) = oracle(&x_curr);

    MinimizationResult {
        x_final: x_curr,
        e_final,
        g_final,
        converged: stop_reason == StopReason::Converged,
        stop_reason,
        oracle_calls,
        trajectory,
        energies: all_energies,
    }
}
