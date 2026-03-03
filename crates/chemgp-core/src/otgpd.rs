//! Optimal Transport GP Dimer (OTGPD) saddle point search.
//!
//! Ports `otgpd.jl`: GP-dimer with initial rotation on true potential,
//! adaptive GP threshold, HOD (hyperparameter oscillation detection),
//! and convergence requiring negative curvature.
//!
//! Reference: Goswami et al., J. Chem. Theory Comput. (2025).

use crate::dimer_utils::{
    curvature, normalize_vec, project_out_translations, rotational_force, translational_force,
    vec_dot, vec_norm,
};
use crate::hod::{HodConfig, HodState};
use crate::kernel::Kernel;
use crate::lbfgs::LbfgsHistory;
use crate::predict::{build_pred_model_full, PredModel};
use crate::sampling::{prune_training_data, select_optim_subset};
use crate::train::train_model;
use crate::trust::{
    adaptive_trust_threshold, remove_rigid_body_modes, trust_distance, trust_min_distance,
    TrustMetric,
};
use crate::types::{GPModel, TrainingData};
use crate::StopReason;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// OTGPD configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OTGPDConfig {
    pub t_dimer: f64,
    pub divisor_t_dimer_gp: f64,
    pub t_angle_rot: f64,
    pub max_outer_iter: usize,
    pub max_inner_iter: usize,
    pub max_rot_iter: usize,
    pub dimer_sep: f64,
    pub eval_image1: bool,
    pub rotation_method: String,
    pub translation_method: String,
    pub lbfgs_memory: usize,
    pub step_convex: f64,
    pub max_step: f64,
    pub alpha_trans: f64,
    pub trust_radius: f64,
    pub ratio_at_limit: f64,
    pub initial_rotation: bool,
    pub max_initial_rot: usize,
    pub gp_train_iter: usize,
    pub n_initial_perturb: usize,
    pub perturb_scale: f64,
    pub max_training_points: usize,
    pub fps_history: usize,
    pub fps_latest_points: usize,
    pub use_hod: bool,
    pub hod_monitoring_window: usize,
    pub hod_flip_threshold: f64,
    pub hod_history_increment: usize,
    pub hod_max_history: usize,
    pub rff_features: usize,
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
    /// SCG initial lambda. Higher = more conservative hyperparameter steps.
    /// C++ gpr_optim default is 10 (Structures.h).
    pub scg_lambda_init: f64,
    /// Student-t prior degrees of freedom on sqrt(sigma2). 0 = Gaussian fallback.
    pub prior_dof: f64,
    /// Student-t prior scale. Default 1.0.
    pub prior_s2: f64,
    /// Student-t prior location. Default 0.0.
    pub prior_mu: f64,
    /// Energy observation noise variance. C++ default 1e-7 (uniform on all observations).
    pub noise_e: f64,
    /// Gradient observation noise variance. C++ applies same sigma2 to ALL observations.
    pub noise_g: f64,
    /// Jitter added to ALL covariance diagonals (EE and GG) for numerical stability.
    /// C++ default 1e-6.
    pub jitter: f64,
    pub verbose: bool,
}

impl Default for OTGPDConfig {
    fn default() -> Self {
        Self {
            t_dimer: 0.01,
            divisor_t_dimer_gp: 10.0,
            t_angle_rot: 5.0_f64.to_radians(),
            max_outer_iter: 50,
            max_inner_iter: 10000,
            max_rot_iter: 0,
            dimer_sep: 0.01,
            eval_image1: false,
            rotation_method: "lbfgs".to_string(),
            translation_method: "lbfgs".to_string(),
            lbfgs_memory: 25,
            step_convex: 0.1,
            max_step: 0.05,
            alpha_trans: 0.01,
            trust_radius: 0.5,
            ratio_at_limit: 2.0 / 3.0,
            initial_rotation: false,
            max_initial_rot: 0,
            gp_train_iter: 300,
            n_initial_perturb: 0,
            perturb_scale: 0.15,
            max_training_points: 0,
            fps_history: 10,
            fps_latest_points: 3,
            use_hod: true,
            hod_monitoring_window: 5,
            hod_flip_threshold: 0.8,
            hod_history_increment: 2,
            hod_max_history: 30,
            rff_features: 0,
            trust_metric: TrustMetric::Emd,
            atom_types: Vec::new(),
            use_adaptive_threshold: false,
            adaptive_t_min: 0.15,
            adaptive_delta_t: 0.35,
            adaptive_n_half: 50,
            adaptive_a: 1.3,
            adaptive_floor: 0.2,
            const_sigma2: 0.0,
            scg_lambda_init: 10.0,
            prior_dof: 28.0,
            prior_s2: 1.0,
            prior_mu: 0.0,
            noise_e: 1e-7,
            noise_g: 1e-7,
            jitter: 1e-6,
            verbose: true,
        }
    }
}

/// OTGPD result.
#[derive(Debug, Clone)]
pub struct OTGPDResult {
    pub r: Vec<f64>,
    pub orient: Vec<f64>,
    pub converged: bool,
    pub stop_reason: StopReason,
    pub oracle_calls: usize,
    pub history: OTGPDHistory,
}

/// Convergence history.
#[derive(Debug, Clone, Default)]
pub struct OTGPDHistory {
    pub e_true: Vec<f64>,
    pub f_true: Vec<f64>,
    pub curv_true: Vec<f64>,
    pub oracle_calls: Vec<usize>,
    pub t_gp: Vec<f64>,
}

/// Oracle function type.
pub type OracleFn = dyn Fn(&[f64]) -> (f64, Vec<f64>);


/// Per-atom step limiting (C++ AtomicDimer.cpp:336-396).
///
/// Checks that no atom moves more than 0.5*(1-ratio_at_limit)*min_interatomic_distance.
/// If exceeded, scales the entire step so the most constrained atom is at 99% of its limit.
/// Returns true if the step was clipped.
fn limit_per_atom_step(r: &[f64], r_new: &mut [f64], n_atoms: usize, ratio_at_limit: f64) -> bool {
    // Per-atom step lengths
    let mut atom_steps = vec![0.0; n_atoms];
    for i in 0..n_atoms {
        let dx = r_new[3 * i] - r[3 * i];
        let dy = r_new[3 * i + 1] - r[3 * i + 1];
        let dz = r_new[3 * i + 2] - r[3 * i + 2];
        atom_steps[i] = (dx * dx + dy * dy + dz * dz).sqrt();
    }

    // Minimum interatomic distance per atom
    let mut min_dists = vec![f64::INFINITY; n_atoms];
    for i in 0..n_atoms {
        for j in (i + 1)..n_atoms {
            let dx = r[3 * i] - r[3 * j];
            let dy = r[3 * i + 1] - r[3 * j + 1];
            let dz = r[3 * i + 2] - r[3 * j + 2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            min_dists[i] = min_dists[i].min(d);
            min_dists[j] = min_dists[j].min(d);
        }
    }

    let factor = 0.5 * (1.0 - ratio_at_limit);
    let mut needs_clip = false;
    let mut min_ratio = f64::INFINITY;

    for i in 0..n_atoms {
        let limit = factor * min_dists[i];
        if atom_steps[i] > 0.99 * limit && atom_steps[i] > 1e-15 {
            needs_clip = true;
            let ratio = limit / atom_steps[i];
            min_ratio = min_ratio.min(ratio);
        }
    }

    if needs_clip {
        let scale = min_ratio * 0.99;
        for j in 0..r_new.len() {
            r_new[j] = r[j] + (r_new[j] - r[j]) * scale;
        }
    }

    needs_clip
}


fn predict_dimer(
    r: &[f64], orient: &[f64], sep: f64, model: &PredModel, e_ref: f64,
) -> (Vec<f64>, Vec<f64>, f64) {
    let r1: Vec<f64> = r.iter().zip(orient.iter()).map(|(r, o)| r + sep * o).collect();
    let pred0 = model.predict(r);
    let pred1 = model.predict(&r1);
    let g0: Vec<f64> = pred0[1..].to_vec();
    let g1: Vec<f64> = pred1[1..].to_vec();
    let e0 = pred0[0] + e_ref;
    (g0, g1, e0)
}

/// Rotate dimer on GP surface with modified Newton parabolic fit.
fn rotate_on_gp(
    r: &[f64],
    orient: &mut Vec<f64>,
    sep: f64,
    model: &PredModel,
    e_ref: f64,
    cfg: &OTGPDConfig,
) -> f64 {
    let mut best_c = 0.0;
    for _ in 0..cfg.max_rot_iter {
        let (g0, g1, _) = predict_dimer(r, orient, sep, model, e_ref);
        let f_rot = rotational_force(&g0, &g1, orient, sep);
        let f_rot_norm = vec_norm(&f_rot);

        if f_rot_norm < 1e-10 {
            best_c = curvature(&g0, &g1, orient, sep);
            break;
        }

        let c0 = curvature(&g0, &g1, orient, sep);
        best_c = c0;

        let dtheta = 0.5 * (0.5 * f_rot_norm / (c0.abs() + 1e-10)).atan();
        if dtheta < cfg.t_angle_rot {
            break;
        }

        let orient_rot: Vec<f64> = f_rot.iter().map(|x| x / f_rot_norm).collect();

        // Trial rotation
        let orient_trial: Vec<f64> = orient.iter().zip(orient_rot.iter())
            .map(|(o, r)| dtheta.cos() * o + dtheta.sin() * r).collect();
        let orient_trial = normalize_vec(&orient_trial);

        let r1_trial: Vec<f64> = r.iter().zip(orient_trial.iter())
            .map(|(r, o)| r + sep * o).collect();
        let pred1_trial = model.predict(&r1_trial);
        let g1_trial: Vec<f64> = pred1_trial[1..].to_vec();

        let f_rot_trial = rotational_force(&g0, &g1_trial, &orient_trial, sep);

        let orient_rot_trial: Vec<f64> = orient.iter().zip(orient_rot.iter())
            .map(|(o, r)| -dtheta.sin() * o + dtheta.cos() * r).collect();
        let orient_rot_trial = normalize_vec(&orient_rot_trial);

        let f_dtheta = vec_dot(&f_rot_trial, &orient_rot_trial);
        let f_0 = vec_dot(&f_rot, &orient_rot);

        let sin2 = (2.0 * dtheta).sin();
        let cos2 = (2.0 * dtheta).cos();

        if sin2.abs() < 1e-12 {
            *orient = orient_trial;
            continue;
        }

        let a1 = (f_dtheta - f_0 * cos2) / (2.0 * sin2);
        let b1 = -0.5 * f_0;
        let angle_rot = 0.5 * (b1 / (a1 + 1e-18)).atan();

        // EON-style curvature improvement check
        let mut angle_final = angle_rot;
        let mut c_est = c0 + a1 * ((2.0 * angle_final).cos() - 1.0) + b1 * (2.0 * angle_final).sin();
        if c_est > c0 {
            angle_final += std::f64::consts::FRAC_PI_2;
            c_est = c0 + a1 * ((2.0 * angle_final).cos() - 1.0) + b1 * (2.0 * angle_final).sin();
        }

        // If rotation still worsens curvature, don't rotate
        if c_est > c0 {
            break;
        }

        let orient_new: Vec<f64> = orient.iter().zip(orient_rot.iter())
            .map(|(o, r)| angle_final.cos() * o + angle_final.sin() * r).collect();
        let new_orient = normalize_vec(&orient_new);
        *orient = new_orient;

        best_c = c_est;
    }
    best_c
}

/// GP-accelerated optimal transport dimer search.
pub fn otgpd(
    oracle: &OracleFn,
    x_init: &[f64],
    orient_init: &[f64],
    kernel: &Kernel,
    config: &OTGPDConfig,
    training_data: Option<TrainingData>,
) -> OTGPDResult {
    let cfg = config;
    let d = x_init.len();

    let mut orient = normalize_vec(orient_init);
    let mut r = x_init.to_vec();

    let mut td = training_data.unwrap_or_else(|| TrainingData::new(d));
    let mut oracle_calls = 0;

    // Initial data generation (C++ AtomicDimer::execute pattern)
    // 1. Evaluate midpoint
    let (e_r, g_r) = oracle(&r);
    td.add_point(&r, e_r, &g_r);
    oracle_calls += 1;

    // Optional random perturbations (off by default for molecular systems)
    if cfg.n_initial_perturb > 0 {
        let mut rng = rand::rng();
        for _ in 0..cfg.n_initial_perturb {
            let perturb: Vec<f64> = (0..d)
                .map(|_| (rng.random::<f64>() - 0.5) * cfg.perturb_scale)
                .collect();
            let x_p: Vec<f64> = r.iter().zip(perturb.iter()).map(|(a, b)| a + b).collect();
            let (e_p, g_p) = oracle(&x_p);
            if e_p.is_finite() && e_p < 1e6 {
                td.add_point(&x_p, e_p, &g_p);
                oracle_calls += 1;
            }
        }
    }

    // 2. Evaluate image1
    let r1: Vec<f64> = r.iter().zip(orient.iter())
        .map(|(r, o)| r + cfg.dimer_sep * o).collect();
    let (e_r1, g_r1) = oracle(&r1);
    td.add_point(&r1, e_r1, &g_r1);
    oracle_calls += 1;

    // Cache midpoint gradient for initial rotation (midpoint position is fixed)
    let g0_cached = g_r.clone();
    let mut g1_cached = g_r1.clone();

    let mut history = OTGPDHistory::default();

    // Phase 1: Initial rotation on true potential
    // C++ pattern: evaluate image1 at START of each rotation, use for next
    if cfg.initial_rotation && cfg.max_initial_rot > 0 {
        for rot_iter in 0..cfg.max_initial_rot {
            // Check convergence (C++: skip on first iteration)
            if rot_iter > 0 {
                let f_rot = rotational_force(&g0_cached, &g1_cached, &orient, cfg.dimer_sep);
                let f_rot_norm = vec_norm(&f_rot);
                let c0 = curvature(&g0_cached, &g1_cached, &orient, cfg.dimer_sep);
                let dtheta = 0.5 * (0.5 * f_rot_norm / (c0.abs() + 1e-10)).atan();
                if dtheta < cfg.t_angle_rot {
                    break;
                }
            }

            // Evaluate image1 at current orientation (1 oracle call per iteration)
            let r1_cur: Vec<f64> = r.iter().zip(orient.iter())
                .map(|(rv, o)| rv + cfg.dimer_sep * o).collect();
            let (e1_cur, g1_cur) = oracle(&r1_cur);
            oracle_calls += 1;
            td.add_point(&r1_cur, e1_cur, &g1_cur);
            g1_cached = g1_cur;

            // Rotate using Kastner-Sherwood parabolic fit
            let f_rot = rotational_force(&g0_cached, &g1_cached, &orient, cfg.dimer_sep);
            let f_rot_norm = vec_norm(&f_rot);
            let c0 = curvature(&g0_cached, &g1_cached, &orient, cfg.dimer_sep);

            if f_rot_norm < 1e-10 { break; }

            let dtheta = 0.5 * (0.5 * f_rot_norm / (c0.abs() + 1e-10)).atan();
            let orient_rot: Vec<f64> = f_rot.iter().map(|x| x / f_rot_norm).collect();

            // Trial rotation
            let orient_trial: Vec<f64> = orient.iter().zip(orient_rot.iter())
                .map(|(o, rv)| dtheta.cos() * o + dtheta.sin() * rv).collect();
            let orient_trial = normalize_vec(&orient_trial);

            // Evaluate image1 at trial angle (1 oracle call)
            let r1_trial: Vec<f64> = r.iter().zip(orient_trial.iter())
                .map(|(rv, o)| rv + cfg.dimer_sep * o).collect();
            let (e1_trial, g1_trial) = oracle(&r1_trial);
            oracle_calls += 1;
            td.add_point(&r1_trial, e1_trial, &g1_trial);

            let c_trial = curvature(&g0_cached, &g1_trial, &orient_trial, cfg.dimer_sep);

            // Parabolic fit for optimal angle
            let f_rot_trial = rotational_force(&g0_cached, &g1_trial, &orient_trial, cfg.dimer_sep);
            let orient_rot_trial: Vec<f64> = orient.iter().zip(orient_rot.iter())
                .map(|(o, rv)| -dtheta.sin() * o + dtheta.cos() * rv).collect();
            let orient_rot_trial = normalize_vec(&orient_rot_trial);

            let f_dtheta = vec_dot(&f_rot_trial, &orient_rot_trial);
            let f_0 = vec_dot(&f_rot, &orient_rot);
            let sin2 = (2.0 * dtheta).sin();
            let cos2 = (2.0 * dtheta).cos();

            if sin2.abs() < 1e-12 {
                orient = orient_trial;
                orient = normalize_vec(&orient);
                continue;
            }

            let a1 = (f_dtheta - f_0 * cos2) / (2.0 * sin2);
            let b1 = -0.5 * f_0;
            let angle_rot = 0.5 * (b1 / (a1 + 1e-18)).atan();

            let mut angle_final = angle_rot;
            let mut c_est = c0 + a1 * ((2.0 * angle_final).cos() - 1.0)
                + b1 * (2.0 * angle_final).sin();
            if c_est > c0 {
                angle_final += std::f64::consts::FRAC_PI_2;
                c_est = c0 + a1 * ((2.0 * angle_final).cos() - 1.0)
                    + b1 * (2.0 * angle_final).sin();
            }

            if c_est < c0 {
                let orient_new: Vec<f64> = orient.iter().zip(orient_rot.iter())
                    .map(|(o, rv)| angle_final.cos() * o + angle_final.sin() * rv).collect();
                orient = normalize_vec(&orient_new);
            } else if c_trial < c0 {
                orient = orient_trial;
            }
            orient = normalize_vec(&orient);
        }
    }

    // After Phase 1 rotation, g1_cached may be stale (evaluated at a previous
    // orient, not the final parabolic-fit orient). Re-evaluate image1 at the
    // final orient to get correct curvature. Without this, c_true_cached can
    // be +58 when the actual curvature is -8 (bug: stale g1/orient mismatch).
    if cfg.initial_rotation && cfg.max_initial_rot > 0 {
        let r1_final: Vec<f64> = r.iter().zip(orient.iter())
            .map(|(rv, o)| rv + cfg.dimer_sep * o).collect();
        let (e1_final, g1_final) = oracle(&r1_final);
        oracle_calls += 1;
        td.add_point(&r1_final, e1_final, &g1_final);
        g1_cached = g1_final;
    }

    // Record initial state in history (before GP loop)
    {
        let g_inf_init = g0_cached.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        let c_init = curvature(&g0_cached, &g1_cached, &orient, cfg.dimer_sep);
        history.e_true.push(e_r);
        history.f_true.push(g_inf_init);
        history.curv_true.push(c_init);
        history.oracle_calls.push(oracle_calls);
        history.t_gp.push(f64::NAN);
        if cfg.verbose {
            eprintln!("OTGPD init: E={:.6} |G|_inf={:.5} C_true={:.4} calls={}", e_r, g_inf_init, c_init, oracle_calls);
        }
    }

    // Cache true curvature from initial rotation for inner loop.
    let mut c_true_cached = curvature(&g0_cached, &g1_cached, &orient, cfg.dimer_sep);

    // Phase 2: GP-accelerated loop
    let mut prev_kern: Option<Kernel> = None;
    let mut stop_reason = StopReason::MaxIterations;
    let mut hod_state = HodState::new(cfg.fps_history);
    let n_atoms = d / 3;
    // Track latest true gradient Linf for C++ threshold formula
    let mut latest_g_inf: f64 = g0_cached.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
    // Restart logic (C++ AtomicDimer.cpp:496-522): track latest converged state
    let r_init = r.clone();
    let orient_init = orient.clone();
    let mut r_latest_conv: Option<Vec<f64>> = None;
    let mut orient_latest_conv: Option<Vec<f64>> = None;

    for outer_iter in 0..cfg.max_outer_iter {
        // Restart from latest converged state (C++ defineStartPath)
        if let Some(ref rc) = r_latest_conv {
            r = rc.clone();
            orient = orient_latest_conv.as_ref().unwrap().clone();
        } else {
            r = r_init.clone();
            orient = orient_init.clone();
        }
        // Pruning
        if cfg.max_training_points > 0 && td.npoints() > cfg.max_training_points {
            let dist_fn = |a: &[f64], b: &[f64]| crate::distances::euclidean_distance(a, b);
            prune_training_data(&mut td, &r, cfg.max_training_points, &dist_fn);
        }

        // FPS subset selection (with HOD-adjusted size)
        let fps_size = hod_state.current_fps_history;
        let td_sub = if fps_size > 0 && td.npoints() > fps_size {
            let dist_fn = |a: &[f64], b: &[f64]| -> f64 {
                trust_distance(cfg.trust_metric, &cfg.atom_types, a, b)
            };
            let sub_idx = select_optim_subset(
                &td, &r, fps_size, cfg.fps_latest_points, &dist_fn,
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

        let e_ref_sub = td_sub.energies[0];
        let mut y_sub: Vec<f64> = td_sub.energies.iter().map(|e| e - e_ref_sub).collect();
        y_sub.extend_from_slice(&td_sub.gradients);

        // Use config-provided kernel for first iteration; SCG optimizes from there.
        // Data-dependent init is pathological for clustered dimer data (0.01A sep).
        let kern = match &prev_kern {
            None => kernel.clone(),
            Some(k) => k.clone(),
        };

        let mut gp_sub = GPModel::new(kern, &td_sub, y_sub.clone(), cfg.noise_e, cfg.noise_g, cfg.jitter);
        // Dynamic constSigma2 (MATLAB atomic_GP_dimer.m:453): max(1, mean_y^2)
        // Uses SHIFTED energies (y_sub[0..n]), not raw. Raw energies at -43 eV would
        // give const_sigma2 = 1849 and corrupt the covariance matrix.
        let const_sigma2 = if cfg.const_sigma2 > 0.0 {
            let n = td_sub.npoints() as f64;
            let mean_y = y_sub[..td_sub.npoints()].iter().sum::<f64>() / n;
            (mean_y * mean_y).max(1.0)
        } else {
            0.0
        };
        gp_sub.const_sigma2 = const_sigma2;
        gp_sub.scg_lambda_init = cfg.scg_lambda_init;
        gp_sub.prior_dof = cfg.prior_dof;
        gp_sub.prior_s2 = cfg.prior_s2;
        gp_sub.prior_mu = cfg.prior_mu;
        train_model(&mut gp_sub, train_iters, cfg.verbose);
        if cfg.verbose {
            eprintln!("  sigma2={:.6e} inv_ls[0..3]={:?}",
                gp_sub.kernel.signal_variance(),
                &gp_sub.kernel.inv_lengthscales()[..gp_sub.kernel.inv_lengthscales().len().min(3)]);
        }
        prev_kern = Some(gp_sub.kernel.clone());

        // HOD check
        if cfg.use_hod {
            let hod_cfg = HodConfig {
                monitoring_window: cfg.hod_monitoring_window,
                flip_threshold: cfg.hod_flip_threshold,
                history_increment: cfg.hod_history_increment,
                max_history: cfg.hod_max_history,
            };
            hod_state.check(&gp_sub, &hod_cfg);
        }

        // Build prediction model on full data (RFF if configured, else exact GP)
        let model = build_pred_model_full(
            &gp_sub.kernel, &td, cfg.rff_features, 42, const_sigma2,
            cfg.noise_e, cfg.noise_g, cfg.jitter,
        );
        let e_ref = td.energies[0];

        // Adaptive GP threshold (C++ AtomicDimer.cpp:1056)
        // Uses Linf of the current true gradient, not historical min L2 force
        let t_gp = if cfg.divisor_t_dimer_gp > 0.0 {
            (latest_g_inf / cfg.divisor_t_dimer_gp).max(cfg.t_dimer * 0.1)
        } else {
            cfg.t_dimer * 0.1
        };

        // Inner loop: optimize on GP surface
        let mut trans_hist = LbfgsHistory::new(cfg.lbfgs_memory);
        let _r_before = r.clone();

        for inner_iter in 0..cfg.max_inner_iter {
            // Predict g0 at midpoint; use true curvature when available.
            // When eval_image1=true (molecular systems), orient is fixed from true
            // rotation and curvature comes from oracle. GP only provides translation
            // direction. When eval_image1=false (2D surfaces), fall back to GP rotation.
            let (g0, c) = if cfg.eval_image1 {
                let pred0 = model.predict(&r);
                (pred0[1..].to_vec(), c_true_cached)
            } else {
                let _c = rotate_on_gp(&r, &mut orient, cfg.dimer_sep, &model, e_ref, cfg);
                orient = normalize_vec(&orient);
                let (g0, g1, _e0) = predict_dimer(&r, &orient, cfg.dimer_sep, &model, e_ref);
                let c = curvature(&g0, &g1, &orient, cfg.dimer_sep);
                (g0, c)
            };
            let mut f_trans = translational_force(&g0, &orient);
            project_out_translations(&mut f_trans);

            // Inner convergence: raw GP gradient Linf (C++ AtomicDimer.cpp:561)
            let g0_inf = g0.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
            if cfg.verbose && inner_iter < 5 {
                eprintln!("  inner {}: g0_inf={:.6} t_gp={:.6} c={:.4}", inner_iter, g0_inf, t_gp, c);
            }
            if g0_inf < t_gp {
                r_latest_conv = Some(r.clone());
                orient_latest_conv = Some(orient.clone());
                break;
            }

            // Translate
            let step = if c < 0.0 {
                // Negative curvature: L-BFGS (pass -f_trans as gradient to minimize)
                // C++ uses raw LBFGS direction, only clips if > max_step
                let neg_f: Vec<f64> = f_trans.iter().map(|x| -x).collect();
                let mut dir = trans_hist.compute_direction(&neg_f);
                let dir_norm = vec_norm(&dir);
                if dir_norm > cfg.max_step {
                    let scale = cfg.max_step / dir_norm;
                    for d in dir.iter_mut() { *d *= scale; }
                    trans_hist.reset();
                }
                dir
            } else {
                // Positive curvature: move uphill along dimer axis (C++ Dimer.cpp:161-180)
                // F_eff = -(F.orient)*orient where F=-G, so F_eff = (G.orient)*orient
                let g_dot_o: f64 = g0.iter().zip(orient.iter()).map(|(g, o)| g * o).sum();
                let f_along: Vec<f64> = orient.iter().map(|o| g_dot_o * o).collect();
                let fn_val = vec_norm(&f_along);
                trans_hist.reset();
                if fn_val < 1e-12 {
                    break;
                }
                f_along.iter().map(|f| cfg.step_convex * f / fn_val).collect::<Vec<f64>>()
            };

            let step_norm = vec_norm(&step);
            let step = if step_norm > cfg.max_step {
                trans_hist.reset(); // C++ Dimer.cpp:481
                step.iter().map(|s| s * cfg.max_step / step_norm).collect()
            } else {
                step
            };

            let mut step_proj = step;
            // 6-DOF projection: remove rigid body translation + rotation (C++ Dimer.cpp:21-101)
            if n_atoms >= 2 {
                remove_rigid_body_modes(&mut step_proj, &r, n_atoms);
            }
            let mut r_new: Vec<f64> = r.iter().zip(step_proj.iter()).map(|(a, b)| a + b).collect();

            // Per-atom step limiting (C++ AtomicDimer.cpp:336-396)
            if n_atoms >= 2 {
                if limit_per_atom_step(&r, &mut r_new, n_atoms, cfg.ratio_at_limit) {
                    trans_hist.reset();
                }
            }

            // Trust region check (C++ AtomicDimer.cpp:614: skip on first inner step)
            // C++ rejects the step outright and breaks (does NOT clip to boundary).
            // The oracle evaluates at the last accepted position.
            if inner_iter > 0 {
                let thresh = adaptive_trust_threshold(
                    cfg.trust_radius, td.npoints(), n_atoms,
                    cfg.use_adaptive_threshold,
                    cfg.adaptive_t_min, cfg.adaptive_delta_t,
                    cfg.adaptive_n_half, cfg.adaptive_a, cfg.adaptive_floor,
                );
                let dist = trust_min_distance(
                    &r_new, &td.data, d, td.npoints(), cfg.trust_metric, &cfg.atom_types,
                );

                if dist > thresh {
                    // Reject step: keep r at last accepted position (do NOT clip)
                    break;
                }
            }

            // Update L-BFGS history: y = grad_new - grad_old where grad = -f_trans
            // So y = (-f_new) - (-f_old) = f_old - f_new
            if c < 0.0 {
                let s: Vec<f64> = r_new.iter().zip(r.iter()).map(|(a, b)| a - b).collect();
                let g0_new = if cfg.eval_image1 {
                    model.predict(&r_new)[1..].to_vec()
                } else {
                    predict_dimer(&r_new, &orient, cfg.dimer_sep, &model, e_ref).0
                };
                let f_trans_new = translational_force(&g0_new, &orient);
                let y: Vec<f64> = f_trans.iter().zip(f_trans_new.iter()).map(|(a, b)| a - b).collect();
                trans_hist.push_pair(s, y);
            }

            r = r_new;
        }

        // Oracle evaluation
        let (e_true, g_true) = oracle(&r);
        oracle_calls += 1;
        td.add_point(&r, e_true, &g_true);
        latest_g_inf = g_true.iter().map(|x| x.abs()).fold(0.0f64, f64::max);

        let mut c_true = f64::NAN;
        if cfg.eval_image1 {
            let r1_cur: Vec<f64> = r.iter().zip(orient.iter())
                .map(|(r, o)| r + cfg.dimer_sep * o).collect();
            let (e_r1, g_r1) = oracle(&r1_cur);
            oracle_calls += 1;
            td.add_point(&r1_cur, e_r1, &g_r1);
            // True rotation (Kastner-Sherwood ImprovedDimer) at evaluated position.
            // GP cannot resolve curvature at dimer_sep ~ 0.01A; oracle-based
            // rotation is essential for molecular systems.
            let mut g1_rot = g_r1;
            for _rot_iter in 0..cfg.max_rot_iter {
                let f_rot = rotational_force(&g_true, &g1_rot, &orient, cfg.dimer_sep);
                let f_rot_norm = vec_norm(&f_rot);
                if f_rot_norm < 1e-10 { break; }

                let c0 = curvature(&g_true, &g1_rot, &orient, cfg.dimer_sep);
                let dtheta = 0.5 * (0.5 * f_rot_norm / (c0.abs() + 1e-10)).atan();
                if dtheta < cfg.t_angle_rot { break; }

                let theta: Vec<f64> = f_rot.iter().map(|x| x / f_rot_norm).collect();
                let tau_trial: Vec<f64> = orient.iter().zip(theta.iter())
                    .map(|(o, t)| dtheta.cos() * o + dtheta.sin() * t).collect();
                let tau_trial = normalize_vec(&tau_trial);

                let r1_trial: Vec<f64> = r.iter().zip(tau_trial.iter())
                    .map(|(rv, o)| rv + cfg.dimer_sep * o).collect();
                let (e1_trial, g1_trial) = oracle(&r1_trial);
                oracle_calls += 1;
                td.add_point(&r1_trial, e1_trial, &g1_trial);

                // Parabolic fit for optimal rotation angle
                let f_rot_trial = rotational_force(&g_true, &g1_trial, &tau_trial, cfg.dimer_sep);
                let theta_perp: Vec<f64> = orient.iter().zip(theta.iter())
                    .map(|(o, t)| -dtheta.sin() * o + dtheta.cos() * t).collect();
                let theta_perp = normalize_vec(&theta_perp);
                let f_dtheta = vec_dot(&f_rot_trial, &theta_perp);
                let f_0 = vec_dot(&f_rot, &theta);
                let sin2 = (2.0 * dtheta).sin();
                let cos2 = (2.0 * dtheta).cos();

                if sin2.abs() < 1e-12 {
                    orient = tau_trial;
                    orient = normalize_vec(&orient);
                    g1_rot = g1_trial;
                    continue;
                }

                let a1 = (f_dtheta - f_0 * cos2) / (2.0 * sin2);
                let b1 = -0.5 * f_0;
                let mut angle_final = 0.5 * (b1 / (a1 + 1e-18)).atan();
                let mut c_est = c0 + a1 * ((2.0 * angle_final).cos() - 1.0)
                    + b1 * (2.0 * angle_final).sin();
                if c_est > c0 {
                    angle_final += std::f64::consts::FRAC_PI_2;
                    c_est = c0 + a1 * ((2.0 * angle_final).cos() - 1.0)
                        + b1 * (2.0 * angle_final).sin();
                }

                if c_est < c0 {
                    let orient_new: Vec<f64> = orient.iter().zip(theta.iter())
                        .map(|(o, t)| angle_final.cos() * o + angle_final.sin() * t).collect();
                    orient = normalize_vec(&orient_new);
                    // Interpolate g1 at optimal angle (Kastner-Sherwood Eq. 8)
                    if dtheta.abs() > 1e-15 {
                        g1_rot = g1_rot.iter().zip(g1_trial.iter()).zip(g_true.iter())
                            .map(|((g1v, g1pv), g0v)| {
                                let sr1 = (dtheta - angle_final).sin() / dtheta.sin();
                                let sr2 = angle_final.sin() / dtheta.sin();
                                let cc = 1.0 - angle_final.cos()
                                    - angle_final.sin() * (dtheta * 0.5).tan();
                                sr1 * g1v + sr2 * g1pv + cc * g0v
                            }).collect();
                    } else {
                        g1_rot = g1_trial;
                    }
                } else {
                    let c_trial = curvature(&g_true, &g1_trial, &tau_trial, cfg.dimer_sep);
                    if c_trial < c0 {
                        orient = tau_trial;
                        orient = normalize_vec(&orient);
                        g1_rot = g1_trial;
                    } else {
                        break;
                    }
                }
            }
            c_true = curvature(&g_true, &g1_rot, &orient, cfg.dimer_sep);
            c_true_cached = c_true;
        }

        // C++ convergence uses ||G||_inf (not ||F_trans||_2).
        // For N-atom systems, L2 grows with sqrt(3N) and never converges
        // with thresholds calibrated for Linf.
        history.e_true.push(e_true);
        history.f_true.push(latest_g_inf);
        history.curv_true.push(c_true);
        history.oracle_calls.push(oracle_calls);
        history.t_gp.push(t_gp);

        if cfg.verbose {
            eprintln!(
                "OTGPD outer {}: E={:.6} |G|_inf={:.5} C={:.4} T_gp={:.5} calls={}",
                outer_iter, e_true, latest_g_inf, c_true, t_gp, oracle_calls,
            );
        }

        // Convergence: ||G||_inf < threshold (C++ isFinalConvergenceReached)
        // C++ does not check curvature in the outer convergence criterion.
        if latest_g_inf < cfg.t_dimer {
            stop_reason = StopReason::Converged;
            break;
        }
    }

    OTGPDResult {
        r,
        orient,
        converged: stop_reason == StopReason::Converged,
        stop_reason,
        oracle_calls,
        history,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::potentials::{leps_energy_gradient, LEPS_REACTANT};

    #[test]
    fn test_otgpd_leps() {
        let oracle = |x: &[f64]| -> (f64, Vec<f64>) { leps_energy_gradient(x) };

        // Start near LEPS reactant with orientation along reaction coordinate
        let x_init = LEPS_REACTANT.to_vec();
        let mut orient_init = vec![0.0; 9];
        orient_init[3] = 1.0; // orient along AB bond direction
        let orient_init = normalize_vec(&orient_init);

        let mut cfg = OTGPDConfig::default();
        cfg.max_outer_iter = 10;
        cfg.max_inner_iter = 50;
        cfg.t_dimer = 1.0;
        cfg.initial_rotation = false;
        cfg.eval_image1 = false;
        cfg.verbose = false;

        let kernel = Kernel::MolInvDist(crate::kernel::MolInvDistSE::isotropic(1.0, 1.0, vec![]));

        let result = otgpd(&oracle, &x_init, &orient_init, &kernel, &cfg, None);
        assert!(result.oracle_calls > 2);
        assert_eq!(result.r.len(), 9);
        assert_eq!(result.orient.len(), 9);
    }
}
