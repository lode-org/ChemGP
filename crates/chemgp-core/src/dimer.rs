//! GP-Dimer saddle point search.
//!
//! Ports `dimer.jl`: rotation + translation on GP surface with oracle calls
//! at trust boundary.
//!
//! Reference: Henkelman & Jonsson, J. Chem. Phys. 111, 7010 (1999).
//! GP-Dimer: Koistinen et al., J. Chem. Theory Comput. 16, 499 (2020).

use crate::dimer_utils::{
    curvature, max_atom_motion, max_atom_motion_applied, normalize_vec, perpendicular_sigma,
    project_out_translations, rotational_force, translational_force, vec_dot, vec_norm,
};
use crate::kernel::Kernel;
use crate::lbfgs::LbfgsHistory;
use crate::predict::{build_pred_model_full_with_prior, GPNoiseParams, PredModel};
use crate::prior_mean::PriorMeanConfig;
use crate::sampling::select_optim_subset;
use crate::train::{adaptive_train_iters, train_model};
use crate::trust::{
    adaptive_trust_threshold, clip_point_to_trust, trust_distance, trust_min_distance,
    AdaptiveTrustParams, TrustClipParams, TrustMetric,
};
use crate::types::{GPModel, TrainingData};
use crate::StopReason;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Current state of the dimer.
#[derive(Debug, Clone)]
pub struct DimerState {
    pub r: Vec<f64>,
    pub orient: Vec<f64>,
    pub dimer_sep: f64,
}

/// Configuration for GP-dimer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimerConfig {
    pub t_force_true: f64,
    pub t_force_gp: f64,
    pub t_angle_rot: f64,
    pub trust_radius: f64,
    pub max_outer_iter: usize,
    pub max_oracle_calls: usize,
    pub max_inner_iter: usize,
    pub max_rot_iter: usize,
    pub alpha_trans: f64,
    pub gp_train_iter: usize,
    pub n_initial_perturb: usize,
    pub perturb_scale: f64,
    pub rotation_method: String,
    pub translation_method: String,
    pub lbfgs_memory: usize,
    pub max_step: f64,
    pub step_convex: f64,
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
    pub const_sigma2: f64,
    pub scg_lambda_init: f64,
    pub prior_dof: f64,
    pub prior_s2: f64,
    pub prior_mu: f64,
    /// Energy observation noise variance. C++ default 1e-7 (uniform on all observations).
    pub noise_e: f64,
    /// Gradient observation noise variance. C++ applies same sigma2 to ALL observations.
    pub noise_g: f64,
    /// Jitter added to ALL covariance diagonals (EE and GG) for numerical stability.
    /// C++ default 1e-6.
    pub jitter: f64,
    /// LCB kappa for inner loop convergence. When > 0, the GP convergence
    /// check uses |F| + kappa * sigma_perp < t_force_gp.
    /// 0.0 = disabled (default, backward compatible).
    pub lcb_kappa: f64,
    /// Variance-based oracle gate (same semantics as OTGPD).
    /// 0.0 = disabled (default, backward compatible).
    pub unc_convergence: f64,
    pub seed: u64,
    pub prior_mean: PriorMeanConfig,
    pub verbose: bool,
}

impl Default for DimerConfig {
    fn default() -> Self {
        Self {
            t_force_true: 1e-3,
            t_force_gp: 1e-2,
            t_angle_rot: 5.0_f64.to_radians(),
            trust_radius: 0.1,
            max_outer_iter: 50,
            max_oracle_calls: 0,
            max_inner_iter: 100,
            max_rot_iter: 0,
            alpha_trans: 0.01,
            gp_train_iter: 300,
            n_initial_perturb: 4,
            perturb_scale: 0.15,
            rotation_method: "lbfgs".to_string(),
            translation_method: "lbfgs".to_string(),
            lbfgs_memory: 25,
            max_step: 0.05,
            step_convex: 0.1,
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
            scg_lambda_init: 10.0,
            prior_dof: 28.0,
            prior_s2: 1.0,
            prior_mu: 0.0,
            noise_e: 1e-7,
            noise_g: 1e-7,
            jitter: 1e-6,
            lcb_kappa: 0.0,
            unc_convergence: 0.0,
            seed: 42,
            prior_mean: PriorMeanConfig::Reference,
            verbose: true,
        }
    }
}

/// Result of GP-dimer search.
#[derive(Debug, Clone)]
pub struct DimerResult {
    pub state: DimerState,
    pub converged: bool,
    pub stop_reason: StopReason,
    pub oracle_calls: usize,
    pub history: DimerHistory,
}

/// Convergence history.
#[derive(Debug, Clone, Default)]
pub struct DimerHistory {
    pub e_true: Vec<f64>,
    pub f_true: Vec<f64>,
    pub curv_true: Vec<f64>,
    pub oracle_calls: Vec<usize>,
    /// Perpendicular gradient sigma at each outer step (0.0 when LCB disabled).
    pub sigma_perp: Vec<f64>,
}

/// Oracle function type.
pub type OracleFn = dyn Fn(&[f64]) -> (f64, Vec<f64>);

// ============================================================================
// Dimer utility functions
// ============================================================================

/// Get the two dimer endpoint images.
fn dimer_images(state: &DimerState) -> (Vec<f64>, Vec<f64>) {
    let r1: Vec<f64> = state
        .r
        .iter()
        .zip(state.orient.iter())
        .map(|(r, o)| r + state.dimer_sep * o)
        .collect();
    let r2: Vec<f64> = state
        .r
        .iter()
        .zip(state.orient.iter())
        .map(|(r, o)| r - state.dimer_sep * o)
        .collect();
    (r1, r2)
}


// ============================================================================
// GP prediction helpers
// ============================================================================

fn predict_dimer_gradients(
    state: &DimerState,
    model: &PredModel,
    y_std: f64,
) -> (Vec<f64>, Vec<f64>, f64) {
    let (r1, _) = dimer_images(state);
    let pred0 = model.predict(&state.r);
    let pred1 = model.predict(&r1);

    let g0: Vec<f64> = pred0[1..].iter().map(|v| v * y_std).collect();
    let g1: Vec<f64> = pred1[1..].iter().map(|v| v * y_std).collect();
    let e0 = pred0[0] * y_std;

    (g0, g1, e0)
}

/// Like predict_dimer_gradients but also returns variance at the midpoint.
fn predict_dimer_gradients_with_variance(
    state: &DimerState,
    model: &PredModel,
    y_std: f64,
) -> (Vec<f64>, Vec<f64>, f64, Vec<f64>) {
    let (r1, _) = dimer_images(state);
    let (pred0, var0) = model.predict_with_variance(&state.r);
    let pred1 = model.predict(&r1);

    let g0: Vec<f64> = pred0[1..].iter().map(|v| v * y_std).collect();
    let g1: Vec<f64> = pred1[1..].iter().map(|v| v * y_std).collect();
    let e0 = pred0[0] * y_std;

    (g0, g1, e0, var0)
}


// ============================================================================
// Rotation with modified Newton (parabolic fit)
// ============================================================================

fn rotate_dimer_newton(
    state: &mut DimerState,
    model: &PredModel,
    f_rot_direction: &[f64],
    config: &DimerConfig,
    y_std: f64,
) -> Option<f64> {
    let orient = state.orient.clone();
    let f_rot_norm = vec_norm(f_rot_direction);

    if f_rot_norm < 1e-10 {
        return None;
    }

    let (g0, g1, _) = predict_dimer_gradients(state, model, y_std);
    let c0 = curvature(&g0, &g1, &orient, state.dimer_sep);

    let dtheta = 0.5 * (0.5 * f_rot_norm / (c0.abs() + 1e-10)).atan();
    if dtheta < config.t_angle_rot {
        return Some(c0);
    }

    let orient_rot: Vec<f64> = f_rot_direction.iter().map(|x| x / f_rot_norm).collect();

    // Trial rotation
    let orient_trial: Vec<f64> = orient
        .iter()
        .zip(orient_rot.iter())
        .map(|(o, r)| dtheta.cos() * o + dtheta.sin() * r)
        .collect();
    let orient_trial = normalize_vec(&orient_trial);

    // Evaluate GP at trial R1
    let r1_trial: Vec<f64> = state
        .r
        .iter()
        .zip(orient_trial.iter())
        .map(|(r, o)| r + state.dimer_sep * o)
        .collect();
    let pred1_trial = model.predict(&r1_trial);
    let g1_trial: Vec<f64> = pred1_trial[1..].iter().map(|v| v * y_std).collect();

    let f_rot_trial = rotational_force(&g0, &g1_trial, &orient_trial, state.dimer_sep);

    // Project trial force onto rotated perpendicular direction
    let orient_rot_trial: Vec<f64> = orient
        .iter()
        .zip(orient_rot.iter())
        .map(|(o, r)| -dtheta.sin() * o + dtheta.cos() * r)
        .collect();
    let orient_rot_trial = normalize_vec(&orient_rot_trial);
    let f_dtheta = vec_dot(&f_rot_trial, &orient_rot_trial);
    let f_0 = vec_dot(f_rot_direction, &orient_rot);

    // Parabolic fit
    let sin2 = (2.0 * dtheta).sin();
    let cos2 = (2.0 * dtheta).cos();

    if sin2.abs() < 1e-12 {
        state.orient = orient_trial;
        return None;
    }

    let a1 = (f_dtheta - f_0 * cos2) / (2.0 * sin2);
    let b1 = -0.5 * f_0;

    let angle_rot = 0.5 * (b1 / (a1 + 1e-18)).atan();

    // Ensure minimum, not maximum (EON ImprovedDimer logic)
    let mut angle_final = angle_rot;
    let mut c_est = c0 + a1 * ((2.0 * angle_final).cos() - 1.0) + b1 * (2.0 * angle_final).sin();
    if c_est > c0 {
        // Curvature got worse: try pi/2 shift
        angle_final += std::f64::consts::FRAC_PI_2;
        c_est = c0 + a1 * ((2.0 * angle_final).cos() - 1.0) + b1 * (2.0 * angle_final).sin();
    }

    // If rotation still makes curvature worse, don't rotate
    if c_est > c0 {
        return Some(c0);
    }

    let orient_new: Vec<f64> = orient
        .iter()
        .zip(orient_rot.iter())
        .map(|(o, r)| angle_final.cos() * o + angle_final.sin() * r)
        .collect();
    state.orient = normalize_vec(&orient_new);

    Some(c_est)
}

// ============================================================================
// Rotation strategies
// ============================================================================

fn rotate_dimer_simple(
    state: &mut DimerState,
    model: &PredModel,
    config: &DimerConfig,
    y_std: f64,
) {
    for _ in 0..config.max_rot_iter {
        let (g0, g1, _) = predict_dimer_gradients(state, model, y_std);
        let f_rot = rotational_force(&g0, &g1, &state.orient, state.dimer_sep);
        let f_rot_norm = vec_norm(&f_rot);

        if f_rot_norm < 1e-10 {
            break;
        }

        let c = curvature(&g0, &g1, &state.orient, state.dimer_sep);
        let dtheta = 0.5 * (f_rot_norm / (c.abs() + 1e-10)).atan();

        if dtheta < config.t_angle_rot {
            break;
        }

        let b1: Vec<f64> = f_rot.iter().map(|x| x / f_rot_norm).collect();
        let orient_new: Vec<f64> = state
            .orient
            .iter()
            .zip(b1.iter())
            .map(|(o, b)| dtheta.cos() * o + dtheta.sin() * b)
            .collect();
        state.orient = normalize_vec(&orient_new);
    }
}

fn rotate_dimer_lbfgs(
    state: &mut DimerState,
    model: &PredModel,
    config: &DimerConfig,
    rot_hist: &mut LbfgsHistory,
    y_std: f64,
) {
    let mut f_rot_prev: Vec<f64> = Vec::new();
    let mut orient_prev: Vec<f64> = Vec::new();

    for _ in 0..config.max_rot_iter {
        let (g0, g1, _) = predict_dimer_gradients(state, model, y_std);
        let f_rot = rotational_force(&g0, &g1, &state.orient, state.dimer_sep);
        let f_rot_norm = vec_norm(&f_rot);

        if f_rot_norm < 1e-10 {
            break;
        }

        if !f_rot_prev.is_empty() {
            let s: Vec<f64> = state
                .orient
                .iter()
                .zip(orient_prev.iter())
                .map(|(a, b)| a - b)
                .collect();
            let y: Vec<f64> = f_rot
                .iter()
                .zip(f_rot_prev.iter())
                .map(|(a, b)| -(a - b))
                .collect();
            rot_hist.push_pair(s, y);
        }

        let neg_f: Vec<f64> = f_rot.iter().map(|x| -x).collect();
        let mut search_dir = rot_hist.compute_direction(&neg_f);

        // Project perpendicular to orient
        let sd_dot = vec_dot(&search_dir, &state.orient);
        for (s, o) in search_dir.iter_mut().zip(state.orient.iter()) {
            *s -= sd_dot * o;
        }
        let sn = vec_norm(&search_dir);
        if sn < 1e-12 {
            search_dir = f_rot.clone();
        } else {
            for s in search_dir.iter_mut() {
                *s /= sn;
            }
        }

        // Project force onto search direction
        let f_proj = vec_dot(&f_rot, &search_dir);
        let f_rot_oriented: Vec<f64> = search_dir.iter().map(|s| f_proj * s).collect();

        f_rot_prev = f_rot;
        orient_prev = state.orient.clone();

        let c_est = rotate_dimer_newton(state, model, &f_rot_oriented, config, y_std);

        if let Some(_c) = c_est {
            let dtheta = vec_dot(&orient_prev, &state.orient)
                .clamp(-1.0, 1.0)
                .acos();
            if dtheta < config.t_angle_rot {
                break;
            }
        } else {
            break;
        }
    }
}

fn rotate_dimer(
    state: &mut DimerState,
    model: &PredModel,
    config: &DimerConfig,
    rot_hist: &mut Option<LbfgsHistory>,
    y_std: f64,
) {
    match config.rotation_method.as_str() {
        "lbfgs" => {
            if let Some(ref mut hist) = rot_hist {
                rotate_dimer_lbfgs(state, model, config, hist, y_std);
            }
        }
        _ => rotate_dimer_simple(state, model, config, y_std),
    }
}

// ============================================================================
// Translation
// ============================================================================

fn translate_dimer_lbfgs(
    state: &DimerState,
    g0: &[f64],
    g1: &[f64],
    config: &DimerConfig,
    trans_hist: &mut LbfgsHistory,
    curvature_override: Option<f64>,
) -> (Vec<f64>, Vec<f64>, f64) {
    let c = curvature_override.unwrap_or_else(|| curvature(g0, g1, &state.orient, state.dimer_sep));

    if c < 0.0 {
        let f_trans = translational_force(g0, &state.orient);
        let neg_f: Vec<f64> = f_trans.iter().map(|x| -x).collect();
        let mut search_dir = trans_hist.compute_direction(&neg_f);

        let step_len = vec_norm(&search_dir);
        if step_len > config.max_step {
            let scale = config.max_step / step_len;
            for s in search_dir.iter_mut() {
                *s *= scale;
            }
            trans_hist.reset();
        }

        let r_new: Vec<f64> = state
            .r
            .iter()
            .zip(search_dir.iter())
            .map(|(r, s)| r + s)
            .collect();
        (r_new, f_trans, c)
    } else {
        let f_along: Vec<f64> = {
            let dot: f64 = g0.iter().zip(state.orient.iter()).map(|(g, o)| g * o).sum();
            state.orient.iter().map(|o| dot * o).collect()
        };
        let fn_val = vec_norm(&f_along);
        if fn_val < 1e-12 {
            return (state.r.clone(), vec![0.0; state.r.len()], c);
        }

        let r_new: Vec<f64> = state
            .r
            .iter()
            .zip(f_along.iter())
            .map(|(r, f)| r + config.step_convex * f / fn_val)
            .collect();
        trans_hist.reset();
        (r_new, f_along, c)
    }
}

// ============================================================================
// Main GP-Dimer
// ============================================================================

/// GP-dimer saddle point search.
pub fn gp_dimer(
    oracle: &OracleFn,
    x_init: &[f64],
    orient_init: &[f64],
    kernel: &Kernel,
    config: &DimerConfig,
    training_data: Option<TrainingData>,
    dimer_sep: f64,
) -> DimerResult {
    let d = x_init.len();
    let cfg = config;

    let orient = normalize_vec(orient_init);
    let mut state = DimerState {
        r: x_init.to_vec(),
        orient,
        dimer_sep,
    };

    let mut td = training_data.unwrap_or_else(|| TrainingData::new(d));

    // Generate initial training data (C++ AtomicDimer::initialize pattern)
    if td.npoints() == 0 {
        if cfg.verbose {
            eprintln!("Generating initial training data...");
        }

        // Evaluate midpoint
        let (e, g) = oracle(x_init);
        td.add_point(x_init, e, &g).expect("add_point failed: invalid data");

        // Evaluate image1 along initial orientation (C++ always evaluates this)
        let (r1_init, _) = dimer_images(&state);
        let (e1, g1) = oracle(&r1_init);
        td.add_point(&r1_init, e1, &g1).expect("add_point failed: invalid data");

        // Optional perturbations
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        for _ in 0..cfg.n_initial_perturb {
            let perturb: Vec<f64> = (0..d)
                .map(|_| (rng.random::<f64>() - 0.5) * cfg.perturb_scale)
                .collect();
            let x_p: Vec<f64> = x_init.iter().zip(perturb.iter()).map(|(a, b)| a + b).collect();
            let (e_p, g_p) = oracle(&x_p);
            if e_p.is_finite() && e_p < 1e6 {
                td.add_point(&x_p, e_p, &g_p).expect("add_point failed: invalid data");
            }
        }
    }

    let mut oracle_calls = td.npoints();
    let mut history = DimerHistory::default();

    // Record initial state in history (before any GP-guided steps)
    {
        let g_init = &td.gradients[0..d];
        let f_trans = translational_force(g_init, &state.orient);
        let f_norm = vec_norm(&f_trans);
        history.e_true.push(td.energies[0]);
        history.f_true.push(f_norm);
        history.curv_true.push(f64::NAN);
        history.oracle_calls.push(oracle_calls);
        history.sigma_perp.push(0.0);
    }

    let mut rot_hist: Option<LbfgsHistory> = if cfg.rotation_method == "lbfgs" {
        Some(LbfgsHistory::new(cfg.lbfgs_memory))
    } else {
        None
    };
    let mut trans_hist = LbfgsHistory::new(cfg.lbfgs_memory);
    trans_hist.default_scaling = cfg.alpha_trans;
    let mut f_trans_prev: Vec<f64> = Vec::new();

    let mut stop_reason = StopReason::MaxIterations;
    let mut stagnation_count = 0;
    let mut prev_f_true = f64::NEG_INFINITY;
    let mut prev_kern: Option<Kernel> = None;
    let n_atoms = d / 3;
    // Track last GP curvature from inner loop for history.
    let mut c_last_gp = f64::NAN;

    for _outer_iter in 0..cfg.max_outer_iter {
        if cfg.max_oracle_calls > 0 && oracle_calls >= cfg.max_oracle_calls {
            stop_reason = StopReason::OracleCap;
            break;
        }

        // FPS subset selection
        let dist_fn = |a: &[f64], b: &[f64]| -> f64 {
            trust_distance(cfg.trust_metric, &cfg.atom_types, a, b)
        };

        let td_sub = if cfg.fps_history > 0 && td.npoints() > cfg.fps_history {
            let sub_idx = select_optim_subset(
                &td,
                &state.r,
                cfg.fps_history,
                cfg.fps_latest_points,
                &dist_fn,
            );
            td.extract_subset(&sub_idx)
        } else {
            td.clone()
        };

        // Train GP
        let train_iters = adaptive_train_iters(cfg.gp_train_iter, prev_kern.is_none());

        let (mut y_sub, grad_sub) = cfg.prior_mean.residualize_training_data(&td_sub);
        y_sub.extend_from_slice(&grad_sub);

        // Use config-provided kernel for first iteration; SCG optimizes from there.
        // Data-dependent init is pathological for clustered dimer data (0.01A sep).
        let kern = match &prev_kern {
            None => kernel.clone(),
            Some(k) => k.clone(),
        };

        let mut gp_sub = GPModel::new(kern, &td_sub, y_sub.clone(), cfg.noise_e, cfg.noise_g, cfg.jitter)
            .expect("GPModel::new failed: invalid training data or kernel params");
        // Dynamic constSigma2 (MATLAB atomic_GP_dimer.m:453): max(1, mean_y^2)
        // Uses SHIFTED energies (y_sub[0..n]), not raw.
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
        prev_kern = Some(gp_sub.kernel.clone());

        // Build prediction model on full data (RFF if configured, else exact GP)
        let y_std = 1.0;
        let model = build_pred_model_full_with_prior(
            &gp_sub.kernel, &td, cfg.rff_features, 42, const_sigma2,
            &GPNoiseParams { noise_e: cfg.noise_e, noise_g: cfg.noise_g, jitter: cfg.jitter },
            &cfg.prior_mean,
        );

        // Reset L-BFGS/CG state for new outer iteration
        if let Some(ref mut rh) = rot_hist {
            rh.reset();
        }
        trans_hist.reset();
        f_trans_prev.clear();

        // Inner loop: optimize on GP surface
        let _r_prev_outer = state.r.clone();

        for inner_iter in 0..cfg.max_inner_iter {
            // GP rotation + GP curvature (matches C++ gpr_optim main relaxation).
            // All rotation and curvature computation on GP surface; oracle evaluates
            // only midpoint after inner convergence.
            rotate_dimer(&mut state, &model, cfg, &mut rot_hist, y_std);
            state.orient = normalize_vec(&state.orient);
            let (g0, g1, e0) = predict_dimer_gradients(&state, &model, y_std);
            let e0_pred = e0;

            // Translate (GP curvature from GP-predicted g0, g1)
            let (r_new, f_trans_cur, c) = if cfg.translation_method == "lbfgs" {
                let (rn, ft, c) =
                    translate_dimer_lbfgs(&state, &g0, &g1, cfg, &mut trans_hist, None);

                // Update L-BFGS history
                if !f_trans_prev.is_empty() {
                    let s: Vec<f64> = rn
                        .iter()
                        .zip(state.r.iter())
                        .map(|(a, b)| a - b)
                        .collect();
                    let y: Vec<f64> = ft
                        .iter()
                        .zip(f_trans_prev.iter())
                        .map(|(a, b)| -(a - b))
                        .collect();
                    trans_hist.push_pair(s, y);
                }
                f_trans_prev = ft.clone();
                (rn, ft, c)
            } else {
                let f_trans = translational_force(&g0, &state.orient);
                let f_norm = vec_norm(&f_trans);
                let c = curvature(&g0, &g1, &state.orient, state.dimer_sep);
                let step_size = if c.abs() > 1e-6 {
                    cfg.alpha_trans.min(0.1 * f_norm / c.abs())
                } else {
                    cfg.alpha_trans
                };
                let rn: Vec<f64> = state
                    .r
                    .iter()
                    .zip(f_trans.iter())
                    .map(|(r, f)| r + step_size * f)
                    .collect();
                (rn, f_trans, c)
            };

            let f_norm = vec_norm(&f_trans_cur);
            c_last_gp = c;

            if cfg.verbose && (inner_iter % 10 == 0 || inner_iter == 0) {
                eprintln!(
                    "  GP step {:3}: E = {:8.4} | |F| = {:.5} | C = {:+.3e}",
                    inner_iter, e0_pred, f_norm, c
                );
            }

            // LCB-augmented convergence: continue until both accurate and confident
            let f_eff = if cfg.lcb_kappa > 0.0 {
                let (_, _, _, var0) = predict_dimer_gradients_with_variance(&state, &model, y_std);
                let sp = perpendicular_sigma(&var0, &state.orient, d);
                f_norm + cfg.lcb_kappa * sp
            } else {
                f_norm
            };
            if f_eff < cfg.t_force_gp {
                break;
            }

            // Trust radius check
            let trust_thresh = adaptive_trust_threshold(
                cfg.trust_radius,
                td.npoints(),
                n_atoms,
                cfg.use_adaptive_threshold,
                &AdaptiveTrustParams {
                    t_min: cfg.adaptive_t_min,
                    delta_t: cfg.adaptive_delta_t,
                    n_half: cfg.adaptive_n_half,
                    a: cfg.adaptive_a,
                    floor: cfg.adaptive_floor,
                },
            );
            let trust_dist = trust_min_distance(
                &r_new,
                &td.data,
                d,
                td.npoints(),
                cfg.trust_metric,
                &cfg.atom_types,
            );
            if trust_dist > trust_thresh {
                let step_vec: Vec<f64> = r_new
                    .iter()
                    .zip(state.r.iter())
                    .map(|(a, b)| a - b)
                    .collect();
                let scale = trust_thresh / trust_dist * 0.95;
                state.r = state
                    .r
                    .iter()
                    .zip(step_vec.iter())
                    .map(|(r, s)| r + scale * s)
                    .collect();
                break;
            }

            state.r = r_new;
        }

        // Post-inner EMD trust clip
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
        clip_point_to_trust(&mut state.r, &td, &trust_params);

        // Compute sigma_perp before oracle for history recording
        let sp_at_r = if cfg.lcb_kappa > 0.0 || cfg.unc_convergence > 0.0 {
            let (_, var0) = model.predict_with_variance(&state.r);
            perpendicular_sigma(&var0, &state.orient, d)
        } else {
            0.0
        };

        // Evaluate oracle at midpoint only (matches C++ main relaxation: 1 call/iter).
        // GP rotation and GP curvature are used for orient and curvature tracking.
        let (e_true, g_true) = oracle(&state.r);
        oracle_calls += 1;
        td.add_point(&state.r, e_true, &g_true).expect("add_point failed: invalid data");

        let f_trans_true = translational_force(&g_true, &state.orient);
        let f_norm_true = vec_norm(&f_trans_true);

        if cfg.verbose {
            eprintln!(
                "  True: E = {:8.4} | |F| = {:.5} | C_gp = {:+.3e}",
                e_true, f_norm_true, c_last_gp
            );
        }

        // Stagnation
        if (f_norm_true - prev_f_true).abs() < 1e-10 {
            stagnation_count += 1;
        } else {
            stagnation_count = 0;
        }
        prev_f_true = f_norm_true;

        if stagnation_count >= 3 {
            stop_reason = StopReason::ForceStagnation;
            break;
        }

        history.e_true.push(e_true);
        history.f_true.push(f_norm_true);
        history.curv_true.push(c_last_gp);
        history.oracle_calls.push(oracle_calls);
        history.sigma_perp.push(sp_at_r);

        // Convergence check: force only (C++ isFinalConvergenceReached).
        // GP rotation maintains saddle orientation; curvature check uses GP.
        if f_norm_true < cfg.t_force_true && c_last_gp < 0.0 {
            if cfg.verbose {
                eprintln!("CONVERGED TO SADDLE POINT!");
                eprintln!("Final Energy:    {:.6}", e_true);
                eprintln!("Final |F|:       {:.6}", f_norm_true);
                eprintln!("Final Curvature: {:+.6} (GP)", c_last_gp);
                eprintln!("Oracle calls:    {}", oracle_calls);
            }
            stop_reason = StopReason::Converged;
            break;
        }
    }

    DimerResult {
        state,
        converged: stop_reason == StopReason::Converged,
        stop_reason,
        oracle_calls,
        history,
    }
}

/// Standard (non-GP) dimer search using direct oracle calls at every step.
///
/// Each iteration: evaluate oracle at midpoint + image1, rotate, translate.
/// Returns the same DimerResult/DimerHistory for uniform plotting.
pub fn standard_dimer(
    oracle: &OracleFn,
    x_init: &[f64],
    orient_init: &[f64],
    config: &DimerConfig,
    dimer_sep: f64,
) -> DimerResult {
    let d = x_init.len();
    let mut r = x_init.to_vec();
    let mut orient = normalize_vec(orient_init);
    let mut oracle_calls = 0;
    let mut history = DimerHistory::default();
    #[allow(unused_assignments)]
    let mut stop_reason = StopReason::MaxIterations;
    let max_step = config.max_step.max(0.05);
    let call_cap = if config.max_oracle_calls > 0 { config.max_oracle_calls } else { 600 };

    let use_lbfgs = config.translation_method == "lbfgs";
    let mut trans_hist = LbfgsHistory::new(config.lbfgs_memory);
    trans_hist.default_scaling = config.alpha_trans;
    let mut f_trans_prev: Vec<f64> = Vec::new();
    let mut r_prev = r.clone();

    loop {
        if oracle_calls >= call_cap {
            stop_reason = StopReason::OracleCap;
            break;
        }
        // Evaluate at midpoint
        let (e0, g0) = oracle(&r);
        oracle_calls += 1;

        // Evaluate at image 1 (along current orient)
        let r1: Vec<f64> = r.iter().zip(orient.iter())
            .map(|(r, o)| r + dimer_sep * o).collect();
        let (_e1, g1) = oracle(&r1);
        oracle_calls += 1;

        let mut c_cur = curvature(&g0, &g1, &orient, dimer_sep);
        let mut g1_cur = g1.clone();

        // Rotation loop (Kastner-Sherwood ImprovedDimer with L-BFGS rotation direction)
        let phi_tol_rad = config.t_angle_rot;
        let max_rots = config.max_rot_iter;
        let mut rot_lbfgs = LbfgsHistory::new(config.lbfgs_memory);
        let mut f_rot_prev: Vec<f64> = Vec::new();
        let mut orient_prev: Vec<f64> = Vec::new();

        for _rot_iter in 0..max_rots {
            if oracle_calls >= call_cap { break; }

            let f_rot = rotational_force(&g0, &g1_cur, &orient, dimer_sep);
            let f_rot_norm = vec_norm(&f_rot);
            if f_rot_norm < 1e-10 { break; }

            // L-BFGS direction for rotation (matches eOn ImprovedDimer)
            let theta = if !f_rot_prev.is_empty() {
                let s: Vec<f64> = orient.iter().zip(orient_prev.iter()).map(|(a, b)| a - b).collect();
                let y: Vec<f64> = f_rot.iter().zip(f_rot_prev.iter()).map(|(a, b)| -(a - b)).collect();
                rot_lbfgs.push_pair(s, y);

                let neg_f: Vec<f64> = f_rot.iter().map(|x| -x).collect();
                let mut dir = rot_lbfgs.compute_direction(&neg_f);
                // Project perpendicular to orient
                let sd_dot = vec_dot(&dir, &orient);
                for (d, o) in dir.iter_mut().zip(orient.iter()) { *d -= sd_dot * o; }
                let sn = vec_norm(&dir);
                if sn < 1e-12 { normalize_vec(&f_rot) }
                else { dir.iter().map(|x| x / sn).collect() }
            } else {
                normalize_vec(&f_rot)
            };

            // Curvature derivative for trial angle estimate
            let d_c_d_phi = 2.0 * g1_cur.iter().zip(g0.iter()).zip(theta.iter())
                .map(|((g1v, g0v), tv)| (g1v - g0v) * tv).sum::<f64>() / dimer_sep;
            let phi_prime = -0.5 * (d_c_d_phi / (2.0 * c_cur.abs() + 1e-10)).atan();

            if phi_prime.abs() < phi_tol_rad { break; }

            // Evaluate at trial rotation
            let tau_prime: Vec<f64> = orient.iter().zip(theta.iter())
                .map(|(o, t)| phi_prime.cos() * o + phi_prime.sin() * t).collect();
            let tau_prime = normalize_vec(&tau_prime);
            let r1_trial: Vec<f64> = r.iter().zip(tau_prime.iter())
                .map(|(r, o)| r + dimer_sep * o).collect();
            let (_, g1_trial) = oracle(&r1_trial);
            oracle_calls += 1;

            let c_trial = curvature(&g0, &g1_trial, &tau_prime, dimer_sep);

            // Parabolic fit for optimal angle (Kastner-Sherwood Eq. 4-6)
            let b1 = 0.5 * d_c_d_phi;
            let a1 = (c_cur - c_trial + b1 * (2.0 * phi_prime).sin())
                / (1.0 - (2.0 * phi_prime).cos());
            let phi_min = 0.5 * (b1 / a1).atan();

            let a0 = 2.0 * (c_cur - a1);
            let mut c_min = 0.5 * a0 + a1 * (2.0 * phi_min).cos() + b1 * (2.0 * phi_min).sin();
            let mut phi_final = phi_min;

            if c_min > c_cur {
                phi_final += std::f64::consts::FRAC_PI_2;
                c_min = 0.5 * a0 + a1 * (2.0 * phi_final).cos() + b1 * (2.0 * phi_final).sin();
            }

            // Wrap angle for L-BFGS accuracy (eOn: if phi_min > pi/2, subtract pi)
            if phi_final > std::f64::consts::FRAC_PI_2 {
                phi_final -= std::f64::consts::PI;
            }

            if c_min >= c_cur { break; }

            // Accept rotation
            f_rot_prev = f_rot;
            orient_prev = orient.clone();

            orient = orient.iter().zip(theta.iter())
                .map(|(o, t)| phi_final.cos() * o + phi_final.sin() * t).collect();
            orient = normalize_vec(&orient);
            orient = normalize_vec(&orient);

            // Interpolate g1 at optimal angle (Kastner-Sherwood Eq. 8)
            if phi_prime.abs() > 1e-15 {
                g1_cur = g1_cur.iter().zip(g1_trial.iter()).zip(g0.iter())
                    .map(|((g1v, g1pv), g0v)| {
                        let sr1 = (phi_prime - phi_final).sin() / phi_prime.sin();
                        let sr2 = phi_final.sin() / phi_prime.sin();
                        let cc = 1.0 - phi_final.cos() - phi_final.sin() * (phi_prime * 0.5).tan();
                        sr1 * g1v + sr2 * g1pv + cc * g0v
                    }).collect();
            } else {
                g1_cur = g1_trial;
            }
            c_cur = c_min;
        }

        // Curvature and translational force with the ROTATED orient
        let c = c_cur;
        let f_trans = translational_force(&g0, &orient);
        let mut f_trans_proj = f_trans;
        project_out_translations(&mut f_trans_proj);
        let f_norm = vec_norm(&f_trans_proj);

        history.e_true.push(e0);
        history.f_true.push(f_norm);
        history.curv_true.push(c);
        history.oracle_calls.push(oracle_calls);
        history.sigma_perp.push(0.0);

        // Convergence check
        if f_norm < config.t_force_true && c < 0.0 {
            stop_reason = StopReason::Converged;
            break;
        }

        // Translate
        if c < 0.0 {
            if use_lbfgs {
                // L-BFGS translation (matches eOn LBFGS.cpp with auto_scale)
                let n_atoms = d / 3;
                let mut h0 = 0.01_f64; // inverse_curvature default (eOn LBFGS.cpp)

                // Update L-BFGS history and compute auto-scaled H0
                if !f_trans_prev.is_empty() {
                    let dr: Vec<f64> = r.iter().zip(r_prev.iter()).map(|(a, b)| a - b).collect();
                    let df: Vec<f64> = f_trans_prev.iter().zip(f_trans_proj.iter())
                        .map(|(fp, fc)| fp - fc).collect();
                    let dr_dot_df: f64 = dr.iter().zip(df.iter()).map(|(a, b)| a * b).sum();
                    let df_dot_df: f64 = df.iter().map(|x| x * x).sum();

                    if dr_dot_df > 1e-18 {
                        let curv = df_dot_df / dr_dot_df;
                        if curv > 0.0 {
                            h0 = 1.0 / curv;
                        } else {
                            // Negative curvature: reset, take max move step
                            trans_hist.reset();
                            let scaled_f: Vec<f64> = f_trans_proj.iter()
                                .map(|x| 1000.0 * x).collect();
                            let step = max_atom_motion_applied(&scaled_f, max_step, n_atoms);
                            for j in 0..d { r[j] += step[j]; }
                            r_prev = r.clone();
                            f_trans_prev = f_trans_proj.clone();
                            continue;
                        }
                        trans_hist.push_pair(dr, df);
                    }
                }

                let neg_f: Vec<f64> = f_trans_proj.iter().map(|x| -x).collect();
                let search_dir = trans_hist.compute_direction(&neg_f);

                // H0 scaling now handled by LbfgsHistory::default_scaling (0.01)
                // When history exists, LBFGS auto-scales via gamma = s.y/y.y

                let d_vec = search_dir.clone();

                // Distance reset: if any atom moves > max_move, reset
                let max_atom = max_atom_motion(&d_vec, n_atoms);
                if max_atom >= max_step {
                    trans_hist.reset();
                    let fallback: Vec<f64> = f_trans_proj.iter()
                        .map(|x| h0 * x).collect();
                    let step = max_atom_motion_applied(&fallback, max_step, n_atoms);
                    r_prev = r.clone();
                    for j in 0..d { r[j] += step[j]; }
                    f_trans_prev = f_trans_proj.clone();
                    continue;
                }

                // Angle reset: if direction > 90 degrees from force
                let d_norm = vec_norm(&d_vec);
                let f_norm_local = vec_norm(&f_trans_proj);
                if d_norm > 1e-18 && f_norm_local > 1e-18 {
                    let cos_angle: f64 = d_vec.iter().zip(f_trans_proj.iter())
                        .map(|(a, b)| a * b).sum::<f64>() / (d_norm * f_norm_local);
                    if cos_angle.clamp(-1.0, 1.0).acos() > std::f64::consts::FRAC_PI_2 {
                        trans_hist.reset();
                        let fallback: Vec<f64> = f_trans_proj.iter()
                            .map(|x| h0 * x).collect();
                        let step = max_atom_motion_applied(&fallback, max_step, n_atoms);
                        r_prev = r.clone();
                        for j in 0..d { r[j] += step[j]; }
                        f_trans_prev = f_trans_proj.clone();
                        continue;
                    }
                }

                // Per-atom clipping on final step
                let step = max_atom_motion_applied(&d_vec, max_step, n_atoms);
                r_prev = r.clone();
                for j in 0..d { r[j] += step[j]; }
                f_trans_prev = f_trans_proj.clone();
            } else {
                let step_size = max_step / f_norm.max(1e-18);
                for j in 0..d {
                    r[j] += step_size * f_trans_proj[j];
                }
            }
        } else {
            // Convex region: move uphill along dimer axis (C++ Dimer.cpp:161-180)
            let g_par: f64 = g0.iter().zip(orient.iter()).map(|(g, o)| g * o).sum();
            let f_along: Vec<f64> = orient.iter().map(|o| g_par * o).collect();
            let fn_val = vec_norm(&f_along);
            trans_hist.reset();
            f_trans_prev.clear();
            if fn_val > 1e-12 {
                let step_len = config.step_convex.min(max_step);
                for j in 0..d {
                    r[j] += step_len * f_along[j] / fn_val;
                }
            }
        }
    }

    DimerResult {
        state: DimerState { r, orient, dimer_sep },
        converged: stop_reason == StopReason::Converged,
        stop_reason,
        oracle_calls,
        history,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimer_utilities() {
        let state = DimerState {
            r: vec![0.0, 0.0],
            orient: vec![1.0, 0.0],
            dimer_sep: 0.01,
        };
        let (r1, r2) = dimer_images(&state);
        assert!((r1[0] - 0.01).abs() < 1e-10);
        assert!((r2[0] + 0.01).abs() < 1e-10);

        let g0 = vec![1.0, 0.0];
        let g1 = vec![2.0, 0.0];
        let c = curvature(&g0, &g1, &state.orient, state.dimer_sep);
        assert!((c - 100.0).abs() < 1e-6);

        let f_trans = translational_force(&g0, &state.orient);
        // F_trans = -G0 + 2*(G0.orient)*orient = [-1,0] + 2*[1,0] = [1, 0]
        assert!((f_trans[0] - 1.0).abs() < 1e-10);
        assert!(f_trans[1].abs() < 1e-10);
    }
}
