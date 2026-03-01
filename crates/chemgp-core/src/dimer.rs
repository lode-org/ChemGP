//! GP-Dimer saddle point search.
//!
//! Ports `dimer.jl`: rotation + translation on GP surface with oracle calls
//! at trust boundary.
//!
//! Reference: Henkelman & Jonsson, J. Chem. Phys. 111, 7010 (1999).
//! GP-Dimer: Koistinen et al., J. Chem. Theory Comput. 16, 499 (2020).

use crate::kernel::MolInvDistSE;
use crate::lbfgs::LbfgsHistory;
use crate::predict::predict;
use crate::sampling::select_optim_subset;
use crate::train::train_model;
use crate::trust::{adaptive_trust_threshold, trust_distance, trust_min_distance, TrustMetric};
use crate::types::{init_mol_invdist_se, GPModel, TrainingData};
use crate::StopReason;
use rand::Rng;
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
    pub verbose: bool,
}

impl Default for DimerConfig {
    fn default() -> Self {
        Self {
            t_force_true: 1e-3,
            t_force_gp: 1e-2,
            t_angle_rot: 1e-3,
            trust_radius: 0.1,
            max_outer_iter: 50,
            max_oracle_calls: 0,
            max_inner_iter: 100,
            max_rot_iter: 10,
            alpha_trans: 0.01,
            gp_train_iter: 300,
            n_initial_perturb: 4,
            perturb_scale: 0.15,
            rotation_method: "lbfgs".to_string(),
            translation_method: "lbfgs".to_string(),
            lbfgs_memory: 5,
            max_step: 0.5,
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

/// Curvature along dimer direction.
fn curvature(g0: &[f64], g1: &[f64], orient: &[f64], dimer_sep: f64) -> f64 {
    let dot: f64 = g1
        .iter()
        .zip(g0.iter())
        .zip(orient.iter())
        .map(|((g1, g0), o)| (g1 - g0) * o)
        .sum();
    dot / dimer_sep
}

/// Rotational force perpendicular to the dimer.
fn rotational_force(g0: &[f64], g1: &[f64], orient: &[f64], dimer_sep: f64) -> Vec<f64> {
    let g_diff: Vec<f64> = g1
        .iter()
        .zip(g0.iter())
        .map(|(a, b)| (a - b) / dimer_sep)
        .collect();
    let dot: f64 = g_diff.iter().zip(orient.iter()).map(|(g, o)| g * o).sum();
    g_diff.iter().zip(orient.iter()).map(|(g, o)| g - dot * o).collect()
}

/// Modified translational force for saddle point search.
fn translational_force(g0: &[f64], orient: &[f64]) -> Vec<f64> {
    let f_par: f64 = g0.iter().zip(orient.iter()).map(|(g, o)| g * o).sum();
    g0.iter()
        .zip(orient.iter())
        .map(|(g, o)| -g + 2.0 * f_par * o)
        .collect()
}

/// Project out translational components from a 3N vector.
///
/// For N atoms in 3D, removes the 3 rigid translation modes so the
/// dimer operates only in the internal coordinate subspace.
fn project_out_translations(v: &mut [f64]) {
    let n = v.len() / 3;
    if n == 0 {
        return;
    }
    // For each Cartesian direction, remove the uniform translation component
    for d in 0..3 {
        let avg: f64 = (0..n).map(|i| v[3 * i + d]).sum::<f64>() / n as f64;
        for i in 0..n {
            v[3 * i + d] -= avg;
        }
    }
}

fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn vec_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn normalize_vec(v: &[f64]) -> Vec<f64> {
    let n = vec_norm(v);
    if n > 1e-18 {
        v.iter().map(|x| x / n).collect()
    } else {
        v.to_vec()
    }
}

// ============================================================================
// GP prediction helpers
// ============================================================================

fn predict_dimer_gradients(
    state: &DimerState,
    model: &GPModel,
    y_std: f64,
) -> (Vec<f64>, Vec<f64>, f64) {
    let (r1, _) = dimer_images(state);
    let pred0 = predict(model, &state.r, 1);
    let pred1 = predict(model, &r1, 1);

    let g0: Vec<f64> = pred0[1..].iter().map(|v| v * y_std).collect();
    let g1: Vec<f64> = pred1[1..].iter().map(|v| v * y_std).collect();
    let e0 = pred0[0] * y_std;

    (g0, g1, e0)
}


// ============================================================================
// Rotation with modified Newton (parabolic fit)
// ============================================================================

fn rotate_dimer_newton(
    state: &mut DimerState,
    model: &GPModel,
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
    let pred1_trial = predict(model, &r1_trial, 1);
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

    let a1 = (f_dtheta - f_0 * cos2) / sin2;
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
    let mut new_orient = normalize_vec(&orient_new);
    project_out_translations(&mut new_orient);
    state.orient = normalize_vec(&new_orient);

    Some(c_est)
}

// ============================================================================
// Rotation strategies
// ============================================================================

fn rotate_dimer_simple(
    state: &mut DimerState,
    model: &GPModel,
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
    model: &GPModel,
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
    model: &GPModel,
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
) -> (Vec<f64>, Vec<f64>, f64) {
    let c = curvature(g0, g1, &state.orient, state.dimer_sep);

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
            state.orient.iter().map(|o| -dot * o).collect()
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
    kernel: &MolInvDistSE,
    config: &DimerConfig,
    training_data: Option<TrainingData>,
    dimer_sep: f64,
) -> DimerResult {
    let d = x_init.len();
    let cfg = config;

    let mut orient = normalize_vec(orient_init);
    project_out_translations(&mut orient);
    orient = normalize_vec(&orient);
    let mut state = DimerState {
        r: x_init.to_vec(),
        orient,
        dimer_sep,
    };

    let mut td = training_data.unwrap_or_else(|| TrainingData::new(d));

    // Generate initial training data
    if td.npoints() == 0 {
        if cfg.verbose {
            eprintln!("Generating initial training data...");
        }

        let (e, g) = oracle(x_init);
        td.add_point(x_init, e, &g);

        let mut rng = rand::rng();
        for _ in 0..cfg.n_initial_perturb {
            let perturb: Vec<f64> = (0..d)
                .map(|_| (rng.random::<f64>() - 0.5) * cfg.perturb_scale)
                .collect();
            let x_p: Vec<f64> = x_init.iter().zip(perturb.iter()).map(|(a, b)| a + b).collect();
            let (e_p, g_p) = oracle(&x_p);
            if e_p.is_finite() && e_p < 1e6 {
                td.add_point(&x_p, e_p, &g_p);
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
    }

    let mut rot_hist: Option<LbfgsHistory> = if cfg.rotation_method == "lbfgs" {
        Some(LbfgsHistory::new(cfg.lbfgs_memory))
    } else {
        None
    };
    let mut trans_hist = LbfgsHistory::new(cfg.lbfgs_memory);
    let mut f_trans_prev: Vec<f64> = Vec::new();

    let mut stop_reason = StopReason::MaxIterations;
    let mut stagnation_count = 0;
    let mut prev_f_true = f64::NEG_INFINITY;
    let mut prev_kern: Option<MolInvDistSE> = None;
    let n_atoms = d / 3;

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
        let train_iters = if prev_kern.is_none() {
            cfg.gp_train_iter
        } else {
            (cfg.gp_train_iter / 3).max(50)
        };

        let e_ref_sub = td_sub.energies[0];
        let mut y_sub: Vec<f64> = td_sub.energies.iter().map(|e| e - e_ref_sub).collect();
        y_sub.extend_from_slice(&td_sub.gradients);

        let kern = match &prev_kern {
            None => init_mol_invdist_se(&td_sub, kernel),
            Some(k) => k.clone(),
        };

        let mut gp_sub = GPModel::new(kern, &td_sub, y_sub.clone(), 1e-6, 1e-4, 1e-6);
        train_model(&mut gp_sub, train_iters, cfg.verbose);
        prev_kern = Some(gp_sub.kernel.clone());

        // Use trained kernel on subset for prediction (fast, O(K^3) where K = fps_history)
        let y_mean = e_ref_sub;
        let y_std = 1.0;
        let model = GPModel::new(gp_sub.kernel.clone(), &td_sub, y_sub, 1e-6, 1e-4, 1e-6);

        // Reset L-BFGS/CG state for new outer iteration
        if let Some(ref mut rh) = rot_hist {
            rh.reset();
        }
        trans_hist.reset();
        f_trans_prev.clear();

        // Inner loop: optimize on GP surface
        let _r_prev_outer = state.r.clone();

        for inner_iter in 0..cfg.max_inner_iter {
            // Rotate dimer (then project out translations from orient)
            rotate_dimer(&mut state, &model, cfg, &mut rot_hist, y_std);
            project_out_translations(&mut state.orient);
            state.orient = normalize_vec(&state.orient);

            // Predict at current position
            let (g0, g1, e0) = predict_dimer_gradients(&state, &model, y_std);
            let e0_pred = e0 + y_mean;

            // Translate
            let (r_new, f_trans_cur, c) = if cfg.translation_method == "lbfgs" {
                let (rn, ft, c) =
                    translate_dimer_lbfgs(&state, &g0, &g1, cfg, &mut trans_hist);

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

            if cfg.verbose && (inner_iter % 10 == 0 || inner_iter == 0) {
                eprintln!(
                    "  GP step {:3}: E = {:8.4} | |F| = {:.5} | C = {:+.3e}",
                    inner_iter, e0_pred, f_norm, c
                );
            }

            if f_norm < cfg.t_force_gp {
                break;
            }

            // Trust radius check
            let trust_thresh = adaptive_trust_threshold(
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
        let dimer_thresh = adaptive_trust_threshold(
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
        let dimer_td = trust_min_distance(
            &state.r,
            &td.data,
            d,
            td.npoints(),
            cfg.trust_metric,
            &cfg.atom_types,
        );
        if dimer_td > dimer_thresh {
            let nearest_idx = (0..td.npoints())
                .min_by(|&a, &b| {
                    let da = trust_distance(cfg.trust_metric, &cfg.atom_types, &state.r, td.col(a));
                    let db = trust_distance(cfg.trust_metric, &cfg.atom_types, &state.r, td.col(b));
                    da.partial_cmp(&db).unwrap()
                })
                .unwrap();
            let nearest = td.col(nearest_idx).to_vec();
            let disp: Vec<f64> = state.r.iter().zip(nearest.iter()).map(|(a, b)| a - b).collect();
            state.r = nearest
                .iter()
                .zip(disp.iter())
                .map(|(a, b)| a + b * (dimer_thresh / dimer_td * 0.95))
                .collect();
        }

        // Call oracle at midpoint and image 1
        let (e_true, g_true) = oracle(&state.r);
        oracle_calls += 1;

        let (r1, _) = dimer_images(&state);
        let (e1_true, g1_true) = oracle(&r1);
        oracle_calls += 1;

        let c_true = curvature(&g_true, &g1_true, &state.orient, state.dimer_sep);
        let f_trans_true = translational_force(&g_true, &state.orient);
        let f_norm_true = vec_norm(&f_trans_true);

        if cfg.verbose {
            eprintln!(
                "  True: E = {:8.4} | |F| = {:.5} | C = {:+.3e}",
                e_true, f_norm_true, c_true
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
        history.curv_true.push(c_true);
        history.oracle_calls.push(oracle_calls);

        td.add_point(&state.r, e_true, &g_true);
        td.add_point(&r1, e1_true, &g1_true);

        // Convergence check
        if f_norm_true < cfg.t_force_true && c_true < 0.0 {
            if cfg.verbose {
                eprintln!("CONVERGED TO SADDLE POINT!");
                eprintln!("Final Energy:    {:.6}", e_true);
                eprintln!("Final |F|:       {:.6}", f_norm_true);
                eprintln!("Final Curvature: {:+.6}", c_true);
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
    project_out_translations(&mut orient);
    orient = normalize_vec(&orient);
    let mut oracle_calls = 0;
    let mut history = DimerHistory::default();
    #[allow(unused_assignments)]
    let mut stop_reason = StopReason::MaxIterations;
    let max_step = 0.05;
    let call_cap = if config.max_oracle_calls > 0 { config.max_oracle_calls } else { 600 };

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

        let c0 = curvature(&g0, &g1, &orient, dimer_sep);

        // Rotate: parabolic fit rotation step (Henkelman & Jonsson 1999)
        let f_rot = rotational_force(&g0, &g1, &orient, dimer_sep);
        let f_rot_norm = vec_norm(&f_rot);
        let mut g1_final = g1.clone();
        if f_rot_norm > 1e-10 {
            let dtheta = 0.5 * (0.5 * f_rot_norm / (c0.abs() + 1e-10)).atan().min(0.3);
            let orient_rot: Vec<f64> = f_rot.iter().map(|x| x / f_rot_norm).collect();

            // Trial rotation at dtheta
            let orient_trial: Vec<f64> = orient.iter().zip(orient_rot.iter())
                .map(|(o, r)| dtheta.cos() * o + dtheta.sin() * r).collect();
            let orient_trial = normalize_vec(&orient_trial);

            let r1_trial: Vec<f64> = r.iter().zip(orient_trial.iter())
                .map(|(r, o)| r + dimer_sep * o).collect();
            let (_, g1_trial) = oracle(&r1_trial);
            oracle_calls += 1;

            // Parabolic fit for optimal angle
            let f_rot_trial = rotational_force(&g0, &g1_trial, &orient_trial, dimer_sep);
            let orient_rot_trial: Vec<f64> = orient.iter().zip(orient_rot.iter())
                .map(|(o, r)| -dtheta.sin() * o + dtheta.cos() * r).collect();
            let orient_rot_trial = normalize_vec(&orient_rot_trial);

            let f_dtheta = vec_dot(&f_rot_trial, &orient_rot_trial);
            let f_0 = vec_dot(&f_rot, &orient_rot);
            let sin2 = (2.0 * dtheta).sin();
            let cos2 = (2.0 * dtheta).cos();

            // Check if trial rotation improves curvature
            let c_trial = curvature(&g0, &g1_trial, &orient_trial, dimer_sep);

            if c_trial < c0 {
                // Trial rotation improved curvature: accept and try parabolic refinement
                if sin2.abs() > 1e-12 {
                    let a1 = (f_dtheta - f_0 * cos2) / sin2;
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
                        // Parabolic fit improvement: use fitted angle
                        let orient_new: Vec<f64> = orient.iter().zip(orient_rot.iter())
                            .map(|(o, r)| angle_final.cos() * o + angle_final.sin() * r).collect();
                        orient = normalize_vec(&orient_new);

                        // Interpolate g1 at fitted angle
                        if dtheta.abs() > 1e-15 {
                            let sin_ratio_1 = (dtheta - angle_final).sin() / dtheta.sin();
                            let sin_ratio_2 = angle_final.sin() / dtheta.sin();
                            let cos_correction = 1.0 - angle_final.cos()
                                - angle_final.sin() * (dtheta * 0.5).tan();
                            g1_final = g1.iter()
                                .zip(g1_trial.iter())
                                .zip(g0.iter())
                                .map(|((a, b), c)| sin_ratio_1 * a + sin_ratio_2 * b + cos_correction * c)
                                .collect();
                        }
                    } else {
                        // Parabolic fit failed: use trial angle
                        orient = orient_trial.clone();
                        g1_final = g1_trial.clone();
                    }
                } else {
                    // sin2 too small: use trial angle directly
                    orient = orient_trial.clone();
                    g1_final = g1_trial.clone();
                }
                project_out_translations(&mut orient);
                orient = normalize_vec(&orient);
            }
            // else: rotation made curvature worse, keep original orient
        }

        // Curvature and translational force with the ROTATED orient and matching g1
        let c = curvature(&g0, &g1_final, &orient, dimer_sep);
        let f_trans = translational_force(&g0, &orient);
        let mut f_trans_proj = f_trans;
        project_out_translations(&mut f_trans_proj);
        let f_norm = vec_norm(&f_trans_proj);

        history.e_true.push(e0);
        history.f_true.push(f_norm);
        history.curv_true.push(c);
        history.oracle_calls.push(oracle_calls);

        // Convergence check
        if f_norm < config.t_force_true && c < 0.0 {
            stop_reason = StopReason::Converged;
            break;
        }

        // Translate: for negative curvature, use modified force; for positive, step along orient
        if c < 0.0 {
            let step_size = (max_step / f_norm.max(1e-18)).min(0.01);
            for j in 0..d {
                r[j] += step_size * f_trans_proj[j];
            }
        } else {
            // Convex region: step along negative gradient projected onto orient (uphill)
            let g_par: f64 = g0.iter().zip(orient.iter()).map(|(g, o)| g * o).sum();
            let mut step: Vec<f64> = orient.iter().map(|o| -g_par * o).collect();
            project_out_translations(&mut step);
            let sn = vec_norm(&step);
            if sn > 1e-18 {
                let scale = max_step.min(sn) / sn;
                for j in 0..d {
                    r[j] += scale * config.step_convex * step[j];
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
