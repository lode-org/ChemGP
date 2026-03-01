//! Optimal Transport GP Dimer (OTGPD) saddle point search.
//!
//! Ports `otgpd.jl`: GP-dimer with initial rotation on true potential,
//! adaptive GP threshold, HOD (hyperparameter oscillation detection),
//! and convergence requiring negative curvature.
//!
//! Reference: Goswami et al., J. Chem. Theory Comput. (2025).

use crate::kernel::MolInvDistSE;
use crate::lbfgs::LbfgsHistory;
use crate::predict::predict;
use crate::sampling::{prune_training_data, select_optim_subset};
use crate::train::train_model;
use crate::trust::{adaptive_trust_threshold, trust_distance, trust_min_distance, TrustMetric};
use crate::types::{init_mol_invdist_se, GPModel, TrainingData};
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
    pub verbose: bool,
}

impl Default for OTGPDConfig {
    fn default() -> Self {
        Self {
            t_dimer: 0.01,
            divisor_t_dimer_gp: 10.0,
            t_angle_rot: 1e-3,
            max_outer_iter: 50,
            max_inner_iter: 10000,
            max_rot_iter: 5,
            dimer_sep: 0.01,
            eval_image1: true,
            rotation_method: "lbfgs".to_string(),
            translation_method: "lbfgs".to_string(),
            lbfgs_memory: 5,
            step_convex: 0.1,
            max_step: 0.5,
            alpha_trans: 0.01,
            trust_radius: 0.5,
            ratio_at_limit: 2.0 / 3.0,
            initial_rotation: true,
            max_initial_rot: 20,
            gp_train_iter: 300,
            n_initial_perturb: 4,
            perturb_scale: 0.15,
            max_training_points: 0,
            fps_history: 5,
            fps_latest_points: 2,
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

/// HOD state for monitoring hyperparameter oscillation.
struct HodState {
    history: Vec<Vec<f64>>,
    current_fps_history: usize,
}

impl HodState {
    fn new(initial_fps: usize) -> Self {
        Self {
            history: Vec::new(),
            current_fps_history: initial_fps,
        }
    }

    /// Extract log-space hyperparameters from GP model.
    fn extract_hyperparams(model: &GPModel) -> Vec<f64> {
        let mut hp = vec![model.kernel.signal_variance.ln()];
        for &ls in &model.kernel.inv_lengthscales {
            hp.push(ls.ln());
        }
        hp.push(model.noise_var.ln());
        hp.push(model.grad_noise_var.ln());
        hp
    }

    /// Check for oscillation and potentially enlarge FPS subset.
    fn check(&mut self, model: &GPModel, cfg: &OTGPDConfig) -> bool {
        let hp = Self::extract_hyperparams(model);
        self.history.push(hp);

        if self.history.len() < 3 {
            return false;
        }

        let window = cfg.hod_monitoring_window.min(self.history.len() - 1);
        let start = self.history.len() - window - 1;

        let n_dims = self.history[start].len();
        let mut n_flips = 0;
        let mut n_pairs = 0;

        for i in start..self.history.len() - 1 {
            if i + 1 >= self.history.len() {
                break;
            }
            let d1: Vec<f64> = self.history[i]
                .iter()
                .zip(self.history[i.saturating_sub(1)].iter())
                .map(|(a, b)| a - b)
                .collect();
            let d2: Vec<f64> = self.history[i + 1]
                .iter()
                .zip(self.history[i].iter())
                .map(|(a, b)| a - b)
                .collect();

            for j in 0..n_dims {
                if d1[j].abs() > 1e-10 && d2[j].abs() > 1e-10 {
                    n_pairs += 1;
                    if d1[j].signum() != d2[j].signum() {
                        n_flips += 1;
                    }
                }
            }
        }

        if n_pairs == 0 {
            return false;
        }

        let flip_ratio = n_flips as f64 / n_pairs as f64;
        if flip_ratio > cfg.hod_flip_threshold {
            let new_size = (self.current_fps_history + cfg.hod_history_increment)
                .min(cfg.hod_max_history);
            if new_size > self.current_fps_history {
                self.current_fps_history = new_size;
                return true;
            }
        }
        false
    }
}

// ============================================================================
// Dimer utility functions (reused from dimer.rs patterns)
// ============================================================================

fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn vec_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn normalize_vec(v: &[f64]) -> Vec<f64> {
    let n = vec_norm(v);
    if n > 1e-18 { v.iter().map(|x| x / n).collect() } else { v.to_vec() }
}

fn curvature_fn(g0: &[f64], g1: &[f64], orient: &[f64], sep: f64) -> f64 {
    let dot: f64 = g1.iter().zip(g0.iter()).zip(orient.iter())
        .map(|((g1, g0), o)| (g1 - g0) * o).sum();
    dot / sep
}

fn rotational_force_fn(g0: &[f64], g1: &[f64], orient: &[f64], sep: f64) -> Vec<f64> {
    let g_diff: Vec<f64> = g1.iter().zip(g0.iter()).map(|(a, b)| (a - b) / sep).collect();
    let dot: f64 = g_diff.iter().zip(orient.iter()).map(|(g, o)| g * o).sum();
    g_diff.iter().zip(orient.iter()).map(|(g, o)| g - dot * o).collect()
}

fn translational_force_fn(g0: &[f64], orient: &[f64]) -> Vec<f64> {
    let f_par: f64 = g0.iter().zip(orient.iter()).map(|(g, o)| g * o).sum();
    g0.iter().zip(orient.iter()).map(|(g, o)| -g + 2.0 * f_par * o).collect()
}

fn predict_dimer(
    r: &[f64], orient: &[f64], sep: f64, model: &GPModel, e_ref: f64,
) -> (Vec<f64>, Vec<f64>, f64) {
    let r1: Vec<f64> = r.iter().zip(orient.iter()).map(|(r, o)| r + sep * o).collect();
    let pred0 = predict(model, r, 1);
    let pred1 = predict(model, &r1, 1);
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
    model: &GPModel,
    e_ref: f64,
    cfg: &OTGPDConfig,
) -> f64 {
    let mut best_c = 0.0;
    for _ in 0..cfg.max_rot_iter {
        let (g0, g1, _) = predict_dimer(r, orient, sep, model, e_ref);
        let f_rot = rotational_force_fn(&g0, &g1, orient, sep);
        let f_rot_norm = vec_norm(&f_rot);

        if f_rot_norm < 1e-10 {
            best_c = curvature_fn(&g0, &g1, orient, sep);
            break;
        }

        let c0 = curvature_fn(&g0, &g1, orient, sep);
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
        let pred1_trial = predict(model, &r1_trial, 1);
        let g1_trial: Vec<f64> = pred1_trial[1..].to_vec();

        let f_rot_trial = rotational_force_fn(&g0, &g1_trial, &orient_trial, sep);

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

        let a1 = (f_dtheta - f_0 * cos2) / sin2;
        let b1 = -0.5 * f_0;
        let mut angle_final = 0.5 * (b1 / (a1 + 1e-18)).atan();

        if a1 * (2.0 * angle_final).cos() + b1 * (2.0 * angle_final).sin() > 0.0 {
            angle_final += std::f64::consts::FRAC_PI_2;
        }

        let orient_new: Vec<f64> = orient.iter().zip(orient_rot.iter())
            .map(|(o, r)| angle_final.cos() * o + angle_final.sin() * r).collect();
        *orient = normalize_vec(&orient_new);

        best_c = c0 + a1 * ((2.0 * angle_final).cos() - 1.0) + b1 * (2.0 * angle_final).sin();
    }
    best_c
}

/// GP-accelerated optimal transport dimer search.
pub fn otgpd(
    oracle: &OracleFn,
    x_init: &[f64],
    orient_init: &[f64],
    kernel: &MolInvDistSE,
    config: &OTGPDConfig,
    training_data: Option<TrainingData>,
) -> OTGPDResult {
    let cfg = config;
    let d = x_init.len();

    let mut orient = normalize_vec(orient_init);
    let mut r = x_init.to_vec();

    let mut td = training_data.unwrap_or_else(|| TrainingData::new(d));
    let mut oracle_calls = 0;

    // Initial data generation
    if td.npoints() == 0 {
        let (e, g) = oracle(&r);
        td.add_point(&r, e, &g);
        oracle_calls += 1;

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

    // Evaluate at midpoint and image 1
    let (e_r, g_r) = oracle(&r);
    td.add_point(&r, e_r, &g_r);
    oracle_calls += 1;

    let r1: Vec<f64> = r.iter().zip(orient.iter())
        .map(|(r, o)| r + cfg.dimer_sep * o).collect();
    let (e_r1, g_r1) = oracle(&r1);
    td.add_point(&r1, e_r1, &g_r1);
    oracle_calls += 1;

    let mut history = OTGPDHistory::default();

    // Phase 1: Initial rotation on true potential
    if cfg.initial_rotation && cfg.max_initial_rot > 0 {
        for _ in 0..cfg.max_initial_rot {
            let (_, g0) = oracle(&r);
            let r1_cur: Vec<f64> = r.iter().zip(orient.iter())
                .map(|(r, o)| r + cfg.dimer_sep * o).collect();
            let (_, g1) = oracle(&r1_cur);
            oracle_calls += 2;
            td.add_point(&r, g0[0], &g0); // Note: using gradient as is
            td.add_point(&r1_cur, g1[0], &g1);

            let f_rot = rotational_force_fn(&g0, &g1, &orient, cfg.dimer_sep);
            let f_rot_norm = vec_norm(&f_rot);
            let c0 = curvature_fn(&g0, &g1, &orient, cfg.dimer_sep);

            let dtheta = 0.5 * (0.5 * f_rot_norm / (c0.abs() + 1e-10)).atan();
            if dtheta < cfg.t_angle_rot {
                break;
            }

            let orient_rot: Vec<f64> = f_rot.iter().map(|x| x / f_rot_norm).collect();
            let orient_trial: Vec<f64> = orient.iter().zip(orient_rot.iter())
                .map(|(o, r)| dtheta.cos() * o + dtheta.sin() * r).collect();
            let orient_trial = normalize_vec(&orient_trial);

            // Trial evaluation
            let r1_trial: Vec<f64> = r.iter().zip(orient_trial.iter())
                .map(|(r, o)| r + cfg.dimer_sep * o).collect();
            let (_, g1_trial) = oracle(&r1_trial);
            oracle_calls += 1;
            td.add_point(&r1_trial, 0.0, &g1_trial);

            let f_rot_trial = rotational_force_fn(&g0, &g1_trial, &orient_trial, cfg.dimer_sep);

            let orient_rot_trial: Vec<f64> = orient.iter().zip(orient_rot.iter())
                .map(|(o, r)| -dtheta.sin() * o + dtheta.cos() * r).collect();
            let orient_rot_trial = normalize_vec(&orient_rot_trial);

            let f_dtheta = vec_dot(&f_rot_trial, &orient_rot_trial);
            let f_0 = vec_dot(&f_rot, &orient_rot);

            let sin2 = (2.0 * dtheta).sin();
            let cos2 = (2.0 * dtheta).cos();

            if sin2.abs() < 1e-12 {
                orient = orient_trial;
                continue;
            }

            let a1 = (f_dtheta - f_0 * cos2) / sin2;
            let b1 = -0.5 * f_0;
            let mut angle_final = 0.5 * (b1 / (a1 + 1e-18)).atan();

            if a1 * (2.0 * angle_final).cos() + b1 * (2.0 * angle_final).sin() > 0.0 {
                angle_final += std::f64::consts::FRAC_PI_2;
            }

            let orient_new: Vec<f64> = orient.iter().zip(orient_rot.iter())
                .map(|(o, r)| angle_final.cos() * o + angle_final.sin() * r).collect();
            orient = normalize_vec(&orient_new);
        }
    }

    // Phase 2: GP-accelerated loop
    let mut prev_kern: Option<MolInvDistSE> = None;
    let mut stop_reason = StopReason::MaxIterations;
    let mut hod_state = HodState::new(cfg.fps_history);
    let n_atoms = d / 3;

    for outer_iter in 0..cfg.max_outer_iter {
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

        let kern = match &prev_kern {
            None => init_mol_invdist_se(&td_sub, kernel),
            Some(k) => k.clone(),
        };

        let mut gp_sub = GPModel::new(kern, &td_sub, y_sub, 1e-6, 1e-4, 1e-6);
        train_model(&mut gp_sub, train_iters, cfg.verbose);
        prev_kern = Some(gp_sub.kernel.clone());

        // HOD check
        if cfg.use_hod {
            hod_state.check(&gp_sub, cfg);
        }

        // Rebuild on full data
        let e_ref = td.energies[0];
        let mut y_gp: Vec<f64> = td.energies.iter().map(|e| e - e_ref).collect();
        y_gp.extend_from_slice(&td.gradients);
        let model = GPModel::new(gp_sub.kernel.clone(), &td, y_gp, 1e-6, 1e-4, 1e-6);

        // Adaptive GP threshold
        let t_gp = if cfg.divisor_t_dimer_gp > 0.0 && !history.f_true.is_empty() {
            let min_f = history.f_true.iter().cloned().fold(f64::INFINITY, f64::min);
            (min_f / cfg.divisor_t_dimer_gp).max(cfg.t_dimer / 10.0)
        } else {
            cfg.t_dimer / 10.0
        };

        // Inner loop: optimize on GP surface
        let mut trans_hist = LbfgsHistory::new(cfg.lbfgs_memory);
        let _r_before = r.clone();

        for _inner_iter in 0..cfg.max_inner_iter {
            // Rotate
            let _c = rotate_on_gp(&r, &mut orient, cfg.dimer_sep, &model, e_ref, cfg);

            // Predict
            let (g0, g1, _e0) = predict_dimer(&r, &orient, cfg.dimer_sep, &model, e_ref);
            let c = curvature_fn(&g0, &g1, &orient, cfg.dimer_sep);
            let f_trans = translational_force_fn(&g0, &orient);
            let f_norm = vec_norm(&f_trans);

            if f_norm < t_gp {
                break;
            }

            // Translate
            let step = if c < 0.0 {
                // Negative curvature: L-BFGS
                let dir = trans_hist.compute_direction(&f_trans);
                let step_size = (cfg.max_step / vec_norm(&dir).max(1e-18)).min(0.01 / f_norm);
                dir.iter().map(|d| step_size * d).collect::<Vec<f64>>()
            } else {
                // Positive curvature: fixed step along force
                f_trans.iter().map(|f| cfg.step_convex * f / f_norm.max(1e-18)).collect::<Vec<f64>>()
            };

            let step_norm = vec_norm(&step);
            let step = if step_norm > cfg.max_step {
                step.iter().map(|s| s * cfg.max_step / step_norm).collect()
            } else {
                step
            };

            let r_new: Vec<f64> = r.iter().zip(step.iter()).map(|(a, b)| a + b).collect();

            // Trust region check
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
                let nearest_idx = (0..td.npoints())
                    .min_by(|&a, &b| {
                        let da = trust_distance(cfg.trust_metric, &cfg.atom_types, &r_new, td.col(a));
                        let db = trust_distance(cfg.trust_metric, &cfg.atom_types, &r_new, td.col(b));
                        da.partial_cmp(&db).unwrap()
                    })
                    .unwrap();
                let nearest = td.col(nearest_idx).to_vec();
                let disp: Vec<f64> = r_new.iter().zip(nearest.iter()).map(|(a, b)| a - b).collect();
                r = nearest.iter().zip(disp.iter())
                    .map(|(a, b)| a + b * (thresh / dist * 0.95))
                    .collect();
                break;
            }

            // Update L-BFGS history
            if c < 0.0 {
                let s: Vec<f64> = r_new.iter().zip(r.iter()).map(|(a, b)| a - b).collect();
                let (g0_new, _g1_new, _) = predict_dimer(&r_new, &orient, cfg.dimer_sep, &model, e_ref);
                let f_trans_new = translational_force_fn(&g0_new, &orient);
                let y: Vec<f64> = f_trans.iter().zip(f_trans_new.iter()).map(|(a, b)| a - b).collect();
                trans_hist.push_pair(s, y);
            }

            r = r_new;
        }

        // Oracle evaluation
        let (e_true, g_true) = oracle(&r);
        oracle_calls += 1;
        td.add_point(&r, e_true, &g_true);

        let mut c_true = f64::NAN;
        if cfg.eval_image1 {
            let r1_cur: Vec<f64> = r.iter().zip(orient.iter())
                .map(|(r, o)| r + cfg.dimer_sep * o).collect();
            let (e_r1, g_r1) = oracle(&r1_cur);
            oracle_calls += 1;
            td.add_point(&r1_cur, e_r1, &g_r1);
            c_true = curvature_fn(&g_true, &g_r1, &orient, cfg.dimer_sep);
        }

        let f_trans_true = translational_force_fn(&g_true, &orient);
        let f_norm_true = vec_norm(&f_trans_true);

        history.e_true.push(e_true);
        history.f_true.push(f_norm_true);
        history.curv_true.push(c_true);
        history.oracle_calls.push(oracle_calls);
        history.t_gp.push(t_gp);

        if cfg.verbose {
            eprintln!(
                "OTGPD outer {}: E={:.6} |F|={:.5} C={:.4} T_gp={:.5} calls={}",
                outer_iter, e_true, f_norm_true, c_true, t_gp, oracle_calls,
            );
        }

        // Convergence: small force AND negative curvature
        if f_norm_true < cfg.t_dimer && (c_true.is_nan() || c_true < 0.0) {
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

        let kernel = MolInvDistSE::isotropic(1.0, 1.0, vec![]);

        let result = otgpd(&oracle, &x_init, &orient_init, &kernel, &cfg, None);
        assert!(result.oracle_calls > 2);
        assert_eq!(result.r.len(), 9);
        assert_eq!(result.orient.len(), 9);
    }
}
