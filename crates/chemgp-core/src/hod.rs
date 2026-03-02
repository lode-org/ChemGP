//! Hyperparameter Oscillation Detection (HOD).
//!
//! Monitors GP hyperparameter trajectories for sign-flip oscillation.
//! When detected, grows the FPS training subset to stabilize training.
//! Ported from gpr_optim GaussianProcessRegression.cpp.

use crate::types::GPModel;

/// HOD configuration.
#[derive(Debug, Clone)]
pub struct HodConfig {
    /// Sliding window size for sign-flip detection.
    pub monitoring_window: usize,
    /// Fraction of sign flips that triggers growth (0.0--1.0).
    pub flip_threshold: f64,
    /// FPS subset growth increment when oscillation detected.
    pub history_increment: usize,
    /// Maximum FPS subset size.
    pub max_history: usize,
}

impl Default for HodConfig {
    fn default() -> Self {
        Self {
            monitoring_window: 5,
            flip_threshold: 0.8,
            history_increment: 2,
            max_history: 30,
        }
    }
}

/// Tracks hyperparameter history and detects oscillation.
pub struct HodState {
    history: Vec<Vec<f64>>,
    pub current_fps_history: usize,
}

impl HodState {
    pub fn new(initial_fps: usize) -> Self {
        Self {
            history: Vec::new(),
            current_fps_history: initial_fps,
        }
    }

    /// Extract log-space hyperparameters from a trained GP model.
    pub fn extract_hyperparams(model: &GPModel) -> Vec<f64> {
        let mut hp = vec![model.kernel.signal_variance().ln()];
        for ls in model.kernel.inv_lengthscales() {
            hp.push(ls.ln());
        }
        hp.push(model.noise_var.ln());
        hp.push(model.grad_noise_var.ln());
        hp
    }

    /// Record hyperparameters and check for oscillation.
    /// Returns true if oscillation detected and FPS subset was grown.
    pub fn check(&mut self, model: &GPModel, cfg: &HodConfig) -> bool {
        let hp = Self::extract_hyperparams(model);
        self.history.push(hp);

        if self.history.len() < 3 {
            return false;
        }

        let window = cfg.monitoring_window.min(self.history.len() - 1);
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
        if flip_ratio > cfg.flip_threshold {
            let new_size =
                (self.current_fps_history + cfg.history_increment).min(cfg.max_history);
            if new_size > self.current_fps_history {
                self.current_fps_history = new_size;
                return true;
            }
        }
        false
    }
}
