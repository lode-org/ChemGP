//! Shared optimizer step (L-BFGS with per-atom max_move).
//!
//! Ports `optim_step.jl`.

use crate::lbfgs::LbfgsHistory;

/// Mutable optimizer state wrapping L-BFGS with previous-state tracking.
pub struct OptimState {
    pub lbfgs: LbfgsHistory,
    pub prev_x: Option<Vec<f64>>,
    pub prev_g: Option<Vec<f64>>,
}

impl OptimState {
    pub fn new(memory: usize) -> Self {
        Self {
            lbfgs: LbfgsHistory::new(memory),
            prev_x: None,
            prev_g: None,
        }
    }

    pub fn reset(&mut self) {
        self.lbfgs.reset();
        self.prev_x = None;
        self.prev_g = None;
    }

    /// Convenience wrapper around the free function `optim_step`.
    pub fn step(
        &mut self,
        x: &[f64],
        force: &[f64],
        max_move: f64,
        n_coords_per_atom: usize,
    ) -> Vec<f64> {
        optim_step(self, x, force, max_move, n_coords_per_atom)
    }
}

/// Clip displacement so no atom moves more than max_move.
pub fn clip_to_max_move(d: &[f64], max_move: f64, n_coords_per_atom: usize) -> Vec<f64> {
    let mut result = d.to_vec();
    let n_atoms = d.len() / n_coords_per_atom;
    let mut max_disp = 0.0f64;

    for a in 0..n_atoms {
        let off = a * n_coords_per_atom;
        let disp: f64 = result[off..off + n_coords_per_atom]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        max_disp = max_disp.max(disp);
    }

    if max_disp > max_move {
        let scale = max_move / max_disp;
        for v in result.iter_mut() {
            *v *= scale;
        }
    }

    result
}

/// Compute an L-BFGS displacement from position and force (downhill direction).
///
/// Returns displacement to add to x.
pub fn optim_step(
    state: &mut OptimState,
    x: &[f64],
    force: &[f64],
    max_move: f64,
    n_coords_per_atom: usize,
) -> Vec<f64> {
    let g: Vec<f64> = force.iter().map(|f| -f).collect();

    // Negative curvature reset
    if let (Some(ref prev_x), Some(ref prev_g)) = (&state.prev_x, &state.prev_g) {
        let s: Vec<f64> = x.iter().zip(prev_x.iter()).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = g.iter().zip(prev_g.iter()).map(|(a, b)| a - b).collect();
        let sy: f64 = s.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

        if sy < 0.0 {
            state.reset();
            state.prev_x = Some(x.to_vec());
            state.prev_g = Some(g);
            return clip_to_max_move(force, max_move, n_coords_per_atom);
        }
        state.lbfgs.push_pair(s, y);
    }

    state.prev_x = Some(x.to_vec());
    state.prev_g = Some(g.clone());

    let direction = state.lbfgs.compute_direction(&g);

    // Angle check
    let fn_norm: f64 = force.iter().map(|x| x * x).sum::<f64>().sqrt();
    let dn: f64 = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
    if fn_norm > 1e-30 && dn > 1e-30 {
        let cos_angle: f64 = direction
            .iter()
            .zip(force.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / (dn * fn_norm);
        if cos_angle < 0.0 {
            state.reset();
            state.prev_x = Some(x.to_vec());
            state.prev_g = Some(force.iter().map(|f| -f).collect());
            return clip_to_max_move(force, max_move, n_coords_per_atom);
        }
    }

    // Distance reset
    let n_atoms = direction.len() / n_coords_per_atom;
    let mut max_disp = 0.0f64;
    for a in 0..n_atoms {
        let off = a * n_coords_per_atom;
        let disp: f64 = direction[off..off + n_coords_per_atom]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        max_disp = max_disp.max(disp);
    }
    if max_disp > max_move {
        state.reset();
        state.prev_x = Some(x.to_vec());
        state.prev_g = Some(force.iter().map(|f| -f).collect());
        return clip_to_max_move(force, max_move, n_coords_per_atom);
    }

    direction
}
