//! Shared dimer geometry and step-limiting utilities.
//!
//! Used by both `dimer.rs` (standard + GP dimer) and `otgpd.rs`.

/// L2 norm of a vector.
pub(crate) fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Dot product of two vectors.
pub(crate) fn vec_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Normalize a vector to unit length. Returns unchanged if near-zero.
pub(crate) fn normalize_vec(v: &[f64]) -> Vec<f64> {
    let n = vec_norm(v);
    if n > 1e-18 {
        v.iter().map(|x| x / n).collect()
    } else {
        v.to_vec()
    }
}

/// Project out translational components from a 3N vector.
///
/// For N atoms in 3D, removes the 3 rigid translation modes so the
/// dimer operates only in the internal coordinate subspace.
pub(crate) fn project_out_translations(v: &mut [f64]) {
    let n = v.len() / 3;
    if n == 0 {
        return;
    }
    for d in 0..3 {
        let avg: f64 = (0..n).map(|i| v[3 * i + d]).sum::<f64>() / n as f64;
        for i in 0..n {
            v[3 * i + d] -= avg;
        }
    }
}

/// Curvature along dimer direction: C = (g1 - g0) . orient / dimer_sep.
pub(crate) fn curvature(g0: &[f64], g1: &[f64], orient: &[f64], dimer_sep: f64) -> f64 {
    let dot: f64 = g1
        .iter()
        .zip(g0.iter())
        .zip(orient.iter())
        .map(|((g1, g0), o)| (g1 - g0) * o)
        .sum();
    dot / dimer_sep
}

/// Rotational force perpendicular to the dimer.
pub(crate) fn rotational_force(
    g0: &[f64],
    g1: &[f64],
    orient: &[f64],
    dimer_sep: f64,
) -> Vec<f64> {
    let g_diff: Vec<f64> = g1
        .iter()
        .zip(g0.iter())
        .map(|(a, b)| (a - b) / dimer_sep)
        .collect();
    let dot: f64 = g_diff.iter().zip(orient.iter()).map(|(g, o)| g * o).sum();
    g_diff
        .iter()
        .zip(orient.iter())
        .map(|(g, o)| g - dot * o)
        .collect()
}

/// Modified translational force for saddle point search.
///
/// F_trans = -g0 + 2*(g0 . orient)*orient
pub(crate) fn translational_force(g0: &[f64], orient: &[f64]) -> Vec<f64> {
    let f_par: f64 = g0.iter().zip(orient.iter()).map(|(g, o)| g * o).sum();
    g0.iter()
        .zip(orient.iter())
        .map(|(g, o)| -g + 2.0 * f_par * o)
        .collect()
}

/// Maximum per-atom displacement in a 3N step vector.
pub(crate) fn max_atom_motion(step: &[f64], n_atoms: usize) -> f64 {
    let mut max_disp = 0.0_f64;
    for i in 0..n_atoms {
        let dx = step[3 * i];
        let dy = step[3 * i + 1];
        let dz = step[3 * i + 2];
        max_disp = max_disp.max((dx * dx + dy * dy + dz * dz).sqrt());
    }
    max_disp
}

/// Clip a 3N step vector so no atom moves more than `max_move`.
pub(crate) fn max_atom_motion_applied(step: &[f64], max_move: f64, n_atoms: usize) -> Vec<f64> {
    if n_atoms < 2 {
        // Non-molecular: clip total norm
        let norm = vec_norm(step);
        if norm > max_move && norm > 1e-18 {
            return step.iter().map(|x| x * max_move / norm).collect();
        }
        return step.to_vec();
    }
    let max_disp = max_atom_motion(step, n_atoms);
    if max_disp > max_move && max_disp > 1e-18 {
        let scale = max_move / max_disp;
        step.iter().map(|x| x * scale).collect()
    } else {
        step.to_vec()
    }
}
