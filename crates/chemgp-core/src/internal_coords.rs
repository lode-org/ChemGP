//! Redundant internal-coordinate utilities for molecular optimization.
//!
//! The current complete-curvilinear path uses redundant inverse-distance
//! coordinates, which are already native to ChemGP's molecular kernels.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoordinateMode {
    Cartesian,
    CompleteRedundantInvDist,
}

impl Default for CoordinateMode {
    fn default() -> Self {
        Self::Cartesian
    }
}

#[derive(Debug, Clone)]
pub struct RedundantInverseDistance {
    pub n_atoms: usize,
    pub pairs: Vec<(usize, usize)>,
}

impl RedundantInverseDistance {
    pub fn new(n_atoms: usize) -> Self {
        let mut pairs = Vec::new();
        for i in 0..n_atoms {
            for j in i + 1..n_atoms {
                pairs.push((i, j));
            }
        }
        Self { n_atoms, pairs }
    }

    pub fn values(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), 3 * self.n_atoms, "coordinate dimension mismatch");
        self.pairs
            .iter()
            .map(|&(i, j)| {
                let dx = x[3 * i] - x[3 * j];
                let dy = x[3 * i + 1] - x[3 * j + 1];
                let dz = x[3 * i + 2] - x[3 * j + 2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-12);
                1.0 / r
            })
            .collect()
    }

    pub fn jacobian(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), 3 * self.n_atoms, "coordinate dimension mismatch");
        let m = self.pairs.len();
        let d = 3 * self.n_atoms;
        let mut b = vec![0.0; m * d];
        for (row, &(i, j)) in self.pairs.iter().enumerate() {
            let dx = x[3 * i] - x[3 * j];
            let dy = x[3 * i + 1] - x[3 * j + 1];
            let dz = x[3 * i + 2] - x[3 * j + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let r = r2.sqrt().max(1e-12);
            let inv_r3 = 1.0 / (r2 * r).max(1e-18);
            let gx = -dx * inv_r3;
            let gy = -dy * inv_r3;
            let gz = -dz * inv_r3;
            b[row * d + 3 * i] = gx;
            b[row * d + 3 * i + 1] = gy;
            b[row * d + 3 * i + 2] = gz;
            b[row * d + 3 * j] = -gx;
            b[row * d + 3 * j + 1] = -gy;
            b[row * d + 3 * j + 2] = -gz;
        }
        b
    }

    pub fn cartesian_to_internal_gradient(
        &self,
        x: &[f64],
        grad_x: &[f64],
        damping: f64,
    ) -> Vec<f64> {
        let d = 3 * self.n_atoms;
        assert_eq!(grad_x.len(), d, "gradient dimension mismatch");
        let b = self.jacobian(x);
        let btb = build_btb(&b, self.pairs.len(), d, damping);
        let y = solve_spd(&btb, grad_x).unwrap_or_else(|| grad_x.to_vec());
        let mut grad_q = vec![0.0; self.pairs.len()];
        for row in 0..self.pairs.len() {
            let mut acc = 0.0;
            for col in 0..d {
                acc += b[row * d + col] * y[col];
            }
            grad_q[row] = acc;
        }
        grad_q
    }

    pub fn backtransform_target(
        &self,
        x_start: &[f64],
        q_target: &[f64],
        damping: f64,
        max_iter: usize,
        tol: f64,
        max_cart_step: f64,
    ) -> Vec<f64> {
        let d = 3 * self.n_atoms;
        assert_eq!(x_start.len(), d, "coordinate dimension mismatch");
        assert_eq!(q_target.len(), self.pairs.len(), "internal dimension mismatch");
        let mut x = x_start.to_vec();

        for _ in 0..max_iter.max(1) {
            let q = self.values(&x);
            let dq: Vec<f64> = q_target.iter().zip(q.iter()).map(|(a, b)| a - b).collect();
            let dq_norm = dq.iter().map(|v| v * v).sum::<f64>().sqrt();
            if dq_norm < tol {
                break;
            }
            let b = self.jacobian(&x);
            let btb = build_btb(&b, self.pairs.len(), d, damping);
            let mut rhs = vec![0.0; d];
            for col in 0..d {
                let mut acc = 0.0;
                for row in 0..self.pairs.len() {
                    acc += b[row * d + col] * dq[row];
                }
                rhs[col] = acc;
            }
            let mut dx = solve_spd(&btb, &rhs).unwrap_or_else(|| vec![0.0; d]);
            clip_cartesian_step(&mut dx, max_cart_step);
            let mut accepted = false;
            let mut trial_scale = 1.0;
            let mut best_x = x.clone();
            let mut best_err = dq_norm;
            for _ in 0..8 {
                let mut x_trial = x.clone();
                for (xi, dxi) in x_trial.iter_mut().zip(dx.iter()) {
                    *xi += trial_scale * dxi;
                }
                let q_trial = self.values(&x_trial);
                let err = q_target
                    .iter()
                    .zip(q_trial.iter())
                    .map(|(a, b)| {
                        let dv = a - b;
                        dv * dv
                    })
                    .sum::<f64>()
                    .sqrt();
                if err < best_err {
                    best_err = err;
                    best_x = x_trial;
                    accepted = true;
                }
                if err < dq_norm {
                    break;
                }
                trial_scale *= 0.5;
            }
            if accepted {
                x = best_x;
            } else {
                break;
            }
        }

        x
    }

    pub fn internal_step(
        &self,
        x: &[f64],
        grad_x: &[f64],
        step_size: f64,
        damping: f64,
        max_backtransform_iter: usize,
        backtransform_tol: f64,
        max_cart_step: f64,
    ) -> Vec<f64> {
        let q = self.values(x);
        let grad_q = self.cartesian_to_internal_gradient(x, grad_x, damping);
        let q_target: Vec<f64> = q
            .iter()
            .zip(grad_q.iter())
            .map(|(qi, gi)| qi - step_size * gi)
            .collect();
        self.backtransform_target(
            x,
            &q_target,
            damping,
            max_backtransform_iter,
            backtransform_tol,
            max_cart_step,
        )
    }
}

fn build_btb(b: &[f64], m: usize, d: usize, damping: f64) -> Vec<f64> {
    let mut out = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut acc = 0.0;
            for row in 0..m {
                acc += b[row * d + i] * b[row * d + j];
            }
            if i == j {
                acc += damping;
            }
            out[i * d + j] = acc;
            out[j * d + i] = acc;
        }
    }
    out
}

fn solve_spd(a: &[f64], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = rhs.len();
    if a.len() != n * n {
        return None;
    }
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 1e-14 {
                    return None;
                }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }

    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = rhs[i];
        for k in 0..i {
            sum -= l[i * n + k] * y[k];
        }
        y[i] = sum / l[i * n + i];
    }

    let mut x = vec![0.0; n];
    for ii in 0..n {
        let i = n - 1 - ii;
        let mut sum = y[i];
        for k in i + 1..n {
            sum -= l[k * n + i] * x[k];
        }
        x[i] = sum / l[i * n + i];
    }
    Some(x)
}

fn clip_cartesian_step(step: &mut [f64], max_cart_step: f64) {
    if max_cart_step <= 0.0 {
        return;
    }
    let max_atom = step
        .chunks_exact(3)
        .map(|chunk| chunk.iter().map(|v| v * v).sum::<f64>().sqrt())
        .fold(0.0f64, f64::max);
    if max_atom <= max_cart_step || max_atom == 0.0 {
        return;
    }
    let scale = max_cart_step / max_atom;
    for v in step.iter_mut() {
        *v *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::RedundantInverseDistance;

    #[test]
    fn inverse_distance_jacobian_matches_finite_difference() {
        let sys = RedundantInverseDistance::new(3);
        let x = vec![0.0, 0.1, 0.0, 1.2, -0.2, 0.0, 2.0, 0.5, -0.1];
        let b = sys.jacobian(&x);
        let q0 = sys.values(&x);
        let h = 1e-6;
        for col in 0..x.len() {
            let mut xp = x.clone();
            xp[col] += h;
            let qp = sys.values(&xp);
            for row in 0..q0.len() {
                let fd = (qp[row] - q0[row]) / h;
                let an = b[row * x.len() + col];
                assert!((fd - an).abs() < 1e-4, "row {row} col {col}: fd={fd} an={an}");
            }
        }
    }

    #[test]
    fn backtransform_hits_small_inverse_distance_target() {
        let sys = RedundantInverseDistance::new(3);
        let x = vec![0.0, 0.0, 0.0, 1.0, 0.2, 0.0, 1.8, 0.7, 0.1];
        let mut q_target = sys.values(&x);
        q_target[0] += 0.005;
        let q_old = sys.values(&x);
        let x_new = sys.backtransform_target(&x, &q_target, 1e-8, 20, 1e-8, 0.1);
        let q_new = sys.values(&x_new);
        let old_err = (q_old[0] - q_target[0]).abs();
        let new_err = (q_new[0] - q_target[0]).abs();
        assert!(new_err < old_err);
        assert!(new_err < 1e-3);
    }
}
