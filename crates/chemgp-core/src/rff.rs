//! Random Fourier Features for GP kernels.
//!
//! Ports `rff.jl`. Approximates the GP with O(N*D*D_rff + D_rff^3) cost.

use crate::kernel::Kernel;
use faer::linalg::solvers::Solve;
use faer::{Mat, Side};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

/// How RFF features are computed from raw coordinates.
#[derive(Debug, Clone)]
pub enum FeatureMode {
    /// Inverse interatomic distances (molecular kernel).
    InverseDistances { frozen: Vec<f64> },
    /// Raw coordinates (Cartesian kernel, identity Jacobian).
    Cartesian,
}

/// RFF model for fast GP prediction.
pub struct RffModel {
    pub w: Mat<f64>,      // (d_rff, d_feat)
    pub b: Vec<f64>,      // (d_rff,)
    pub c: f64,           // sigma * sqrt(2/d_rff)
    pub feature_mode: FeatureMode,
    pub alpha: Vec<f64>,  // (d_eff,) where d_eff = d_rff + 1 (constant feature)
    pub a_chol: Mat<f64>, // Lower-triangular Cholesky factor of regularized Gram
    pub dim: usize,       // Coordinate dimension D
    pub const_sigma2: f64, // Constant kernel variance
}

impl RffModel {
    /// Compute RFF features z(x) and Jacobian J_z = dz/dx.
    fn features(&self, x: &[f64]) -> (Vec<f64>, Mat<f64>) {
        let d = x.len();
        let d_rff = self.w.nrows();

        let (phi, j_phi) = match &self.feature_mode {
            FeatureMode::InverseDistances { frozen } => {
                crate::kernel::invdist_jacobian(x, frozen)
            }
            FeatureMode::Cartesian => {
                // Features = coordinates, Jacobian = identity
                let phi = x.to_vec();
                let d_feat = d;
                let mut j_phi = Mat::<f64>::zeros(d_feat, d);
                for i in 0..d_feat {
                    j_phi[(i, i)] = 1.0;
                }
                (phi, j_phi)
            }
        };

        let d_feat = phi.len();

        // u = W * phi + b
        let mut u = vec![0.0; d_rff];
        for i in 0..d_rff {
            let mut s = self.b[i];
            for f in 0..d_feat {
                s += self.w[(i, f)] * phi[f];
            }
            u[i] = s;
        }

        let mut z: Vec<f64> = u.iter().map(|&v| self.c * v.cos()).collect();

        // w_jphi = W * J_phi (d_rff x d)
        let mut w_jphi = Mat::<f64>::zeros(d_rff, d);
        for i in 0..d_rff {
            for j in 0..d {
                let mut s = 0.0;
                for f in 0..d_feat {
                    s += self.w[(i, f)] * j_phi[(f, j)];
                }
                w_jphi[(i, j)] = s;
            }
        }

        // J_z = -c * sin(u) .* (W * J_phi)
        // Append constant feature: z gets sqrt(const_sigma2), j_z gets zero row.
        let d_eff = d_rff + 1;
        let mut j_z = Mat::<f64>::zeros(d_eff, d);
        for i in 0..d_rff {
            let sin_u = u[i].sin();
            for j in 0..d {
                j_z[(i, j)] = -self.c * sin_u * w_jphi[(i, j)];
            }
        }
        // Row d_rff (constant feature) is already zeros.

        z.push(self.const_sigma2.sqrt());

        (z, j_z)
    }
}

/// Build an RFF model from a trained kernel and all training data.
pub fn build_rff(
    kernel: &Kernel,
    x_train: &[f64],
    dim: usize,
    n: usize,
    y_train: &[f64],
    d_rff: usize,
    noise_var: f64,
    grad_noise_var: f64,
    seed: u64,
    const_sigma2: f64,
) -> RffModel {
    let inv_ls = kernel.inv_lengthscales();
    let d_feat = kernel.n_features(dim);

    let feature_mode = match kernel {
        Kernel::MolInvDist(k) => FeatureMode::InverseDistances {
            frozen: k.frozen_coords.clone(),
        },
        Kernel::Cartesian(_) => FeatureMode::Cartesian,
    };

    let mut rng = StdRng::seed_from_u64(seed);

    // Sample frequencies from N(0, 2*theta^2 * I)
    let feat_map = kernel.feature_params_map();
    let mut w = Mat::<f64>::zeros(d_rff, d_feat);
    if feat_map.is_empty() && inv_ls.len() == 1 {
        let scale = (2.0f64).sqrt() * inv_ls[0];
        for i in 0..d_rff {
            for f in 0..d_feat {
                w[(i, f)] = rng.sample::<f64, _>(StandardNormal) * scale;
            }
        }
    } else {
        for f in 0..d_feat {
            let idx = if feat_map.is_empty() { 0 } else { feat_map[f] };
            let scale = (2.0f64).sqrt() * inv_ls[idx];
            for i in 0..d_rff {
                w[(i, f)] = rng.sample::<f64, _>(StandardNormal) * scale;
            }
        }
    }

    let mut b = vec![0.0; d_rff];
    for i in 0..d_rff {
        b[i] = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
    }

    let sigma = kernel.signal_variance().sqrt();
    let c = sigma * (2.0 / d_rff as f64).sqrt();

    // Build design matrix Z: [energy rows; gradient rows]
    // d_eff = d_rff + 1: extra column for constant kernel feature.
    let d_eff = d_rff + 1;
    let n_obs = n * (1 + dim);
    let mut z_mat = Mat::<f64>::zeros(n_obs, d_eff);

    let model = RffModel {
        w: w.clone(),
        b: b.clone(),
        c,
        feature_mode: feature_mode.clone(),
        alpha: vec![0.0; d_eff],
        a_chol: Mat::<f64>::zeros(d_eff, d_eff),
        dim,
        const_sigma2,
    };

    for i in 0..n {
        let xi = &x_train[i * dim..(i + 1) * dim];
        let (z, j_z) = model.features(xi);

        // Energy row i (d_eff features including constant)
        for f in 0..d_eff {
            z_mat[(i, f)] = z[f];
        }

        // Gradient rows (constant feature derivative is zero)
        for d in 0..dim {
            for f in 0..d_eff {
                z_mat[(n + i * dim + d, f)] = j_z[(f, d)];
            }
        }
    }

    // Noise precision
    let mut prec = vec![0.0; n_obs];
    for i in 0..n {
        prec[i] = 1.0 / noise_var;
    }
    for i in n..n_obs {
        prec[i] = 1.0 / grad_noise_var;
    }

    // A = Z^T diag(prec) Z + I
    // ztp = Z^T * diag(prec)  (d_eff x n_obs)
    let mut ztp = Mat::<f64>::zeros(d_eff, n_obs);
    for i in 0..d_eff {
        for j in 0..n_obs {
            ztp[(i, j)] = z_mat[(j, i)] * prec[j];
        }
    }

    // a = ztp * z_mat + I  (d_eff x d_eff)
    let mut a = Mat::<f64>::zeros(d_eff, d_eff);
    for i in 0..d_eff {
        for j in 0..d_eff {
            let mut s = 0.0;
            for k in 0..n_obs {
                s += ztp[(i, k)] * z_mat[(k, j)];
            }
            a[(i, j)] = s;
        }
        a[(i, i)] += 1.0;
    }

    let llt = a.llt(Side::Lower).expect("RFF Cholesky failed");
    let a_chol_l = {
        let l = llt.L();
        let mut m = Mat::<f64>::zeros(d_eff, d_eff);
        for i in 0..d_eff {
            for j in 0..=i {
                m[(i, j)] = l[(i, j)];
            }
        }
        m
    };

    // rhs = ztp * y  (d_eff x 1)
    let mut rhs = Mat::<f64>::zeros(d_eff, 1);
    for i in 0..d_eff {
        let mut s = 0.0;
        for j in 0..n_obs {
            s += ztp[(i, j)] * y_train[j];
        }
        rhs[(i, 0)] = s;
    }

    let alpha_mat = llt.solve(&rhs);
    let alpha: Vec<f64> = (0..d_eff).map(|i| alpha_mat[(i, 0)]).collect();

    RffModel {
        w,
        b,
        c,
        feature_mode,
        alpha,
        a_chol: a_chol_l,
        dim,
        const_sigma2,
    }
}

/// RFF prediction with interleaved output: [E1, G1_1..G1_D, E2, ...].
pub fn rff_predict(rff: &RffModel, x_test: &[f64], n_test: usize) -> Vec<f64> {
    let d = rff.dim;
    let dim_block = d + 1;
    let mut result = vec![0.0; n_test * dim_block];

    for i in 0..n_test {
        let xi = &x_test[i * d..(i + 1) * d];
        let (z, j_z) = rff.features(xi);

        let offset = i * dim_block;
        // Energy: z . alpha
        result[offset] = z.iter().zip(rff.alpha.iter()).map(|(a, b)| a * b).sum();
        // Gradient: J_z^T * alpha
        for dd in 0..d {
            let mut s = 0.0;
            for f in 0..z.len() {
                s += j_z[(f, dd)] * rff.alpha[f];
            }
            result[offset + 1 + dd] = s;
        }
    }

    result
}

/// RFF prediction with variance.
pub fn rff_predict_with_variance(
    rff: &RffModel,
    x_test: &[f64],
    n_test: usize,
) -> (Vec<f64>, Vec<f64>) {
    let d = rff.dim;
    let d_rff = rff.alpha.len();
    let dim_block = d + 1;
    let mut mu = vec![0.0; n_test * dim_block];
    let mut var = vec![0.0; n_test * dim_block];

    for i in 0..n_test {
        let xi = &x_test[i * d..(i + 1) * d];
        let (z, j_z) = rff.features(xi);

        let offset = i * dim_block;

        // Energy mean
        mu[offset] = z.iter().zip(rff.alpha.iter()).map(|(a, b)| a * b).sum();

        // Energy variance: v = L^{-1} z, var = v^T v
        let mut v = Mat::from_fn(d_rff, 1, |r, _| z[r]);
        rff.a_chol.solve_lower_triangular_in_place(&mut v);
        let mut vdot = 0.0;
        for r in 0..d_rff {
            vdot += v[(r, 0)] * v[(r, 0)];
        }
        var[offset] = vdot;

        // Gradient mean and variance
        for dd in 0..d {
            let mut s = 0.0;
            for f in 0..d_rff {
                s += j_z[(f, dd)] * rff.alpha[f];
            }
            mu[offset + 1 + dd] = s;

            let mut v_d = Mat::from_fn(d_rff, 1, |r, _| j_z[(r, dd)]);
            rff.a_chol.solve_lower_triangular_in_place(&mut v_d);
            let mut vd_dot = 0.0;
            for r in 0..d_rff {
                vd_dot += v_d[(r, 0)] * v_d[(r, 0)];
            }
            var[offset + 1 + dd] = vd_dot;
        }
    }

    (mu, var)
}
