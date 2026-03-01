//! GP prediction: mean and variance.
//!
//! Ports `predict` and `predict_with_variance` from `functions.jl`.
//! Also provides `PredModel` enum for unified exact-GP / RFF prediction.

use crate::covariance::{build_full_covariance, robust_cholesky};
use crate::kernel::Kernel;
use crate::rff::{build_rff, rff_predict, rff_predict_with_variance, RffModel};
use crate::types::{GPModel, TrainingData};
use faer::linalg::solvers::Solve;
use faer::Mat;

/// Prediction model: either exact GP or RFF approximation.
pub enum PredModel {
    Gp(GPModel),
    Rff(RffModel),
}

impl PredModel {
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        match self {
            PredModel::Gp(m) => predict(m, x, 1),
            PredModel::Rff(m) => rff_predict(m, x, 1),
        }
    }

    pub fn predict_with_variance(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        match self {
            PredModel::Gp(m) => predict_with_variance(m, x, 1),
            PredModel::Rff(m) => rff_predict_with_variance(m, x, 1),
        }
    }
}

/// Build a PredModel from trained kernel + full training data.
///
/// When `rff_features > 0`, builds an RFF approximation with the given
/// deterministic `seed` for reproducibility. Otherwise builds exact GP.
pub fn build_pred_model(
    kernel: &Kernel,
    td: &TrainingData,
    rff_features: usize,
    seed: u64,
) -> PredModel {
    let e_ref = td.energies[0];
    if rff_features > 0 {
        let mut y_rff: Vec<f64> = td.energies.iter().map(|e| e - e_ref).collect();
        y_rff.extend_from_slice(&td.gradients);
        let rff = build_rff(
            kernel,
            &td.data,
            td.dim,
            td.npoints(),
            &y_rff,
            rff_features,
            1e-6,
            1e-4,
            seed,
        );
        PredModel::Rff(rff)
    } else {
        let mut y_gp: Vec<f64> = td.energies.iter().map(|e| e - e_ref).collect();
        y_gp.extend_from_slice(&td.gradients);
        let gp_model = GPModel::new(kernel.clone(), td, y_gp, 1e-6, 1e-4, 1e-6);
        PredModel::Gp(gp_model)
    }
}

/// Convert a Vec<f64> to an (n, 1) column Mat.
fn vec_to_col(v: &[f64]) -> Mat<f64> {
    Mat::from_fn(v.len(), 1, |i, _| v[i])
}

/// Compute GP posterior mean at test points.
///
/// Returns predictions in interleaved layout: [E1, G1_1..G1_D, E2, ...].
pub fn predict(model: &GPModel, x_test: &[f64], n_test: usize) -> Vec<f64> {
    let d = model.dim;
    let n_train = model.n_train;

    let k_train = build_full_covariance(
        &model.kernel,
        &model.x_data,
        d,
        n_train,
        model.noise_var,
        model.grad_noise_var,
        model.jitter,
    );

    let llt = robust_cholesky(&k_train, 8).expect("Cholesky failed in predict");
    let y_col = vec_to_col(&model.y);
    let alpha = llt.solve(&y_col); // (train_len, 1)

    let dim_test_block = 1 + d;
    let n_out = n_test * dim_test_block;
    let train_len = model.y.len();
    let mut k_star = Mat::<f64>::zeros(n_out, train_len);

    for i in 0..n_test {
        let xt = &x_test[i * d..(i + 1) * d];
        let r_e = i * dim_test_block;

        for j in 0..n_train {
            let xtrain = model.train_col(j);
            let b = model.kernel.kernel_blocks(xt, xtrain);

            let c_e = j;
            let c_g_start = n_train + j * d;

            k_star[(r_e, c_e)] = b.k_ee;
            for dd in 0..d {
                k_star[(r_e, c_g_start + dd)] = b.k_ef[dd];
                k_star[(r_e + 1 + dd, c_e)] = b.k_fe[dd];
            }
            for di in 0..d {
                for dj in 0..d {
                    k_star[(r_e + 1 + di, c_g_start + dj)] = b.k_ff[(di, dj)];
                }
            }
        }
    }

    // result = k_star * alpha (manual matmul, alpha is (train_len, 1))
    let mut result = vec![0.0; n_out];
    for i in 0..n_out {
        let mut s = 0.0;
        for j in 0..train_len {
            s += k_star[(i, j)] * alpha[(j, 0)];
        }
        result[i] = s;
    }

    result
}

/// GP prediction with Bayesian variance.
///
/// Returns (mean, variance) in interleaved layout.
pub fn predict_with_variance(
    model: &GPModel,
    x_test: &[f64],
    n_test: usize,
) -> (Vec<f64>, Vec<f64>) {
    let d = model.dim;
    let n_train = model.n_train;

    let k_train = build_full_covariance(
        &model.kernel,
        &model.x_data,
        d,
        n_train,
        model.noise_var,
        model.grad_noise_var,
        model.jitter,
    );

    let llt = robust_cholesky(&k_train, 8).expect("Cholesky failed");
    let y_col = vec_to_col(&model.y);
    let alpha = llt.solve(&y_col);

    let dim_test_block = 1 + d;
    let n_out = n_test * dim_test_block;
    let train_len = model.y.len();
    let mut k_star = Mat::<f64>::zeros(n_out, train_len);

    for i in 0..n_test {
        let xt = &x_test[i * d..(i + 1) * d];
        let r_e = i * dim_test_block;

        for j in 0..n_train {
            let xtrain = model.train_col(j);
            let b = model.kernel.kernel_blocks(xt, xtrain);

            let c_e = j;
            let c_g_start = n_train + j * d;

            k_star[(r_e, c_e)] = b.k_ee;
            for dd in 0..d {
                k_star[(r_e, c_g_start + dd)] = b.k_ef[dd];
                k_star[(r_e + 1 + dd, c_e)] = b.k_fe[dd];
            }
            for di in 0..d {
                for dj in 0..d {
                    k_star[(r_e + 1 + di, c_g_start + dj)] = b.k_ff[(di, dj)];
                }
            }
        }
    }

    // mu = k_star * alpha
    let mut mu = vec![0.0; n_out];
    for i in 0..n_out {
        let mut s = 0.0;
        for j in 0..train_len {
            s += k_star[(i, j)] * alpha[(j, 0)];
        }
        mu[i] = s;
    }

    // V = L^{-1} K_*^T  (solve L * V = K_*^T)
    let l_ref = llt.L();
    let mut k_star_t = Mat::<f64>::zeros(train_len, n_out);
    for i in 0..n_out {
        for j in 0..train_len {
            k_star_t[(j, i)] = k_star[(i, j)];
        }
    }
    l_ref.solve_lower_triangular_in_place(&mut k_star_t);
    // Now k_star_t holds V = L^{-1} K_*^T

    let mut variance = vec![0.0; n_out];

    // Prior variance (diagonal of K_**)
    for i in 0..n_test {
        let xt = &x_test[i * d..(i + 1) * d];
        let b = model.kernel.kernel_blocks(xt, xt);
        let r_e = i * dim_test_block;
        variance[r_e] = b.k_ee;
        for dd in 0..d {
            variance[r_e + 1 + dd] = b.k_ff[(dd, dd)];
        }
    }

    // Subtract explained variance: var[idx] -= ||V[:, idx]||^2
    for idx in 0..n_out {
        let mut explained = 0.0;
        for r in 0..train_len {
            explained += k_star_t[(r, idx)] * k_star_t[(r, idx)];
        }
        variance[idx] = (variance[idx] - explained).max(0.0);
    }

    (mu, variance)
}
