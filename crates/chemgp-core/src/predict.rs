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

/// Cached exact GP model with pre-computed Cholesky and alpha.
///
/// Avoids recomputing the O(n^3) Cholesky decomposition on every prediction.
pub struct CachedGpModel {
    pub kernel: Kernel,
    pub x_data: Vec<f64>,
    pub dim: usize,
    pub n_train: usize,
    pub alpha: Mat<f64>,  // K^{-1} y, shape (train_len, 1)
    pub l_factor: Mat<f64>, // Lower Cholesky factor of K, shape (train_len, train_len)
    pub const_sigma2: f64,
}

impl CachedGpModel {
    /// Build a cached GP model from a GPModel by pre-computing Cholesky + alpha.
    pub fn from_gp(model: &GPModel) -> Self {
        let k_train = build_full_covariance(
            &model.kernel,
            &model.x_data,
            model.dim,
            model.n_train,
            model.noise_var,
            model.grad_noise_var,
            model.jitter,
            model.const_sigma2,
        );
        let llt = robust_cholesky(&k_train, 8).expect("Cholesky failed in CachedGpModel");
        let y_col = Mat::from_fn(model.y.len(), 1, |i, _| model.y[i]);
        let alpha = llt.solve(&y_col);

        let train_len = model.y.len();
        let l_ref = llt.L();
        let mut l_factor = Mat::<f64>::zeros(train_len, train_len);
        for i in 0..train_len {
            for j in 0..=i {
                l_factor[(i, j)] = l_ref[(i, j)];
            }
        }

        Self {
            kernel: model.kernel.clone(),
            x_data: model.x_data.clone(),
            dim: model.dim,
            n_train: model.n_train,
            alpha,
            l_factor,
            const_sigma2: model.const_sigma2,
        }
    }

    fn train_col(&self, i: usize) -> &[f64] {
        let start = i * self.dim;
        &self.x_data[start..start + self.dim]
    }
}

/// Prediction model: either cached exact GP or RFF approximation.
pub enum PredModel {
    Gp(CachedGpModel),
    Rff(RffModel),
}

impl PredModel {
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        match self {
            PredModel::Gp(m) => cached_predict(m, x, 1),
            PredModel::Rff(m) => rff_predict(m, x, 1),
        }
    }

    pub fn predict_with_variance(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        match self {
            PredModel::Gp(m) => cached_predict_with_variance(m, x, 1),
            PredModel::Rff(m) => rff_predict_with_variance(m, x, 1),
        }
    }
}

/// Build a PredModel from trained kernel + full training data.
///
/// When `rff_features > 0`, builds an RFF approximation with the given
/// deterministic `seed` for reproducibility. Otherwise builds exact GP
/// with cached Cholesky for fast repeated predictions.
pub fn build_pred_model(
    kernel: &Kernel,
    td: &TrainingData,
    rff_features: usize,
    seed: u64,
    const_sigma2: f64,
) -> PredModel {
    build_pred_model_full(kernel, td, rff_features, seed, const_sigma2, 1e-6, 1e-4, 1e-6)
}

/// Build a PredModel with explicit noise and jitter.
///
/// C++ gpr_optim defaults: noise_e=1e-7, noise_g=1e-5, jitter=0 for molecular systems.
pub fn build_pred_model_full(
    kernel: &Kernel,
    td: &TrainingData,
    rff_features: usize,
    seed: u64,
    const_sigma2: f64,
    noise_e: f64,
    noise_g: f64,
    jitter: f64,
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
            noise_e,
            noise_g,
            seed,
            const_sigma2,
        );
        PredModel::Rff(rff)
    } else {
        let mut y_gp: Vec<f64> = td.energies.iter().map(|e| e - e_ref).collect();
        y_gp.extend_from_slice(&td.gradients);
        let mut gp_model = GPModel::new(kernel.clone(), td, y_gp, noise_e, noise_g, jitter)
            .expect("GPModel::new failed: invalid training data or kernel params");
        gp_model.const_sigma2 = const_sigma2;
        let cached = CachedGpModel::from_gp(&gp_model);
        PredModel::Gp(cached)
    }
}

/// Build k_star row for one test point (shared between predict and predict_with_variance).
fn build_k_star_row(
    model: &CachedGpModel,
    xt: &[f64],
    row_offset: usize,
    k_star: &mut Mat<f64>,
) {
    let d = model.dim;
    let n_train = model.n_train;

    for j in 0..n_train {
        let xtrain = model.train_col(j);
        let b = model.kernel.kernel_blocks(xt, xtrain);

        let c_e = j;
        let c_g_start = n_train + j * d;

        k_star[(row_offset, c_e)] = b.k_ee + model.const_sigma2;
        for dd in 0..d {
            k_star[(row_offset, c_g_start + dd)] = b.k_ef[dd];
            k_star[(row_offset + 1 + dd, c_e)] = b.k_fe[dd];
        }
        for di in 0..d {
            for dj in 0..d {
                k_star[(row_offset + 1 + di, c_g_start + dj)] = b.k_ff[(di, dj)];
            }
        }
    }
}

/// Fast GP prediction using cached alpha (no Cholesky recomputation).
fn cached_predict(model: &CachedGpModel, x_test: &[f64], n_test: usize) -> Vec<f64> {
    let d = model.dim;
    let dim_test_block = 1 + d;
    let n_out = n_test * dim_test_block;
    let train_len = model.alpha.nrows();
    let mut k_star = Mat::<f64>::zeros(n_out, train_len);

    for i in 0..n_test {
        let xt = &x_test[i * d..(i + 1) * d];
        build_k_star_row(model, xt, i * dim_test_block, &mut k_star);
    }

    let mut result = vec![0.0; n_out];
    for i in 0..n_out {
        let mut s = 0.0;
        for j in 0..train_len {
            s += k_star[(i, j)] * model.alpha[(j, 0)];
        }
        result[i] = s;
    }

    result
}

/// Fast GP prediction with variance using cached L factor.
fn cached_predict_with_variance(
    model: &CachedGpModel,
    x_test: &[f64],
    n_test: usize,
) -> (Vec<f64>, Vec<f64>) {
    let d = model.dim;
    let dim_test_block = 1 + d;
    let n_out = n_test * dim_test_block;
    let train_len = model.alpha.nrows();
    let mut k_star = Mat::<f64>::zeros(n_out, train_len);

    for i in 0..n_test {
        let xt = &x_test[i * d..(i + 1) * d];
        build_k_star_row(model, xt, i * dim_test_block, &mut k_star);
    }

    // mu = k_star * alpha
    let mut mu = vec![0.0; n_out];
    for i in 0..n_out {
        let mut s = 0.0;
        for j in 0..train_len {
            s += k_star[(i, j)] * model.alpha[(j, 0)];
        }
        mu[i] = s;
    }

    // V = L^{-1} K_*^T
    let mut k_star_t = Mat::<f64>::zeros(train_len, n_out);
    for i in 0..n_out {
        for j in 0..train_len {
            k_star_t[(j, i)] = k_star[(i, j)];
        }
    }
    model.l_factor.solve_lower_triangular_in_place(&mut k_star_t);

    let mut variance = vec![0.0; n_out];

    // Prior variance (includes constant kernel for energy)
    for i in 0..n_test {
        let xt = &x_test[i * d..(i + 1) * d];
        let b = model.kernel.kernel_blocks(xt, xt);
        let r_e = i * dim_test_block;
        variance[r_e] = b.k_ee + model.const_sigma2;
        for dd in 0..d {
            variance[r_e + 1 + dd] = b.k_ff[(dd, dd)];
        }
    }

    // Subtract explained variance
    for idx in 0..n_out {
        let mut explained = 0.0;
        for r in 0..train_len {
            explained += k_star_t[(r, idx)] * k_star_t[(r, idx)];
        }
        variance[idx] = (variance[idx] - explained).max(0.0);
    }

    (mu, variance)
}

/// Convert a Vec<f64> to an (n, 1) column Mat.
fn vec_to_col(v: &[f64]) -> Mat<f64> {
    Mat::from_fn(v.len(), 1, |i, _| v[i])
}

/// Compute GP posterior mean at test points (uncached, used by train/test code).
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
        model.const_sigma2,
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

            k_star[(r_e, c_e)] = b.k_ee + model.const_sigma2;
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

/// GP prediction with Bayesian variance (uncached, used by train/test code).
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
        model.const_sigma2,
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

            k_star[(r_e, c_e)] = b.k_ee + model.const_sigma2;
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

    // Prior variance (diagonal of K_**, includes constant kernel for energy)
    for i in 0..n_test {
        let xt = &x_test[i * d..(i + 1) * d];
        let b = model.kernel.kernel_blocks(xt, xt);
        let r_e = i * dim_test_block;
        variance[r_e] = b.k_ee + model.const_sigma2;
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
