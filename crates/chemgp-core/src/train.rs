//! GP hyperparameter training via SCG MAP NLL.
//!
//! Ports `train_model!` from `functions.jl` (MolInvDistSE + fix_noise path).

use crate::nll::nll_and_grad;
use crate::scg::{scg_optimize, ScgConfig};
use crate::types::GPModel;

/// Train GP model hyperparameters using SCG with MAP NLL.
///
/// Optimizes signal_variance and inv_lengthscales in log-space,
/// keeping noise fixed. Matches C++ gpr_optim / MATLAB gpstuff.
pub fn train_model(model: &mut GPModel, iterations: usize, verbose: bool) {
    let frozen = model.kernel.frozen_coords.clone();
    let feat_map = model.kernel.feature_params_map.clone();
    let n_ls = model.kernel.inv_lengthscales.len();

    // Pack to log-space
    let mut w0 = Vec::with_capacity(1 + n_ls);
    w0.push(model.kernel.signal_variance.max(1e-30).ln());
    for &l in &model.kernel.inv_lengthscales {
        w0.push(l.max(1e-30).ln());
    }
    let w_prior = w0.clone();

    // Adaptive MAP prior: sigma2 gets s2=2.0, lengthscales scaled by features per param
    let mut n_feat_per_param = vec![0usize; n_ls];
    if !feat_map.is_empty() {
        for &p in &feat_map {
            n_feat_per_param[p] += 1;
        }
    } else {
        n_feat_per_param.fill(n_ls.max(1));
    }

    let mut prior_var = Vec::with_capacity(1 + n_ls);
    prior_var.push(2.0);
    for p in 0..n_ls {
        let v = 0.5 * (n_feat_per_param[p] as f64 / 3.0).clamp(0.3, 1.0);
        prior_var.push(v);
    }

    let noise_e = model.noise_var;
    let noise_g = model.grad_noise_var;
    let jit = model.jitter;
    let x_data = model.x_data.clone();
    let dim = model.dim;
    let n = model.n_train;
    let y = model.y.clone();

    let mut fg = |w: &[f64]| -> (f64, Vec<f64>) {
        nll_and_grad(
            w, &x_data, dim, n, &y, &frozen, &feat_map, noise_e, noise_g, jit, &w_prior,
            &prior_var,
        )
    };

    let config = ScgConfig {
        max_iter: iterations,
        tol_f: 1e-4,
        verbose,
        ..Default::default()
    };

    let result = scg_optimize(&mut fg, &w0, &config);

    if result.converged || result.f_best < f64::INFINITY {
        let sigma2_opt = result.w_best[0].exp();
        let inv_ls_opt: Vec<f64> = result.w_best[1..].iter().map(|v| v.exp()).collect();
        model.kernel = model.kernel.with_params(sigma2_opt, inv_ls_opt);
        if verbose {
            eprintln!("SCG Training Complete. Final MAP NLL: {:.4}", result.f_best);
        }
    } else if verbose {
        eprintln!("SCG did not converge.");
    }
}
