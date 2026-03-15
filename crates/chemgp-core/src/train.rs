//! GP hyperparameter training via SCG MAP NLL.
//!
//! Ports `train_model!` from `functions.jl` (Kernel + fix_noise path).

use crate::nll::{nll_and_grad, NllData, NllNoise, NllPrior};
use crate::scg::{scg_optimize, ScgConfig};
use crate::types::GPModel;

/// Adaptive GP training iteration count.
///
/// First iteration uses full budget; subsequent iterations use 1/3 (min 50)
/// since hyperparameters are warm-started from the previous kernel.
pub fn adaptive_train_iters(base_iters: usize, is_first: bool) -> usize {
    if is_first {
        base_iters
    } else {
        (base_iters / 3).max(50)
    }
}

/// Prepare GP training targets: energy-shifted energies + gradient observations.
///
/// Returns `(y_sub, e_ref)` where `y_sub = [e_0 - e_ref, e_1 - e_ref, ..., g_0, g_1, ...]`
/// and `e_ref = td.energies[0]`.
pub fn prepare_training_targets(td: &crate::types::TrainingData) -> (Vec<f64>, f64) {
    let e_ref = td.energies[0];
    let mut y: Vec<f64> = td.energies.iter().map(|e| e - e_ref).collect();
    y.extend_from_slice(&td.gradients);
    (y, e_ref)
}

/// Train GP model hyperparameters using SCG with MAP NLL.
///
/// Optimizes signal_variance and inv_lengthscales in log-space,
/// keeping noise fixed. Matches C++ gpr_optim / MATLAB gpstuff.
pub fn train_model(model: &mut GPModel, iterations: usize, verbose: bool) {
    let n_ls = model.kernel.n_ls_params();
    let feat_map = model.kernel.feature_params_map().to_vec();

    // Pack to log-space
    let inv_ls = model.kernel.inv_lengthscales();
    let mut w0 = Vec::with_capacity(1 + n_ls);
    w0.push(model.kernel.signal_variance().max(1e-30).ln());
    for &l in &inv_ls {
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
    prior_var.push(1.0);
    for nfpp in n_feat_per_param.iter().take(n_ls) {
        let v = 0.5 * (*nfpp as f64 / 3.0).clamp(0.3, 1.0);
        prior_var.push(v);
    }

    let noise_e = model.noise_var;
    let noise_g = model.grad_noise_var;
    let jit = model.jitter;
    let x_data = model.x_data.clone();
    let dim = model.dim;
    let n = model.n_train;
    let y = model.y.clone();
    let template = model.kernel.clone();

    let const_s2 = model.const_sigma2;
    let p_dof = model.prior_dof;
    let p_s2 = model.prior_s2;
    let p_mu = model.prior_mu;
    let nll_data = NllData { x_data: &x_data, dim, n, y: &y, template: &template };
    let nll_noise = NllNoise { noise_e, noise_g, jitter: jit, const_sigma2: const_s2 };
    let nll_prior = NllPrior {
        w_prior: &w_prior, prior_var: &prior_var,
        prior_dof: p_dof, prior_s2: p_s2, prior_mu: p_mu,
    };
    let mut fg = |w: &[f64]| -> (f64, Vec<f64>) {
        nll_and_grad(w, &nll_data, &nll_noise, &nll_prior)
    };

    let config = ScgConfig {
        max_iter: iterations,
        tol_f: 1e-4,
        lambda_init: model.scg_lambda_init,
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
