//! MAP Negative Log-Likelihood + analytical gradient.
//!
//! Ports `nll_and_grad` from `functions.jl`.

use crate::covariance::robust_cholesky;
use crate::kernel::Kernel;
use faer::linalg::solvers::{DenseSolveCore, Solve};
use faer::Mat;
use std::f64::consts::PI;

/// Lanczos approximation for ln(Gamma(x)), x > 0. Accurate to ~15 digits.
fn ln_gamma(x: f64) -> f64 {
    // Coefficients from Numerical Recipes (Lanczos, g=7, n=9).
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let x = x - 1.0;
    let mut sum = C[0];
    for i in 1..9 {
        sum += C[i] / (x + i as f64);
    }
    let t = x + 7.5;
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// Compute MAP NLL and gradient w.r.t. log-space hyperparameters.
///
/// w = [log(sigma2), log(theta_1), ..., log(theta_P)]
///
/// Uses `template` kernel to reconstruct the kernel with new hyperparameters.
/// Returns (nll, gradient). Returns (Inf, zeros) on Cholesky failure.
pub fn nll_and_grad(
    w: &[f64],
    x_data: &[f64],
    dim: usize,
    n: usize,
    y: &[f64],
    template: &Kernel,
    noise_e: f64,
    noise_g: f64,
    jitter: f64,
    w_prior: &[f64],
    prior_var: &[f64],
    const_sigma2: f64,
    prior_dof: f64,
    prior_s2: f64,
    prior_mu: f64,
) -> (f64, Vec<f64>) {
    let sigma2 = w[0].exp();
    let inv_ls: Vec<f64> = w[1..].iter().map(|v| v.exp()).collect();
    let n_params = w.len();

    let kern = template.with_params(sigma2, inv_ls);

    let total_dim = n * (1 + dim);
    let mut k_mat = Mat::<f64>::zeros(total_dim, total_dim);
    let mut dk: Vec<Mat<f64>> = (0..n_params)
        .map(|_| Mat::<f64>::zeros(total_dim, total_dim))
        .collect();

    for i in 0..n {
        let xi = &x_data[i * dim..(i + 1) * dim];
        let s_gi = n + i * dim;

        // Diagonal
        let bg = kern.kernel_blocks_and_hypergrads(xi, xi);
        let b = &bg.blocks;

        k_mat[(i, i)] = b.k_ee + const_sigma2 + noise_e + jitter;
        for d in 0..dim {
            k_mat[(i, s_gi + d)] = b.k_ef[d];
            k_mat[(s_gi + d, i)] = b.k_fe[d];
        }
        for di in 0..dim {
            for dj in 0..dim {
                k_mat[(s_gi + di, s_gi + dj)] = b.k_ff[(di, dj)];
            }
            k_mat[(s_gi + di, s_gi + di)] += noise_g + jitter;
        }

        for jp in 0..n_params {
            let gb = &bg.grad_blocks[jp];
            dk[jp][(i, i)] += gb.k_ee;
            for d in 0..dim {
                dk[jp][(i, s_gi + d)] += gb.k_ef[d];
                dk[jp][(s_gi + d, i)] += gb.k_fe[d];
            }
            for di in 0..dim {
                for dj in 0..dim {
                    dk[jp][(s_gi + di, s_gi + dj)] += gb.k_ff[(di, dj)];
                }
            }
        }

        // Off-diagonal
        for j in (i + 1)..n {
            let xj = &x_data[j * dim..(j + 1) * dim];
            let s_gj = n + j * dim;

            let bg = kern.kernel_blocks_and_hypergrads(xi, xj);
            let b = &bg.blocks;

            k_mat[(i, j)] = b.k_ee + const_sigma2;
            k_mat[(j, i)] = b.k_ee + const_sigma2;

            for d in 0..dim {
                k_mat[(i, s_gj + d)] = b.k_ef[d];
                k_mat[(s_gj + d, i)] = b.k_ef[d];
                k_mat[(s_gi + d, j)] = b.k_fe[d];
                k_mat[(j, s_gi + d)] = b.k_fe[d];
            }

            for di in 0..dim {
                for dj in 0..dim {
                    k_mat[(s_gi + di, s_gj + dj)] = b.k_ff[(di, dj)];
                    k_mat[(s_gj + dj, s_gi + di)] = b.k_ff[(di, dj)];
                }
            }

            for jp in 0..n_params {
                let gb = &bg.grad_blocks[jp];
                dk[jp][(i, j)] += gb.k_ee;
                dk[jp][(j, i)] += gb.k_ee;
                for d in 0..dim {
                    dk[jp][(i, s_gj + d)] += gb.k_ef[d];
                    dk[jp][(s_gj + d, i)] += gb.k_ef[d];
                    dk[jp][(s_gi + d, j)] += gb.k_fe[d];
                    dk[jp][(j, s_gi + d)] += gb.k_fe[d];
                }
                for di in 0..dim {
                    for dj in 0..dim {
                        dk[jp][(s_gi + di, s_gj + dj)] += gb.k_ff[(di, dj)];
                        dk[jp][(s_gj + dj, s_gi + di)] += gb.k_ff[(di, dj)];
                    }
                }
            }
        }
    }

    // Truncate near-zero
    let eps = f64::EPSILON;
    for r in 0..total_dim {
        for c in 0..total_dim {
            if k_mat[(r, c)].abs() < eps {
                k_mat[(r, c)] = 0.0;
            }
        }
    }

    // Cholesky factorization
    let llt = match robust_cholesky(&k_mat, 8) {
        Some(llt) => llt,
        None => return (f64::INFINITY, vec![0.0; n_params]),
    };

    let y_col = Mat::from_fn(total_dim, 1, |i, _| y[i]);
    let alpha = llt.solve(&y_col); // (total_dim, 1)

    // NLL = 0.5*y'*alpha + 0.5*logdet(K) + 0.5*n*log(2pi)
    let mut data_fit = 0.0;
    for i in 0..total_dim {
        data_fit += y[i] * alpha[(i, 0)];
    }
    data_fit *= 0.5;

    // logdet from L diagonal: logdet(K) = 2 * sum(log(diag(L)))
    let l_ref = llt.L();
    let mut log_det = 0.0;
    for i in 0..total_dim {
        log_det += l_ref[(i, i)].ln();
    }
    log_det *= 2.0;

    let complexity = 0.5 * log_det;
    let constant = 0.5 * total_dim as f64 * (2.0 * PI).ln();
    let mut nll = data_fit + complexity + constant;

    // MAP prior contribution: sigma2 (w[0]) and length scales (w[1..])
    if prior_dof > 0.0 {
        // Student-t prior on sqrt(sigma2), matching C++ PriorSqrtt + SexpatCF Jacobian.
        // log p = StudentT(sqrt(sigma2); mu, s2, nu) - log(2*sqrt(sigma2)) + log(sigma2)
        let sqrt_s2 = sigma2.sqrt();
        let nu = prior_dof;
        let diff = sqrt_s2 - prior_mu;
        let lp = ln_gamma((nu + 1.0) / 2.0) - ln_gamma(nu / 2.0)
            - 0.5 * (nu * PI * prior_s2).ln()
            - (nu + 1.0) / 2.0 * (1.0 + diff * diff / (nu * prior_s2)).ln()
            - (2.0 * sqrt_s2).ln()
            + w[0];
        nll -= lp;
    } else {
        // Gaussian prior on log(sigma2)
        nll += 0.5 * (w[0] - w_prior[0]).powi(2) / prior_var[0];
    }
    // Gaussian prior on length scale parameters (unchanged)
    for i in 1..n_params {
        nll += 0.5 * (w[i] - w_prior[i]).powi(2) / prior_var[i];
    }

    // magnSigma2 barrier: log-barrier penalty keeping sigma^2 bounded.
    // Matches gpr_optim C++ SCG barrier: max_log_magnSigma2 = log(2.0).
    // Barrier strength grows with dataset size to tighten as more data
    // constrains the model (initial_strength=1e-4, growth=1e-3 per point).
    let max_log_sigma2 = (2.0f64).ln();
    let barrier_strength = (1e-4 + 1e-3 * n as f64).min(0.5);
    let gap = max_log_sigma2 - w[0];
    if gap <= 0.0 {
        return (f64::INFINITY, vec![0.0; n_params]);
    }
    nll -= barrier_strength * gap.ln();

    // Gradient: W = K_inv - alpha*alpha', grad_j = 0.5 * tr(W * dK_j)
    let k_inv = llt.inverse();

    // W = K_inv - alpha * alpha^T  (element-wise)
    let mut w_mat = Mat::<f64>::zeros(total_dim, total_dim);
    for r in 0..total_dim {
        for c in 0..total_dim {
            w_mat[(r, c)] = k_inv[(r, c)] - alpha[(r, 0)] * alpha[(c, 0)];
        }
    }

    let mut grad = vec![0.0; n_params];
    for jp in 0..n_params {
        // tr(W * dK_j) = sum(W .* dK_j) element-wise (both symmetric)
        let mut trace = 0.0;
        for r in 0..total_dim {
            for c in 0..total_dim {
                trace += w_mat[(r, c)] * dk[jp][(r, c)];
            }
        }
        grad[jp] = 0.5 * trace;
        // MAP prior gradient for length scales (w[1..])
        if jp > 0 {
            grad[jp] += (w[jp] - w_prior[jp]) / prior_var[jp];
        }
    }

    // sigma2 prior gradient (w[0])
    if prior_dof > 0.0 {
        // Student-t gradient: d(-lp)/dw[0]
        let sqrt_s2 = sigma2.sqrt();
        let nu = prior_dof;
        let diff = sqrt_s2 - prior_mu;
        // d(sqrtt_log_prior)/d(sigma2)
        let dsqrtt = 0.5 / sqrt_s2
            * (-(nu + 1.0) * diff / (nu * prior_s2 + diff * diff))
            - 0.5 / sigma2;
        // Chain: d/dw[0] = dsqrtt * sigma2 + 1.0 (Jacobian)
        let dlp_dw0 = dsqrtt * sigma2 + 1.0;
        grad[0] -= dlp_dw0;
    } else {
        grad[0] += (w[0] - w_prior[0]) / prior_var[0];
    }

    // magnSigma2 barrier gradient: d/dw[0] of -strength * ln(gap) = +strength / gap
    grad[0] += barrier_strength / gap;

    (nll, grad)
}
