//! Covariance matrix assembly for GP with derivative observations.
//!
//! Ports `build_full_covariance` from `functions.jl`.

use crate::kernel::Kernel;
use faer::linalg::solvers::Llt;
use faer::{Mat, Side};

/// Assemble the full N*(1+D) x N*(1+D) covariance matrix.
///
/// Block structure:
///   [ K_EE + noise_e*I    K_EG          ]
///   [ K_GE                K_GG + noise_g*I ]
pub fn build_full_covariance(
    kernel: &Kernel,
    x_data: &[f64],
    dim: usize,
    n: usize,
    noise_e: f64,
    noise_g: f64,
    jitter: f64,
    const_sigma2: f64,
) -> Mat<f64> {
    let total = n * (1 + dim);
    let mut k_mat = Mat::<f64>::zeros(total, total);

    for i in 0..n {
        let xi = &x_data[i * dim..(i + 1) * dim];

        // Diagonal blocks
        let b = kernel.kernel_blocks(xi, xi);

        // Energy index
        k_mat[(i, i)] = b.k_ee + const_sigma2 + noise_e + jitter;

        // Gradient indices
        let s_g = n + i * dim;
        for d in 0..dim {
            k_mat[(i, s_g + d)] = b.k_ef[d];
            k_mat[(s_g + d, i)] = b.k_fe[d];
        }
        for di in 0..dim {
            for dj in 0..dim {
                k_mat[(s_g + di, s_g + dj)] = b.k_ff[(di, dj)];
            }
            k_mat[(s_g + di, s_g + di)] += noise_g + jitter;
        }

        // Off-diagonal interactions
        for j in (i + 1)..n {
            let xj = &x_data[j * dim..(j + 1) * dim];
            let b = kernel.kernel_blocks(xi, xj);

            let j_s = n + j * dim;

            k_mat[(i, j)] = b.k_ee + const_sigma2;
            k_mat[(j, i)] = b.k_ee + const_sigma2;

            for d in 0..dim {
                k_mat[(i, j_s + d)] = b.k_ef[d];
                k_mat[(j_s + d, i)] = b.k_ef[d];
                k_mat[(s_g + d, j)] = b.k_fe[d];
                k_mat[(j, s_g + d)] = b.k_fe[d];
            }

            for di in 0..dim {
                for dj in 0..dim {
                    k_mat[(s_g + di, j_s + dj)] = b.k_ff[(di, dj)];
                    k_mat[(j_s + dj, s_g + di)] = b.k_ff[(di, dj)];
                }
            }
        }
    }

    // Truncate near-zero entries (matching MATLAB GPstuff: C(C<eps)=0)
    let eps = f64::EPSILON;
    for r in 0..total {
        for c in 0..total {
            if k_mat[(r, c)].abs() < eps {
                k_mat[(r, c)] = 0.0;
            }
        }
    }

    k_mat
}

/// Robust Cholesky with adaptive jitter.
/// Returns an Llt decomposition such that K ~ L * L^T.
pub fn robust_cholesky(k: &Mat<f64>, max_attempts: usize) -> Option<Llt<f64>> {
    // Try without jitter first
    if let Ok(llt) = k.llt(Side::Lower) {
        return Some(llt);
    }

    let n = k.nrows();
    let mut max_diag = 0.0f64;
    for i in 0..n {
        max_diag = max_diag.max(k[(i, i)]);
    }
    let scale = max_diag.max(1.0);
    let mut jitter = scale * 1e-8;

    for _ in 1..max_attempts {
        let mut jittered = k.clone();
        for i in 0..n {
            jittered[(i, i)] += jitter;
        }
        if let Ok(llt) = jittered.llt(Side::Lower) {
            return Some(llt);
        }
        jitter *= 10.0;
    }

    None
}
