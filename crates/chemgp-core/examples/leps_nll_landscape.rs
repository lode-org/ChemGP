//! Visualize the NLL landscape on the LEPS surface.
//!
//! Tutorial T4 (Hyperparameter Training): evaluates the MAP-NLL on a grid
//! of (log_sigma2, log_theta) values to show the optimization landscape
//! that SCG navigates during hyperparameter training.
//!
//! Uses MolInvDistSE kernel with isotropic length scales (single theta).
//! Generates training data from 5 oracle calls near the LEPS reactant,
//! then sweeps a 2D grid of hyperparameter values.
//!
//! Outputs `leps_nll_landscape.jsonl` for contour plotting.

use chemgp_core::kernel::{Kernel, MolInvDistSE};
use chemgp_core::nll::nll_and_grad;
use chemgp_core::potentials::{leps_energy_gradient, LEPS_REACTANT};
use chemgp_core::types::TrainingData;

use std::io::Write;

fn main() {
    let oracle = |x: &[f64]| -> (f64, Vec<f64>) { leps_energy_gradient(x) };
    let dim = 9; // 3 atoms x 3D

    // Collect training data: reactant + small perturbations
    let mut td = TrainingData::new(dim);
    let x0 = LEPS_REACTANT.to_vec();
    let (e0, g0) = oracle(&x0);
    td.add_point(&x0, e0, &g0);

    let perturbations = [
        [0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.05, 0.0, 0.0],
    ];

    for pert in &perturbations {
        let x: Vec<f64> = x0.iter().zip(pert.iter()).map(|(a, b)| a + b).collect();
        let (e, g) = oracle(&x);
        td.add_point(&x, e, &g);
    }

    let n = td.npoints();
    let (y, _y_mean, _y_std) = td.normalize();

    // Flatten training data
    let x_data: Vec<f64> = (0..n).flat_map(|i| td.col(i).to_vec()).collect();

    // Template kernel (isotropic, 1 length scale parameter)
    let template = Kernel::MolInvDist(MolInvDistSE::isotropic(1.0, 1.0, vec![]));
    let n_params = 1 + template.n_ls_params(); // sigma2 + theta

    let noise_e = 0.001;
    let noise_g = 0.001;
    let jitter = 1e-8;
    let const_sigma2 = 0.0;

    // Grid sweep
    let n_grid = 40;
    let log_sigma2_range: Vec<f64> = (0..n_grid)
        .map(|i| -3.0 + 6.0 * i as f64 / (n_grid - 1) as f64)
        .collect();
    let log_theta_range: Vec<f64> = (0..n_grid)
        .map(|i| -2.0 + 5.0 * i as f64 / (n_grid - 1) as f64)
        .collect();

    // Prior centered at grid center
    let w_prior = vec![0.0; n_params];
    let prior_var = vec![2.0; n_params];

    let outfile = "leps_nll_landscape.jsonl";
    let mut f = std::fs::File::create(outfile).unwrap();

    let mut n_finite = 0;
    let mut n_inf = 0;

    for &ls2 in &log_sigma2_range {
        for &lt in &log_theta_range {
            // For isotropic MolInvDistSE, all theta params share the same value
            let mut w = vec![ls2];
            for _ in 0..template.n_ls_params() {
                w.push(lt);
            }

            let (nll, grad) = nll_and_grad(
                &w, &x_data, dim, n, &y, &template,
                noise_e, noise_g, jitter, &w_prior, &prior_var, const_sigma2,
                0.0, 1.0, 0.0,
            );

            if nll.is_finite() {
                let grad_norm: f64 = grad.iter().map(|v| v * v).sum::<f64>().sqrt();
                writeln!(f,
                    r#"{{"log_sigma2":{},"log_theta":{},"nll":{},"grad_norm":{}}}"#,
                    ls2, lt, nll, grad_norm
                ).unwrap();
                n_finite += 1;
            } else {
                n_inf += 1;
            }
        }
    }

    eprintln!("NLL landscape: {} finite points, {} infeasible (Cholesky failure or barrier)",
        n_finite, n_inf);
    eprintln!("Grid: log_sigma2 in [{:.1}, {:.1}], log_theta in [{:.1}, {:.1}]",
        log_sigma2_range[0], log_sigma2_range.last().unwrap(),
        log_theta_range[0], log_theta_range.last().unwrap());
    eprintln!("Training points: {}, dim: {}", n, dim);
    eprintln!("Output: {}", outfile);
}
