//! Visualize the NLL landscape on the LEPS surface.
//!
//! Tutorial T4 (Hyperparameter Training): evaluates the MAP-NLL on a grid
//! of (log_sigma2, log_theta) values to show the optimization landscape
//! that SCG navigates during hyperparameter training.
//!
//! Uses MolInvDistSE kernel with isotropic length scales (single theta).
//! Generates training data from 15 oracle calls along the LEPS reaction
//! coordinate (reactant to product), then sweeps a 2D grid of
//! hyperparameter values. SCG-trained optimum is marked for reference.
//!
//! Outputs `leps_nll_landscape.jsonl` for contour plotting.

use chemgp_core::kernel::{Kernel, MolInvDistSE};
use chemgp_core::nll::{nll_and_grad, NllData, NllNoise, NllPrior};
use chemgp_core::potentials::{leps_energy_gradient, LEPS_REACTANT, LEPS_PRODUCT};
use chemgp_core::train::train_model;
use chemgp_core::types::{init_kernel, GPModel, TrainingData};

use std::io::Write;

fn main() {
    let oracle = |x: &[f64]| -> (f64, Vec<f64>) { leps_energy_gradient(x) };
    let dim = 9; // 3 atoms x 3D

    // Collect training data: 15 points along reaction coordinate + off-path
    let mut td = TrainingData::new(dim);
    let x0 = LEPS_REACTANT.to_vec();
    let x1 = LEPS_PRODUCT.to_vec();

    // Interpolate along the reaction path
    for i in 0..10 {
        let t = i as f64 / 9.0;
        let x: Vec<f64> = x0.iter().zip(x1.iter()).map(|(a, b)| a + t * (b - a)).collect();
        let (e, g) = oracle(&x);
        let _ = td.add_point(&x, e, &g);
    }

    // Off-path perturbations near transition state region
    let x_mid: Vec<f64> = x0.iter().zip(x1.iter()).map(|(a, b)| 0.5 * (a + b)).collect();
    let perp_offsets: &[f64] = &[0.05, -0.05, 0.08, -0.08, 0.03];
    for (idx, &off) in perp_offsets.iter().enumerate() {
        let mut x = x_mid.clone();
        // Perturb different coordinates for diversity
        x[3 + (idx % 3)] += off;
        let (e, g) = oracle(&x);
        let _ = td.add_point(&x, e, &g);
    }

    let n = td.npoints();
    eprintln!("Training points: {}, dim: {}", n, dim);

    // Train GP with SCG to find MAP optimum
    let kernel = Kernel::MolInvDist(MolInvDistSE::isotropic(1.0, 1.0, vec![]));
    let kernel = init_kernel(&td, &kernel);
    let (y, _y_mean, _y_std) = td.normalize();
    let mut gp = GPModel::new(kernel.clone(), &td, y.clone(), 0.001, 0.001, 1e-8)
        .expect("Failed to create GP model");
    train_model(&mut gp, 200, false);

    // Extract trained hyperparameters and convert to log-space for plotting
    let (sv, ils) = match &gp.kernel {
        Kernel::MolInvDist(k) => (k.signal_variance, k.inv_lengthscales[0]),
        Kernel::Cartesian(k) => (k.signal_variance, k.inv_lengthscale),
    };
    let opt_ls2 = sv.ln();  // log(sigma^2)
    let opt_lt = ils.ln();  // log(theta) (isotropic: all same)
    eprintln!("SCG MAP optimum: log_sigma2 = {:.3}, log_theta = {:.3}", opt_ls2, opt_lt);

    // Flatten training data for NLL evaluation
    let x_data: Vec<f64> = (0..n).flat_map(|i| td.col(i).to_vec()).collect();

    // Template kernel (isotropic, 1 length scale parameter)
    let template = Kernel::MolInvDist(MolInvDistSE::isotropic(1.0, 1.0, vec![]));
    let n_params = 1 + template.n_ls_params(); // sigma2 + theta

    let noise_e = 0.001;
    let noise_g = 0.001;
    let jitter = 1e-8;
    let const_sigma2 = 0.0;

    // Grid sweep centered on SCG optimum (+/- 3 in each direction)
    let n_grid = 50;
    let ls2_lo = (opt_ls2 - 3.0).max(-4.0);
    let ls2_hi = (opt_ls2 + 3.0).min(4.0);
    let lt_lo = (opt_lt - 3.0).max(-3.0);
    let lt_hi = (opt_lt + 3.0).min(5.0);

    let log_sigma2_range: Vec<f64> = (0..n_grid)
        .map(|i| ls2_lo + (ls2_hi - ls2_lo) * i as f64 / (n_grid - 1) as f64)
        .collect();
    let log_theta_range: Vec<f64> = (0..n_grid)
        .map(|i| lt_lo + (lt_hi - lt_lo) * i as f64 / (n_grid - 1) as f64)
        .collect();

    // Prior centered at init_kernel defaults
    let w_prior = vec![0.0; n_params];
    let prior_var = vec![2.0; n_params];

    let outfile = "leps_nll_landscape.jsonl";
    let mut f = std::fs::File::create(outfile).expect("Failed to create output file");

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
                &w,
                &NllData { x_data: &x_data, dim, n, y: &y, template: &template },
                &NllNoise { noise_e, noise_g, jitter, const_sigma2 },
                &NllPrior { w_prior: &w_prior, prior_var: &prior_var, prior_dof: 0.0, prior_s2: 1.0, prior_mu: 0.0 },
            );

            if nll.is_finite() {
                let grad_norm: f64 = grad.iter().map(|v| v * v).sum::<f64>().sqrt();
                writeln!(f,
                    r#"{{"log_sigma2":{},"log_theta":{},"nll":{},"grad_norm":{}}}"#,
                    ls2, lt, nll, grad_norm
                ).expect("Failed to write to output file");
                n_finite += 1;
            } else {
                n_inf += 1;
            }
        }
    }

    // Write SCG optimum as a separate record for the plotter
    writeln!(f,
        r#"{{"type":"scg_optimum","log_sigma2":{},"log_theta":{}}}"#,
        opt_ls2, opt_lt
    ).expect("Failed to write optimum");

    eprintln!("NLL landscape: {} finite points, {} infeasible (Cholesky failure or barrier)",
        n_finite, n_inf);
    eprintln!("Grid: log_sigma2 in [{:.1}, {:.1}], log_theta in [{:.1}, {:.1}]",
        log_sigma2_range[0], log_sigma2_range.last().unwrap(),
        log_theta_range[0], log_theta_range.last().unwrap());
    eprintln!("Output: {}", outfile);
}
