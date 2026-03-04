//! GP predictions with 9 fixed hyperparameter combos on 1D MB slice (Fig 4).
//!
//! Outputs JSONL: training points and predictions for each (sigma_f, ell) pair.

use chemgp_core::kernel::{CartesianSE, Kernel};
use chemgp_core::potentials::muller_brown_energy_gradient;
use chemgp_core::predict::build_pred_model_full;
use chemgp_core::types::{GPModel, TrainingData};

use std::fs::File;
use std::io::{BufWriter, Write};

fn main() {
    let y_slice = 0.5;

    // Training data: 10 quasi-uniform points along x at y=0.5
    let train_x: Vec<f64> = vec![
        -1.3, -1.0, -0.7, -0.4, -0.1, 0.2, 0.4, 0.6, 0.8, 1.1,
    ];
    let mut td = TrainingData::new(2);
    for &x in &train_x {
        let (e, g) = muller_brown_energy_gradient(&[x, y_slice]);
        td.add_point(&[x, y_slice], e, &g);
    }

    // Prediction grid along x
    let n_pred = 200;
    let x_min = -1.5f64;
    let x_max = 1.2;
    let x_pred: Vec<f64> = (0..n_pred)
        .map(|i| x_min + (x_max - x_min) * i as f64 / (n_pred - 1) as f64)
        .collect();

    // 9 hyperparameter combos: sigma_f^2 x inv_ell
    // sigma_f^2 in {10, 100, 10000}, ell in {0.05, 0.3, 2.0} -> inv_ell in {20, 3.33, 0.5}
    let sigma_f2_vals = [10.0, 100.0, 10000.0];
    let ell_vals = [0.05, 0.3, 2.0];

    let outfile = "mb_hyperparams.jsonl";
    let file = File::create(outfile).expect("Cannot create output file");
    let mut w = BufWriter::new(file);

    // Write training points
    for &x in &train_x {
        let (e, _) = muller_brown_energy_gradient(&[x, y_slice]);
        writeln!(
            w,
            r#"{{"type":"train_point","x":{},"y":{},"energy":{}}}"#,
            x, y_slice, e
        )
        .unwrap();
    }

    // For each combo, build GP with fixed params and predict
    for &sigma_f2 in &sigma_f2_vals {
        for &ell in &ell_vals {
            let inv_ell = 1.0 / ell;
            let kernel = Kernel::Cartesian(CartesianSE::new(sigma_f2, inv_ell));

            // Build GP model with fixed hyperparameters (no SCG training)
            let (y, _mean, _std) = td.normalize();
            let gp = GPModel::new(kernel.clone(), &td, y, 1e-6, 1e-4, 1e-6);

            // Build prediction model (exact GP, no RFF)
            let pred = build_pred_model_full(
                &gp.kernel, &td, 0, 42, 0.0, gp.noise_var, gp.grad_noise_var, 1e-6,
            );

            for &x in &x_pred {
                let (vals, vars) = pred.predict_with_variance(&[x, y_slice]);
                // Un-shift prediction
                let gp_mean = vals[0] + td.energies[0];
                let gp_var = vars[0].max(0.0);
                let (true_e, _) = muller_brown_energy_gradient(&[x, y_slice]);
                writeln!(
                    w,
                    r#"{{"type":"prediction","sigma_f2":{},"ell":{},"x":{},"gp_mean":{},"gp_var":{},"true_e":{}}}"#,
                    sigma_f2, ell, x, gp_mean, gp_var, true_e
                )
                .unwrap();
            }

            eprintln!("  sigma_f2={}, ell={} done", sigma_f2, ell);
        }
    }

    eprintln!("Output: {}", outfile);
}
