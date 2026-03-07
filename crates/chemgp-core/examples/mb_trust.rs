//! GP on clustered training data, predictions along 1D MB slice (Fig 6).
//!
//! Shows trust region divergence: GP reliable near data, unreliable far away.

use chemgp_core::kernel::{CartesianSE, Kernel};
use chemgp_core::potentials::muller_brown_energy_gradient;
use chemgp_core::predict::build_pred_model;
use chemgp_core::train::train_model;
use chemgp_core::types::{init_kernel, GPModel, TrainingData};

use std::fs::File;
use std::io::{BufWriter, Write};

fn main() {
    let y_slice = 0.5;

    // 10 clustered training points near x in [-0.25, 0.35]
    let train_x: Vec<f64> = vec![
        -0.25, -0.15, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.28, 0.35,
    ];
    let mut td = TrainingData::new(2);
    for &x in &train_x {
        let (e, g) = muller_brown_energy_gradient(&[x, y_slice]);
        td.add_point(&[x, y_slice], e, &g);
    }

    // Train GP
    let kernel = Kernel::Cartesian(CartesianSE::new(100.0, 2.0));
    let kernel = init_kernel(&td, &kernel);
    let (y, _mean, _std) = td.normalize();
    let mut gp = GPModel::new(kernel, &td, y, 1e-6, 1e-4, 1e-6);
    train_model(&mut gp, 100, false);

    let pred = build_pred_model(&gp.kernel, &td, 0, 42, 0.0);

    // Predictions along 1D slice
    let n_pred = 400;
    let x_min = -1.5f64;
    let x_max = 1.2;

    let outfile = "mb_trust.jsonl";
    let file = File::create(outfile).expect("Cannot create output file");
    let mut w = BufWriter::new(file);

    // Trust metadata
    let trust_radius = 0.4;
    writeln!(w, r#"{{"type":"trust_meta","trust_radius":{}}}"#, trust_radius).expect("Failed to write to output file");

    // Training points
    for &x in &train_x {
        let (e, _) = muller_brown_energy_gradient(&[x, y_slice]);
        writeln!(
            w,
            r#"{{"type":"train_point","x":{},"energy":{}}}"#,
            x, e
        )
        .expect("Operation failed");
    }

    // Predictions
    for i in 0..n_pred {
        let x = x_min + (x_max - x_min) * i as f64 / (n_pred - 1) as f64;
        let (vals, vars) = pred.predict_with_variance(&[x, y_slice]);
        let gp_mean = vals[0] + td.energies[0];
        let gp_var = vars[0].max(0.0);
        let (true_e, _) = muller_brown_energy_gradient(&[x, y_slice]);
        writeln!(
            w,
            r#"{{"type":"prediction","x":{},"gp_mean":{},"gp_var":{},"true_e":{}}}"#,
            x, gp_mean, gp_var, true_e
        )
        .expect("Operation failed");
    }

    eprintln!("Output: {}", outfile);
}
