//! RFF approximation quality on LEPS: compare RFF vs exact GP vs true surface.
//!
//! Trains a GP on 20 LEPS oracle calls, then evaluates exact GP and RFF
//! predictions at 50 test points for various D_rff values.
//!
//! Usage: cargo run --release --example leps_rff_quality

use chemgp_core::kernel::{Kernel, MolInvDistSE};
use chemgp_core::potentials::leps_energy_gradient;
use chemgp_core::predict::build_pred_model;
use chemgp_core::train::train_model;
use chemgp_core::types::{init_kernel, GPModel, TrainingData};

use std::fs::File;
use std::io::{BufWriter, Write};

fn main() {
    let outfile = "leps_rff_quality.jsonl";
    let file = File::create(outfile).expect("Cannot create output file");
    let mut w = BufWriter::new(file);

    // LEPS equilibrium geometry: collinear H-H-H with r=0.7414
    let base = [0.0, 0.0, 0.0, 0.7414, 0.0, 0.0, 1.4828, 0.0, 0.0];

    // Generate training data: small perturbations around equilibrium
    let mut td = TrainingData::new(9);
    let perturbations: Vec<f64> = vec![
        0.0, 0.02, -0.02, 0.05, -0.05, 0.08, -0.08, 0.1, -0.1, 0.15,
        -0.15, 0.03, -0.03, 0.07, -0.07, 0.12, -0.12, 0.04, -0.06, 0.09,
    ];

    for (i, &p) in perturbations.iter().enumerate() {
        let mut x = base.to_vec();
        // Perturb the middle atom along x-axis (bond stretching coordinate)
        x[3] += p;
        // Also small perpendicular perturbation for some points
        if i % 3 == 0 {
            x[4] += p * 0.1;
        }
        let (e, g) = leps_energy_gradient(&x);
        td.add_point(&x, e, &g);
    }

    println!("Training data: {} points, dim={}", td.npoints(), td.dim);

    // Train exact GP
    let kernel = Kernel::MolInvDist(MolInvDistSE::isotropic(1.0, 1.0, vec![]));
    let kernel = init_kernel(&td, &kernel);
    let (y, _mean, _std) = td.normalize();
    let mut gp = GPModel::new(kernel.clone(), &td, y.clone(), 1e-6, 1e-4, 1e-6);
    train_model(&mut gp, 100, false);

    // Build exact GP prediction model
    let exact_pred = build_pred_model(&gp.kernel, &td, 0, 42, 0.0);

    // Generate test points (different from training)
    let test_perts: Vec<f64> = vec![
        0.01, -0.01, 0.035, -0.035, 0.06, -0.06, 0.085, -0.085,
        0.11, -0.11, 0.13, -0.13, 0.16, -0.16, 0.025, -0.025,
        0.045, -0.045, 0.065, -0.065, 0.095, -0.095, 0.105, -0.105,
        0.14, -0.14, 0.018, -0.018, 0.055, -0.055, 0.072, -0.072,
        0.088, -0.088, 0.115, -0.115, 0.125, -0.125, 0.145, -0.145,
        0.005, -0.005, 0.042, -0.042, 0.078, -0.078, 0.098, -0.098,
        0.135, -0.135,
    ];

    // Evaluate exact GP on test points
    let mut true_energies = Vec::new();
    let mut true_gradients = Vec::new();
    let mut exact_energies = Vec::new();
    let mut exact_gradients = Vec::new();
    let mut test_points = Vec::new();

    for &p in &test_perts {
        let mut x = base.to_vec();
        x[3] += p;
        let (e, g) = leps_energy_gradient(&x);
        let pred = exact_pred.predict(&x);
        let gp_e = pred[0] + td.energies[0];
        let gp_g: Vec<f64> = pred[1..].to_vec();

        true_energies.push(e);
        true_gradients.push(g);
        exact_energies.push(gp_e);
        exact_gradients.push(gp_g);
        test_points.push(x);
    }

    // Exact GP error
    let exact_e_mae: f64 = true_energies.iter().zip(exact_energies.iter())
        .map(|(t, p)| (t - p).abs()).sum::<f64>() / true_energies.len() as f64;
    let exact_g_mae: f64 = true_gradients.iter().zip(exact_gradients.iter())
        .map(|(tg, pg)| {
            tg.iter().zip(pg.iter()).map(|(a, b)| (a - b).abs()).sum::<f64>() / tg.len() as f64
        }).sum::<f64>() / true_gradients.len() as f64;

    writeln!(w, r#"{{"type":"exact_gp","energy_mae":{},"gradient_mae":{}}}"#,
        exact_e_mae, exact_g_mae).expect("Failed to write to output file");
    println!("Exact GP: energy MAE = {:.6}, gradient MAE = {:.6}", exact_e_mae, exact_g_mae);

    // Evaluate RFF at various D_rff
    for &d_rff in &[10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000] {
        let rff_pred = build_pred_model(&gp.kernel, &td, d_rff, 42, 0.0);

        let mut rff_e_errors = Vec::new();
        let mut rff_g_errors = Vec::new();
        let mut rff_vs_gp_e_errors = Vec::new();
        let mut rff_vs_gp_g_errors = Vec::new();

        for (i, x) in test_points.iter().enumerate() {
            let pred = rff_pred.predict(x);
            let rff_e = pred[0] + td.energies[0];
            let rff_g: Vec<f64> = pred[1..].to_vec();

            rff_e_errors.push((true_energies[i] - rff_e).abs());
            rff_g_errors.push(
                true_gradients[i].iter().zip(rff_g.iter())
                    .map(|(a, b)| (a - b).abs()).sum::<f64>() / 9.0
            );
            rff_vs_gp_e_errors.push((exact_energies[i] - rff_e).abs());
            rff_vs_gp_g_errors.push(
                exact_gradients[i].iter().zip(rff_g.iter())
                    .map(|(a, b)| (a - b).abs()).sum::<f64>() / 9.0
            );
        }

        let e_mae: f64 = rff_e_errors.iter().sum::<f64>() / rff_e_errors.len() as f64;
        let g_mae: f64 = rff_g_errors.iter().sum::<f64>() / rff_g_errors.len() as f64;
        let vs_gp_e: f64 = rff_vs_gp_e_errors.iter().sum::<f64>() / rff_vs_gp_e_errors.len() as f64;
        let vs_gp_g: f64 = rff_vs_gp_g_errors.iter().sum::<f64>() / rff_vs_gp_g_errors.len() as f64;

        writeln!(w, r#"{{"type":"rff","d_rff":{},"energy_mae_vs_true":{},"gradient_mae_vs_true":{},"energy_mae_vs_gp":{},"gradient_mae_vs_gp":{}}}"#,
            d_rff, e_mae, g_mae, vs_gp_e, vs_gp_g).expect("Failed to write to output file");
        println!("D_rff={:>3}: energy MAE = {:.6}, gradient MAE = {:.6}", d_rff, e_mae, g_mae);
    }

    println!("Output: {}", outfile);
}
