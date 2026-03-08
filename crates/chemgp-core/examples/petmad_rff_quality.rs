//! RFF approximation quality on C2H4NO (system100) via PET-MAD RPC.
//!
//! Trains a GP on perturbed geometries around the minimized reactant,
//! then compares exact GP vs RFF predictions at held-out test points
//! for various D_rff values.
//!
//! Requires: pixi run -e rpc serve-petmad
//! Usage: cargo run --release --features io,rgpot --example petmad_rff_quality

use std::cell::RefCell;
use std::fs::File;
use std::io::{BufWriter, Write};

use chemgp_core::io::read_con;
use chemgp_core::kernel::{Kernel, MolInvDistSE};
use chemgp_core::oracle::RpcOracle;
use chemgp_core::predict::build_pred_model;
use chemgp_core::train::train_model;
use chemgp_core::types::{init_kernel, GPModel, TrainingData};

fn main() {
    let host = std::env::var("RGPOT_HOST").unwrap_or_else(|_| "localhost".into());
    let port: u16 = std::env::var("RGPOT_PORT")
        .unwrap_or_else(|_| "12345".into())
        .parse()
        .expect("RGPOT_PORT must be a valid port number");

    let frames = read_con("data/system100/reactant_minimized.con")
        .expect("Failed to read reactant");
    let reactant = &frames[0];
    let atomic_numbers = reactant.atomic_numbers.clone();
    let n_atoms = atomic_numbers.len();
    let ndim = 3 * n_atoms;
    let box_matrix = [
        reactant.cell[0][0], reactant.cell[0][1], reactant.cell[0][2],
        reactant.cell[1][0], reactant.cell[1][1], reactant.cell[1][2],
        reactant.cell[2][0], reactant.cell[2][1], reactant.cell[2][2],
    ];

    eprintln!("C2H4NO RFF quality (PET-MAD via RPC)");
    eprintln!("  atoms: {} ({:?})", n_atoms, atomic_numbers);
    eprintln!("  connecting to {}:{}", host, port);

    let rpc_oracle = RpcOracle::new(&host, port, atomic_numbers.clone(), box_matrix)
        .expect("Failed to connect to eOn serve");
    let oracle_cell = RefCell::new(rpc_oracle);
    let oracle = |x: &[f64]| -> (f64, Vec<f64>) {
        oracle_cell
            .borrow_mut()
            .evaluate(x)
            .unwrap_or_else(|e| panic!("RPC oracle failed: {}", e))
    };

    let base = reactant.positions.clone();

    // Generate training data: 30 random perturbations around equilibrium
    // Use deterministic perturbations (seeded pattern) for reproducibility
    let mut td = TrainingData::new(ndim);
    let n_train = 30;
    let n_test = 20;
    let pert_scale = 0.05; // 0.05 A perturbations

    eprintln!("Generating {} training + {} test points...", n_train, n_test);

    // Simple LCG for reproducible perturbations (seed=12345)
    let mut rng_state: u64 = 12345;
    let mut next_f64 = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng_state >> 33) as f64) / (1u64 << 31) as f64 * 2.0 - 1.0
    };

    // Training points
    let mut train_points = Vec::new();
    for _ in 0..n_train {
        let mut x = base.clone();
        for j in 0..ndim {
            x[j] += next_f64() * pert_scale;
        }
        let (e, g) = oracle(&x);
        let _ = td.add_point(&x, e, &g);
        train_points.push(x);
    }
    eprintln!("  {} training points evaluated", n_train);

    // Test points (different perturbations)
    let mut test_points = Vec::new();
    let mut true_energies: Vec<f64> = Vec::new();
    let mut true_gradients: Vec<Vec<f64>> = Vec::new();
    for _ in 0..n_test {
        let mut x = base.clone();
        for j in 0..ndim {
            x[j] += next_f64() * pert_scale;
        }
        let (e, g) = oracle(&x);
        true_energies.push(e);
        true_gradients.push(g);
        test_points.push(x);
    }
    eprintln!("  {} test points evaluated ({} total oracle calls)", n_test, n_train + n_test);

    // Train exact GP
    let kernel = Kernel::MolInvDist(MolInvDistSE::from_atomic_numbers(
        &atomic_numbers, vec![], &[], 1.0, 1.0,
    ));
    let kernel = init_kernel(&td, &kernel);
    let (y, _mean, _std) = td.normalize();
    let mut gp = GPModel::new(kernel.clone(), &td, y.clone(), 1e-7, 1e-7, 1e-7)
        .expect("Failed to create GP model");
    train_model(&mut gp, 100, false);

    // Exact GP predictions
    let exact_pred = build_pred_model(&gp.kernel, &td, 0, 42, 0.0);
    let mut exact_energies: Vec<f64> = Vec::new();
    let mut exact_gradients: Vec<Vec<f64>> = Vec::new();
    for x in &test_points {
        let pred = exact_pred.predict(x);
        let gp_e = pred[0] + td.energies[0];
        let gp_g: Vec<f64> = pred[1..].to_vec();
        exact_energies.push(gp_e);
        exact_gradients.push(gp_g);
    }

    let exact_e_mae: f64 = true_energies.iter().zip(exact_energies.iter())
        .map(|(t, p)| (t - p).abs()).sum::<f64>() / n_test as f64;
    let exact_g_mae: f64 = true_gradients.iter().zip(exact_gradients.iter())
        .map(|(tg, pg)| {
            tg.iter().zip(pg.iter()).map(|(a, b)| (a - b).abs()).sum::<f64>() / ndim as f64
        }).sum::<f64>() / n_test as f64;

    eprintln!("Exact GP: energy MAE = {:.6} eV, gradient MAE = {:.6} eV/A",
        exact_e_mae, exact_g_mae);

    // Write output
    let outfile = "petmad_rff_quality.jsonl";
    let file = File::create(outfile).expect("Cannot create output file");
    let mut w = BufWriter::new(file);

    writeln!(w, r#"{{"type":"exact_gp","system":"C2H4NO","n_atoms":{},"n_features":{},"n_train":{},"n_test":{},"energy_mae":{},"gradient_mae":{}}}"#,
        n_atoms, td.dim, n_train, n_test, exact_e_mae, exact_g_mae).expect("Failed to write to output file");

    // RFF at various D_rff
    for &d_rff in &[10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000] {
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
                    .map(|(a, b)| (a - b).abs()).sum::<f64>() / ndim as f64
            );
            rff_vs_gp_e_errors.push((exact_energies[i] - rff_e).abs());
            rff_vs_gp_g_errors.push(
                exact_gradients[i].iter().zip(rff_g.iter())
                    .map(|(a, b)| (a - b).abs()).sum::<f64>() / ndim as f64
            );
        }

        let e_mae: f64 = rff_e_errors.iter().sum::<f64>() / n_test as f64;
        let g_mae: f64 = rff_g_errors.iter().sum::<f64>() / n_test as f64;
        let vs_gp_e: f64 = rff_vs_gp_e_errors.iter().sum::<f64>() / n_test as f64;
        let vs_gp_g: f64 = rff_vs_gp_g_errors.iter().sum::<f64>() / n_test as f64;

        writeln!(w, r#"{{"type":"rff","system":"C2H4NO","d_rff":{},"energy_mae_vs_true":{},"gradient_mae_vs_true":{},"energy_mae_vs_gp":{},"gradient_mae_vs_gp":{}}}"#,
            d_rff, e_mae, g_mae, vs_gp_e, vs_gp_g).expect("Failed to write to output file");
        eprintln!("D_rff={:>4}: e_mae={:.6}, g_mae={:.6} (vs GP: e={:.6}, g={:.6})",
            d_rff, e_mae, g_mae, vs_gp_e, vs_gp_g);
    }

    eprintln!("Output: {}", outfile);
}
