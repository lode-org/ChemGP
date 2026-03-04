//! GP quality visualization on Muller-Brown: grid predictions + variance.
//!
//! Outputs JSONL with:
//! - Grid predictions (true energy, GP mean, GP variance) at each (x,y)
//! - Training point locations
//! - Stationary points (minima, saddles)
//!
//! Usage: cargo run --release --example mb_gp_quality

use chemgp_core::kernel::{CartesianSE, Kernel};
use chemgp_core::potentials::muller_brown_energy_gradient;
use chemgp_core::predict::build_pred_model;
use chemgp_core::train::train_model;
use chemgp_core::types::{init_kernel, GPModel, TrainingData};

use std::fs::File;
use std::io::{BufWriter, Write};

/// Known MB stationary points.
const MINIMA: [(f64, f64); 3] = [
    (-0.558, 1.442),
    (0.623, 0.028),
    (-0.050, 0.467),
];
const SADDLES: [(f64, f64); 2] = [(-0.822, 0.624), (0.212, 0.293)];

/// Sample N training points near the stationary points (clustered).
fn generate_clustered(n: usize) -> TrainingData {
    let mut td = TrainingData::new(2);

    // Points near minima and saddles -- clustered around interesting features
    let seeds: Vec<(f64, f64)> = vec![
        MINIMA[0], MINIMA[1], MINIMA[2], SADDLES[0], SADDLES[1],
        (-1.0, 0.5), (0.0, 1.0), (-0.3, 0.8), (0.5, 0.3), (-0.7, 1.2),
        (-0.2, 1.5), (0.3, 0.15), (-0.9, 0.9), (-0.4, 0.5), (0.1, 0.7),
        (-0.6, 1.0), (-0.1, 0.3), (0.4, 0.1), (-0.5, 0.7), (-0.3, 1.3),
        (-0.8, 1.4), (0.2, 0.5), (-0.15, 1.1), (0.55, 0.15), (-0.45, 1.6),
        (-0.7, 0.3), (0.1, 0.1), (-0.65, 1.5), (-0.35, 0.4), (0.0, 0.6),
    ];

    for i in 0..n.min(seeds.len()) {
        let (x, y) = seeds[i];
        let xy = [x, y];
        let (e, g) = muller_brown_energy_gradient(&xy);
        td.add_point(&xy, e, &g);
    }
    td
}

/// Sample N training points scattered uniformly via simple LCG PRNG.
fn generate_scattered(n: usize, x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> TrainingData {
    let mut td = TrainingData::new(2);
    // Simple LCG for reproducible random points
    let mut state: u64 = 42;
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = (state >> 33) as f64 / (1u64 << 31) as f64;
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (state >> 33) as f64 / (1u64 << 31) as f64;
        let x = x_min + (x_max - x_min) * u1;
        let y = y_min + (y_max - y_min) * u2;
        let xy = [x, y];
        let (e, g) = muller_brown_energy_gradient(&xy);
        td.add_point(&xy, e, &g);
    }
    td
}

/// Grid parameters shared by both strategies.
const NX: usize = 100;
const NY: usize = 100;
const X_MIN: f64 = -1.5;
const X_MAX: f64 = 1.2;
const Y_MIN: f64 = -0.3;
const Y_MAX: f64 = 2.0;

/// Write header (grid_meta + stationary points) to a JSONL writer.
fn write_header(w: &mut BufWriter<File>) {
    writeln!(w, r#"{{"type":"grid_meta","nx":{},"ny":{},"x_min":{},"x_max":{},"y_min":{},"y_max":{}}}"#,
        NX, NY, X_MIN, X_MAX, Y_MIN, Y_MAX).unwrap();

    for (i, (mx, my)) in MINIMA.iter().enumerate() {
        let (e, _) = muller_brown_energy_gradient(&[*mx, *my]);
        writeln!(w, r#"{{"type":"minimum","id":{},"x":{},"y":{},"energy":{}}}"#, i, mx, my, e).unwrap();
    }
    for (i, (sx, sy)) in SADDLES.iter().enumerate() {
        let (e, _) = muller_brown_energy_gradient(&[*sx, *sy]);
        writeln!(w, r#"{{"type":"saddle","id":{},"x":{},"y":{},"energy":{}}}"#, i, sx, sy, e).unwrap();
    }
}

/// Train GP on `td` and write grid predictions to JSONL.
fn evaluate_gp(w: &mut BufWriter<File>, td: &TrainingData, n_train: usize) {
    // Write training points
    for i in 0..td.npoints() {
        let col = td.col(i);
        writeln!(w, r#"{{"type":"train_point","n_train":{},"x":{},"y":{},"energy":{}}}"#,
            n_train, col[0], col[1], td.energies[i]).unwrap();
    }

    // Initialize and train GP
    let kernel = Kernel::Cartesian(CartesianSE::new(100.0, 2.0));
    let kernel = init_kernel(td, &kernel);
    let (y, _mean, _std) = td.normalize();
    let mut gp = GPModel::new(kernel.clone(), td, y, 1e-6, 1e-4, 1e-6);
    train_model(&mut gp, 100, false);

    // Build prediction model
    let pred = build_pred_model(&gp.kernel, td, 0, 42, 0.0);

    // Evaluate on grid
    for iy in 0..NY {
        let y_val = Y_MIN + (Y_MAX - Y_MIN) * iy as f64 / (NY - 1) as f64;
        for ix in 0..NX {
            let x_val = X_MIN + (X_MAX - X_MIN) * ix as f64 / (NX - 1) as f64;
            let xy = [x_val, y_val];
            let (true_e, _) = muller_brown_energy_gradient(&xy);
            let (pred_vals, var_vals) = pred.predict_with_variance(&xy);
            let gp_e = pred_vals[0] + td.energies[0]; // un-shift
            let gp_var_e = var_vals[0];

            writeln!(w, r#"{{"type":"grid","n_train":{},"ix":{},"iy":{},"x":{},"y":{},"true_e":{},"gp_e":{},"gp_var":{}}}"#,
                n_train, ix, iy, x_val, y_val, true_e, gp_e, gp_var_e).unwrap();
        }
    }
}

fn run_strategy<F>(outfile: &str, strategy_name: &str, gen: F)
where
    F: Fn(usize) -> TrainingData,
{
    let file = File::create(outfile).expect("Cannot create output file");
    let mut w = BufWriter::new(file);
    write_header(&mut w);

    for &n_train in &[3, 8, 15, 30] {
        println!("  {} N={}...", strategy_name, n_train);
        let td = gen(n_train);
        evaluate_gp(&mut w, &td, n_train);
    }
    println!("  -> {}", outfile);
}

fn main() {
    println!("Clustered training data:");
    run_strategy("mb_gp_quality.jsonl", "clustered", generate_clustered);

    println!("Scattered training data:");
    run_strategy("mb_gp_scattered.jsonl", "scattered", |n| {
        generate_scattered(n, X_MIN, X_MAX, Y_MIN, Y_MAX)
    });
}
