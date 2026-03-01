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

/// Sample N training points near the stationary points + some random.
fn generate_training_data(n: usize) -> TrainingData {
    let mut td = TrainingData::new(2);

    // Place points near minima and saddles to get good coverage
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

fn main() {
    let outfile = "mb_gp_quality.jsonl";
    let file = File::create(outfile).expect("Cannot create output file");
    let mut w = BufWriter::new(file);

    // Grid parameters
    let nx = 100;
    let ny = 100;
    let x_min = -1.5f64;
    let x_max = 1.2;
    let y_min = -0.3f64;
    let y_max = 2.0;

    // Write grid metadata
    writeln!(w, r#"{{"type":"grid_meta","nx":{},"ny":{},"x_min":{},"x_max":{},"y_min":{},"y_max":{}}}"#,
        nx, ny, x_min, x_max, y_min, y_max).unwrap();

    // Write stationary points
    for (i, (mx, my)) in MINIMA.iter().enumerate() {
        let (e, _) = muller_brown_energy_gradient(&[*mx, *my]);
        writeln!(w, r#"{{"type":"minimum","id":{},"x":{},"y":{},"energy":{}}}"#, i, mx, my, e).unwrap();
    }
    for (i, (sx, sy)) in SADDLES.iter().enumerate() {
        let (e, _) = muller_brown_energy_gradient(&[*sx, *sy]);
        writeln!(w, r#"{{"type":"saddle","id":{},"x":{},"y":{},"energy":{}}}"#, i, sx, sy, e).unwrap();
    }

    // Generate GP quality data for multiple training set sizes
    for &n_train in &[5, 15, 21, 30] {
        println!("Training GP with {} points...", n_train);
        let td = generate_training_data(n_train);

        // Write training points
        for i in 0..td.npoints() {
            let col = td.col(i);
            writeln!(w, r#"{{"type":"train_point","n_train":{},"x":{},"y":{},"energy":{}}}"#,
                n_train, col[0], col[1], td.energies[i]).unwrap();
        }

        // Initialize and train GP
        let kernel = Kernel::Cartesian(CartesianSE::new(100.0, 2.0));
        let kernel = init_kernel(&td, &kernel);
        let (y, _mean, _std) = td.normalize();
        let mut gp = GPModel::new(kernel.clone(), &td, y, 1e-6, 1e-4, 1e-6);
        train_model(&mut gp, 100, false);

        // Build prediction model
        let pred = build_pred_model(&gp.kernel, &td, 0, 42);

        // Evaluate on grid
        for iy in 0..ny {
            let y_val = y_min + (y_max - y_min) * iy as f64 / (ny - 1) as f64;
            for ix in 0..nx {
                let x_val = x_min + (x_max - x_min) * ix as f64 / (nx - 1) as f64;
                let xy = [x_val, y_val];
                let (true_e, _) = muller_brown_energy_gradient(&xy);
                let (pred_vals, var_vals) = pred.predict_with_variance(&xy);
                let gp_e = pred_vals[0] + td.energies[0]; // un-shift
                let gp_var_e = var_vals[0];

                writeln!(w, r#"{{"type":"grid","n_train":{},"ix":{},"iy":{},"x":{},"y":{},"true_e":{},"gp_e":{},"gp_var":{}}}"#,
                    n_train, ix, iy, x_val, y_val, true_e, gp_e, gp_var_e).unwrap();
            }
        }

        println!("  Grid {}x{} done for N={}", nx, ny, n_train);
    }

    println!("Output: {}", outfile);
}
