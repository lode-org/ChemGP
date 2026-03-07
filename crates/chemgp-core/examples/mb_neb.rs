//! Standard NEB on Muller-Brown 2D surface.
//!
//! Outputs JSONL: grid (200x200), stationary points, NEB path (Figs 1, 7).

use chemgp_core::neb::neb_optimize;
use chemgp_core::neb_path::NEBConfig;
use chemgp_core::potentials::{
    muller_brown_energy_gradient, MULLER_BROWN_MINIMA, MULLER_BROWN_SADDLES,
};

use std::io::Write;

fn main() {
    let oracle = |x: &[f64]| -> (f64, Vec<f64>) { muller_brown_energy_gradient(x) };

    // A -> B through saddle S2
    let x_start = MULLER_BROWN_MINIMA[0].to_vec();
    let x_end = MULLER_BROWN_MINIMA[1].to_vec();

    let mut cfg = NEBConfig::default();
    cfg.images = 11;
    cfg.max_iter = 500;
    cfg.conv_tol = 0.1;
    cfg.spring_constant = 10.0;
    cfg.climbing_image = true;
    cfg.verbose = false;

    eprintln!("Running standard NEB on Muller-Brown (A -> B, 11 images, CI)...");
    let result = neb_optimize(&oracle, &x_start, &x_end, &cfg);
    eprintln!(
        "  NEB: {} calls, max|F| = {:.5}, converged = {}",
        result.oracle_calls,
        result.history.max_force.last().unwrap_or(&f64::NAN),
        result.converged
    );

    let outfile = "mb_neb.jsonl";
    let mut f = std::fs::File::create(outfile).expect("Failed to create output file");

    // Grid metadata
    let nx = 200;
    let ny = 200;
    let x_min = -1.5f64;
    let x_max = 1.2;
    let y_min = -0.5f64;
    let y_max = 2.0;
    writeln!(
        f,
        r#"{{"type":"grid_meta","nx":{},"ny":{},"x_min":{},"x_max":{},"y_min":{},"y_max":{}}}"#,
        nx, ny, x_min, x_max, y_min, y_max
    )
    .expect("Operation failed");

    // Grid points
    for iy in 0..ny {
        let y = y_min + (y_max - y_min) * iy as f64 / (ny - 1) as f64;
        for ix in 0..nx {
            let x = x_min + (x_max - x_min) * ix as f64 / (nx - 1) as f64;
            let (e, _) = muller_brown_energy_gradient(&[x, y]);
            writeln!(
                f,
                r#"{{"type":"grid","ix":{},"iy":{},"x":{},"y":{},"energy":{}}}"#,
                ix, iy, x, y, e
            )
            .expect("Operation failed");
        }
    }

    // Stationary points
    for (i, m) in MULLER_BROWN_MINIMA.iter().enumerate() {
        let (e, _) = muller_brown_energy_gradient(m);
        writeln!(
            f,
            r#"{{"type":"minimum","id":{},"x":{},"y":{},"energy":{}}}"#,
            i, m[0], m[1], e
        )
        .expect("Operation failed");
    }
    for (i, s) in MULLER_BROWN_SADDLES.iter().enumerate() {
        let (e, _) = muller_brown_energy_gradient(s);
        writeln!(
            f,
            r#"{{"type":"saddle","id":{},"x":{},"y":{},"energy":{}}}"#,
            i, s[0], s[1], e
        )
        .expect("Operation failed");
    }

    // NEB path
    for (i, img) in result.path.images.iter().enumerate() {
        let (e, _) = muller_brown_energy_gradient(img);
        writeln!(
            f,
            r#"{{"type":"neb_path","image":{},"x":{},"y":{},"energy":{}}}"#,
            i, img[0], img[1], e
        )
        .expect("Operation failed");
    }

    // Endpoints
    writeln!(
        f,
        r#"{{"type":"endpoint","label":"A","x":{},"y":{}}}"#,
        x_start[0], x_start[1]
    )
    .expect("Operation failed");
    writeln!(
        f,
        r#"{{"type":"endpoint","label":"B","x":{},"y":{}}}"#,
        x_end[0], x_end[1]
    )
    .expect("Operation failed");

    // Climbing image index
    writeln!(
        f,
        r#"{{"type":"climbing_image","image":{}}}"#,
        result.max_energy_image
    )
    .expect("Operation failed");

    // Convergence history
    for (i, (&mf, &oc)) in result
        .history
        .max_force
        .iter()
        .zip(result.history.oracle_calls.iter())
        .enumerate()
    {
        writeln!(
            f,
            r#"{{"method":"neb","step":{},"max_force":{},"oracle_calls":{}}}"#,
            i, mf, oc
        )
        .expect("Operation failed");
    }

    eprintln!("Output: {}", outfile);
}
