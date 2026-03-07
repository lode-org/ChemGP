//! Standard dimer + GP dimer on Muller-Brown near saddle S2 (Fig 5).
//!
//! Outputs JSONL convergence data: method, step, energy, force, oracle_calls.

use chemgp_core::dimer::{gp_dimer, standard_dimer, DimerConfig};
use chemgp_core::kernel::{CartesianSE, Kernel};
use chemgp_core::potentials::{muller_brown_energy_gradient, MULLER_BROWN_SADDLES};

use std::io::Write;

/// Compute finite-difference Hessian and return eigenvector of minimum eigenvalue.
fn softest_mode(oracle: &dyn Fn(&[f64]) -> (f64, Vec<f64>), x: &[f64], h: f64) -> Vec<f64> {
    let d = x.len();
    let (_, g0) = oracle(x);
    let mut hess = vec![vec![0.0; d]; d];
    for i in 0..d {
        let mut xp = x.to_vec();
        xp[i] += h;
        let (_, gp) = oracle(&xp);
        for j in 0..d {
            hess[i][j] = (gp[j] - g0[j]) / h;
        }
    }
    // Symmetrize
    for i in 0..d {
        for j in (i + 1)..d {
            let avg = 0.5 * (hess[i][j] + hess[j][i]);
            hess[i][j] = avg;
            hess[j][i] = avg;
        }
    }
    // 2x2 eigenvalue via quadratic formula
    let a = hess[0][0];
    let b = hess[0][1];
    let dd = hess[1][1];
    let trace = a + dd;
    let det = a * dd - b * b;
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
    let lam_min = 0.5 * (trace - disc);
    eprintln!("  Hessian eigenvalues: {:.2}, {:.2}", lam_min, 0.5 * (trace + disc));
    // Eigenvector for lam_min
    let vx = b;
    let vy = lam_min - a;
    let n = (vx * vx + vy * vy).sqrt();
    if n < 1e-12 {
        vec![1.0, 0.0]
    } else {
        vec![vx / n, vy / n]
    }
}

fn main() {
    let oracle = |x: &[f64]| -> (f64, Vec<f64>) { muller_brown_energy_gradient(x) };

    // Compute softest mode at saddle S2
    let s2 = MULLER_BROWN_SADDLES[1];
    eprintln!("Computing Hessian at S2 = ({:.4}, {:.4})...", s2[0], s2[1]);
    let orient_init = softest_mode(&oracle, &s2, 1e-5);
    eprintln!("  Softest mode: ({:.4}, {:.4})", orient_init[0], orient_init[1]);

    // Displace along softest mode toward minimum B
    let dist_sp = 0.1;
    let x_init: Vec<f64> = s2
        .iter()
        .zip(orient_init.iter())
        .map(|(s, o)| s + dist_sp * o)
        .collect();

    let kernel = Kernel::Cartesian(CartesianSE::new(100.0, 2.0));
    // MB surface scale: gradients ~10-100, distances ~0.1-1.0
    let dimer_sep = 0.005;
    let t_conv = 5.0;

    // Standard dimer
    let mut std_cfg = DimerConfig::default();
    std_cfg.t_force_true = t_conv;
    std_cfg.max_oracle_calls = 500;
    std_cfg.step_convex = 0.002;
    std_cfg.max_step = 0.05;
    std_cfg.verbose = false;

    eprintln!("Running standard dimer on MB near S2...");
    let std_result = standard_dimer(&oracle, &x_init, &orient_init, &std_cfg, dimer_sep);
    eprintln!(
        "  Std dimer: {} calls, |F| = {:.5}, converged = {}",
        std_result.oracle_calls,
        std_result.history.f_true.last().unwrap_or(&f64::NAN),
        std_result.converged
    );

    // GP dimer (exact GP for strongly curved 2D surface)
    let mut dimer_cfg = DimerConfig::default();
    dimer_cfg.max_outer_iter = 50;
    dimer_cfg.max_inner_iter = 3;
    dimer_cfg.t_force_true = t_conv;
    dimer_cfg.t_force_gp = 1.0;
    dimer_cfg.gp_train_iter = 100;
    dimer_cfg.fps_history = 30;
    dimer_cfg.n_initial_perturb = 3;
    dimer_cfg.perturb_scale = 0.03;
    dimer_cfg.max_step = 0.02;
    dimer_cfg.step_convex = 0.002;
    dimer_cfg.trust_radius = 0.2;
    dimer_cfg.rff_features = 0;
    dimer_cfg.max_rot_iter = 0;
    dimer_cfg.verbose = false;

    eprintln!("Running GP-dimer on MB near S2...");
    let dimer_result = gp_dimer(
        &oracle,
        &x_init,
        &orient_init,
        &kernel,
        &dimer_cfg,
        None,
        dimer_sep,
    );
    eprintln!(
        "  GP-dimer: {} calls, |F| = {:.5}, converged = {}",
        dimer_result.oracle_calls,
        dimer_result.history.f_true.last().unwrap_or(&f64::NAN),
        dimer_result.converged
    );

    // Write comparison data
    let outfile = "mb_dimer_comparison.jsonl";
    let mut f = std::fs::File::create(outfile).expect("Failed to create output file");

    for (i, ((&e, &fv), &oc)) in std_result
        .history
        .e_true
        .iter()
        .zip(std_result.history.f_true.iter())
        .zip(std_result.history.oracle_calls.iter())
        .enumerate()
    {
        writeln!(
            f,
            r#"{{"method":"standard_dimer","step":{},"energy":{},"force":{},"oracle_calls":{}}}"#,
            i, e, fv, oc
        )
        .expect("Operation failed");
    }

    for (i, ((&e, &fv), &oc)) in dimer_result
        .history
        .e_true
        .iter()
        .zip(dimer_result.history.f_true.iter())
        .zip(dimer_result.history.oracle_calls.iter())
        .enumerate()
    {
        writeln!(
            f,
            r#"{{"method":"gp_dimer","step":{},"energy":{},"force":{},"oracle_calls":{}}}"#,
            i, e, fv, oc
        )
        .expect("Operation failed");
    }

    writeln!(
        f,
        r#"{{"summary":true,"standard_calls":{},"dimer_calls":{}}}"#,
        std_result.oracle_calls, dimer_result.oracle_calls
    )
    .expect("Operation failed");

    eprintln!("Output: {}", outfile);
}
