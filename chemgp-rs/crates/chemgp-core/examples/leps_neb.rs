//! NEB vs GP-NEB AIE vs GP-NEB OIE on LEPS surface.
//!
//! Outputs JSONL data showing oracle call efficiency:
//! GP OIE > GP AIE > standard NEB.

use chemgp_core::kernel::MolInvDistSE;
use chemgp_core::neb::gp_neb_aie;
use chemgp_core::neb::neb_optimize;
use chemgp_core::neb_oie::gp_neb_oie;
use chemgp_core::neb_path::NEBConfig;
use chemgp_core::potentials::{leps_energy_gradient, LEPS_PRODUCT, LEPS_REACTANT};

use std::io::Write;

fn main() {
    let oracle = |x: &[f64]| -> (f64, Vec<f64>) { leps_energy_gradient(x) };
    let x_start = LEPS_REACTANT.to_vec();
    let x_end = LEPS_PRODUCT.to_vec();

    let kernel = MolInvDistSE::isotropic(1.0, 1.0, vec![]);

    // Standard NEB
    let mut neb_cfg = NEBConfig::default();
    neb_cfg.images = 5;
    neb_cfg.max_iter = 200;
    neb_cfg.conv_tol = 0.1;
    neb_cfg.climbing_image = false;
    neb_cfg.verbose = false;

    eprintln!("Running standard NEB...");
    let neb_result = neb_optimize(&oracle, &x_start, &x_end, &neb_cfg);
    eprintln!("  NEB: {} calls, max|F| = {:.5}, converged = {}",
        neb_result.oracle_calls,
        neb_result.history.max_force.last().unwrap_or(&f64::NAN),
        neb_result.converged);

    // GP-NEB AIE (per-bead subset + RFF for fast inner relax)
    let mut aie_cfg = NEBConfig::default();
    aie_cfg.images = 5;
    aie_cfg.max_outer_iter = 40;
    aie_cfg.max_iter = 100;
    aie_cfg.conv_tol = 0.1;
    aie_cfg.climbing_image = false;
    aie_cfg.gp_train_iter = 100;
    aie_cfg.max_gp_points = 50;
    aie_cfg.rff_features = 500;
    aie_cfg.verbose = false;

    eprintln!("Running GP-NEB AIE...");
    let aie_result = gp_neb_aie(&oracle, &x_start, &x_end, &kernel, &aie_cfg);
    eprintln!("  AIE: {} calls, max|F| = {:.5}, converged = {}",
        aie_result.oracle_calls,
        aie_result.history.max_force.last().unwrap_or(&f64::NAN),
        aie_result.converged);

    // GP-NEB OIE (one oracle call per outer iteration + RFF)
    let mut oie_cfg = NEBConfig::default();
    oie_cfg.images = 5;
    oie_cfg.max_outer_iter = 60;
    oie_cfg.max_iter = 100;
    oie_cfg.conv_tol = 0.1;
    oie_cfg.climbing_image = false;
    oie_cfg.gp_train_iter = 100;
    oie_cfg.max_gp_points = 50;
    oie_cfg.rff_features = 500;
    oie_cfg.verbose = false;

    eprintln!("Running GP-NEB OIE...");
    let oie_result = gp_neb_oie(&oracle, &x_start, &x_end, &kernel, &oie_cfg);
    eprintln!("  OIE: {} calls, max|F| = {:.5}, converged = {}",
        oie_result.oracle_calls,
        oie_result.history.max_force.last().unwrap_or(&f64::NAN),
        oie_result.converged);

    // Write comparison data
    let outfile = "leps_neb_comparison.jsonl";
    let mut f = std::fs::File::create(outfile).unwrap();

    // NEB convergence history
    for (i, (&mf, &oc)) in neb_result.history.max_force.iter()
        .zip(neb_result.history.oracle_calls.iter()).enumerate()
    {
        writeln!(f, r#"{{"method":"neb","step":{},"max_force":{},"oracle_calls":{}}}"#,
            i, mf, oc).unwrap();
    }

    // AIE convergence history
    for (i, (&mf, &oc)) in aie_result.history.max_force.iter()
        .zip(aie_result.history.oracle_calls.iter()).enumerate()
    {
        writeln!(f, r#"{{"method":"gp_neb_aie","step":{},"max_force":{},"oracle_calls":{}}}"#,
            i, mf, oc).unwrap();
    }

    // OIE convergence history
    for (i, (&mf, &oc)) in oie_result.history.max_force.iter()
        .zip(oie_result.history.oracle_calls.iter()).enumerate()
    {
        writeln!(f, r#"{{"method":"gp_neb_oie","step":{},"max_force":{},"oracle_calls":{}}}"#,
            i, mf, oc).unwrap();
    }

    // Summary
    writeln!(f, r#"{{"summary":true,"neb_calls":{},"aie_calls":{},"oie_calls":{}}}"#,
        neb_result.oracle_calls, aie_result.oracle_calls, oie_result.oracle_calls).unwrap();

    eprintln!("\nSummary: NEB={} calls, AIE={} calls, OIE={} calls",
        neb_result.oracle_calls, aie_result.oracle_calls, oie_result.oracle_calls);
    eprintln!("Output: {}", outfile);
}
