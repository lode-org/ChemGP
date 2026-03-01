//! GP-Dimer vs OTGPD on LEPS surface.
//!
//! Outputs JSONL data showing OTGPD finds saddle with fewer oracle calls.

use chemgp_core::dimer::{gp_dimer, DimerConfig};
use chemgp_core::kernel::MolInvDistSE;
use chemgp_core::otgpd::{otgpd, OTGPDConfig};
use chemgp_core::potentials::{leps_energy_gradient, LEPS_REACTANT};

use std::io::Write;

fn main() {
    let oracle = |x: &[f64]| -> (f64, Vec<f64>) { leps_energy_gradient(x) };
    let x_init = LEPS_REACTANT.to_vec();

    // Orient along the AB bond direction
    let mut orient_init = vec![0.0; 9];
    orient_init[3] = 1.0;
    let on = orient_init.iter().map(|x| x * x).sum::<f64>().sqrt();
    let orient_init: Vec<f64> = orient_init.iter().map(|x| x / on).collect();

    let kernel = MolInvDistSE::isotropic(1.0, 1.0, vec![]);

    // GP-Dimer
    let mut dimer_cfg = DimerConfig::default();
    dimer_cfg.max_outer_iter = 30;
    dimer_cfg.max_inner_iter = 50;
    dimer_cfg.t_force_true = 0.5;
    dimer_cfg.t_force_gp = 0.1;
    dimer_cfg.gp_train_iter = 100;
    dimer_cfg.verbose = false;

    eprintln!("Running GP-Dimer...");
    let dimer_result = gp_dimer(
        &oracle, &x_init, &orient_init, &kernel, &dimer_cfg, None, 0.01,
    );
    eprintln!(
        "  GP-Dimer: {} calls, converged = {}",
        dimer_result.oracle_calls, dimer_result.converged
    );

    // OTGPD
    let mut otgpd_cfg = OTGPDConfig::default();
    otgpd_cfg.max_outer_iter = 30;
    otgpd_cfg.max_inner_iter = 50;
    otgpd_cfg.t_dimer = 0.5;
    otgpd_cfg.initial_rotation = false;
    otgpd_cfg.eval_image1 = false;
    otgpd_cfg.gp_train_iter = 100;
    otgpd_cfg.verbose = false;

    eprintln!("Running OTGPD...");
    let otgpd_result = otgpd(&oracle, &x_init, &orient_init, &kernel, &otgpd_cfg, None);
    eprintln!(
        "  OTGPD: {} calls, converged = {}",
        otgpd_result.oracle_calls, otgpd_result.converged
    );

    // Write comparison data
    let outfile = "leps_dimer_comparison.jsonl";
    let mut f = std::fs::File::create(outfile).unwrap();

    for (i, (&e, &fv)) in dimer_result
        .history
        .e_true
        .iter()
        .zip(dimer_result.history.f_true.iter())
        .enumerate()
    {
        writeln!(
            f,
            r#"{{"method":"gp_dimer","step":{},"energy":{},"force":{},"oracle_calls":{}}}"#,
            i,
            e,
            fv,
            dimer_result.history.oracle_calls[i]
        )
        .unwrap();
    }

    for (i, (&e, &fv)) in otgpd_result
        .history
        .e_true
        .iter()
        .zip(otgpd_result.history.f_true.iter())
        .enumerate()
    {
        writeln!(
            f,
            r#"{{"method":"otgpd","step":{},"energy":{},"force":{},"oracle_calls":{}}}"#,
            i,
            e,
            fv,
            otgpd_result.history.oracle_calls[i]
        )
        .unwrap();
    }

    writeln!(
        f,
        r#"{{"summary":true,"dimer_calls":{},"otgpd_calls":{}}}"#,
        dimer_result.oracle_calls, otgpd_result.oracle_calls
    )
    .unwrap();

    eprintln!(
        "\nSummary: GP-Dimer={} calls, OTGPD={} calls",
        dimer_result.oracle_calls, otgpd_result.oracle_calls
    );
    eprintln!("Output: {}", outfile);
}
