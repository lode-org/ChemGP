//! GP minimize vs direct minimize on LEPS surface.
//!
//! Outputs JSONL data for plotting: GP should converge in fewer oracle calls.

use chemgp_core::kernel::MolInvDistSE;
use chemgp_core::minimize::{gp_minimize, MinimizationConfig};
use chemgp_core::potentials::{leps_energy_gradient, LEPS_REACTANT};

use std::io::Write;

fn main() {
    let oracle = |x: &[f64]| -> (f64, Vec<f64>) { leps_energy_gradient(x) };
    let x_init = LEPS_REACTANT.to_vec();

    // GP minimize
    let kernel = MolInvDistSE::isotropic(1.0, 1.0, vec![]);
    let mut gp_cfg = MinimizationConfig::default();
    gp_cfg.max_iter = 100;
    gp_cfg.max_oracle_calls = 50;
    gp_cfg.conv_tol = 0.01;
    gp_cfg.verbose = false;

    let gp_result = gp_minimize(&oracle, &x_init, &kernel, &gp_cfg, None);

    // Direct gradient descent (no GP, just repeated oracle calls)
    let mut x = x_init.clone();
    let mut direct_energies = Vec::new();
    let mut direct_calls = 0;
    let max_step = 0.05;

    for _ in 0..200 {
        let (e, g) = oracle(&x);
        direct_energies.push(e);
        direct_calls += 1;

        let g_norm: f64 = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if g_norm < 0.01 {
            break;
        }

        // Adaptive step: cap displacement at max_step
        let step_size = (max_step / g_norm).min(0.01);
        for j in 0..x.len() {
            x[j] -= step_size * g[j];
        }
    }

    // Output JSONL
    let outfile = "leps_minimize_comparison.jsonl";
    let mut f = std::fs::File::create(outfile).unwrap();

    // GP trajectory
    for (i, e) in gp_result.energies.iter().enumerate() {
        writeln!(f, r#"{{"method":"gp_minimize","step":{},"energy":{},"oracle_calls":{}}}"#,
            i, e, i + 1).unwrap();
    }

    // Direct trajectory
    for (i, e) in direct_energies.iter().enumerate() {
        writeln!(f, r#"{{"method":"direct_minimize","step":{},"energy":{},"oracle_calls":{}}}"#,
            i, e, i + 1).unwrap();
    }

    eprintln!("GP minimize: {} oracle calls, final E = {:.6}, converged = {}",
        gp_result.oracle_calls, gp_result.e_final, gp_result.converged);
    eprintln!("Direct minimize: {} oracle calls, final E = {:.6}",
        direct_calls, direct_energies.last().unwrap_or(&f64::NAN));
    eprintln!("Output: {}", outfile);
}
