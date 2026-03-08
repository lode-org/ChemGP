//! GP minimize vs direct minimize on LEPS surface.
//!
//! Outputs JSONL data for plotting: GP should converge in fewer oracle calls.

use chemgp_core::kernel::{Kernel, MolInvDistSE};
use chemgp_core::lbfgs::LbfgsHistory;
use chemgp_core::minimize::{gp_minimize, MinimizationConfig};
use chemgp_core::potentials::{leps_energy_gradient, LEPS_REACTANT};

use std::io::Write;

fn main() {
    let oracle = |x: &[f64]| -> (f64, Vec<f64>) { leps_energy_gradient(x) };
    let x_init = LEPS_REACTANT.to_vec();

    // GP minimize
    let kernel = Kernel::MolInvDist(MolInvDistSE::isotropic(1.0, 1.0, vec![]));
    let mut gp_cfg = MinimizationConfig::default();
    gp_cfg.max_iter = 100;
    gp_cfg.max_oracle_calls = 50;
    gp_cfg.conv_tol = 0.01;
    gp_cfg.verbose = false;

    let gp_result = gp_minimize(&oracle, &x_init, &kernel, &gp_cfg, None);

    // Re-evaluate forces at each GP trajectory point for plotting
    let mut gp_forces: Vec<f64> = Vec::new();
    for pt in &gp_result.trajectory {
        let (_, g) = oracle(pt);
        let max_f = g.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        gp_forces.push(max_f);
    }

    // Direct L-BFGS for comparison
    let mut x = x_init.clone();
    let mut direct_energies = Vec::new();
    let mut direct_forces = Vec::new();
    let mut direct_calls = 0usize;
    let max_step = 0.1;
    let mut lbfgs = LbfgsHistory::new(10);
    let mut prev_x: Option<Vec<f64>> = None;
    let mut prev_g: Option<Vec<f64>> = None;

    for _ in 0..200 {
        let (e, g) = oracle(&x);
        let max_f = g.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        direct_energies.push(e);
        direct_forces.push(max_f);
        direct_calls += 1;

        if max_f < 0.01 {
            break;
        }

        // L-BFGS step computation
        let grad: Vec<f64> = g.iter().map(|v| -v).collect();  // gradient = -force
        let step = lbfgs.compute_direction(&grad);

        // Update L-BFGS history
        if let (Some(ref px), Some(ref pg)) = (&prev_x, &prev_g) {
            let s: Vec<f64> = x.iter().zip(px.iter()).map(|(a, b)| a - b).collect();
            let y: Vec<f64> = grad.iter().zip(pg.iter()).map(|(a, b)| a - b).collect();
            lbfgs.push_pair(s, y);
        }

        prev_x = Some(x.clone());
        prev_g = Some(grad);

        // Clip step to max_step per atom
        let step_norm: f64 = step.iter().map(|v| v * v).sum::<f64>().sqrt();
        let scale = if step_norm > max_step {
            max_step / step_norm
        } else {
            1.0
        };

        for j in 0..x.len() {
            x[j] += scale * step[j];
        }
    }

    // Output JSONL

    // Output JSONL
    let outfile = "leps_minimize_comparison.jsonl";
    let mut f = std::fs::File::create(outfile).expect("Failed to create output file");

    // GP trajectory
    for (i, (e, max_f)) in gp_result.energies.iter().zip(gp_forces.iter()).enumerate() {
        writeln!(f, r#"{{"method":"gp_minimize","step":{},"energy":{},"max_fatom":{},"oracle_calls":{}}}"#,
            i, e, max_f, i + 1).expect("Failed to write to output file");
    }

    // Direct trajectory
    for (i, (e, max_f)) in direct_energies.iter().zip(direct_forces.iter()).enumerate() {
        writeln!(f, r#"{{"method":"direct_minimize","step":{},"energy":{},"max_fatom":{},"oracle_calls":{}}}"#,
            i, e, max_f, i + 1).expect("Failed to write to output file");
    }

    // Summary
    writeln!(f, r#"{{"summary":true,"gp_calls":{},"gp_energy":{},"gp_converged":{},"direct_calls":{},"direct_energy":{},"conv_tol":{}}}"#,
        gp_result.oracle_calls, gp_result.e_final, gp_result.converged,
        direct_calls, direct_energies.last().unwrap_or(&f64::NAN), gp_cfg.conv_tol)
        .expect("Failed to write to output file");

    eprintln!("GP minimize: {} oracle calls, final E = {:.6}, converged = {}",
        gp_result.oracle_calls, gp_result.e_final, gp_result.converged);
    eprintln!("Direct minimize: {} oracle calls, final E = {:.6}",
        direct_calls, direct_energies.last().unwrap_or(&f64::NAN));
    eprintln!("Output: {}", outfile);
}
