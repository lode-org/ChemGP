//! GP minimize vs direct minimize on Muller-Brown surface.
//!
//! Demonstrates CartesianSE kernel on a 2D analytical potential.
//! GP should converge in fewer oracle calls than direct gradient descent.

use chemgp_core::kernel::{CartesianSE, Kernel};
use chemgp_core::minimize::{gp_minimize, MinimizationConfig};
use chemgp_core::potentials::{muller_brown_energy_gradient, MULLER_BROWN_MINIMA};

use std::io::Write;

fn main() {
    let oracle = |x: &[f64]| -> (f64, Vec<f64>) { muller_brown_energy_gradient(x) };

    // Start displaced from the second minimum [0.623, 0.028].
    // GP should converge to a local minimum efficiently.
    let x_init = vec![0.4, 0.2];

    // GP minimize with CartesianSE kernel
    let kernel = Kernel::Cartesian(CartesianSE::new(100.0, 2.0));
    let mut gp_cfg = MinimizationConfig::default();
    gp_cfg.max_iter = 200;
    gp_cfg.max_oracle_calls = 50;
    gp_cfg.conv_tol = 1.0;
    gp_cfg.dedup_tol = 0.001; // MB coordinates span ~3 units; default (conv_tol*0.1) is way too large
    gp_cfg.n_initial_perturb = 3;
    gp_cfg.perturb_scale = 0.15;
    gp_cfg.trust_radius = 0.3;
    gp_cfg.penalty_coeff = 1000.0;
    gp_cfg.max_move = 0.3;
    gp_cfg.trust_metric = chemgp_core::trust::TrustMetric::Euclidean;
    gp_cfg.verbose = false;

    let gp_result = gp_minimize(&oracle, &x_init, &kernel, &gp_cfg, None);

    // Direct gradient descent
    let mut x = x_init.clone();
    let mut direct_energies = Vec::new();
    let mut direct_calls = 0;

    for _ in 0..200 {
        let (e, g) = oracle(&x);
        direct_energies.push(e);
        direct_calls += 1;

        let g_norm: f64 = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if g_norm < 1.0 {
            break;
        }

        let step_size = (0.01 / g_norm).min(0.002);
        for j in 0..x.len() {
            x[j] -= step_size * g[j];
        }
    }

    // Output JSONL
    let outfile = "mb_minimize_comparison.jsonl";
    let mut f = std::fs::File::create(outfile).unwrap();

    for (i, e) in gp_result.energies.iter().enumerate() {
        writeln!(
            f,
            r#"{{"method":"gp_minimize","step":{},"energy":{},"oracle_calls":{}}}"#,
            i, e, i + 1
        )
        .unwrap();
    }

    for (i, e) in direct_energies.iter().enumerate() {
        writeln!(
            f,
            r#"{{"method":"direct_minimize","step":{},"energy":{},"oracle_calls":{}}}"#,
            i, e, i + 1
        )
        .unwrap();
    }

    let (min_idx, dist_to_min) = MULLER_BROWN_MINIMA
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let d: f64 = gp_result
                .x_final
                .iter()
                .zip(m.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            (i, d)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    eprintln!(
        "GP minimize: {} oracle calls, final E = {:.4}, converged = {}, nearest_min = {} (d={:.4})",
        gp_result.oracle_calls, gp_result.e_final, gp_result.converged, min_idx, dist_to_min
    );
    eprintln!(
        "Direct minimize: {} oracle calls, final E = {:.4}",
        direct_calls,
        direct_energies.last().unwrap_or(&f64::NAN)
    );
    eprintln!("Output: {}", outfile);
}
