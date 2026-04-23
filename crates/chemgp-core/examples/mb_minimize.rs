//! GP minimize vs direct minimize on Muller-Brown surface.
//!
//! Demonstrates CartesianSE kernel on a 2D analytical potential.
//! GP should converge in fewer oracle calls than direct gradient descent.

use chemgp_core::benchmarking::{
    linear_prior, nearest_linear_prior, output_path, seed_training_data, select_adaptive_prior,
    BenchmarkVariant,
};
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
    gp_cfg.n_initial_perturb = 3;
    gp_cfg.perturb_scale = 0.15;
    gp_cfg.trust_radius = 0.3;
    gp_cfg.penalty_coeff = 1000.0;
    gp_cfg.max_move = 0.3;
    gp_cfg.trust_metric = chemgp_core::trust::TrustMetric::Euclidean;
    gp_cfg.verbose = false;

    let variant = BenchmarkVariant::from_env();
    let mut gp_training_data = None;
    if variant.uses_prior() {
        let (td_seed, observations) = seed_training_data(
            &oracle,
            &x_init,
            gp_cfg.n_initial_perturb,
            gp_cfg.perturb_scale,
            gp_cfg.seed,
        );
        let best_obs = observations
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("No benchmark observations found");
        let worst_obs = observations
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("No benchmark observations found");
        gp_cfg.prior_mean = match variant {
            BenchmarkVariant::Chemgp => gp_cfg.prior_mean.clone(),
            BenchmarkVariant::PhysicalPrior => {
                linear_prior(&observations[0].0, observations[0].1, &observations[0].2, "initial")
            }
            BenchmarkVariant::AdaptivePrior => select_adaptive_prior(
                &td_seed,
                &[
                    (
                        "initial",
                        observations[0].0.as_slice(),
                        observations[0].1,
                        observations[0].2.as_slice(),
                    ),
                    (
                        "best_sample",
                        best_obs.0.as_slice(),
                        best_obs.1,
                        best_obs.2.as_slice(),
                    ),
                ],
            ),
            BenchmarkVariant::RecycledLocalPes => nearest_linear_prior(&[
                (
                    "initial",
                    observations[0].0.as_slice(),
                    observations[0].1,
                    observations[0].2.as_slice(),
                ),
                (
                    "best_sample",
                    best_obs.0.as_slice(),
                    best_obs.1,
                    best_obs.2.as_slice(),
                ),
                (
                    "worst_sample",
                    worst_obs.0.as_slice(),
                    worst_obs.1,
                    worst_obs.2.as_slice(),
                ),
            ]),
        };
        gp_training_data = Some(td_seed);
    }

    let gp_label = variant.label();
    let gp_result = gp_minimize(&oracle, &x_init, &kernel, &gp_cfg, gp_training_data);

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
    let outfile = output_path("mb_minimize_comparison.jsonl");
    let mut f = std::fs::File::create(&outfile).expect("Failed to create output file");

    for (i, e) in gp_result.energies.iter().enumerate() {
        writeln!(
            f,
            r#"{{"method":"{}","step":{},"energy":{},"oracle_calls":{}}}"#,
            gp_label, i, e, i + 1
        )
        .expect("Operation failed");
    }

    for (i, e) in direct_energies.iter().enumerate() {
        writeln!(
            f,
            r#"{{"method":"classical","step":{},"energy":{},"oracle_calls":{}}}"#,
            i, e, i + 1
        )
        .expect("Operation failed");
    }

    // Summary
    writeln!(
        f,
        r#"{{"summary":true,"gp_method":"{}","gp_calls":{},"gp_energy":{},"gp_converged":{},"direct_calls":{},"direct_energy":{},"conv_tol":{}}}"#,
        gp_label,
        gp_result.oracle_calls, gp_result.e_final, gp_result.converged,
        direct_calls, direct_energies.last().unwrap_or(&f64::NAN), gp_cfg.conv_tol
    )
    .expect("Operation failed");

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
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .expect("Operation failed");

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
