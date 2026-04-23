//! Standard Dimer vs GP-Dimer vs OTGPD on LEPS surface.
//!
//! Starts near the saddle point (as in the C++ gpr_optim reference)
//! to demonstrate GP-accelerated dimer convergence with fewer oracle calls.

use chemgp_core::benchmarking::{
    linear_prior, nearest_linear_prior, output_path, seed_training_data, select_adaptive_prior,
    BenchmarkVariant,
};
use chemgp_core::dimer::{gp_dimer, standard_dimer, DimerConfig};
use chemgp_core::kernel::{Kernel, MolInvDistSE};
use chemgp_core::otgpd::{otgpd, OTGPDConfig};
use chemgp_core::potentials::{leps_energy_gradient, LEPS_SADDLE};

use std::io::Write;

fn main() {
    let oracle = |x: &[f64]| -> (f64, Vec<f64>) { leps_energy_gradient(x) };

    // Orient along the reaction coordinate (from Hessian negative eigenmode).
    // In practice this comes from the NEB climbing image tangent.
    // Negative mode: A moves -x, B moves +x, C moves -x (bond transfer).
    let orient_init = vec![-0.606, 0.0, 0.0, 0.777, 0.0, 0.0, -0.171, 0.0, 0.0];
    let on = orient_init.iter().map(|x| x * x).sum::<f64>().sqrt();
    let orient_init: Vec<f64> = orient_init.iter().map(|x| x / on).collect();

    // Start displaced from the saddle (like C++ gpr_optim dist_sp parameter).
    // dist_sp=0.05 at the boundary of negative curvature (C ~ -3.8 along neg mode).
    let dist_sp = 0.05;
    let x_init: Vec<f64> = LEPS_SADDLE
        .iter()
        .zip(orient_init.iter())
        .map(|(s, o)| s + dist_sp * o)
        .collect();

    let (e_init, g_init) = oracle(&x_init);
    let gnorm: f64 = g_init.iter().map(|v| v * v).sum::<f64>().sqrt();
    eprintln!("Initial: E={:.4}, |G|={:.4}", e_init, gnorm);

    let (e_saddle, _) = oracle(&LEPS_SADDLE);
    eprintln!("Target saddle: E={:.4}", e_saddle);

    let kernel = Kernel::MolInvDist(MolInvDistSE::isotropic(1.0, 1.0, vec![]));
    let dimer_sep = 0.005;
    let gp_dimer_sep = 0.005; // same sep; energy-based curvature is robust at any sep
    let t_conv = 0.1; // force convergence: 0.1 eV/A

    // Standard Dimer (direct oracle, no GP)
    let mut std_cfg = DimerConfig::default();
    std_cfg.t_force_true = t_conv;
    std_cfg.max_oracle_calls = 200;
    std_cfg.step_convex = 0.5;
    std_cfg.verbose = false;

    eprintln!("Running Standard Dimer...");
    let std_result = standard_dimer(&oracle, &x_init, &orient_init, &std_cfg, dimer_sep);
    eprintln!(
        "  Standard Dimer: {} calls, |F| = {:.5}, C = {:.3}, converged = {}",
        std_result.oracle_calls,
        std_result.history.f_true.last().unwrap_or(&f64::NAN),
        std_result.history.curv_true.last().unwrap_or(&f64::NAN),
        std_result.converged
    );
    // Print first 5 curvatures
    for (i, c) in std_result.history.curv_true.iter().take(5).enumerate() {
        eprintln!("    step {}: C = {:+.4e}, |F| = {:.5}", i,
            c, std_result.history.f_true[i]);
    }

    // GP-Dimer (with FPS subset)
    let mut dimer_cfg = DimerConfig::default();
    dimer_cfg.max_outer_iter = 50;
    dimer_cfg.max_inner_iter = 5; // few GP steps, then oracle (user guidance: don't wander)
    dimer_cfg.t_force_true = t_conv;
    dimer_cfg.t_force_gp = 0.01;
    dimer_cfg.gp_train_iter = 50;
    dimer_cfg.fps_history = 15;
    dimer_cfg.n_initial_perturb = 4;
    dimer_cfg.perturb_scale = 0.02;
    dimer_cfg.max_step = 0.01;
    dimer_cfg.step_convex = 0.001; // minimal convex step: near saddle, trust the neg-curv branch
    dimer_cfg.trust_radius = 0.03;
    dimer_cfg.max_rot_iter = 0; // skip GP rotation: orient from NEB tangent is reliable
    dimer_cfg.verbose = false;

    let variant = BenchmarkVariant::from_env();
    let mut gp_training_data = None;
    if variant.uses_prior() {
        let (td_seed, observations) = seed_training_data(
            &oracle,
            &x_init,
            dimer_cfg.n_initial_perturb,
            dimer_cfg.perturb_scale,
            dimer_cfg.seed,
        );
        let best_obs = observations
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("No benchmark observations found");
        let worst_obs = observations
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("No benchmark observations found");
        dimer_cfg.prior_mean = match variant {
            BenchmarkVariant::Chemgp => dimer_cfg.prior_mean.clone(),
            BenchmarkVariant::PhysicalPrior => {
                linear_prior(&observations[0].0, observations[0].1, &observations[0].2, "initial")
            }
            BenchmarkVariant::AdaptivePrior => select_adaptive_prior(
                observations[0].0.as_slice(),
                observations[0].1,
                observations[0].2.as_slice(),
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
    eprintln!("Running GP-Dimer...");
    let dimer_result = gp_dimer(
        &oracle,
        &x_init,
        &orient_init,
        &kernel,
        &dimer_cfg,
        gp_training_data,
        gp_dimer_sep,
    );
    eprintln!(
        "  GP-Dimer: {} calls, |F| = {:.5}, converged = {}",
        dimer_result.oracle_calls,
        dimer_result.history.f_true.last().unwrap_or(&f64::NAN),
        dimer_result.converged
    );

    // OTGPD (adaptive threshold GP dimer)
    let mut otgpd_cfg = OTGPDConfig::default();
    otgpd_cfg.max_outer_iter = 30;
    otgpd_cfg.max_inner_iter = 5;
    otgpd_cfg.t_dimer = t_conv;
    otgpd_cfg.dimer_sep = dimer_sep;
    otgpd_cfg.initial_rotation = false;
    otgpd_cfg.max_rot_iter = 0; // skip GP rotation (orient from eigenmode is reliable)
    otgpd_cfg.eval_image1 = true;
    otgpd_cfg.gp_train_iter = 50;
    otgpd_cfg.n_initial_perturb = 4;
    otgpd_cfg.perturb_scale = 0.02;
    otgpd_cfg.max_step = 0.01;
    otgpd_cfg.step_convex = 0.001;
    otgpd_cfg.trust_radius = 0.03;
    otgpd_cfg.fps_history = 30; // don't FPS-subset until we have enough data
    otgpd_cfg.verbose = false;

    eprintln!("Running OTGPD...");
    let otgpd_result = otgpd(&oracle, &x_init, &orient_init, &kernel, &otgpd_cfg, None);
    eprintln!(
        "  OTGPD: {} calls, |F| = {:.5}, converged = {}",
        otgpd_result.oracle_calls,
        otgpd_result.history.f_true.last().unwrap_or(&f64::NAN),
        otgpd_result.converged
    );

    // Write comparison data
    let outfile = output_path("leps_dimer_comparison.jsonl");
    let mut f = std::fs::File::create(&outfile).expect("Failed to create output file");

    for (i, (&e, &fv)) in std_result
        .history
        .e_true
        .iter()
        .zip(std_result.history.f_true.iter())
        .enumerate()
    {
        writeln!(
            f,
            r#"{{"method":"classical","step":{},"energy":{},"force":{},"oracle_calls":{}}}"#,
            i,
            e,
            fv,
            std_result.history.oracle_calls[i]
        )
        .expect("Operation failed");
    }

    for (i, (&e, &fv)) in dimer_result
        .history
        .e_true
        .iter()
        .zip(dimer_result.history.f_true.iter())
        .enumerate()
    {
        writeln!(
            f,
            r#"{{"method":"{}","step":{},"energy":{},"force":{},"oracle_calls":{},"sigma_perp":{}}}"#,
            gp_label,
            i,
            e,
            fv,
            dimer_result.history.oracle_calls[i],
            dimer_result.history.sigma_perp[i]
        )
        .expect("Operation failed");
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
            r#"{{"method":"otgpd","step":{},"energy":{},"force":{},"oracle_calls":{},"sigma_perp":{}}}"#,
            i,
            e,
            fv,
            otgpd_result.history.oracle_calls[i],
            otgpd_result.history.sigma_perp[i]
        )
        .expect("Operation failed");
    }

    writeln!(
        f,
        r#"{{"summary":true,"gp_method":"{}","standard_calls":{},"dimer_calls":{},"otgpd_calls":{}}}"#,
        gp_label,
        std_result.oracle_calls,
        dimer_result.oracle_calls,
        otgpd_result.oracle_calls
    )
    .expect("Operation failed");

    eprintln!(
        "\nSummary: Standard={} calls, GP-Dimer={} calls, OTGPD={} calls",
        std_result.oracle_calls, dimer_result.oracle_calls, otgpd_result.oracle_calls
    );
    eprintln!("Output: {}", outfile);
}
