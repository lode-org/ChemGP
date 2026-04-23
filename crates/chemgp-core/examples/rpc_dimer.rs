// GP-Dimer / OTGPD / Standard dimer saddle search via either the rgpot RPC
// client or the direct local metatomic backend.
//
// Generic CLI: reads any pos.con + displacement.con pair,
// connects to a running eOn serve instance for energy/gradient.
//
// RPC mode:
//   1. Export model:  uvx --from metatrain mtt export model.ckpt -o model.pt
//   2. Start server:  pixi run -e rpc eonclient -p metatomic --config <ini> --serve-port 12345
//
// Direct local mode:
//   export RGPOT_BUILD_DIR=/path/to/rgpot/bbdir
//   cargo run --release --example metatomic_dimer --features io,rgpot_local -- \
//     --pos data/d000/pos.con --disp data/d000/displacement.con \
//     --method {standard,dimer,otgpd,all} [--max-calls 20]
//
// Usage:
//   cargo run --release --example rpc_dimer --features io,rgpot -- \
//     --pos data/d000/pos.con --disp data/d000/displacement.con \
//     --method {standard,dimer,otgpd,all} [--max-calls 20]
//
// Outputs rpc_dimer.jsonl for plotting.

use std::cell::RefCell;
use std::io::Write;

use chemgp_core::benchmarking::{
    linear_prior, nearest_linear_prior, output_path, seed_training_data, select_adaptive_prior,
    BenchmarkVariant,
};
use chemgp_core::dimer::{gp_dimer, standard_dimer, DimerConfig};
use chemgp_core::io::read_con;
use chemgp_core::kernel::{Kernel, MolInvDistSE};
#[cfg(feature = "rgpot_local")]
use chemgp_core::oracle::{LocalMetatomicConfig, LocalMetatomicOracle};
#[cfg(feature = "rgpot")]
use chemgp_core::oracle::RpcOracle;
use chemgp_core::otgpd::{otgpd, OTGPDConfig};


fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

pub fn main() {
    let args: Vec<String> = std::env::args().collect();

    let pos_path = get_arg(&args, "--pos").unwrap_or_else(|| "data/d000/pos.con".into());
    let disp_path =
        get_arg(&args, "--disp").unwrap_or_else(|| "data/d000/displacement.con".into());
    let method = get_arg(&args, "--method").unwrap_or_else(|| "all".into());
    let max_calls: usize = get_arg(&args, "--max-calls")
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    #[cfg(feature = "rgpot")]
    let host = std::env::var("RGPOT_HOST").unwrap_or_else(|_| "localhost".into());
    #[cfg(feature = "rgpot")]
    let port: u16 = std::env::var("RGPOT_PORT")
        .unwrap_or_else(|_| "12345".into())
        .parse()
        .expect("RGPOT_PORT must be a valid port number");
    #[cfg(feature = "rgpot_local")]
    let local_cfg = LocalMetatomicConfig {
        model_path: std::env::var("RGPOT_MODEL_PATH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("models/pet-mad-xs-v1.5.0.pt")),
        device: std::env::var("RGPOT_DEVICE").unwrap_or_else(|_| "cpu".into()),
        length_unit: std::env::var("RGPOT_LENGTH_UNIT").unwrap_or_else(|_| "angstrom".into()),
        extensions_directory: std::env::var("RGPOT_EXTENSIONS_DIRECTORY")
            .ok()
            .map(std::path::PathBuf::from),
        check_consistency: false,
        uncertainty_threshold: -1.0,
        dtype_override: std::env::var("RGPOT_DTYPE_OVERRIDE").ok(),
    };

    // Load geometry
    let pos_frames = read_con(&pos_path).unwrap_or_else(|e| panic!("Failed to read {}: {}", pos_path, e));
    let disp_frames =
        read_con(&disp_path).unwrap_or_else(|e| panic!("Failed to read {}: {}", disp_path, e));

    let pos = &pos_frames[0];
    let disp = &disp_frames[0];
    let atomic_numbers = pos.atomic_numbers.clone();
    let n_atoms = atomic_numbers.len();
    let box_matrix = [
        pos.cell[0][0], pos.cell[0][1], pos.cell[0][2],
        pos.cell[1][0], pos.cell[1][1], pos.cell[1][2],
        pos.cell[2][0], pos.cell[2][1], pos.cell[2][2],
    ];

    eprintln!("RPC dimer search");
    eprintln!("  pos:    {}", pos_path);
    eprintln!("  disp:   {}", disp_path);
    eprintln!("  atoms:  {} ({:?})", n_atoms, atomic_numbers);
    #[cfg(feature = "rgpot")]
    eprintln!("  server: {}:{}", host, port);
    #[cfg(feature = "rgpot_local")]
    eprintln!("  local model: {}", local_cfg.model_path.display());
    eprintln!("  method: {}", method);
    eprintln!("  max_calls: {}", max_calls);

    #[cfg(feature = "rgpot")]
    let oracle_impl = RpcOracle::new(&host, port, atomic_numbers.clone(), box_matrix)
        .expect("Failed to connect to eOn serve");
    #[cfg(feature = "rgpot_local")]
    let oracle_impl = LocalMetatomicOracle::new(&local_cfg, atomic_numbers.clone(), box_matrix)
        .expect("Failed to create local metatomic oracle");

    let oracle_cell = RefCell::new(oracle_impl);
    let oracle = move |x: &[f64]| -> (f64, Vec<f64>) {
        oracle_cell
            .borrow_mut()
            .evaluate(x)
            .unwrap_or_else(|e| panic!("RPC oracle failed: {}", e))
    };

    let x_start: Vec<f64> = pos.positions.clone();

    // Orientation: displacement direction (disp - pos), normalized
    let mut orient: Vec<f64> = pos
        .positions
        .iter()
        .zip(disp.positions.iter())
        .map(|(p, d)| d - p)
        .collect();
    let orient_norm: f64 = orient.iter().map(|v| v * v).sum::<f64>().sqrt();
    eprintln!("  |displacement|: {:.6} A", orient_norm);
    if orient_norm < 1e-15 {
        panic!("Displacement is zero; pos.con and displacement.con are identical");
    }
    for v in &mut orient {
        *v /= orient_norm;
    }

    let kernel = Kernel::MolInvDist(MolInvDistSE::from_atomic_numbers(
        &atomic_numbers, vec![], &[], 1.0, 1.0,
    ));

    let dimer_sep = 0.01;
    let run_probe = method == "probe";
    let run_std = method == "standard" || method == "all";
    let run_dimer = method == "dimer" || method == "all";
    let run_otgpd = method == "otgpd" || method == "all";
    let variant = BenchmarkVariant::from_env();

    // --- Probe: single-point evaluation ---
    if run_probe {
        eprintln!("\n=== Single-point probe ===");
        let (e0, g0) = oracle(&x_start);
        let g_inf: f64 = g0.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        let g_l2 = g0.iter().map(|x| x * x).sum::<f64>().sqrt();
        let g_rms = (g0.iter().map(|x| x * x).sum::<f64>() / g0.len() as f64).sqrt();

        // Curvature along displacement direction (finite difference)
        let r1: Vec<f64> = x_start.iter().zip(orient.iter())
            .map(|(r, o)| r + dimer_sep * o).collect();
        let (e1, g1) = oracle(&r1);
        let c_along: f64 = g1.iter().zip(g0.iter()).zip(orient.iter())
            .map(|((a, b), o)| (a - b) * o).sum::<f64>() / dimer_sep;

        eprintln!("  Energy:         {:.8} eV", e0);
        eprintln!("  |grad|_inf:     {:.8} eV/A", g_inf);
        eprintln!("  |grad|_L2:      {:.8} eV/A", g_l2);
        eprintln!("  |grad|_RMS:     {:.8} eV/A", g_rms);
        eprintln!("  E(image1):      {:.8} eV", e1);
        eprintln!("  Curvature:      {:+.8} eV/A^2 (along displacement)", c_along);
        eprintln!("  Saddle sign:    {} (need C < 0 for saddle)", if c_along < 0.0 { "YES" } else { "NO" });

        // Per-atom gradient norms
        eprintln!("\n  Per-atom |grad| (eV/A):");
        for i in 0..n_atoms {
            let gx = g0[3*i];
            let gy = g0[3*i+1];
            let gz = g0[3*i+2];
            let gnorm = (gx*gx + gy*gy + gz*gz).sqrt();
            eprintln!("    atom {}: {:.6}  ({:+.4}, {:+.4}, {:+.4})", i, gnorm, gx, gy, gz);
        }

        // Finite-difference gradient check (3 random components)
        eprintln!("\n  FD gradient check (h=1e-4 A):");
        let h = 1e-4;
        for &idx in &[0, 3, 7] {
            let idx3 = idx % (n_atoms * 3);
            let mut x_plus = x_start.clone();
            let mut x_minus = x_start.clone();
            x_plus[idx3] += h;
            x_minus[idx3] -= h;
            let (e_plus, _) = oracle(&x_plus);
            let (e_minus, _) = oracle(&x_minus);
            let fd_grad = (e_plus - e_minus) / (2.0 * h);
            let anal_grad = g0[idx3];
            let err = (fd_grad - anal_grad).abs();
            eprintln!("    coord[{}]: analytic={:+.6}, FD={:+.6}, |err|={:.2e}",
                idx3, anal_grad, fd_grad, err);
        }

        eprintln!("\nOutput: (probe mode, no JSONL)");
        return;
    }

    let outfile = output_path("rpc_dimer.jsonl");
    let mut f = std::fs::File::create(&outfile).expect("Failed to create output file");

    let conv_threshold = 0.1; // eV/A, consistent with NEB examples
    let mut standard_summary = (0usize, false, f64::NAN, f64::NAN);
    let mut gp_dimer_summary = (0usize, false, f64::NAN, f64::NAN);
    let mut otgpd_summary = (0usize, false, f64::NAN, f64::NAN);

    // --- Standard Dimer ---
    if run_std {
        let mut cfg = DimerConfig::default();
        cfg.t_force_true = conv_threshold;
        cfg.max_oracle_calls = max_calls;
        cfg.max_outer_iter = max_calls;
        cfg.verbose = true;

        eprintln!("\n=== Standard Dimer ===");
        let result = standard_dimer(&oracle, &x_start, &orient, &cfg, dimer_sep);
        eprintln!(
            "  Result: {} calls, |F| = {:.5}, curv = {:.4}, conv = {}, stop = {:?}",
            result.oracle_calls,
            result.history.f_true.last().unwrap_or(&f64::NAN),
            result.history.curv_true.last().unwrap_or(&f64::NAN),
            result.converged,
            result.stop_reason,
        );
        standard_summary = (
            result.oracle_calls,
            result.converged,
            result.history.f_true.last().copied().unwrap_or(f64::NAN),
            result.history.curv_true.last().copied().unwrap_or(f64::NAN),
        );

        for (i, ((&e, &ft), &oc)) in result
            .history
            .e_true
            .iter()
            .zip(result.history.f_true.iter())
            .zip(result.history.oracle_calls.iter())
            .enumerate()
        {
            writeln!(
                f,
                r#"{{"method":"classical","step":{},"energy":{},"force":{},"oracle_calls":{}}}"#,
                i, e, ft, oc
            )
            .expect("Operation failed");
        }
    }

    // --- GP-Dimer ---
    if run_dimer {
        let mut cfg = DimerConfig::default();
        cfg.t_force_true = conv_threshold;
        cfg.t_force_gp = conv_threshold / 10.0;
        cfg.trust_radius = 0.05;
        cfg.max_outer_iter = max_calls;
        cfg.max_oracle_calls = max_calls;
        cfg.max_rot_iter = 0;
        cfg.gp_train_iter = 400;
        cfg.atom_types = atomic_numbers.clone();
        cfg.const_sigma2 = 1.0;

        let mut gp_training_data = None;
        if variant.uses_prior() {
            let (td_seed, observations) = seed_training_data(
                &oracle,
                &x_start,
                cfg.n_initial_perturb,
                cfg.perturb_scale,
                cfg.seed,
            );
            let best_obs = observations
                .iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .expect("No benchmark observations found");
            let worst_obs = observations
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .expect("No benchmark observations found");
            cfg.prior_mean = match variant {
                BenchmarkVariant::Chemgp => cfg.prior_mean.clone(),
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
        eprintln!("\n=== GP-Dimer ===");
        let result = gp_dimer(
            &oracle,
            &x_start,
            &orient,
            &kernel,
            &cfg,
            gp_training_data,
            dimer_sep,
        );
        eprintln!(
            "  Result: {} calls, |F| = {:.5}, curv = {:.4}, conv = {}, stop = {:?}",
            result.oracle_calls,
            result.history.f_true.last().unwrap_or(&f64::NAN),
            result.history.curv_true.last().unwrap_or(&f64::NAN),
            result.converged,
            result.stop_reason,
        );
        gp_dimer_summary = (
            result.oracle_calls,
            result.converged,
            result.history.f_true.last().copied().unwrap_or(f64::NAN),
            result.history.curv_true.last().copied().unwrap_or(f64::NAN),
        );

        for (i, ((&e, &ft), &oc)) in result
            .history
            .e_true
            .iter()
            .zip(result.history.f_true.iter())
            .zip(result.history.oracle_calls.iter())
            .enumerate()
        {
            let sp = result.history.sigma_perp[i];
            writeln!(
                f,
                r#"{{"method":"{}","step":{},"energy":{},"force":{},"oracle_calls":{},"sigma_perp":{}}}"#,
                gp_label,
                i, e, ft, oc, sp
            )
            .expect("Operation failed");
        }
    }

    // --- OTGPD ---
    if run_otgpd {
        let mut cfg = OTGPDConfig::default();
        cfg.t_dimer = conv_threshold;
        cfg.trust_radius = 0.05;
        cfg.max_outer_iter = max_calls;
        cfg.dimer_sep = dimer_sep;
        cfg.max_rot_iter = 0;
        cfg.max_initial_rot = 0;
        cfg.gp_train_iter = 400;
        cfg.fps_latest_points = 3;
        cfg.atom_types = atomic_numbers.clone();
        cfg.const_sigma2 = 1.0;
        cfg.hod_max_history = 60;
        cfg.use_adaptive_threshold = true;
        cfg.divisor_t_dimer_gp = 3.0;
        cfg.rff_features = 500;
        if variant.uses_prior() {
            let (td_seed, observations) = seed_training_data(
                &oracle,
                &x_start,
                cfg.n_initial_perturb,
                cfg.perturb_scale,
                cfg.seed,
            );
            let best_obs = observations
                .iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .expect("No benchmark observations found");
            let worst_obs = observations
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .expect("No benchmark observations found");
            cfg.prior_mean = match variant {
                BenchmarkVariant::Chemgp => cfg.prior_mean.clone(),
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
        }

        eprintln!("\n=== OTGPD ===");
        let result = otgpd(&oracle, &x_start, &orient, &kernel, &cfg, None);
        eprintln!(
            "  Result: {} calls, |F| = {:.5}, curv = {:.4}, conv = {}, stop = {:?}",
            result.oracle_calls,
            result.history.f_true.last().unwrap_or(&f64::NAN),
            result.history.curv_true.last().unwrap_or(&f64::NAN),
            result.converged,
            result.stop_reason,
        );
        eprintln!("  Reference (gprdzbl NWChem): 15 oracle calls, converged");
        otgpd_summary = (
            result.oracle_calls,
            result.converged,
            result.history.f_true.last().copied().unwrap_or(f64::NAN),
            result.history.curv_true.last().copied().unwrap_or(f64::NAN),
        );

        for (i, ((&e, &ft), &oc)) in result
            .history
            .e_true
            .iter()
            .zip(result.history.f_true.iter())
            .zip(result.history.oracle_calls.iter())
            .enumerate()
        {
            let sp = result.history.sigma_perp[i];
            writeln!(
                f,
                r#"{{"method":"otgpd","step":{},"energy":{},"force":{},"oracle_calls":{},"sigma_perp":{}}}"#,
                i, e, ft, oc, sp
            )
            .expect("Operation failed");
        }
    }

    // Summary with convergence threshold
    writeln!(
        f,
        r#"{{"summary":true,"conv_tol":{},"variant":"{}","standard_calls":{},"standard_converged":{},"standard_force":{},"standard_curv":{},"gpdimer_calls":{},"gpdimer_converged":{},"gpdimer_force":{},"gpdimer_curv":{},"otgpd_calls":{},"otgpd_converged":{},"otgpd_force":{},"otgpd_curv":{}}}"#,
        conv_threshold,
        variant.label(),
        standard_summary.0,
        standard_summary.1,
        standard_summary.2,
        standard_summary.3,
        gp_dimer_summary.0,
        gp_dimer_summary.1,
        gp_dimer_summary.2,
        gp_dimer_summary.3,
        otgpd_summary.0,
        otgpd_summary.1,
        otgpd_summary.2,
        otgpd_summary.3
    )
    .expect("Operation failed");

    eprintln!("\nOutput: {}", outfile);
}
