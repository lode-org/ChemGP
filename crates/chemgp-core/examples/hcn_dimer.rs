//! GP-Dimer and OTGPD saddle point search on d_000 (C3H5) via RPC oracle.
//!
//! Tutorial T10 (Production Saddle Search): demonstrates GP-accelerated
//! saddle point refinement on a real PES, using the softest-mode
//! displacement from gprdzbl benchmark data.
//!
//! System: d_000 (allyl radical C3H5, 8 atoms, doublet)
//! Reference: gprdzbl OTGPD converged in 15 oracle calls.
//!
//! Requires a running eOn serve instance:
//!   pixi run -e rpc serve-petmad
//!
//! Usage:
//!   cargo run --release --example hcn_dimer --features io,rgpot -- --method {standard,dimer,otgpd,all}
//!
//! Outputs `hcn_dimer_comparison.jsonl` for plotting.

use std::cell::RefCell;
use std::io::Write;

use chemgp_core::dimer::{gp_dimer, standard_dimer, DimerConfig};
use chemgp_core::io::read_con;
use chemgp_core::kernel::{Kernel, MolInvDistSE};
use chemgp_core::oracle::RpcOracle;
use chemgp_core::otgpd::{otgpd, OTGPDConfig};
use chemgp_core::trust::TrustMetric;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let method = args
        .iter()
        .position(|a| a == "--method")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("all");

    let host = std::env::var("RGPOT_HOST").unwrap_or_else(|_| "localhost".into());
    let port: u16 = std::env::var("RGPOT_PORT")
        .unwrap_or_else(|_| "12345".into())
        .parse()
        .expect("RGPOT_PORT must be a valid port number");

    // Load d_000 geometry: pos.con (initial) and displacement.con (softest mode displaced)
    let pos_frames = read_con("data/d000/pos.con").expect("Failed to read pos.con");
    let disp_frames =
        read_con("data/d000/displacement.con").expect("Failed to read displacement.con");

    let pos = &pos_frames[0];
    let disp = &disp_frames[0];
    let atomic_numbers = pos.atomic_numbers.clone();
    let n_atoms = atomic_numbers.len();
    let box_matrix = [
        pos.cell[0][0], pos.cell[0][1], pos.cell[0][2],
        pos.cell[1][0], pos.cell[1][1], pos.cell[1][2],
        pos.cell[2][0], pos.cell[2][1], pos.cell[2][2],
    ];

    eprintln!("d_000 (C3H5) dimer search (PET-MAD via RPC)");
    eprintln!("  atoms: {} ({:?})", n_atoms, atomic_numbers);
    eprintln!("  connecting to {}:{}", host, port);
    eprintln!("  method: {}", method);

    let rpc_oracle = RpcOracle::new(&host, port, atomic_numbers.clone(), box_matrix)
        .expect("Failed to connect to eOn serve");

    let oracle_cell = RefCell::new(rpc_oracle);
    let oracle = move |x: &[f64]| -> (f64, Vec<f64>) {
        oracle_cell
            .borrow_mut()
            .evaluate(x)
            .unwrap_or_else(|e| panic!("RPC oracle failed: {}", e))
    };

    // Starting geometry from pos.con
    let x_start: Vec<f64> = pos.positions.clone();

    // Initial orientation: displacement direction (disp - pos), normalized
    // This is the softest vibrational mode from the gprdzbl benchmark
    let mut orient: Vec<f64> = pos
        .positions
        .iter()
        .zip(disp.positions.iter())
        .map(|(p, d)| d - p)
        .collect();
    let orient_norm: f64 = orient.iter().map(|v| v * v).sum::<f64>().sqrt();
    eprintln!("  displacement magnitude: {:.6} A", orient_norm);
    for v in &mut orient {
        *v /= orient_norm;
    }

    let kernel = Kernel::MolInvDist(MolInvDistSE::from_atomic_numbers(
        &atomic_numbers, vec![], &[], 1.0, 1.0,
    ));

    // Match gprdzbl config: dimer_separation = 0.01, max_step_size = 0.05
    let dimer_sep = 0.01;
    let run_std = method == "standard" || method == "all";
    let run_dimer = method == "dimer" || method == "all";
    let run_otgpd = method == "otgpd" || method == "all";

    let outfile = "hcn_dimer_comparison.jsonl";
    let mut f = std::fs::File::create(outfile).expect("Failed to create output file");

    // --- Standard Dimer ---
    if run_std {
        let mut cfg = DimerConfig::default();
        cfg.t_force_true = 0.01;
        cfg.max_oracle_calls = 50;
        cfg.max_outer_iter = 50;
        cfg.verbose = true;

        eprintln!("Running Standard Dimer...");
        let result = standard_dimer(&oracle, &x_start, &orient, &cfg, dimer_sep);
        eprintln!(
            "  Standard Dimer: {} calls, |F| = {:.5}, curv = {:.4}, conv = {}, stop = {:?}",
            result.oracle_calls,
            result.history.f_true.last().unwrap_or(&f64::NAN),
            result.history.curv_true.last().unwrap_or(&f64::NAN),
            result.converged,
            result.stop_reason,
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
                r#"{{"method":"standard_dimer","step":{},"energy":{},"force":{},"oracle_calls":{}}}"#,
                i, e, ft, oc
            )
            .expect("Operation failed");
        }
    }

    // --- GP-Dimer ---
    if run_dimer {
        let mut cfg = DimerConfig::default();
        cfg.t_force_true = 0.01;       // gprdzbl: converged_force = 0.01
        cfg.t_force_gp = 0.001;
        cfg.trust_radius = 0.05;       // gprdzbl: max_step_size = 0.05
        cfg.max_outer_iter = 30;
        cfg.max_oracle_calls = 30;
        cfg.max_rot_iter = 0;          // GP rotation breaks molecular systems (degenerate features)
        cfg.gp_train_iter = 400;       // gprdzbl: opt_max_iterations = 400
        cfg.fps_history = 10;          // gprdzbl: fps_history = 10
        cfg.fps_latest_points = 3;
        cfg.trust_metric = TrustMetric::Emd;
        cfg.atom_types = atomic_numbers.clone();
        cfg.const_sigma2 = 1.0;
        cfg.translation_method = "lbfgs".to_string();
        cfg.lbfgs_memory = 25;        // gprdzbl: lbfgs_memory = 25

        eprintln!("Running GP-Dimer...");
        let result = gp_dimer(&oracle, &x_start, &orient, &kernel, &cfg, None, dimer_sep);
        eprintln!(
            "  GP-Dimer: {} calls, |F| = {:.5}, curv = {:.4}, conv = {}, stop = {:?}",
            result.oracle_calls,
            result.history.f_true.last().unwrap_or(&f64::NAN),
            result.history.curv_true.last().unwrap_or(&f64::NAN),
            result.converged,
            result.stop_reason,
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
                r#"{{"method":"gp_dimer","step":{},"energy":{},"force":{},"oracle_calls":{},"sigma_perp":{}}}"#,
                i, e, ft, oc, sp
            )
            .expect("Operation failed");
        }
    }

    // --- OTGPD ---
    if run_otgpd {
        let mut cfg = OTGPDConfig::default();
        cfg.t_dimer = 0.01;           // gprdzbl: converged_force = 0.01
        cfg.divisor_t_dimer_gp = 3.0; // Molecular: looser inner threshold (2D: 10)
        cfg.trust_radius = 0.05;       // gprdzbl: max_step_size = 0.05
        cfg.max_outer_iter = 30;
        cfg.dimer_sep = dimer_sep;
        cfg.max_rot_iter = 0;          // GP rotation breaks molecular systems
        cfg.max_initial_rot = 0;
        cfg.initial_rotation = false;
        cfg.gp_train_iter = 400;       // gprdzbl: opt_max_iterations = 400
        cfg.fps_history = 10;          // gprdzbl: fps_history = 10
        cfg.fps_latest_points = 3;
        cfg.trust_metric = TrustMetric::Emd;
        cfg.atom_types = atomic_numbers.clone();
        cfg.const_sigma2 = 1.0;
        cfg.use_hod = true;
        cfg.hod_monitoring_window = 5;
        cfg.hod_max_history = 60;
        cfg.translation_method = "lbfgs".to_string();
        cfg.lbfgs_memory = 25;        // gprdzbl: lbfgs_memory = 25
        cfg.use_adaptive_threshold = true;
        cfg.rff_features = 500;        // Smooths GP surface for inner loop
        cfg.max_inner_iter = 200;

        eprintln!("Running OTGPD...");
        let result = otgpd(&oracle, &x_start, &orient, &kernel, &cfg, None);
        eprintln!(
            "  OTGPD: {} calls, |F| = {:.5}, curv = {:.4}, conv = {}, stop = {:?}",
            result.oracle_calls,
            result.history.f_true.last().unwrap_or(&f64::NAN),
            result.history.curv_true.last().unwrap_or(&f64::NAN),
            result.converged,
            result.stop_reason,
        );

        // Print known saddle for comparison
        eprintln!("  Reference (gprdzbl NWChem): 15 oracle calls, converged");

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

    eprintln!("Output: {}", outfile);
}
