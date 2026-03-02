//! GP-Dimer and OTGPD saddle point search on system100 via RPC oracle.
//!
//! Tutorial T10 (Production Saddle Search): demonstrates GP-accelerated
//! saddle point refinement on a real PES, seeded from an NEB climbing
//! image tangent.
//!
//! Requires a running eOn serve instance:
//!   pixi run -e rpc serve-petmad
//!
//! Usage:
//!   cargo run --release --example hcn_dimer --features io,rgpot -- --method {dimer,otgpd,all}
//!
//! Outputs `hcn_dimer_comparison.jsonl` for plotting.

use std::cell::RefCell;
use std::io::Write;

use chemgp_core::dimer::{gp_dimer, DimerConfig};
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

    // Load system100 endpoints for initial geometry
    let react_frames =
        read_con("data/system100/reactant_minimized.con").expect("Failed to read reactant");
    let prod_frames =
        read_con("data/system100/product_minimized.con").expect("Failed to read product");

    let reactant = &react_frames[0];
    let product = &prod_frames[0];
    let atomic_numbers = reactant.atomic_numbers.clone();
    let n_atoms = atomic_numbers.len();
    let box_matrix = [
        reactant.cell[0][0], reactant.cell[0][1], reactant.cell[0][2],
        reactant.cell[1][0], reactant.cell[1][1], reactant.cell[1][2],
        reactant.cell[2][0], reactant.cell[2][1], reactant.cell[2][2],
    ];

    eprintln!("System100 dimer search (PET-MAD via RPC)");
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

    // Initial guess: midpoint between reactant and product (approximate TS)
    let x_start: Vec<f64> = reactant
        .positions
        .iter()
        .zip(product.positions.iter())
        .map(|(r, p)| 0.5 * (r + p))
        .collect();

    // Initial orientation: reactant-to-product direction (normalized)
    let mut orient: Vec<f64> = reactant
        .positions
        .iter()
        .zip(product.positions.iter())
        .map(|(r, p)| p - r)
        .collect();
    let orient_norm: f64 = orient.iter().map(|v| v * v).sum::<f64>().sqrt();
    for v in &mut orient {
        *v /= orient_norm;
    }

    let kernel = Kernel::MolInvDist(MolInvDistSE::from_atomic_numbers(
        &atomic_numbers, vec![], &[], 1.0, 1.0,
    ));

    let dimer_sep = 0.01;
    let run_dimer = method == "dimer" || method == "all";
    let run_otgpd = method == "otgpd" || method == "all";

    let outfile = "hcn_dimer_comparison.jsonl";
    let mut f = std::fs::File::create(outfile).unwrap();

    // --- GP-Dimer ---
    if run_dimer {
        let mut cfg = DimerConfig::default();
        cfg.t_force_true = 0.1;
        cfg.t_force_gp = 0.01;
        cfg.trust_radius = 0.05;
        cfg.max_outer_iter = 100;
        cfg.max_oracle_calls = 80;
        cfg.max_rot_iter = 0; // skip rotation (use provided orient)
        cfg.gp_train_iter = 100;
        cfg.fps_history = 20;
        cfg.fps_latest_points = 3;
        cfg.trust_metric = TrustMetric::Emd;
        cfg.atom_types = atomic_numbers.clone();
        cfg.const_sigma2 = 1.0;
        cfg.translation_method = "lbfgs".to_string();
        cfg.lbfgs_memory = 10;

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
            writeln!(
                f,
                r#"{{"method":"gp_dimer","step":{},"energy":{},"force":{},"oracle_calls":{}}}"#,
                i, e, ft, oc
            )
            .unwrap();
        }
    }

    // --- OTGPD ---
    if run_otgpd {
        let mut cfg = OTGPDConfig::default();
        cfg.t_dimer = 0.1;
        cfg.divisor_t_dimer_gp = 10.0;
        cfg.trust_radius = 0.05;
        cfg.max_outer_iter = 100;
        cfg.dimer_sep = dimer_sep;
        cfg.max_rot_iter = 0;
        cfg.initial_rotation = false;
        cfg.gp_train_iter = 100;
        cfg.fps_history = 20;
        cfg.fps_latest_points = 3;
        cfg.trust_metric = TrustMetric::Emd;
        cfg.atom_types = atomic_numbers.clone();
        cfg.const_sigma2 = 1.0;
        cfg.use_hod = true;
        cfg.hod_monitoring_window = 5;
        cfg.hod_max_history = 60;
        cfg.translation_method = "lbfgs".to_string();
        cfg.lbfgs_memory = 10;

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
                r#"{{"method":"otgpd","step":{},"energy":{},"force":{},"oracle_calls":{}}}"#,
                i, e, ft, oc
            )
            .unwrap();
        }
    }

    eprintln!("Output: {}", outfile);
}
