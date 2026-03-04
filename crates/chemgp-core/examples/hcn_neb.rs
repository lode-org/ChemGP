//! GP-NEB on system100 cycloaddition via eOn serve RPC.
//!
//! Uses PET-MAD/OMAT-minimized endpoints from the nebviz reference.
//! Requires a running eOn serve instance:
//!   pixi run -e rpc serve-petmad
//!
//! Usage: cargo run --release --example hcn_neb --features io,rgpot -- --method {neb,aie,oie,all}
//!
//! Outputs `hcn_neb_comparison.jsonl` for plotting.

use std::cell::RefCell;
use std::io::Write;

use chemgp_core::io::{read_con, write_con, write_neb_dat, MolConfig};
use chemgp_core::kernel::{Kernel, MolInvDistSE};
use chemgp_core::minimize::{gp_minimize, MinimizationConfig};
use chemgp_core::neb::{gp_neb_aie, neb_optimize, NEBResult};
use chemgp_core::neb_oie::gp_neb_oie;
use chemgp_core::neb_path::NEBConfig;
use chemgp_core::oracle::RpcOracle;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let method = args.iter()
        .position(|a| a == "--method")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("oie");

    let host = std::env::var("RGPOT_HOST").unwrap_or_else(|_| "localhost".into());
    let port: u16 = std::env::var("RGPOT_PORT")
        .unwrap_or_else(|_| "12345".into())
        .parse()
        .expect("RGPOT_PORT must be a valid port number");

    // Load minimized endpoints (from nebviz reference pipeline)
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

    eprintln!("System100 cycloaddition NEB (PET-MAD via RPC)");
    eprintln!("  atoms: {} ({:?})", n_atoms, atomic_numbers);
    eprintln!("  box: [{:.1}, {:.1}, {:.1}]", box_matrix[0], box_matrix[4], box_matrix[8]);
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

    let x_start = reactant.positions.clone();
    let x_end = product.positions.clone();

    // Verify endpoints are minimized
    let (e_r, g_r) = oracle(&x_start);
    let (e_p, g_p) = oracle(&x_end);
    let gr_norm = g_r.iter().map(|v| v * v).sum::<f64>().sqrt();
    let gp_norm = g_p.iter().map(|v| v * v).sum::<f64>().sqrt();
    eprintln!("  Reactant: E = {:.6} eV, |G| = {:.6} eV/A", e_r, gr_norm);
    eprintln!("  Product:  E = {:.6} eV, |G| = {:.6} eV/A", e_p, gp_norm);

    // Kernel with proper pair types from atomic numbers
    let kernel = Kernel::MolInvDist(MolInvDistSE::from_atomic_numbers(
        &atomic_numbers, vec![], &[], 1.0, 1.0,
    ));
    eprintln!("  Kernel pair types from {:?}", atomic_numbers);

    // GP-minimize endpoints on PET-MAD surface
    let mut min_cfg = MinimizationConfig::default();
    min_cfg.max_iter = 50;
    min_cfg.max_oracle_calls = 20;
    min_cfg.conv_tol = 0.05;
    min_cfg.trust_metric = chemgp_core::trust::TrustMetric::Emd;
    min_cfg.atom_types = atomic_numbers.clone();
    min_cfg.fps_history = 15;
    min_cfg.fps_latest_points = 3;
    min_cfg.verbose = true;

    eprintln!("GP-minimizing reactant endpoint...");
    let r_min = gp_minimize(&oracle, &x_start, &kernel, &min_cfg, None);
    eprintln!("  Reactant: {} calls, E = {:.6}, |G| = {:.4}, conv = {}",
        r_min.oracle_calls, r_min.e_final,
        r_min.g_final.iter().map(|v| v*v).sum::<f64>().sqrt(), r_min.converged);
    let x_start = r_min.x_final.clone();

    eprintln!("GP-minimizing product endpoint...");
    let p_min = gp_minimize(&oracle, &x_end, &kernel, &min_cfg, None);
    eprintln!("  Product: {} calls, E = {:.6}, |G| = {:.4}, conv = {}",
        p_min.oracle_calls, p_min.e_final,
        p_min.g_final.iter().map(|v| v*v).sum::<f64>().sqrt(), p_min.converged);
    let x_end = p_min.x_final.clone();

    let min_calls = r_min.oracle_calls + p_min.oracle_calls + 2;
    eprintln!("  Endpoint minimization: {} total oracle calls", min_calls);

    let n_img = 10usize;
    let run_neb = method == "neb" || method == "all";
    let run_aie = method == "aie" || method == "all";
    let run_oie = method == "oie" || method == "all";

    // --- Standard NEB ---
    let neb_result = if run_neb {
        let mut neb_cfg = NEBConfig::default();
        neb_cfg.images = n_img;
        neb_cfg.max_iter = 1000;
        neb_cfg.conv_tol = 0.0514221;
        neb_cfg.climbing_image = true;
        neb_cfg.ci_activation_tol = 0.5;
        neb_cfg.ci_trigger_rel = 0.8;
        neb_cfg.ci_converged_only = true;
        neb_cfg.energy_weighted = true;
        neb_cfg.ew_k_min = 0.972;
        neb_cfg.ew_k_max = 9.72;
        neb_cfg.max_move = 0.1;
        neb_cfg.lbfgs_memory = 20;
        neb_cfg.initializer = "sidpp".to_string();
        neb_cfg.verbose = true;

        eprintln!("Running standard NEB (eOn-matched config)...");
        let r = neb_optimize(&oracle, &x_start, &x_end, &neb_cfg);
        eprintln!(
            "  NEB: {} calls, {} iters, max|F| = {:.5}, stop = {:?}",
            r.oracle_calls,
            r.history.max_force.len(),
            r.history.max_force.last().unwrap_or(&f64::NAN),
            r.stop_reason
        );
        Some(r)
    } else {
        None
    };
    let _neb_calls = neb_result.as_ref().map_or(402, |r| r.oracle_calls);

    // --- GP-NEB AIE ---
    let aie_result = if run_aie {
        let mut aie_cfg = NEBConfig::default();
        aie_cfg.images = n_img;
        aie_cfg.max_outer_iter = 50;
        aie_cfg.max_iter = 500;
        aie_cfg.conv_tol = 0.1;
        aie_cfg.climbing_image = true;
        aie_cfg.ci_activation_tol = 0.5;
        aie_cfg.ci_trigger_rel = 0.8;
        aie_cfg.ci_converged_only = true;
        aie_cfg.energy_weighted = true;
        aie_cfg.ew_k_min = 0.972;
        aie_cfg.ew_k_max = 9.72;
        aie_cfg.max_move = 0.1;
        aie_cfg.lbfgs_memory = 20;
        aie_cfg.initializer = "sidpp".to_string();
        aie_cfg.gp_train_iter = 100;
        aie_cfg.max_gp_points = 50;
        aie_cfg.rff_features = 300;
        aie_cfg.fps_history = 30;
        aie_cfg.fps_latest_points = 3;
        aie_cfg.trust_radius = 0.05;
        aie_cfg.trust_metric = chemgp_core::trust::TrustMetric::Emd;
        aie_cfg.atom_types = atomic_numbers.clone();
        aie_cfg.const_sigma2 = 1.0;
        aie_cfg.verbose = true;

        eprintln!("Running GP-NEB AIE...");
        let r = gp_neb_aie(&oracle, &x_start, &x_end, &kernel, &aie_cfg);
        eprintln!(
            "  AIE: {} calls, max|F| = {:.5}, stop = {:?}",
            r.oracle_calls,
            r.history.max_force.last().unwrap_or(&f64::NAN),
            r.stop_reason
        );
        Some(r)
    } else {
        None
    };

    // --- GP-NEB OIE ---
    let oie_result = if run_oie {
        let mut oie_cfg = NEBConfig::default();
        oie_cfg.images = n_img;
        oie_cfg.max_outer_iter = 200;
        oie_cfg.max_neb_oracle_calls = 150;
        oie_cfg.max_iter = 5;
        oie_cfg.conv_tol = 0.1;
        oie_cfg.climbing_image = true;
        oie_cfg.ci_activation_tol = 0.5;
        oie_cfg.ci_trigger_rel = 0.8;
        oie_cfg.ci_converged_only = true;
        oie_cfg.energy_weighted = true;
        oie_cfg.ew_k_min = 0.972;
        oie_cfg.ew_k_max = 9.72;
        oie_cfg.max_move = 0.05;
        oie_cfg.initializer = "sidpp".to_string();
        oie_cfg.gp_train_iter = 100;
        oie_cfg.max_gp_points = 50;
        oie_cfg.rff_features = 300;
        oie_cfg.ci_force_tol = -1.0;
        oie_cfg.inner_ci_threshold = 0.5;
        oie_cfg.gp_tol_divisor = 10;
        oie_cfg.max_step_frac = 0.1;
        oie_cfg.bond_stretch_limit = 2.0 / 3.0;
        oie_cfg.lcb_kappa = 2.0;
        oie_cfg.fps_history = 30;
        oie_cfg.fps_latest_points = 3;
        oie_cfg.trust_radius = 0.05;
        oie_cfg.trust_metric = chemgp_core::trust::TrustMetric::Emd;
        oie_cfg.atom_types = atomic_numbers.clone();
        oie_cfg.unc_convergence = 0.05;
        oie_cfg.unc_revert_tol = 0.0;
        oie_cfg.hod_max_history = 80;
        oie_cfg.const_sigma2 = 1.0;
        oie_cfg.verbose = true;

        eprintln!("Running GP-NEB OIE...");
        let r = gp_neb_oie(&oracle, &x_start, &x_end, &kernel, &oie_cfg);
        eprintln!(
            "  OIE: {} calls, max|F| = {:.5}, stop = {:?}",
            r.oracle_calls,
            r.history.max_force.last().unwrap_or(&f64::NAN),
            r.stop_reason
        );
        Some(r)
    } else {
        None
    };

    // Write JSONL
    let outfile = "hcn_neb_comparison.jsonl";
    let mut f = std::fs::File::create(outfile).unwrap();

    if let Some(ref r) = neb_result {
        for (i, ((&mf, &cf), &oc)) in r.history.max_force.iter()
            .zip(r.history.ci_force.iter())
            .zip(r.history.oracle_calls.iter()).enumerate()
        {
            writeln!(f, r#"{{"method":"neb","step":{},"max_force":{},"ci_force":{},"oracle_calls":{}}}"#, i, mf, cf, oc).unwrap();
        }
    }
    if let Some(ref r) = aie_result {
        for (i, ((&mf, &cf), &oc)) in r.history.max_force.iter()
            .zip(r.history.ci_force.iter())
            .zip(r.history.oracle_calls.iter()).enumerate()
        {
            writeln!(f, r#"{{"method":"gp_neb_aie","step":{},"max_force":{},"ci_force":{},"oracle_calls":{}}}"#, i, mf, cf, oc).unwrap();
        }
    }
    if let Some(ref r) = oie_result {
        for (i, ((&mf, &cf), &oc)) in r.history.max_force.iter()
            .zip(r.history.ci_force.iter())
            .zip(r.history.oracle_calls.iter()).enumerate()
        {
            writeln!(f, r#"{{"method":"gp_neb_oie","step":{},"max_force":{},"ci_force":{},"oracle_calls":{}}}"#, i, mf, cf, oc).unwrap();
        }
    }

    // Path energy profiles for all converged methods
    let e_ref = neb_result.as_ref()
        .map(|r| r.path.energies[0])
        .or(oie_result.as_ref().map(|r| r.path.energies[0]))
        .unwrap_or(0.0);

    for (label, res) in [("neb", &neb_result), ("gp_neb_aie", &aie_result), ("gp_neb_oie", &oie_result)] {
        if let Some(ref r) = res {
            for (img, e) in r.path.energies.iter().enumerate() {
                writeln!(f, r#"{{"type":"path_energy","method":"{}","image":{},"energy":{}}}"#,
                    label, img, e - e_ref).unwrap();
            }
        }
    }

    // Write .con + .dat for rgpycrumbs
    let cell = reactant.cell;
    let write_path = |label: &str, r: &NEBResult| {
        let configs: Vec<MolConfig> = r.path.images.iter().zip(r.path.energies.iter())
            .map(|(pos, &e)| MolConfig {
                positions: pos.clone(),
                atomic_numbers: atomic_numbers.clone(),
                energy: Some(e),
                forces: None,
                cell,
            })
            .collect();

        let con_path = format!("hcn_neb_path_{}.con", label);
        write_con(&con_path, &configs).unwrap_or_else(|e| eprintln!("  warn: {}", e));

        let dat_path = format!("hcn_neb_{}.dat", label);
        write_neb_dat(&dat_path, &r.path.images, &r.path.energies, &r.path.gradients)
            .unwrap_or_else(|e| eprintln!("  warn: {}", e));

        eprintln!("  wrote {} + {}", con_path, dat_path);
    };

    for (label, res) in [("neb", &neb_result), ("aie", &aie_result), ("oie", &oie_result)] {
        if let Some(ref r) = res {
            write_path(label, r);
        }
    }

    let neb_c = neb_result.as_ref().map_or(0, |r| r.oracle_calls);
    let aie_c = aie_result.as_ref().map_or(0, |r| r.oracle_calls);
    let oie_c = oie_result.as_ref().map_or(0, |r| r.oracle_calls);
    eprintln!("\nSummary: NEB={} calls, AIE={} calls, OIE={} calls", neb_c, aie_c, oie_c);
    eprintln!("Output: {}", outfile);
}
