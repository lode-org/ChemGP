//! GP-NEB benchmark on system100 cycloaddition via eOn serve RPC.
//!
//! Runs standard NEB, GP-NEB AIE, and/or GP-NEB OIE on the N2O + C2H4 system.
//! Requires a running eOn serve instance:
//!   pixi run -e rpc serve-petmad
//!
//! Usage:
//!   cargo run --release --features rgpot,io,cli --example system100_neb -- --help
//!   cargo run --release --features rgpot,io,cli --example system100_neb -- --method oie
//!   cargo run --release --features rgpot,io,cli --example system100_neb -- --method all

use std::cell::RefCell;
use std::io::Write;

use clap::{Parser, ValueEnum};

use chemgp_core::io::{read_con, write_con, write_neb_dat, MolConfig};
use chemgp_core::kernel::{Kernel, MolInvDistSE};
use chemgp_core::minimize::{gp_minimize, MinimizationConfig};
use chemgp_core::neb::{gp_neb_aie, neb_optimize, NEBResult};
use chemgp_core::neb_oie::gp_neb_oie;
use chemgp_core::neb_path::{AcquisitionStrategy, NEBConfig};
use chemgp_core::oracle::RpcOracle;

#[derive(Debug, Clone, ValueEnum)]
enum Method {
    /// Standard NEB only
    Neb,
    /// GP-NEB All Images Evaluated
    Aie,
    /// GP-NEB OIE baseline (variance selection, exact GP, path reset)
    Oie,
    /// GP-NEB OIE enhanced (LCB, RFF, FPS, EMD trust)
    OieEnhanced,
    /// Compare all acquisition strategies for OIE
    OieCompare,
    /// Run all methods
    All,
}

#[derive(Parser, Debug)]
#[command(name = "system100_neb", about = "GP-NEB benchmark on system100 cycloaddition")]
struct Args {
    /// NEB method to run
    #[arg(short, long, value_enum, default_value = "all")]
    method: Method,

    /// RPC server host
    #[arg(long, default_value = "localhost", env = "RGPOT_HOST")]
    host: String,

    /// RPC server port
    #[arg(long, default_value_t = 12345, env = "RGPOT_PORT")]
    port: u16,

    /// Number of intermediate images
    #[arg(long, default_value_t = 10)]
    images: usize,

    /// Skip GP endpoint minimization
    #[arg(long)]
    skip_minimize: bool,

    /// Acquisition strategy for OIE (ucb, ei, thompson, max-force, max-variance)
    #[arg(long, default_value = "ucb")]
    acquisition: String,

    /// Output JSONL file
    #[arg(short, long, default_value = "system100_neb_comparison.jsonl")]
    output: String,
}

fn parse_acquisition(s: &str) -> AcquisitionStrategy {
    match s {
        "max-variance" | "maxvar" => AcquisitionStrategy::MaxVariance,
        "max-force" | "maxf" => AcquisitionStrategy::MaxForce,
        "ucb" => AcquisitionStrategy::Ucb,
        "ei" => AcquisitionStrategy::ExpectedImprovement,
        "thompson" | "ts" => AcquisitionStrategy::ThompsonSampling,
        _ => panic!("Unknown acquisition strategy: {}. Options: ucb, ei, thompson, max-force, max-variance", s),
    }
}

/// Shared NEB config matching eOn 2.11.1 exactly.
fn base_neb_config(images: usize) -> NEBConfig {
    let mut cfg = NEBConfig::default();
    cfg.images = images;
    cfg.max_iter = 1000;
    cfg.conv_tol = 0.5;              // molecular CI force threshold
    cfg.climbing_image = true;
    cfg.ci_activation_tol = 0.5;
    cfg.ci_trigger_rel = 0.8;
    cfg.ci_converged_only = true;
    cfg.energy_weighted = true;
    cfg.ew_k_min = 0.972;
    cfg.ew_k_max = 9.72;
    cfg.max_move = 0.05;
    cfg.lbfgs_memory = 20;
    cfg.initializer = "sidpp".to_string();
    cfg.verbose = true;
    cfg
}

fn main() {
    let args = Args::parse();

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
    eprintln!("  method: {:?}", args.method);
    eprintln!("  connecting to {}:{}", args.host, args.port);

    let rpc_oracle = RpcOracle::new(&args.host, args.port, atomic_numbers.clone(), box_matrix)
        .expect("Failed to connect to eOn serve");

    let oracle_cell = RefCell::new(rpc_oracle);
    let oracle = move |x: &[f64]| -> (f64, Vec<f64>) {
        oracle_cell
            .borrow_mut()
            .evaluate(x)
            .unwrap_or_else(|e| panic!("RPC oracle failed: {}", e))
    };

    let mut x_start = reactant.positions.clone();
    let mut x_end = product.positions.clone();

    // Verify endpoints
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
    let mut min_calls = 2; // initial evals
    if !args.skip_minimize {
        let mut min_cfg = MinimizationConfig::default();
        min_cfg.max_iter = 50;
        min_cfg.max_oracle_calls = 20;
        min_cfg.conv_tol = 0.05;
        min_cfg.trust_metric = chemgp_core::trust::TrustMetric::Emd;
        min_cfg.atom_types = atomic_numbers.clone();
        min_cfg.const_sigma2 = 1.0;
        min_cfg.fps_history = 15;
        min_cfg.fps_latest_points = 3;
        min_cfg.verbose = false;

        eprintln!("GP-minimizing reactant endpoint...");
        let r_min = gp_minimize(&oracle, &x_start, &kernel, &min_cfg, None);
        eprintln!("  Reactant: {} calls, E = {:.6}, |G| = {:.4}, conv = {}",
            r_min.oracle_calls, r_min.e_final,
            r_min.g_final.iter().map(|v| v*v).sum::<f64>().sqrt(), r_min.converged);
        x_start = r_min.x_final;

        eprintln!("GP-minimizing product endpoint...");
        let p_min = gp_minimize(&oracle, &x_end, &kernel, &min_cfg, None);
        eprintln!("  Product: {} calls, E = {:.6}, |G| = {:.4}, conv = {}",
            p_min.oracle_calls, p_min.e_final,
            p_min.g_final.iter().map(|v| v*v).sum::<f64>().sqrt(), p_min.converged);
        x_end = p_min.x_final;

        min_calls += r_min.oracle_calls + p_min.oracle_calls;
        eprintln!("  Endpoint minimization: {} total oracle calls", min_calls);
    }

    let run_neb = matches!(args.method, Method::Neb | Method::All);
    let run_aie = matches!(args.method, Method::Aie | Method::All);
    let run_oie = matches!(args.method, Method::Oie);  // baseline only on explicit request
    let run_oie_enh = matches!(args.method, Method::OieEnhanced | Method::All);
    let run_oie_compare = matches!(args.method, Method::OieCompare);

    // --- Standard NEB ---
    let neb_result: Option<NEBResult> = if run_neb {
        eprintln!("\n=== Standard NEB (eOn-matched config) ===");
        let neb_cfg = base_neb_config(args.images);
        let r = neb_optimize(&oracle, &x_start, &x_end, &neb_cfg);
        eprintln!(
            "  NEB: {} calls, {} iters, max|F| = {:.5}, stop = {:?}",
            r.oracle_calls, r.history.max_force.len(),
            r.history.max_force.last().unwrap_or(&f64::NAN), r.stop_reason
        );
        Some(r)
    } else {
        None
    };
    let neb_calls = neb_result.as_ref().map_or(472, |r| r.oracle_calls);

    // --- GP-NEB AIE ---
    let aie_result: Option<NEBResult> = if run_aie {
        eprintln!("\n=== GP-NEB AIE ===");
        let n_img = args.images;
        let max_outer = ((neb_calls.saturating_sub(12)) / n_img).min(40);
        eprintln!("  Budget: {} outer iters (from {} NEB calls)", max_outer, neb_calls);

        let mut cfg = base_neb_config(n_img);
        cfg.max_outer_iter = max_outer;
        cfg.max_iter = 100;
        cfg.max_move = 0.1;
        cfg.gp_train_iter = 150;
        cfg.max_gp_points = 20;
        cfg.rff_features = 0;  // exact GP for this system size
        cfg.fps_history = 30;
        cfg.fps_latest_points = 3;
        cfg.trust_radius = 0.1;
        cfg.trust_metric = chemgp_core::trust::TrustMetric::Emd;
        cfg.atom_types = atomic_numbers.clone();
        cfg.const_sigma2 = 1.0;

        let r = gp_neb_aie(&oracle, &x_start, &x_end, &kernel, &cfg);
        eprintln!(
            "  AIE: {} calls, max|F| = {:.5}, stop = {:?}",
            r.oracle_calls, r.history.max_force.last().unwrap_or(&f64::NAN), r.stop_reason
        );
        Some(r)
    } else {
        None
    };

    // --- GP-NEB OIE (baseline) ---
    // Koistinen et al. (2019): energy variance selection, exact GP, path reset.
    let oie_result: Option<NEBResult> = if run_oie {
        eprintln!("\n=== GP-NEB OIE (baseline) ===");
        let max_outer = neb_calls.min(400);
        eprintln!("  Budget: {} outer iters (1 call/iter, cap from {} NEB calls)", max_outer, neb_calls);

        let mut cfg = base_neb_config(args.images);
        cfg.max_outer_iter = max_outer;
        cfg.max_iter = 1000;
        cfg.max_move = 0.05;
        cfg.gp_train_iter = 150;
        cfg.max_gp_points = 0;        // no FPS, use all data
        cfg.rff_features = 0;         // exact GP
        cfg.ci_force_tol = -1.0;      // use conv_tol
        cfg.inner_ci_threshold = 0.5;
        cfg.gp_tol_divisor = 10;      // adaptive inner tolerance
        cfg.max_step_frac = 0.5;
        cfg.bond_stretch_limit = 2.0 / 3.0;
        cfg.lcb_kappa = 0.0;          // unused for MaxVariance
        cfg.acquisition = AcquisitionStrategy::MaxVariance;
        cfg.trust_radius = 0.0;       // no EMD trust
        cfg.use_quickmin = true;
        cfg.qm_dt = 0.1;
        cfg.atom_types = atomic_numbers.clone();
        cfg.const_sigma2 = 1.0;

        let r = gp_neb_oie(&oracle, &x_start, &x_end, &kernel, &cfg);
        eprintln!(
            "  OIE (baseline): {} calls, max|F| = {:.5}, stop = {:?}",
            r.oracle_calls, r.history.max_force.last().unwrap_or(&f64::NAN), r.stop_reason
        );
        Some(r)
    } else {
        None
    };

    // --- GP-NEB OIE (enhanced: triplet + FPS + EMD) ---
    let oie_enh_result: Option<NEBResult> = if run_oie_enh {
        eprintln!("\n=== GP-NEB OIE (enhanced) ===");
        let max_outer = neb_calls.min(400);
        eprintln!("  Budget: {} outer iters, cap from {} NEB calls", max_outer, neb_calls);

        let mut cfg = base_neb_config(args.images);
        cfg.max_outer_iter = max_outer;
        cfg.max_iter = 100;             // match AIE: enough inner iters for GP relaxation
        cfg.max_move = 0.05;
        cfg.gp_train_iter = 150;        // match AIE: thorough hyperparameter training
        cfg.max_gp_points = 20;         // match AIE: FPS subset for exact GP
        cfg.rff_features = 0;           // exact GP (RFF too approximate for 27D TS)
        cfg.ci_force_tol = -1.0;
        cfg.inner_ci_threshold = 0.5;
        cfg.gp_tol_divisor = 5;
        cfg.max_step_frac = 0.1;
        cfg.bond_stretch_limit = 2.0 / 3.0;
        cfg.lcb_kappa = 0.0;
        cfg.fps_history = 30;
        cfg.fps_latest_points = 3;
        cfg.trust_radius = 0.1;         // match AIE
        cfg.trust_metric = chemgp_core::trust::TrustMetric::Emd;
        cfg.atom_types = atomic_numbers.clone();
        cfg.unc_convergence = 0.0;
        cfg.evals_per_iter = 3;         // triplet {i-1, i, i+1}
        cfg.max_pred_points = 0;        // no KNN: use full FPS subset for prediction
        cfg.unc_revert_tol = 0.0;
        cfg.hod_max_history = 80;
        cfg.const_sigma2 = 1.0;
        cfg.acquisition = parse_acquisition(&args.acquisition);

        let r = gp_neb_oie(&oracle, &x_start, &x_end, &kernel, &cfg);
        eprintln!(
            "  OIE (enhanced): {} calls, max|F| = {:.5}, stop = {:?}",
            r.oracle_calls, r.history.max_force.last().unwrap_or(&f64::NAN), r.stop_reason
        );
        Some(r)
    } else {
        None
    };

    // --- GP-NEB OIE: compare all acquisition strategies ---
    let mut compare_results: Vec<(String, NEBResult)> = Vec::new();
    if run_oie_compare {
        let strategies = [
            ("ucb", AcquisitionStrategy::Ucb),
            ("ei", AcquisitionStrategy::ExpectedImprovement),
            ("thompson", AcquisitionStrategy::ThompsonSampling),
            ("max_force", AcquisitionStrategy::MaxForce),
            ("max_variance", AcquisitionStrategy::MaxVariance),
        ];
        let max_outer = neb_calls.min(400);

        for (name, acq) in &strategies {
            eprintln!("\n=== GP-NEB OIE ({}) ===", name);
            let mut cfg = base_neb_config(args.images);
            cfg.max_outer_iter = max_outer;
            cfg.max_iter = 30;
            cfg.max_move = 0.05;
            cfg.gp_train_iter = 150;
            cfg.max_gp_points = 50;
            cfg.rff_features = 500;
            cfg.ci_force_tol = 1.0;
            cfg.inner_ci_threshold = 0.5;
            cfg.gp_tol_divisor = 10;
            cfg.max_step_frac = 0.1;
            cfg.bond_stretch_limit = 2.0 / 3.0;
            cfg.lcb_kappa = 2.0;
            cfg.fps_history = 30;
            cfg.fps_latest_points = 3;
            cfg.trust_radius = 0.05;
            cfg.trust_metric = chemgp_core::trust::TrustMetric::Emd;
            cfg.atom_types = atomic_numbers.clone();
            cfg.const_sigma2 = 1.0;
            cfg.acquisition = acq.clone();

            let r = gp_neb_oie(&oracle, &x_start, &x_end, &kernel, &cfg);
            eprintln!(
                "  OIE ({}): {} calls, max|F| = {:.5}, stop = {:?}",
                name, r.oracle_calls,
                r.history.max_force.last().unwrap_or(&f64::NAN),
                r.stop_reason
            );
            compare_results.push((name.to_string(), r));
        }
    }

    // --- Write JSONL ---
    let mut f = std::fs::File::create(&args.output).expect("Failed to create output file");

    // Helper to write convergence records (includes CI force when available)
    let write_convergence = |f: &mut std::fs::File, method: &str, result: &NEBResult| {
        for (i, (&mf, &oc)) in result.history.max_force.iter()
            .zip(result.history.oracle_calls.iter()).enumerate()
        {
            let ci_f = result.history.ci_force.get(i).copied().unwrap_or(f64::NAN);
            writeln!(f, r#"{{"method":"{}","step":{},"max_force":{},"ci_force":{},"oracle_calls":{}}}"#,
                method, i, mf, ci_f, oc).expect("Failed to write to output file");
        }
    };

    if let Some(ref r) = neb_result {
        write_convergence(&mut f, "neb", r);
    }
    if let Some(ref r) = aie_result {
        write_convergence(&mut f, "gp_neb_aie", r);
    }
    if let Some(ref r) = oie_result {
        write_convergence(&mut f, "gp_neb_oie", r);
    }
    if let Some(ref r) = oie_enh_result {
        write_convergence(&mut f, "gp_neb_oie_enh", r);
    }
    for (name, ref r) in &compare_results {
        write_convergence(&mut f, &format!("oie_{}", name), r);
    }

    // Path energies from best GP result
    let best_gp = oie_enh_result.as_ref()
        .or(compare_results.first().map(|(_, r)| r))
        .or(oie_result.as_ref())
        .or(aie_result.as_ref());
    if let Some(ref r) = best_gp {
        for (img, e) in r.path.energies.iter().enumerate() {
            writeln!(f, r#"{{"type":"path_energy","image":{},"energy":{}}}"#, img, e).expect("Failed to write to output file");
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

        let con_path = format!("system100_neb_path_{}.con", label);
        write_con(&con_path, &configs).unwrap_or_else(|e| eprintln!("  warn: {}", e));

        let dat_path = format!("system100_neb_{}.dat", label);
        write_neb_dat(&dat_path, &r.path.images, &r.path.energies, &r.path.gradients)
            .unwrap_or_else(|e| eprintln!("  warn: {}", e));

        eprintln!("  wrote {} + {}", con_path, dat_path);

        // Write SP .con (highest-energy image = climbing image saddle)
        let sp_idx = r.path.energies.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(r.path.images.len() / 2);
        let sp_config = vec![MolConfig {
            positions: r.path.images[sp_idx].clone(),
            atomic_numbers: atomic_numbers.clone(),
            energy: Some(r.path.energies[sp_idx]),
            forces: None,
            cell,
        }];
        let sp_path = format!("system100_neb_sp_{}.con", label);
        write_con(&sp_path, &sp_config).unwrap_or_else(|e| eprintln!("  warn: {}", e));
        eprintln!("  wrote {} (image {})", sp_path, sp_idx);
    };

    for (label, res) in [("neb", &neb_result), ("aie", &aie_result), ("oie", &oie_result), ("oie_enh", &oie_enh_result)] {
        if let Some(ref r) = res {
            write_path(label, r);
        }
    }

    // Summary
    writeln!(f, r#"{{"summary":true,"neb_calls":{},"aie_calls":{},"oie_calls":{},"oie_enh_calls":{}}}"#,
        neb_result.as_ref().map_or(0, |r| r.oracle_calls),
        aie_result.as_ref().map_or(0, |r| r.oracle_calls),
        oie_result.as_ref().map_or(0, |r| r.oracle_calls),
        oie_enh_result.as_ref().map_or(0, |r| r.oracle_calls),
    ).expect("Operation failed");
    for (name, ref r) in &compare_results {
        writeln!(f, r#"{{"summary_acq":"{}","calls":{},"converged":{},"final_max_f":{},"final_ci_f":{}}}"#,
            name, r.oracle_calls, r.converged,
            r.history.max_force.last().unwrap_or(&f64::NAN),
            r.history.ci_force.last().unwrap_or(&f64::NAN),
        ).expect("Operation failed");
    }

    eprintln!("\n=== Summary ===");
    if let Some(ref r) = neb_result {
        eprintln!("  NEB:      {} calls, converged = {}", r.oracle_calls, r.converged);
    }
    if let Some(ref r) = aie_result {
        eprintln!("  AIE:      {} calls, converged = {}", r.oracle_calls, r.converged);
    }
    if let Some(ref r) = oie_result {
        eprintln!("  OIE:      {} calls, converged = {}", r.oracle_calls, r.converged);
    }
    if let Some(ref r) = oie_enh_result {
        eprintln!("  OIE-enh:  {} calls, converged = {}", r.oracle_calls, r.converged);
    }
    for (name, ref r) in &compare_results {
        eprintln!("  OIE-{}:  {} calls, converged = {}, max|F| = {:.5}",
            name, r.oracle_calls, r.converged,
            r.history.max_force.last().unwrap_or(&f64::NAN));
    }
    eprintln!("Output: {}", args.output);
}
