// GP-NEB benchmark on system100 cycloaddition using either the rgpot RPC
// client or the direct local metatomic backend.
//
// Runs standard NEB, GP-NEB AIE, and/or GP-NEB OIE on the N2O + C2H4 system.
// RPC mode requires a running eOn serve instance:
//   pixi run -e rpc serve-petmad
//
// Usage:
//   cargo run --release --features rgpot,io,cli --example system100_neb -- --help
//   cargo run --release --features rgpot,io,cli --example system100_neb -- --method oie
//   cargo run --release --features rgpot,io,cli --example system100_neb -- --method all
//
// Direct local mode:
//   export RGPOT_BUILD_DIR=/path/to/rgpot/bbdir
//   cargo run --release --features rgpot_local,io,cli --example system100_neb_local -- --method all

use std::cell::RefCell;
use std::io::Write;

use clap::{Parser, ValueEnum};

use chemgp_core::benchmarking::{
    artifact_path, nearest_linear_prior, output_path, sampled_taylor_prior, BenchmarkVariant,
};
use chemgp_core::io::{read_con, write_con, write_neb_dat, MolConfig};
use chemgp_core::kernel::{Kernel, MolInvDistSE};
use chemgp_core::minimize::{gp_minimize, MinimizationConfig};
use chemgp_core::neb::{gp_neb_aie, neb_optimize, NEBResult};
use chemgp_core::neb_oie::gp_neb_oie;
use chemgp_core::neb_path::{linear_interpolation, AcquisitionStrategy, NEBConfig};
#[cfg(feature = "rgpot_local")]
use chemgp_core::oracle::{LocalMetatomicConfig, LocalMetatomicOracle};
#[cfg(feature = "rgpot")]
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

    /// Convergence threshold on the climbing-image force
    #[arg(long, default_value_t = 0.5)]
    conv_tol: f64,

    /// Optional cap for GP-NEB outer iterations in tutorial-style runs
    #[arg(long)]
    max_outer: Option<usize>,

    /// Skip GP endpoint minimization
    #[arg(long)]
    skip_minimize: bool,

    /// Convergence threshold for endpoint GP minimization
    #[arg(long, default_value_t = 0.05)]
    endpoint_conv_tol: f64,

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

/// Shared NEB config matching eOn 2.11.1 exactly unless overridden by CLI.
fn base_neb_config(images: usize, conv_tol: f64) -> NEBConfig {
    let mut cfg = NEBConfig::default();
    cfg.images = images;
    cfg.max_iter = 1000;
    cfg.conv_tol = conv_tol;         // molecular CI force threshold
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

pub fn main() {
    let args = Args::parse();
    let variant = BenchmarkVariant::from_env();

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

    eprintln!("System100 cycloaddition NEB");
    eprintln!("  atoms: {} ({:?})", n_atoms, atomic_numbers);
    eprintln!("  box: [{:.1}, {:.1}, {:.1}]", box_matrix[0], box_matrix[4], box_matrix[8]);
    eprintln!("  method: {:?}", args.method);
    #[cfg(feature = "rgpot")]
    eprintln!("  connecting to {}:{}", args.host, args.port);
    #[cfg(feature = "rgpot_local")]
    eprintln!("  local model: {}", local_cfg.model_path.display());

    #[cfg(feature = "rgpot")]
    let oracle_impl = RpcOracle::new(&args.host, args.port, atomic_numbers.clone(), box_matrix)
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

    #[cfg(feature = "rgpot")]
    let prior_host = std::env::var("RGPOT_PRIOR_HOST").unwrap_or_else(|_| args.host.clone());
    #[cfg(feature = "rgpot")]
    let prior_port: u16 = std::env::var("RGPOT_PRIOR_PORT")
        .unwrap_or_else(|_| args.port.to_string())
        .parse()
        .expect("RGPOT_PRIOR_PORT must be a valid port number");
    #[cfg(feature = "rgpot")]
    let prior_oracle_impl =
        RpcOracle::new(&prior_host, prior_port, atomic_numbers.clone(), box_matrix)
            .expect("Failed to connect to prior eOn serve");
    #[cfg(feature = "rgpot_local")]
    let prior_local_cfg = LocalMetatomicConfig {
        model_path: std::env::var("RGPOT_PRIOR_MODEL_PATH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| local_cfg.model_path.clone()),
        device: local_cfg.device.clone(),
        length_unit: local_cfg.length_unit.clone(),
        extensions_directory: local_cfg.extensions_directory.clone(),
        check_consistency: local_cfg.check_consistency,
        uncertainty_threshold: local_cfg.uncertainty_threshold,
        dtype_override: local_cfg.dtype_override.clone(),
    };
    #[cfg(feature = "rgpot_local")]
    let prior_oracle_impl =
        LocalMetatomicOracle::new(&prior_local_cfg, atomic_numbers.clone(), box_matrix)
            .expect("Failed to create local prior metatomic oracle");
    let prior_oracle_cell = RefCell::new(prior_oracle_impl);
    let prior_oracle = move |x: &[f64]| -> (f64, Vec<f64>) {
        prior_oracle_cell
            .borrow_mut()
            .evaluate(x)
            .unwrap_or_else(|e| panic!("Prior oracle failed: {}", e))
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
    let prior_path = linear_interpolation(&x_start, &x_end, args.images.max(5));
    let neb_prior = match variant {
        BenchmarkVariant::Chemgp => None,
        BenchmarkVariant::PhysicalPrior => Some(sampled_taylor_prior(
            &prior_oracle,
            &prior_path,
            "prior_path",
        )),
        BenchmarkVariant::AdaptivePrior | BenchmarkVariant::RecycledLocalPes => {
            Some(nearest_linear_prior(&[
                ("reactant", x_start.as_slice(), e_r, g_r.as_slice()),
                ("product", x_end.as_slice(), e_p, g_p.as_slice()),
            ]))
        }
    };

    // GP-minimize endpoints on PET-MAD surface
    let mut min_calls = 2; // initial evals
    if !args.skip_minimize {
        let mut min_cfg = MinimizationConfig::default();
        min_cfg.max_iter = 50;
        min_cfg.max_oracle_calls = 20;
        min_cfg.conv_tol = args.endpoint_conv_tol;
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
        let neb_cfg = base_neb_config(args.images, args.conv_tol);
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
        let max_outer = args
            .max_outer
            .unwrap_or_else(|| ((neb_calls.saturating_sub(12)) / n_img).min(40));
        eprintln!("  Budget: {} outer iters (from {} NEB calls)", max_outer, neb_calls);

        let mut cfg = base_neb_config(n_img, args.conv_tol);
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
        if let Some(prior) = neb_prior.clone() {
            cfg.prior_mean = prior;
        }

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
        let max_outer = args.max_outer.unwrap_or_else(|| neb_calls.min(400));
        eprintln!("  Budget: {} outer iters (1 call/iter, cap from {} NEB calls)", max_outer, neb_calls);

        let mut cfg = base_neb_config(args.images, args.conv_tol);
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
        if let Some(prior) = neb_prior.clone() {
            cfg.prior_mean = prior;
        }

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
        let max_outer = args.max_outer.unwrap_or_else(|| neb_calls.min(400));
        eprintln!("  Budget: {} outer iters, cap from {} NEB calls", max_outer, neb_calls);

        let mut cfg = base_neb_config(args.images, args.conv_tol);
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
        cfg.fps_history = 30;
        cfg.fps_latest_points = 3;
        cfg.trust_radius = 0.1;         // match AIE
        cfg.trust_metric = chemgp_core::trust::TrustMetric::Emd;
        cfg.atom_types = atomic_numbers.clone();
        cfg.unc_convergence = 0.0;
        cfg.evals_per_iter = 3;         // triplet {i-1, i, i+1}
        cfg.use_adaptive_triplet_exploration = false;
        cfg.max_pred_points = 0;        // no KNN: use full FPS subset for prediction
        cfg.unc_revert_tol = 0.0;
        cfg.hod_max_history = 80;
        cfg.const_sigma2 = 1.0;
        cfg.acquisition = parse_acquisition(&args.acquisition);
        cfg.lcb_kappa = match cfg.acquisition {
            AcquisitionStrategy::Ucb => 2.0,
            _ => 0.0,
        };
        if let Some(prior) = neb_prior.clone() {
            cfg.prior_mean = prior;
        }

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
        let max_outer = args.max_outer.unwrap_or_else(|| neb_calls.min(400));

        for (name, acq) in &strategies {
            eprintln!("\n=== GP-NEB OIE ({}) ===", name);
            let mut cfg = base_neb_config(args.images, args.conv_tol);
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
            cfg.use_adaptive_triplet_exploration = false;
            if let Some(prior) = neb_prior.clone() {
                cfg.prior_mean = prior;
            }

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
    let output_path = output_path(&args.output);
    let mut f = std::fs::File::create(&output_path).expect("Failed to create output file");
    let aie_label = format!("{}_aie", variant.label());
    let oie_label = format!("{}_oie", variant.label());

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
        write_convergence(&mut f, "classical", r);
    }
    if let Some(ref r) = aie_result {
        write_convergence(&mut f, &aie_label, r);
    }
    if let Some(ref r) = oie_result {
        write_convergence(&mut f, &oie_label, r);
    }
    if let Some(ref r) = oie_enh_result {
        write_convergence(&mut f, &oie_label, r);
    }
    for (name, ref r) in &compare_results {
        write_convergence(&mut f, &format!("{}_oie_{}", variant.label(), name), r);
    }

    // Path energies for each method
    let write_path_energies = |f: &mut std::fs::File, method: &str, result: &NEBResult| {
        for (img, e) in result.path.energies.iter().enumerate() {
            writeln!(f, r#"{{"type":"path_energy","method":"{}","image":{},"energy":{}}}"#,
                method, img, e).expect("Failed to write to output file");
        }
    };
    if let Some(ref r) = neb_result {
        write_path_energies(&mut f, "classical", r);
    }
    if let Some(ref r) = aie_result {
        write_path_energies(&mut f, &aie_label, r);
    }
    if let Some(ref r) = oie_result {
        write_path_energies(&mut f, &oie_label, r);
    }
    if let Some(ref r) = oie_enh_result {
        write_path_energies(&mut f, &oie_label, r);
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

        let con_path = artifact_path(&format!("system100_neb_path_{}.con", label));
        write_con(&con_path, &configs).unwrap_or_else(|e| eprintln!("  warn: {}", e));

        let dat_path = artifact_path(&format!("system100_neb_{}.dat", label));
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
        let sp_path = artifact_path(&format!("system100_neb_sp_{}.con", label));
        write_con(&sp_path, &sp_config).unwrap_or_else(|e| eprintln!("  warn: {}", e));
        eprintln!("  wrote {} (image {})", sp_path, sp_idx);
    };

    for (label, res) in [("neb", &neb_result), ("aie", &aie_result), ("oie", &oie_result), ("oie", &oie_enh_result)] {
        if let Some(ref r) = res {
            write_path(label, r);
        }
    }

    // Summary
    let oie_calls = oie_enh_result.as_ref().or(oie_result.as_ref()).map_or(0, |r| r.oracle_calls);
    let base_cfg = base_neb_config(args.images, args.conv_tol);
    writeln!(f, r#"{{"summary":true,"variant":"{}","conv_tol":{},"neb_calls":{},"neb_converged":{},"neb_max_force":{},"neb_ci_force":{},"aie_calls":{},"aie_converged":{},"aie_max_force":{},"aie_ci_force":{},"oie_calls":{},"oie_converged":{},"oie_max_force":{},"oie_ci_force":{}}}"#,
        variant.label(),
        base_cfg.conv_tol,
        neb_result.as_ref().map_or(0, |r| r.oracle_calls),
        neb_result.as_ref().is_some_and(|r| r.converged),
        neb_result.as_ref().and_then(|r| r.history.max_force.last().copied()).unwrap_or(f64::NAN),
        neb_result.as_ref().and_then(|r| r.history.ci_force.last().copied()).unwrap_or(f64::NAN),
        aie_result.as_ref().map_or(0, |r| r.oracle_calls),
        aie_result.as_ref().is_some_and(|r| r.converged),
        aie_result.as_ref().and_then(|r| r.history.max_force.last().copied()).unwrap_or(f64::NAN),
        aie_result.as_ref().and_then(|r| r.history.ci_force.last().copied()).unwrap_or(f64::NAN),
        oie_calls,
        oie_enh_result
            .as_ref()
            .or(oie_result.as_ref())
            .is_some_and(|r| r.converged),
        oie_enh_result
            .as_ref()
            .or(oie_result.as_ref())
            .and_then(|r| r.history.max_force.last().copied())
            .unwrap_or(f64::NAN),
        oie_enh_result
            .as_ref()
            .or(oie_result.as_ref())
            .and_then(|r| r.history.ci_force.last().copied())
            .unwrap_or(f64::NAN),
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
        eprintln!("  OIE:      {} calls, converged = {}", r.oracle_calls, r.converged);
    }
    for (name, ref r) in &compare_results {
        eprintln!("  OIE-{}:  {} calls, converged = {}, max|F| = {:.5}",
            name, r.oracle_calls, r.converged,
            r.history.max_force.last().unwrap_or(&f64::NAN));
    }
    eprintln!("Output: {}", output_path);
}
