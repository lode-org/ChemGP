//! GP minimize on a 9-atom organic fragment via eOn serve RPC (PET-MAD).
//!
//! Requires a running eOn serve instance:
//!   pixi run -e rpc serve-petmad
//!
//! Outputs `petmad_minimize_comparison.jsonl` for plotting.

use std::cell::RefCell;
use std::io::Write;

use chemgp_core::kernel::{Kernel, MolInvDistSE};
use chemgp_core::lbfgs::LbfgsHistory;
use chemgp_core::minimize::{gp_minimize, MinimizationConfig};
use chemgp_core::oracle::RpcOracle;

/// System100 reactant (9-atom organic fragment from ORCA).
/// Atomic numbers: C=6, O=8, N=7, H=1
const SYSTEM100_ATNRS: [i32; 9] = [6, 6, 8, 7, 7, 1, 1, 1, 1];
#[rustfmt::skip]
const SYSTEM100_POSITIONS: [f64; 27] = [
    -1.585722911,  -0.841608472, -0.000003399,
    -0.530569712,  -1.657223032,  0.000004347,
     1.827673209,   0.452908283, -0.000021873,
     0.974426793,   1.269970207,  0.000060316,
     0.157217553,   2.050138136, -0.000040563,
    -2.042098335,  -0.488660077,  0.930399293,
    -2.042089853,  -0.488665692, -0.930412531,
    -0.071757070,  -2.007396792,  0.930065129,
    -0.071749674,  -2.007402560, -0.930050720,
];

fn main() {
    let host = std::env::var("RGPOT_HOST").unwrap_or_else(|_| "localhost".into());
    let port: u16 = std::env::var("RGPOT_PORT")
        .unwrap_or_else(|_| "12345".into())
        .parse()
        .expect("RGPOT_PORT must be a valid port number");

    let atomic_numbers = SYSTEM100_ATNRS.to_vec();
    let n_atoms = atomic_numbers.len();
    // Non-periodic box
    let box_matrix = [20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0];

    eprintln!("PET-MAD GP minimize on system100 (9 atoms)");
    eprintln!("  atoms: {} ({:?})", n_atoms, atomic_numbers);
    eprintln!("  connecting to {}:{}", host, port);

    let rpc_oracle = RpcOracle::new(&host, port, atomic_numbers.clone(), box_matrix)
        .expect("Failed to connect to eOn serve");

    let oracle_cell = RefCell::new(rpc_oracle);
    let oracle = move |x: &[f64]| -> (f64, Vec<f64>) {
        oracle_cell
            .borrow_mut()
            .evaluate(x)
            .unwrap_or_else(|e| panic!("RPC oracle failed: {}", e))
    };

    let x_init = SYSTEM100_POSITIONS.to_vec();

    // Verify oracle works
    let (e0, g0) = oracle(&x_init);
    let g_norm = g0.iter().map(|v| v * v).sum::<f64>().sqrt();
    eprintln!("  Initial E = {:.6} eV, |G| = {:.6}", e0, g_norm);

    let kernel = Kernel::MolInvDist(MolInvDistSE::isotropic(1.0, 1.0, vec![]));

    // GP minimize
    let mut gp_cfg = MinimizationConfig::default();
    gp_cfg.max_iter = 50;
    gp_cfg.max_oracle_calls = 30;
    gp_cfg.conv_tol = 0.01;
    gp_cfg.trust_metric = chemgp_core::trust::TrustMetric::Emd;
    gp_cfg.atom_types = atomic_numbers.clone();
    gp_cfg.const_sigma2 = 1.0;
    gp_cfg.fps_history = 20;
    gp_cfg.fps_latest_points = 3;
    gp_cfg.verbose = true;

    eprintln!("Running GP minimize...");
    let gp_result = gp_minimize(&oracle, &x_init, &kernel, &gp_cfg, None);
    eprintln!(
        "  GP: {} oracle calls, final E = {:.6}, converged = {}",
        gp_result.oracle_calls, gp_result.e_final, gp_result.converged
    );

    // Direct L-BFGS for comparison
    let mut x = x_init.clone();
    let mut direct_energies = Vec::new();
    let mut direct_calls = 0;
    let max_step = 0.1;
    let mut lbfgs = LbfgsHistory::new(20);
    let mut prev_x: Option<Vec<f64>> = None;
    let mut prev_g: Option<Vec<f64>> = None;

    eprintln!("Running direct L-BFGS...");
    for _ in 0..200 {
        let (e, g) = oracle(&x);
        direct_energies.push(e);
        direct_calls += 1;

        let g_norm: f64 = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if g_norm < 0.01 {
            break;
        }

        // Update L-BFGS history
        if let (Some(xp), Some(gp)) = (&prev_x, &prev_g) {
            let s: Vec<f64> = x.iter().zip(xp.iter()).map(|(a, b)| a - b).collect();
            let y: Vec<f64> = g.iter().zip(gp.iter()).map(|(a, b)| a - b).collect();
            lbfgs.push_pair(s, y);
        }

        let d = lbfgs.compute_direction(&g);
        let d_norm: f64 = d.iter().map(|v| v * v).sum::<f64>().sqrt();
        let scale = if d_norm > max_step {
            max_step / d_norm
        } else {
            1.0
        };

        prev_x = Some(x.clone());
        prev_g = Some(g);
        for j in 0..x.len() {
            x[j] += scale * d[j];
        }
    }

    eprintln!(
        "  Direct L-BFGS: {} calls, final E = {:.6}",
        direct_calls,
        direct_energies.last().unwrap_or(&f64::NAN)
    );

    // Write JSONL
    let outfile = "petmad_minimize_comparison.jsonl";
    let mut f = std::fs::File::create(outfile).unwrap();

    for (i, e) in gp_result.energies.iter().enumerate() {
        writeln!(
            f,
            r#"{{"method":"gp_minimize","step":{},"energy":{},"oracle_calls":{}}}"#,
            i,
            e,
            i + 1
        )
        .unwrap();
    }

    for (i, e) in direct_energies.iter().enumerate() {
        writeln!(
            f,
            r#"{{"method":"direct_lbfgs","step":{},"energy":{},"oracle_calls":{}}}"#,
            i,
            e,
            i + 1
        )
        .unwrap();
    }

    writeln!(
        f,
        r#"{{"summary":true,"gp_calls":{},"gp_energy":{},"gp_converged":{},"direct_calls":{},"direct_energy":{}}}"#,
        gp_result.oracle_calls,
        gp_result.e_final,
        gp_result.converged,
        direct_calls,
        direct_energies.last().unwrap_or(&f64::NAN)
    )
    .unwrap();

    eprintln!("Output: {}", outfile);
}
