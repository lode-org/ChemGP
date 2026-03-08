//! GP minimize on a 9-atom organic fragment via eOn serve RPC (PET-MAD).
//!
//! Requires a running eOn serve instance:
//!   pixi run -e rpc serve-petmad
//!
//! Outputs `petmad_minimize_comparison.jsonl` for plotting.

use std::cell::RefCell;
use std::io::Write;

use chemgp_core::kernel::{Kernel, MolInvDistSE};
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

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn max_atom_motion(d: &[f64]) -> f64 {
    let n_atoms = d.len() / 3;
    let mut max_disp = 0.0f64;
    for a in 0..n_atoms {
        let off = a * 3;
        let disp: f64 = d[off..off + 3].iter().map(|v| v * v).sum::<f64>().sqrt();
        max_disp = max_disp.max(disp);
    }
    max_disp
}

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

    // Compute max per-atom force norm for GP trajectory
    let mut gp_max_fatom: Vec<f64> = Vec::new();
    for pt in &gp_result.trajectory {
        let (_, grad) = oracle(pt);
        // Oracle returns gradient; force magnitude is the same
        let max_f = (0..n_atoms).map(|a| {
            let off = a * 3;
            grad[off..off + 3].iter().map(|v| v * v).sum::<f64>().sqrt()
        }).fold(0.0f64, f64::max);
        gp_max_fatom.push(max_f);
    }

    // Direct L-BFGS for comparison (eOn client algorithm: LBFGS.cpp)
    //
    // Fixed H0 = inverse_curvature = 0.01 (eOn default, no auto_scale).
    // Two-loop on gradient (q = -force), step = -z.
    // Angle reset, distance reset, per-atom max displacement clipping.
    let mut x = x_init.clone();
    let mut direct_data: Vec<(usize, f64, f64)> = Vec::new();
    let mut direct_calls = 0;
    let max_move = 0.1;
    let n_dof = x.len();
    let h0 = 0.01f64; // eOn inverse_curvature default

    let lbfgs_memory = 20;
    let mut s_buf: Vec<Vec<f64>> = Vec::new();
    let mut y_buf: Vec<Vec<f64>> = Vec::new();
    let mut rho_buf: Vec<f64> = Vec::new();
    let mut prev_x: Option<Vec<f64>> = None;
    let mut prev_forces: Option<Vec<f64>> = None;

    eprintln!("Running direct L-BFGS...");
    for _ in 0..200 {
        let (e, grad) = oracle(&x);
        direct_calls += 1;
        // Oracle returns gradient; negate for forces
        let forces: Vec<f64> = grad.iter().map(|v| -v).collect();
        let max_fatom = (0..n_atoms).map(|a| {
            let off = a * 3;
            forces[off..off + 3].iter().map(|v| v * v).sum::<f64>().sqrt()
        }).fold(0.0f64, f64::max);
        direct_data.push((direct_calls, e, max_fatom));

        if max_fatom < 0.01 {
            break;
        }

        // Update history from previous step
        if let (Some(ref px), Some(ref pf)) = (&prev_x, &prev_forces) {
            let s: Vec<f64> = x.iter().zip(px.iter()).map(|(a, b)| a - b).collect();
            // y = f_old - f_new = grad_new - grad_old (standard L-BFGS convention)
            let y: Vec<f64> = pf.iter().zip(forces.iter()).map(|(a, b)| a - b).collect();
            let sy: f64 = dot(&s, &y);
            if sy > 1e-18 {
                if s_buf.len() >= lbfgs_memory {
                    s_buf.remove(0); y_buf.remove(0); rho_buf.remove(0);
                }
                rho_buf.push(1.0 / sy);
                s_buf.push(s);
                y_buf.push(y);
            }
        }

        // Two-loop recursion (eOn convention: q = -force = gradient)
        let step = if s_buf.is_empty() {
            forces.iter().map(|v| h0 * v).collect::<Vec<f64>>()
        } else {
            let m = s_buf.len();
            let mut q: Vec<f64> = forces.iter().map(|v| -v).collect();

            let mut alpha_vec = vec![0.0; m];
            for i in (0..m).rev() {
                alpha_vec[i] = rho_buf[i] * dot(&s_buf[i], &q);
                for j in 0..n_dof { q[j] -= alpha_vec[i] * y_buf[i][j]; }
            }

            let mut z: Vec<f64> = q.iter().map(|v| h0 * v).collect();

            for i in 0..m {
                let beta = rho_buf[i] * dot(&y_buf[i], &z);
                for j in 0..n_dof { z[j] += (alpha_vec[i] - beta) * s_buf[i][j]; }
            }

            z.iter().map(|v| -v).collect::<Vec<f64>>()
        };

        // Angle reset: step should be along force direction
        let step_dot_f = dot(&step, &forces);
        let f_norm2: f64 = forces.iter().map(|v| v * v).sum();
        let s_norm2: f64 = step.iter().map(|v| v * v).sum();
        let cos_angle = if f_norm2 > 0.0 && s_norm2 > 0.0 {
            step_dot_f / (f_norm2.sqrt() * s_norm2.sqrt())
        } else { 1.0 };
        let step = if cos_angle < 0.0 && !s_buf.is_empty() {
            s_buf.clear(); y_buf.clear(); rho_buf.clear();
            forces.iter().map(|v| h0 * v).collect::<Vec<f64>>()
        } else {
            step
        };

        // Distance reset: if step > max_move, reset and use SD
        let max_disp = max_atom_motion(&step);
        let step = if max_disp > max_move && !s_buf.is_empty() {
            s_buf.clear(); y_buf.clear(); rho_buf.clear();
            let sd: Vec<f64> = forces.iter().map(|v| h0 * v).collect();
            let sd_disp = max_atom_motion(&sd);
            let sc = if sd_disp > max_move { max_move / sd_disp } else { 1.0 };
            sd.iter().map(|v| sc * v).collect::<Vec<f64>>()
        } else if max_disp > max_move {
            let sc = max_move / max_disp;
            step.iter().map(|v| sc * v).collect::<Vec<f64>>()
        } else {
            step
        };

        prev_x = Some(x.clone());
        prev_forces = Some(forces);
        for j in 0..n_dof { x[j] += step[j]; }
    }

    // Write JSONL
    let outfile = "petmad_minimize_comparison.jsonl";
    let mut f = std::fs::File::create(outfile).expect("Failed to create output file");

    for (i, (e, max_f)) in gp_result.energies.iter().zip(gp_max_fatom.iter()).enumerate() {
        writeln!(
            f,
            r#"{{"method":"gp_minimize","step":{},"energy":{},"max_fatom":{},"oracle_calls":{}}}"#,
            i, e, max_f, i + 1
        )
        .expect("Operation failed");
    }

    for (i, (oc, e, max_f)) in direct_data.iter().enumerate() {
        writeln!(
            f,
            r#"{{"method":"direct_lbfgs","step":{},"energy":{},"max_fatom":{},"oracle_calls":{}}}"#,
            i, e, max_f, oc
        )
        .expect("Operation failed");
    }

    writeln!(
        f,
        r#"{{"summary":true,"gp_calls":{},"gp_energy":{},"gp_converged":{},"direct_calls":{},"direct_energy":{},"direct_max_fatom":{},"conv_tol":{}}}"#,
        gp_result.oracle_calls,
        gp_result.e_final,
        gp_result.converged,
        direct_calls,
        direct_data.last().map(|d| d.1).unwrap_or(f64::NAN),
        direct_data.last().map(|d| d.2).unwrap_or(f64::NAN),
        gp_cfg.conv_tol
    )
    .expect("Operation failed");

    eprintln!("Output: {}", outfile);
}
