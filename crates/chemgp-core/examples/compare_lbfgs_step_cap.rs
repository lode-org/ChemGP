//! Compare baseline L-BFGS with and without the 0.1/g_norm step cap.
//!
//! This demonstrates that removing the cap allows baseline L-BFGS to converge.

use chemgp_core::lbfgs::LbfgsHistory;
use chemgp_core::potentials::{leps_energy_gradient, LEPS_REACTANT};

fn run_lbfgs_with_cap(trust_radius: f64, with_cap: bool) -> (bool, usize, f64, f64) {
    let oracle = |x: &[f64]| -> (f64, Vec<f64>) { leps_energy_gradient(x) };
    let x_init = LEPS_REACTANT.to_vec();
    
    let mut x = x_init.clone();
    let mut lbfgs = LbfgsHistory::new(10);
    let mut prev_grad: Option<Vec<f64>> = None;
    let mut x_inner_prev = x.clone();
    
    let mut converged = false;
    let mut iterations = 0;
    
    for inner in 0..200 {
        let (_, g_pred) = oracle(&x);
        let g_norm: f64 = g_pred.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if g_norm < 1e-4 {
            converged = true;
            iterations = inner;
            break;
        }
        
        // L-BFGS direction
        if let Some(ref pg) = prev_grad {
            let s: Vec<f64> = x.iter().zip(x_inner_prev.iter()).map(|(a, b)| a - b).collect();
            let y: Vec<f64> = g_pred.iter().zip(pg.iter()).map(|(a, b)| a - b).collect();
            lbfgs.push_pair(s, y);
        }
        prev_grad = Some(g_pred.clone());
        x_inner_prev = x.clone();
        
        let dir = lbfgs.compute_direction(&g_pred);
        let dir_norm: f64 = dir.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        // OLD vs NEW step size computation
        let step_size = if lbfgs.count > 0 {
            (1.0f64).min(trust_radius / (dir_norm + 1e-30))
        } else {
            let base = trust_radius * 0.5 / (dir_norm + 1e-30);
            if with_cap {
                // OLD: Artificially capped
                base.min(0.1 / g_norm)
            } else {
                // NEW: No cap
                base
            }
        };
        
        for j in 0..x.len() {
            x[j] += step_size * dir[j];
        }
    }
    
    let (e_final, g_final) = oracle(&x);
    let g_final_norm: f64 = g_final.iter().map(|v| v * v).sum::<f64>().sqrt();
    
    (converged, iterations, e_final, g_final_norm)
}

fn main() {
    println!("Baseline L-BFGS Convergence Test");
    println!("=================================\n");
    
    let trust_radius = 0.1;
    
    println!("Configuration:");
    println!("  trust_radius = {} Å", trust_radius);
    println!("  max_iterations = 200");
    println!("  convergence_tol = 1e-4 eV/Å\n");
    
    // Test with OLD behavior (with cap)
    println!("OLD behavior (with 0.1/g_norm cap):");
    let (conv_old, iter_old, _e_old, g_old) = run_lbfgs_with_cap(trust_radius, true);
    println!("  Converged: {}", conv_old);
    println!("  Iterations: {}", if iter_old > 0 { iter_old.to_string() } else { "N/A".to_string() });
    println!("  Final |G| = {:.6} eV/Å", g_old);
    println!();
    
    // Test with NEW behavior (no cap)
    println!("NEW behavior (no cap):");
    let (conv_new, iter_new, _e_new, g_new) = run_lbfgs_with_cap(trust_radius, false);
    println!("  Converged: {}", conv_new);
    println!("  Iterations: {}", iter_new);
    println!("  Final |G| = {:.6} eV/Å", g_new);
    println!();
    
    // Summary
    println!("Summary:");
    if !conv_old && conv_new {
        println!("✓ FIX VERIFIED: Removing the cap enables convergence!");
        println!("  OLD: Did not converge (|G| = {:.6})", g_old);
        println!("  NEW: Converged in {} iterations (|G| = {:.6})", iter_new, g_new);
        std::process::exit(0);
    } else if conv_old && conv_new {
        println!("⚠ Both versions converged");
        println!("  OLD: {} iterations", iter_old);
        println!("  NEW: {} iterations", iter_new);
        if iter_new < iter_old {
            println!("  Speedup: {:.1}x", iter_old as f64 / iter_new as f64);
        }
        std::process::exit(0);
    } else {
        println!("✗ UNEXPECTED: Neither version converged");
        std::process::exit(1);
    }
}
