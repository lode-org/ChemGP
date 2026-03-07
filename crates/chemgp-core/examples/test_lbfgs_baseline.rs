//! Test baseline LBFGS convergence on LEPS surface.
//!
//! This test verifies that baseline LBFGS (without GP acceleration)
//! converges properly after removing the 0.1/g_norm step size cap.

use chemgp_core::lbfgs::LbfgsHistory;
use chemgp_core::potentials::{leps_energy_gradient, LEPS_REACTANT};

fn main() {
    let oracle = |x: &[f64]| -> (f64, Vec<f64>) { leps_energy_gradient(x) };
    let x_init = LEPS_REACTANT.to_vec();
    
    // Verify initial state
    let (e0, g0) = oracle(&x_init);
    let g0_norm: f64 = g0.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("Initial state:");
    println!("  E = {:.6} eV", e0);
    println!("  |G| = {:.6} eV/Å", g0_norm);
    println!();
    
    // Run baseline L-BFGS (matching the structure in minimize.rs inner loop)
    let mut x = x_init.clone();
    let mut lbfgs = LbfgsHistory::new(10);
    let mut prev_grad: Option<Vec<f64>> = None;
    let mut x_inner_prev = x.clone();
    let trust_radius = 0.1; // Default from MinimizationConfig
    
    println!("Running baseline L-BFGS (no GP)...");
    let mut converged = false;
    let mut final_norm = f64::NAN;
    let mut iterations = 0;
    
    for inner in 0..200 {
        let (_, g_pred) = oracle(&x);
        
        let g_norm: f64 = g_pred.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if g_norm < 1e-4 {
            converged = true;
            final_norm = g_norm;
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
        
        // NEW step size computation (without 0.1/g_norm cap)
        let step_size = if lbfgs.count > 0 {
            // L-BFGS: trust the direction, clip by trust radius
            (1.0f64).min(trust_radius / (dir_norm + 1e-30))
        } else {
            // Steepest descent: step = trust_radius / (2 * |dir|)
            trust_radius * 0.5 / (dir_norm + 1e-30)
        };
        
        // Apply step
        for j in 0..x.len() {
            x[j] += step_size * dir[j];
        }
        
        if inner % 20 == 0 {
            println!("  Iter {}: |G| = {:.6}, step = {:.4}, dir_norm = {:.4}", 
                     inner, g_norm, step_size, dir_norm);
        }
    }
    
    let (e_final, g_final) = oracle(&x);
    let g_final_norm: f64 = g_final.iter().map(|v| v * v).sum::<f64>().sqrt();
    
    println!();
    println!("Results:");
    println!("  Converged: {}", converged);
    println!("  Final iterations: {}", iterations);
    println!("  Final E = {:.6} eV", e_final);
    println!("  Final |G| = {:.6} eV/Å", g_final_norm);
    println!("  Energy change: ΔE = {:.6} eV", e0 - e_final);
    
    // Expected LEPS minimum energy (approximately)
    let leps_min_expected = -4.0; // Approximate value for LEPS reactant well
    
    if converged && g_final_norm < 1e-4 {
        println!();
        println!("✓ SUCCESS: Baseline L-BFGS converged!");
        std::process::exit(0);
    } else {
        println!();
        println!("✗ FAILURE: Baseline L-BFGS did not converge");
        println!("  Final gradient norm: {:.6} (target: <1e-4)", g_final_norm);
        std::process::exit(1);
    }
}
