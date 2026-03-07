//! Compare baseline L-BFGS on Muller-Brown surface with and without step cap.
//!
//! Muller-Brown is a challenging 2D potential with multiple minima and saddle points.

use chemgp_core::lbfgs::LbfgsHistory;
use chemgp_core::potentials::{muller_brown_energy_gradient, MULLER_BROWN_MINIMA};

fn run_lbfgs_mb(x_init: &[f64], trust_radius: f64, with_cap: bool, max_iter: usize) -> (bool, usize, f64, f64, Vec<f64>) {
    let oracle = |x: &[f64]| -> (f64, Vec<f64>) { muller_brown_energy_gradient(x) };
    
    let mut x = x_init.to_vec();
    let mut lbfgs = LbfgsHistory::new(10);
    let mut prev_grad: Option<Vec<f64>> = None;
    let mut x_inner_prev = x.clone();
    let mut trajectory: Vec<f64> = Vec::new();
    
    let mut converged = false;
    let mut iterations = 0;
    
    for inner in 0..max_iter {
        let (e, g_pred) = oracle(&x);
        let g_norm: f64 = g_pred.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        trajectory.push(e);
        
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
                base.min(0.1 / g_norm)
            } else {
                base
            }
        };
        
        for j in 0..x.len() {
            x[j] += step_size * dir[j];
        }
    }
    
    let (e_final, g_final) = oracle(&x);
    let g_final_norm: f64 = g_final.iter().map(|v| v * v).sum::<f64>().sqrt();
    
    // Find nearest minimum
    let (min_idx, dist) = MULLER_BROWN_MINIMA
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let d: f64 = x.iter().zip(m.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
            (i, d)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    
    println!("  Converged to minimum {} (distance = {:.4})", min_idx, dist);
    
    (converged, iterations, e_final, g_final_norm, trajectory)
}

fn main() {
    println!("Muller-Brown Baseline L-BFGS Test");
    println!("==================================\n");
    
    let trust_radius = 0.3;
    let max_iter = 500;
    
    // Test from different starting points
    let test_points = vec![
        (vec![-0.5, 1.5], "Near minimum 1"),
        (vec![0.4, 0.2], "Between minima"),
        (vec![-1.0, 0.5], "Far from minima"),
    ];
    
    for (x_init, description) in test_points {
        println!("Starting point: {} {:?}", description, x_init);
        println!("  trust_radius = {} Å", trust_radius);
        println!();
        
        // OLD behavior
        println!("OLD (with 0.1/g_norm cap):");
        let (conv_old, iter_old, e_old, g_old, _traj_old) = 
            run_lbfgs_mb(&x_init, trust_radius, true, max_iter);
        println!("  Converged: {}", conv_old);
        println!("  Iterations: {}", if iter_old > 0 { iter_old.to_string() } else { "N/A".to_string() });
        println!("  Final E = {:.6}, |G| = {:.6}", e_old, g_old);
        println!();
        
        // NEW behavior
        println!("NEW (no cap):");
        let (conv_new, iter_new, e_new, g_new, _traj_new) = 
            run_lbfgs_mb(&x_init, trust_radius, false, max_iter);
        println!("  Converged: {}", conv_new);
        println!("  Iterations: {}", iter_new);
        println!("  Final E = {:.6}, |G| = {:.6}", e_new, g_new);
        println!();
        
        // Compare
        if conv_old != conv_new {
            if conv_new {
                println!("✓ FIX VERIFIED: NEW converges, OLD does not!");
            } else {
                println!("✗ REGRESSION: OLD converges, NEW does not!");
            }
        } else if conv_new {
            if iter_new < iter_old {
                println!("✓ NEW is {:.1}x faster", iter_old as f64 / iter_new as f64);
            } else {
                println!("⚠ Both converge, OLD is {:.1}x faster", iter_new as f64 / iter_old as f64);
            }
        }
        println!();
        println!("{}", "=".repeat(50));
        println!();
    }
}
