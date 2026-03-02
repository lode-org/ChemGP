//! Smoke test for RPC oracle: single energy+forces call.
//!
//! Start eOn serve: eonclient -p lj --serve-port 12345
//! Then run: cargo run --features rgpot --example rpc_smoke_test

use chemgp_core::oracle::RpcOracle;

fn main() {
    let host = std::env::var("RGPOT_HOST").unwrap_or_else(|_| "localhost".into());
    let port: u16 = std::env::var("RGPOT_PORT")
        .unwrap_or_else(|_| "12345".into())
        .parse()
        .expect("RGPOT_PORT must be a valid port number");

    eprintln!("Connecting to {}:{}...", host, port);

    // Simple 2-atom Ar system for LJ
    let atomic_numbers = vec![18, 18]; // Ar-Ar
    let box_matrix = [20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0];

    let mut oracle = RpcOracle::new(&host, port, atomic_numbers, box_matrix)
        .expect("Failed to connect");

    // Two Ar atoms at r = 3.4 Angstrom (near LJ minimum)
    let positions = vec![0.0, 0.0, 0.0, 3.4, 0.0, 0.0];

    let (energy, gradient) = oracle.evaluate(&positions).expect("Evaluation failed");

    eprintln!("Energy: {:.6} eV", energy);
    eprintln!("Gradient: {:?}", gradient);
    eprintln!("Force on atom 1: [{:.6}, {:.6}, {:.6}]",
        -gradient[0], -gradient[1], -gradient[2]);
    eprintln!("Force on atom 2: [{:.6}, {:.6}, {:.6}]",
        -gradient[3], -gradient[4], -gradient[5]);

    // Sanity checks
    assert!(energy.is_finite(), "Energy should be finite");
    assert!(gradient.iter().all(|g| g.is_finite()), "Gradients should be finite");
    // Forces should be equal and opposite (Newton's 3rd law)
    let f_diff: f64 = (0..3).map(|i| (gradient[i] + gradient[3+i]).powi(2)).sum::<f64>().sqrt();
    assert!(f_diff < 1e-10, "Newton's 3rd law violated: f_diff = {}", f_diff);

    eprintln!("RPC smoke test PASSED");
}
