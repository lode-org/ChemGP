//! Compare CartesianSE vs MolInvDistSE kernel on LEPS minimization.
//!
//! Tutorial T3 (Molecular Kernels): demonstrates that the invariant
//! inverse-distance kernel converges in far fewer oracle calls than
//! the naive Cartesian kernel on a molecular potential.
//!
//! Outputs `leps_kernel_comparison.jsonl` for plotting.

use chemgp_core::kernel::{CartesianSE, Kernel, MolInvDistSE};
use chemgp_core::minimize::{gp_minimize, MinimizationConfig};
use chemgp_core::potentials::{leps_energy_gradient, LEPS_REACTANT};
use chemgp_core::trust::TrustMetric;

use std::io::Write;

fn main() {
    let oracle = |x: &[f64]| -> (f64, Vec<f64>) { leps_energy_gradient(x) };
    let x_init = LEPS_REACTANT.to_vec();

    let outfile = "leps_kernel_comparison.jsonl";
    let mut f = std::fs::File::create(outfile).expect("Failed to create output file");

    // --- MolInvDistSE (invariant kernel) ---
    let kernel_mol = Kernel::MolInvDist(MolInvDistSE::isotropic(1.0, 1.0, vec![]));
    let mut cfg_mol = MinimizationConfig::default();
    cfg_mol.max_iter = 100;
    cfg_mol.max_oracle_calls = 50;
    cfg_mol.conv_tol = 0.01;
    cfg_mol.verbose = false;

    let result_mol = gp_minimize(&oracle, &x_init, &kernel_mol, &cfg_mol, None);

    let mut mol_forces: Vec<f64> = Vec::new();
    for pt in &result_mol.trajectory {
        let (_, g) = oracle(pt);
        let max_f = g.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        mol_forces.push(max_f);
    }

    for (i, (e, max_f)) in result_mol.energies.iter().zip(mol_forces.iter()).enumerate() {
        writeln!(f, r#"{{"kernel":"MolInvDistSE","step":{},"energy":{},"max_fatom":{},"oracle_calls":{}}}"#,
            i, e, max_f, i + 1).expect("Failed to write to output file");
    }

    eprintln!("MolInvDistSE: {} oracle calls, E = {:.6}, converged = {}",
        result_mol.oracle_calls, result_mol.e_final, result_mol.converged);

    // --- CartesianSE (non-invariant kernel) ---
    // LEPS is 3 atoms in 3D = 9 Cartesian coords.
    // CartesianSE treats these as raw input features.
    let kernel_cart = Kernel::Cartesian(CartesianSE::new(100.0, 2.0));
    let mut cfg_cart = MinimizationConfig::default();
    cfg_cart.max_iter = 100;
    cfg_cart.max_oracle_calls = 80;
    cfg_cart.conv_tol = 0.01;
    cfg_cart.trust_metric = TrustMetric::Euclidean;
    cfg_cart.verbose = false;

    let result_cart = gp_minimize(&oracle, &x_init, &kernel_cart, &cfg_cart, None);

    let mut cart_forces: Vec<f64> = Vec::new();
    for pt in &result_cart.trajectory {
        let (_, g) = oracle(pt);
        let max_f = g.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        cart_forces.push(max_f);
    }

    for (i, (e, max_f)) in result_cart.energies.iter().zip(cart_forces.iter()).enumerate() {
        writeln!(f, r#"{{"kernel":"CartesianSE","step":{},"energy":{},"max_fatom":{},"oracle_calls":{}}}"#,
            i, e, max_f, i + 1).expect("Failed to write to output file");
    }

    eprintln!("CartesianSE:  {} oracle calls, E = {:.6}, converged = {}",
        result_cart.oracle_calls, result_cart.e_final, result_cart.converged);

    eprintln!("\nThe MolInvDistSE kernel exploits rotational and translational");
    eprintln!("invariance via inverse distance features, converging faster on");
    eprintln!("molecular potentials where the physics is invariant under rigid");
    eprintln!("body motion.");
    eprintln!("Output: {}", outfile);
}
