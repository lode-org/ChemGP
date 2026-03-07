//! FPS selection from candidate pool on LEPS (Fig 11).
//!
//! Generates a set of candidate configurations by evaluating the LEPS surface
//! along the NEB path at multiple perturbations, then selects a subset via FPS.
//! Outputs candidates + selection status for PCA visualization.

use chemgp_core::potentials::{leps_energy_gradient, LEPS_PRODUCT, LEPS_REACTANT};
use chemgp_core::sampling::farthest_point_sampling;

use std::io::Write;

fn main() {
    // Generate candidates: interpolate between reactant/product + perturbations
    let n_interp = 12;
    let n_perturb = 4;
    let perturb_scale = 0.05;
    let dim = 9;

    let mut candidates: Vec<Vec<f64>> = Vec::new();
    let mut rng_state: u64 = 42;

    // Simple LCG for reproducible perturbations
    let next_rand = |state: &mut u64| -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    };

    // Linear interpolation + perturbations
    for i in 0..n_interp {
        let t = i as f64 / (n_interp - 1) as f64;
        let base: Vec<f64> = LEPS_REACTANT
            .iter()
            .zip(LEPS_PRODUCT.iter())
            .map(|(&a, &b)| a + t * (b - a))
            .collect();
        candidates.push(base.clone());

        for _ in 0..n_perturb {
            let perturbed: Vec<f64> = base
                .iter()
                .map(|&v| v + perturb_scale * next_rand(&mut rng_state))
                .collect();
            candidates.push(perturbed);
        }
    }

    let n_cand = candidates.len();
    let n_select = 20;

    eprintln!("Generated {} candidates, selecting {} via FPS", n_cand, n_select);

    // Flatten candidates for FPS
    let flat_cand: Vec<f64> = candidates.iter().flat_map(|c| c.iter().copied()).collect();

    // Seed FPS with reactant configuration
    let seed = LEPS_REACTANT.to_vec();
    let euclidean = |a: &[f64], b: &[f64]| -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
    };

    let selected_idx = farthest_point_sampling(
        &flat_cand, dim, n_cand, &seed, 1, n_select, &euclidean,
    );

    let mut is_selected = vec![false; n_cand];
    for &idx in &selected_idx {
        is_selected[idx] = true;
    }

    // Write JSONL
    let outfile = "leps_fps.jsonl";
    let mut f = std::fs::File::create(outfile).expect("Failed to create output file");

    writeln!(
        f,
        r#"{{"type":"fps_meta","n_candidates":{},"n_selected":{},"feature_dim":{}}}"#,
        n_cand, selected_idx.len(), dim
    )
    .expect("Operation failed");

    for (idx, (cand, &sel)) in candidates.iter().zip(is_selected.iter()).enumerate() {
        // Compute rAB, rBC for labeling
        let rab = (cand[3] - cand[0]).hypot((cand[4] - cand[1]).hypot(cand[5] - cand[2]));
        let rbc = (cand[6] - cand[3]).hypot((cand[7] - cand[4]).hypot(cand[8] - cand[5]));
        let (e, _) = leps_energy_gradient(cand);

        // Flatten features as JSON array
        let feat_str: Vec<String> = cand.iter().map(|v| format!("{:.6}", v)).collect();
        writeln!(
            f,
            r#"{{"type":"candidate","idx":{},"selected":{},"rAB":{},"rBC":{},"energy":{},"features":[{}]}}"#,
            idx, sel, rab, rbc, e, feat_str.join(",")
        )
        .expect("Operation failed");
    }

    eprintln!("Output: {}", outfile);
}
