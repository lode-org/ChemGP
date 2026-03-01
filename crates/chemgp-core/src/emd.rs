//! Intensive Earth Mover's Distance (iEMD).
//!
//! Ports `distances_emd.jl`.

/// Intensive Earth Mover's Distance between two molecular configurations.
///
/// For each element type, solves the assignment problem (brute-force for <=8,
/// greedy otherwise) and reports max per-type mean displacement.
pub fn emd_distance(x1: &[f64], x2: &[f64], atom_types: &[i32]) -> f64 {
    assert_eq!(x1.len(), x2.len());
    assert!(x1.len() % 3 == 0);
    let n_atoms = x1.len() / 3;

    let types: Vec<i32> = if atom_types.is_empty() {
        vec![1; n_atoms]
    } else {
        assert_eq!(atom_types.len(), n_atoms);
        atom_types.to_vec()
    };

    let mut unique_types: Vec<i32> = types.clone();
    unique_types.sort();
    unique_types.dedup();

    let mut max_mean_disp = 0.0f64;

    for &t in &unique_types {
        let idx: Vec<usize> = types
            .iter()
            .enumerate()
            .filter(|(_, &ti)| ti == t)
            .map(|(i, _)| i)
            .collect();
        let nt = idx.len();

        if nt == 1 {
            let i = idx[0];
            let d = ((x1[3 * i] - x2[3 * i]).powi(2)
                + (x1[3 * i + 1] - x2[3 * i + 1]).powi(2)
                + (x1[3 * i + 2] - x2[3 * i + 2]).powi(2))
            .sqrt();
            max_mean_disp = max_mean_disp.max(d);
            continue;
        }

        // Build cost matrix
        let mut cost = vec![vec![0.0; nt]; nt];
        for i in 0..nt {
            let ai = idx[i];
            for j in 0..nt {
                let aj = idx[j];
                cost[i][j] = ((x1[3 * ai] - x2[3 * aj]).powi(2)
                    + (x1[3 * ai + 1] - x2[3 * aj + 1]).powi(2)
                    + (x1[3 * ai + 2] - x2[3 * aj + 2]).powi(2))
                .sqrt();
            }
        }

        let min_cost = if nt <= 8 {
            bruteforce_assignment(&cost)
        } else {
            greedy_assignment(&cost)
        };

        max_mean_disp = max_mean_disp.max(min_cost / nt as f64);
    }

    max_mean_disp
}

fn bruteforce_assignment(cost: &[Vec<f64>]) -> f64 {
    let n = cost.len();
    if n == 1 {
        return cost[0][0];
    }

    let perms = permutations(n);
    let mut best = f64::INFINITY;
    for p in &perms {
        let c: f64 = (0..n).map(|i| cost[i][p[i]]).sum();
        best = best.min(c);
    }
    best
}

fn greedy_assignment(cost: &[Vec<f64>]) -> f64 {
    let n = cost.len();
    let mut assigned = vec![false; n];
    let mut total = 0.0;

    for i in 0..n {
        let mut best_j = 0;
        let mut best_c = f64::INFINITY;
        for j in 0..n {
            if !assigned[j] && cost[i][j] < best_c {
                best_c = cost[i][j];
                best_j = j;
            }
        }
        assigned[best_j] = true;
        total += best_c;
    }

    total
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 1 {
        return vec![vec![0]];
    }
    let sub = permutations(n - 1);
    let mut result = Vec::new();
    for p in &sub {
        for i in 0..n {
            let mut new_p = p.clone();
            new_p.insert(i, n - 1);
            result.push(new_p);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emd_identical() {
        let x = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        assert!(emd_distance(&x, &x, &[]) < 1e-15);
    }

    #[test]
    fn test_emd_permutation_invariant() {
        // Two atoms of same type, swapped
        let x1 = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let x2 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // swapped
        let d = emd_distance(&x1, &x2, &[1, 1]);
        assert!(d < 1e-15, "EMD should be 0 for permuted identical configs, got {}", d);
    }
}
