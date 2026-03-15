//! Inverse distance features and pair-type scheme for molecular kernels.
//!
//! Ports `invdist.jl`: compute_inverse_distances, build_feature_map, build_pair_scheme.

use std::collections::{BTreeMap, BTreeSet};

/// Compute inverse interatomic distance features (1/r) for all Moving-Moving
/// and Moving-Frozen atom pairs.
///
/// Features are ordered: MM upper triangle (j < i), then MF (all combos).
/// Total features: N_mov*(N_mov-1)/2 + N_mov*N_fro.
pub fn compute_inverse_distances(x_flat: &[f64], frozen_flat: &[f64]) -> Vec<f64> {
    assert!(x_flat.len().is_multiple_of(3), "Moving coordinates must be 3D");
    let n_mov = x_flat.len() / 3;
    let n_fro = frozen_flat.len() / 3;

    let n_mm = n_mov * (n_mov - 1) / 2;
    let n_mf = n_mov * n_fro;
    let mut features = Vec::with_capacity(n_mm + n_mf);

    // Moving-Moving pairs (upper triangle: j < i)
    for j in 0..n_mov {
        let (xj, yj, zj) = (x_flat[3 * j], x_flat[3 * j + 1], x_flat[3 * j + 2]);
        for i in (j + 1)..n_mov {
            let dx = x_flat[3 * i] - xj;
            let dy = x_flat[3 * i + 1] - yj;
            let dz = x_flat[3 * i + 2] - zj;
            let d2 = dx * dx + dy * dy + dz * dz;
            features.push(1.0 / (d2 + 1e-18).sqrt());
        }
    }

    // Moving-Frozen pairs
    if n_fro > 0 {
        for j in 0..n_mov {
            let (xj, yj, zj) = (x_flat[3 * j], x_flat[3 * j + 1], x_flat[3 * j + 2]);
            for k in 0..n_fro {
                let dx = xj - frozen_flat[3 * k];
                let dy = yj - frozen_flat[3 * k + 1];
                let dz = zj - frozen_flat[3 * k + 2];
                let d2 = dx * dx + dy * dy + dz * dz;
                features.push(1.0 / (d2 + 1e-18).sqrt());
            }
        }
    }

    features
}

/// Build the feature-to-parameter index map.
///
/// Maps each inverse distance feature to its pair-type parameter index.
pub fn build_feature_map(
    n_mov: usize,
    n_fro: usize,
    mov_types: &[usize],
    fro_types: &[usize],
    pair_map: &[Vec<usize>], // n_types x n_types
) -> Vec<usize> {
    let mut map_indices = Vec::new();

    // Moving-Moving
    for j in 0..n_mov.saturating_sub(1) {
        for i in (j + 1)..n_mov {
            let t1 = mov_types[j];
            let t2 = mov_types[i];
            map_indices.push(pair_map[t1][t2]);
        }
    }

    // Moving-Frozen
    if n_fro > 0 {
        for mt in mov_types.iter().take(n_mov) {
            for ft in fro_types.iter().take(n_fro) {
                map_indices.push(pair_map[*mt][*ft]);
            }
        }
    }

    map_indices
}

/// Result of `build_pair_scheme`.
#[derive(Debug, Clone)]
pub struct PairScheme {
    pub mov_types: Vec<usize>,
    pub fro_types: Vec<usize>,
    /// Symmetric matrix: pair_map[t1][t2] = parameter index (0-based)
    pub pair_map: Vec<Vec<usize>>,
    pub n_params: usize,
    pub species: Vec<i32>,
}

/// Build the pair-type mapping from atomic numbers, matching C++ gpr_optim convention.
///
/// Only pair types that actually appear in the feature set get assigned
/// parameter indices.
pub fn build_pair_scheme(
    atomic_numbers_mov: &[i32],
    atomic_numbers_fro: &[i32],
) -> PairScheme {
    let mut all_species: Vec<i32> = atomic_numbers_mov
        .iter()
        .chain(atomic_numbers_fro.iter())
        .copied()
        .collect();
    all_species.sort();
    all_species.dedup();

    let species_to_type: BTreeMap<i32, usize> = all_species
        .iter()
        .enumerate()
        .map(|(i, &z)| (z, i))
        .collect();

    let mov_types: Vec<usize> = atomic_numbers_mov
        .iter()
        .map(|z| species_to_type[z])
        .collect();
    let fro_types: Vec<usize> = atomic_numbers_fro
        .iter()
        .map(|z| species_to_type[z])
        .collect();

    let n_types = all_species.len();
    let n_mov = mov_types.len();
    let n_fro = fro_types.len();

    // Scan which pair types actually appear
    let mut used_pairs = BTreeSet::new();

    // Moving-Moving (upper triangle)
    for j in 0..n_mov.saturating_sub(1) {
        for i in (j + 1)..n_mov {
            let (t1, t2) = if mov_types[j] <= mov_types[i] {
                (mov_types[j], mov_types[i])
            } else {
                (mov_types[i], mov_types[j])
            };
            used_pairs.insert((t1, t2));
        }
    }

    // Moving-Frozen
    for mt in mov_types.iter().take(n_mov) {
        for ft in fro_types.iter().take(n_fro) {
            let (t1, t2) = if *mt <= *ft {
                (*mt, *ft)
            } else {
                (*ft, *mt)
            };
            used_pairs.insert((t1, t2));
        }
    }

    let sorted_pairs: Vec<(usize, usize)> = used_pairs.into_iter().collect();
    let n_params = sorted_pairs.len();
    let pair_to_idx: BTreeMap<(usize, usize), usize> = sorted_pairs
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, i))
        .collect();

    let mut pair_map = vec![vec![0usize; n_types]; n_types];
    for (&(t1, t2), &idx) in &pair_to_idx {
        pair_map[t1][t2] = idx;
        pair_map[t2][t1] = idx;
    }

    PairScheme {
        mov_types,
        fro_types,
        pair_map,
        n_params,
        species: all_species,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invdist_3atom() {
        // 3 atoms along x-axis at 0, 1, 2
        let x = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let features = compute_inverse_distances(&x, &[]);
        assert_eq!(features.len(), 3); // 3 pairs
        assert!((features[0] - 1.0).abs() < 1e-10); // d(0,1)=1
        assert!((features[1] - 0.5).abs() < 1e-10); // d(0,2)=2
        assert!((features[2] - 1.0).abs() < 1e-10); // d(1,2)=1
    }

    #[test]
    fn test_pair_scheme_hcn() {
        // HCN: H=1, C=6, N=7
        let scheme = build_pair_scheme(&[1, 6, 7], &[]);
        assert_eq!(scheme.n_params, 3); // H-C, H-N, C-N (no self-pairs)
        assert_eq!(scheme.species, vec![1, 6, 7]);
    }

    #[test]
    fn test_pair_scheme_cu2h() {
        // 2 Cu (Z=29) moving, 1 H (Z=1) frozen
        let scheme = build_pair_scheme(&[29, 29], &[1]);
        // Pairs: Cu-Cu (MM), Cu-H (MF) = 2 pair types
        assert_eq!(scheme.n_params, 2);
    }
}
