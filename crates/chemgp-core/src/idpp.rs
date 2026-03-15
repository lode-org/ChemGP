//! IDPP and S-IDPP path interpolation.
//!
//! Ports `idpp.jl`: Image Dependent Pair Potential for generating initial
//! NEB paths with smooth pairwise distance variation.
//!
//! Reference: Smidstrup et al., J. Chem. Phys. 140, 214106 (2014).

use crate::neb_path::linear_interpolation;
use crate::optim_step::OptimState;

/// Configuration for IDPP/S-IDPP path interpolation.
#[derive(Clone, Copy)]
pub struct IdppConfig {
    pub n_images: usize,
    pub n_coords_per_atom: usize,
    pub max_iter: usize,
    pub max_move: f64,
    pub force_tol: f64,
    pub lbfgs_memory: usize,
}

/// IDPP interpolation: per-image independent optimization.
pub fn idpp_interpolation(
    x_start: &[f64],
    x_end: &[f64],
    cfg: &IdppConfig,
) -> Vec<Vec<f64>> {
    let IdppConfig { n_images, n_coords_per_atom, max_iter, max_move, force_tol, lbfgs_memory } = *cfg;
    let mut images = linear_interpolation(x_start, x_end, n_images);
    let n_atoms = x_start.len() / n_coords_per_atom;

    let d_start = pairwise_distances(x_start, n_atoms, n_coords_per_atom);
    let d_end = pairwise_distances(x_end, n_atoms, n_coords_per_atom);

    for (img_idx, image) in images.iter_mut().enumerate().take(n_images - 1).skip(1) {
        let xi = img_idx as f64 / (n_images - 1) as f64;
        let d_target: Vec<f64> = d_start
            .iter()
            .zip(d_end.iter())
            .map(|(&a, &b)| (1.0 - xi) * a + xi * b)
            .collect();

        let mut optim = OptimState::new(lbfgs_memory);
        let mut x = image.clone();

        for _ in 0..max_iter {
            let (_, force) =
                idpp_energy_force(&x, &d_target, n_atoms, n_coords_per_atom);
            if max_atom_force(&force, n_atoms, n_coords_per_atom) < force_tol {
                break;
            }
            let disp = optim.step(&x, &force, max_move, n_coords_per_atom);
            for j in 0..x.len() {
                x[j] += disp[j];
            }
        }

        *image = x;
    }

    images
}

/// S-IDPP: sequential growth from both endpoints with collective relaxation.
pub fn sidpp_interpolation(
    x_start: &[f64],
    x_end: &[f64],
    cfg: &IdppConfig,
    spring_constant: f64,
    growth_alpha: f64,
) -> Vec<Vec<f64>> {
    let n_images = cfg.n_images;
    let n_coords_per_atom = cfg.n_coords_per_atom;
    let n_atoms = x_start.len() / n_coords_per_atom;

    let d_init = pairwise_distances(x_start, n_atoms, n_coords_per_atom);
    let d_final = pairwise_distances(x_end, n_atoms, n_coords_per_atom);

    let mut path = vec![x_start.to_vec(), x_end.to_vec()];
    let n_target = n_images.saturating_sub(2);
    let mut n_left = 0;
    let mut n_right = 0;
    let mut n_intermediate = 0;

    while n_intermediate < n_target {
        if n_intermediate < n_target {
            let frontier = &path[n_left];
            let next = &path[n_left + 1];
            let new_img: Vec<f64> = frontier
                .iter()
                .zip(next.iter())
                .map(|(&a, &b)| (1.0 - growth_alpha) * a + growth_alpha * b)
                .collect();
            path.insert(n_left + 1, new_img);
            n_left += 1;
            n_intermediate += 1;
        }

        if n_intermediate < n_target {
            let right_idx = path.len() - 1 - n_right;
            let frontier = path[right_idx].clone();
            let prev = path[right_idx - 1].clone();
            let new_img: Vec<f64> = frontier
                .iter()
                .zip(prev.iter())
                .map(|(&a, &b)| (1.0 - growth_alpha) * a + growth_alpha * b)
                .collect();
            path.insert(right_idx, new_img);
            n_right += 1;
            n_intermediate += 1;
        }

        relax_collective_idpp(
            &mut path, &d_init, &d_final, n_atoms, cfg, spring_constant,
        );
    }

    // Final full-path relaxation
    let final_cfg = IdppConfig { max_iter: 500, ..*cfg };
    relax_collective_idpp(
        &mut path, &d_init, &d_final, n_atoms, &final_cfg, spring_constant,
    );

    path
}

/// Collective IDPP-NEB relaxation.
fn relax_collective_idpp(
    path: &mut [Vec<f64>],
    d_init: &[f64],
    d_final: &[f64],
    n_atoms: usize,
    cfg: &IdppConfig,
    spring_constant: f64,
) {
    let IdppConfig { n_coords_per_atom: n_coords, max_iter, max_move, force_tol, lbfgs_memory, .. } = *cfg;
    let n_images = path.len();
    let n_mov = if n_images >= 2 { n_images - 2 } else { return };
    let d = path[0].len();

    let mut optim = OptimState::new(lbfgs_memory);

    for _ in 0..max_iter {
        let forces =
            collective_idpp_forces(path, d_init, d_final, n_atoms, n_coords, spring_constant);

        let mut cur_force = Vec::with_capacity(n_mov * d);
        for f in forces.iter().take(n_mov + 1).skip(1) {
            cur_force.extend_from_slice(f);
        }

        let total_atoms = cur_force.len() / n_coords;
        if max_atom_force(&cur_force, total_atoms, n_coords) < force_tol {
            break;
        }

        let mut cur_x = Vec::with_capacity(n_mov * d);
        for p in path.iter().take(n_mov + 1).skip(1) {
            cur_x.extend_from_slice(p);
        }

        let disp = optim.step(&cur_x, &cur_force, max_move, n_coords);
        let new_x: Vec<f64> = cur_x.iter().zip(disp.iter()).map(|(a, b)| a + b).collect();

        for i in 0..n_mov {
            let off = i * d;
            path[i + 1] = new_x[off..off + d].to_vec();
        }
    }
}

/// Perpendicular IDPP forces + spring forces parallel to tangent.
fn collective_idpp_forces(
    path: &[Vec<f64>],
    d_init: &[f64],
    d_final: &[f64],
    n_atoms: usize,
    n_coords: usize,
    k_spring: f64,
) -> Vec<Vec<f64>> {
    let n_images = path.len();
    let d = path[0].len();
    let mut forces: Vec<Vec<f64>> = (0..n_images).map(|_| vec![0.0; d]).collect();

    for i in 1..n_images - 1 {
        let xi = i as f64 / (n_images - 1) as f64;
        let d_target: Vec<f64> = d_init
            .iter()
            .zip(d_final.iter())
            .map(|(&a, &b)| (1.0 - xi) * a + xi * b)
            .collect();

        let (_, f_idpp) = idpp_energy_force(&path[i], &d_target, n_atoms, n_coords);

        // Simple tangent: (next - prev), normalized
        let tau_raw: Vec<f64> = path[i + 1]
            .iter()
            .zip(path[i - 1].iter())
            .map(|(a, b)| a - b)
            .collect();
        let tn: f64 = tau_raw.iter().map(|x| x * x).sum::<f64>().sqrt();
        let tau: Vec<f64> = if tn > 1e-18 {
            tau_raw.iter().map(|x| x / tn).collect()
        } else {
            vec![0.0; d]
        };

        // Perpendicular IDPP + parallel spring
        let f_dot_t: f64 = f_idpp.iter().zip(tau.iter()).map(|(f, t)| f * t).sum();
        let d_next: f64 = path[i + 1]
            .iter()
            .zip(path[i].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let d_prev: f64 = path[i]
            .iter()
            .zip(path[i - 1].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        for j in 0..d {
            let f_perp = f_idpp[j] - f_dot_t * tau[j];
            let f_spr = k_spring * (d_next - d_prev) * tau[j];
            forces[i][j] = f_perp + f_spr;
        }
    }

    forces
}

/// Pairwise distances as a flat upper-triangular vector.
pub fn pairwise_distances(x: &[f64], n_atoms: usize, n_coords: usize) -> Vec<f64> {
    let mut d = vec![0.0; n_atoms * n_atoms];
    for i in 0..n_atoms {
        for j in i + 1..n_atoms {
            let r: f64 = (0..n_coords)
                .map(|k| (x[i * n_coords + k] - x[j * n_coords + k]).powi(2))
                .sum::<f64>()
                .sqrt();
            d[i * n_atoms + j] = r;
            d[j * n_atoms + i] = r;
        }
    }
    d
}

/// IDPP energy and force: E = 0.5 * sum_{i<j} (1/r^4) * (r - d_target)^2.
fn idpp_energy_force(
    x: &[f64],
    d_target: &[f64],
    n_atoms: usize,
    n_coords: usize,
) -> (f64, Vec<f64>) {
    let mut energy = 0.0;
    let mut force = vec![0.0; x.len()];

    for i in 0..n_atoms {
        for j in i + 1..n_atoms {
            let dr: Vec<f64> = (0..n_coords)
                .map(|k| x[i * n_coords + k] - x[j * n_coords + k])
                .collect();
            let r: f64 = dr.iter().map(|d| d * d).sum::<f64>().sqrt().max(1e-4);

            let diff = r - d_target[i * n_atoms + j];
            let r4 = r.powi(4);

            energy += 0.5 * diff * diff / r4;

            let de_dr = diff * (1.0 - 2.0 * diff / r) / r4;
            for k in 0..n_coords {
                let f = -de_dr / r * dr[k];
                force[i * n_coords + k] += f;
                force[j * n_coords + k] -= f;
            }
        }
    }

    (energy, force)
}

/// Max per-atom force norm.
fn max_atom_force(force: &[f64], n_atoms: usize, n_coords: usize) -> f64 {
    let mut max_f = 0.0f64;
    for a in 0..n_atoms {
        let off = a * n_coords;
        let f: f64 = (0..n_coords).map(|d| force[off + d].powi(2)).sum::<f64>().sqrt();
        max_f = max_f.max(f);
    }
    max_f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idpp_preserves_endpoints() {
        let start = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let end = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let images = idpp_interpolation(&start, &end, &IdppConfig {
            n_images: 5, n_coords_per_atom: 3, max_iter: 100, max_move: 0.1, force_tol: 0.01, lbfgs_memory: 10,
        });
        assert_eq!(images.len(), 5);
        assert!((images[0][3] - 1.0).abs() < 1e-10);
        assert!((images[4][3] - 2.0).abs() < 1e-10);
    }
}
