//! Farthest Point Sampling (FPS) and subset selection.
//!
//! Ports `sampling.jl`.

use crate::types::TrainingData;

/// Select `n_select` points from `candidates` farthest from `selected`.
///
/// candidates and selected are column-major: D*N flat arrays.
/// Returns indices into candidates.
pub fn farthest_point_sampling<F>(
    candidates: &[f64],
    dim: usize,
    n_cand: usize,
    selected: &[f64],
    n_sel: usize,
    n_select: usize,
    distance_fn: &F,
) -> Vec<usize>
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    let mut selected_indices = Vec::new();
    let mut min_dists = vec![f64::INFINITY; n_cand];

    // Pre-compute min distances from each candidate to selected set
    for i in 0..n_cand {
        let ci = &candidates[i * dim..(i + 1) * dim];
        for j in 0..n_sel {
            let sj = &selected[j * dim..(j + 1) * dim];
            let d = distance_fn(ci, sj);
            min_dists[i] = min_dists[i].min(d);
        }
    }

    for _ in 0..n_select.min(n_cand) {
        let best_idx = min_dists
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0);  // Safe fallback

        if min_dists[best_idx] <= 0.0 || min_dists[best_idx] == f64::NEG_INFINITY {
            break;
        }

        selected_indices.push(best_idx);
        let new_point_start = best_idx * dim;
        let new_point = &candidates[new_point_start..new_point_start + dim];

        min_dists[best_idx] = f64::NEG_INFINITY;

        for i in 0..n_cand {
            if min_dists[i] > 0.0 {
                let ci = &candidates[i * dim..(i + 1) * dim];
                let d = distance_fn(ci, new_point);
                min_dists[i] = min_dists[i].min(d);
            }
        }
    }

    selected_indices
}

/// Select a subset of training data for hyperparameter optimization.
///
/// Always includes the n_latest most recent points, fills rest via FPS.
/// Returns column indices into td.
pub fn select_optim_subset<F>(
    td: &TrainingData,
    _x_current: &[f64],
    n_select: usize,
    n_latest: usize,
    distance_fn: &F,
) -> Vec<usize>
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    let n = td.npoints();
    if n_select == 0 || n_select >= n {
        return (0..n).collect();
    }

    let n_latest = n_latest.min(n).min(n_select);
    let latest_start = n - n_latest;
    let latest_idx: Vec<usize> = (latest_start..n).collect();

    let n_fps = n_select - n_latest;
    if n_fps == 0 {
        return latest_idx;
    }

    // Candidates: everything except latest
    let cand_idx: Vec<usize> = (0..latest_start).collect();
    if cand_idx.is_empty() {
        return latest_idx;
    }

    // Build flat arrays
    let dim = td.dim;
    let mut cand_data = Vec::with_capacity(cand_idx.len() * dim);
    for &i in &cand_idx {
        cand_data.extend_from_slice(td.col(i));
    }
    let mut sel_data = Vec::with_capacity(latest_idx.len() * dim);
    for &i in &latest_idx {
        sel_data.extend_from_slice(td.col(i));
    }

    let fps_idx = farthest_point_sampling(
        &cand_data,
        dim,
        cand_idx.len(),
        &sel_data,
        latest_idx.len(),
        n_fps,
        distance_fn,
    );

    let mut result: Vec<usize> = latest_idx;
    for &fi in &fps_idx {
        result.push(cand_idx[fi]);
    }
    result.sort();
    result
}

/// Prune training data to keep at most max_points closest to x_current.
pub fn prune_training_data<F>(
    td: &mut TrainingData,
    x_current: &[f64],
    max_points: usize,
    distance_fn: &F,
) -> usize
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    let n = td.npoints();
    if max_points == 0 || n <= max_points {
        return 0;
    }

    let mut dists: Vec<(usize, f64)> = (0..n)
        .map(|i| (i, distance_fn(td.col(i), x_current)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let keep_idx: Vec<usize> = {
        let mut v: Vec<usize> = dists[..max_points].iter().map(|(i, _)| *i).collect();
        v.sort();
        v
    };

    *td = td.extract_subset(&keep_idx);
    n - max_points
}
