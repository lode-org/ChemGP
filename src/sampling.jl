# ==============================================================================
# Farthest Point Sampling (FPS)
# ==============================================================================
#
# In GP-guided optimization, we accumulate training data over many iterations.
# Not all data points are equally informative -- points that are "far" from
# existing data in configuration space provide the most new information to the GP.
#
# Farthest Point Sampling is the greedy algorithm used in gpr_optim for
# training set management. Given a set of candidate configurations and an
# existing selected set, it iteratively picks the candidate that is farthest
# from all already-selected points.
#
# This is related to the concept of "space-filling designs" in experimental
# design and ensures the GP training set covers the configuration space well.

"""
    farthest_point_sampling(candidates::Matrix{Float64},
                            X_selected::Matrix{Float64},
                            n_select::Int;
                            distance_fn=max_1d_log_distance)

Select `n_select` points from `candidates` (D x N_cand) that are farthest from
the already-selected set `X_selected` (D x N_sel).

Algorithm:
1. For each candidate, compute its minimum distance to any selected point.
2. Pick the candidate with the largest minimum distance (the "farthest" point).
3. Add it to the selected set.
4. Update minimum distances and repeat.

The `distance_fn` should accept two flat coordinate vectors and return a scalar.
Default is `max_1d_log_distance` (the primary metric in gpr_optim).

Returns the indices of selected columns from `candidates`.
"""
function farthest_point_sampling(
    candidates::Matrix{Float64},
    X_selected::Matrix{Float64},
    n_select::Int;
    distance_fn::Function = max_1d_log_distance,
)
    D, N_cand = size(candidates)
    N_sel = size(X_selected, 2)

    selected_indices = Int[]

    # Pre-compute min distances from each candidate to selected set
    min_dists = fill(Inf, N_cand)
    for i in 1:N_cand
        for j in 1:N_sel
            d = distance_fn(candidates[:, i], X_selected[:, j])
            min_dists[i] = min(min_dists[i], d)
        end
    end

    for _ in 1:min(n_select, N_cand)
        # Pick the candidate with the largest minimum distance
        best_idx = argmax(min_dists)

        if min_dists[best_idx] <= 0.0 || min_dists[best_idx] == -Inf
            break  # All remaining candidates overlap with selected set
        end

        push!(selected_indices, best_idx)
        new_point = candidates[:, best_idx]

        # Mark as taken
        min_dists[best_idx] = -Inf

        # Update min distances for remaining candidates
        for i in 1:N_cand
            if min_dists[i] > 0.0
                d = distance_fn(candidates[:, i], new_point)
                min_dists[i] = min(min_dists[i], d)
            end
        end
    end

    return selected_indices
end
