# ==============================================================================
# Configuration Distance Metrics
# ==============================================================================
#
# In GP-guided molecular optimization, we need to measure how "different" two
# molecular configurations are. This is essential for:
# - Trust region management (is the proposed step too far from known data?)
# - Farthest Point Sampling (which configuration is most informative to add?)
# - Early stopping (has the configuration stopped changing?)
#
# The key metric from gpr_optim is MAX_1D_LOG, which compares the maximum
# log-ratio of interatomic distances. This is rotationally and translationally
# invariant -- a critical property since rigid body motions don't change the
# physics.

"""
    interatomic_distances(x::AbstractVector{<:Real})

Given a flat coordinate vector `[x1,y1,z1, x2,y2,z2, ...]`, returns a vector
of all pairwise interatomic distances in canonical order (j < i).

The number of pairs for N atoms is N*(N-1)/2.
"""
function interatomic_distances(x::AbstractVector{<:Real})
    N = div(length(x), 3)
    n_pairs = div(N * (N - 1), 2)
    dists = zeros(eltype(x), n_pairs)
    idx = 1
    for j in 1:(N-1)
        xj = @view x[(3j-2):(3j)]
        for i in (j+1):N
            xi = @view x[(3i-2):(3i)]
            dists[idx] = sqrt(sum((xi .- xj) .^ 2))
            idx += 1
        end
    end
    return dists
end

"""
    max_1d_log_distance(x1::AbstractVector, x2::AbstractVector)

MAX_1D_LOG distance from gpr_optim: the maximum absolute log-ratio of
corresponding interatomic distances between two configurations.

    d(x1, x2) = max_k |log(r1_k / r2_k)|

Properties:
- Rotationally and translationally invariant (uses interatomic distances)
- Sensitive to large relative changes in any single pair
- Zero when configurations are identical
- This is the primary distance metric used in gpr_optim for FPS, trust region,
  and early stopping checks.
"""
function max_1d_log_distance(x1::AbstractVector, x2::AbstractVector)
    d1 = interatomic_distances(x1)
    d2 = interatomic_distances(x2)
    return maximum(abs.(log.(d1 ./ (d2 .+ 1e-18))))
end

"""
    rmsd_distance(x1::AbstractVector, x2::AbstractVector)

Root-mean-square deviation between two flat coordinate vectors.
No alignment is performed -- assumes the configurations share a common frame.

    RMSD = sqrt(1/N * sum_i ||r_i - r_i'||^2)
"""
function rmsd_distance(x1::AbstractVector, x2::AbstractVector)
    N = div(length(x1), 3)
    return sqrt(sum((x1 .- x2) .^ 2) / N)
end
