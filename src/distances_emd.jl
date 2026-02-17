# ==============================================================================
# Intensive Earth Mover's Distance (iEMD)
# ==============================================================================
#
# A permutation-invariant, intensive distance metric for comparing molecular
# configurations. Uses the Hungarian algorithm (linear assignment) to find the
# optimal matching between atoms of the same element type, then reports the
# maximum per-type mean displacement.
#
# This is the primary distance metric advocated in:
#   Goswami, R. & Jónsson, H. (2025). Adaptive Pruning for Increased Robustness
#   and Reduced Computational Overhead in Gaussian Process Accelerated Saddle
#   Point Searches. ChemPhysChem, doi:10.1002/cphc.202500730.
#
#   Goswami, R. (2025). Efficient exploration of chemical kinetics.
#   arXiv:2510.21368.
#
# Advantages over max_1d_log_distance:
# - Permutation invariant (optimal atom matching via assignment)
# - Intensive (size-independent: per-type mean, not sum)
# - Chemically meaningful (Cartesian displacement, not log-ratio)

"""
    emd_distance(x1::AbstractVector, x2::AbstractVector;
                 atom_types::Vector{Int} = Int[])

Intensive Earth Mover's Distance between two molecular configurations.

For each element type `t`, solves the linear assignment problem to find the
optimal permutation minimizing total displacement, then computes the per-type
mean displacement. Returns the maximum over all types:

```math
D(x_1, x_2) = \\max_t \\frac{1}{N_t} \\min_\\pi \\sum_{k=1}^{N_t} \\|r_{k,t}^{(1)} - r_{\\pi(k),t}^{(2)}\\|
```

# Arguments
- `x1`, `x2`: Flat coordinate vectors `[x₁,y₁,z₁, x₂,y₂,z₂, ...]`
- `atom_types`: Integer type label per atom (length `n_atoms`). If empty,
  all atoms are treated as the same type (single assignment problem).

# Notes
Uses a brute-force O(N_t!) assignment for small groups (≤ 8 atoms per type)
and falls back to a greedy assignment for larger groups. For production use
with large systems, consider adding the `Hungarian.jl` package.

See also: [`max_1d_log_distance`](@ref), [`rmsd_distance`](@ref)
"""
function emd_distance(
    x1::AbstractVector,
    x2::AbstractVector;
    atom_types::Vector{Int} = Int[],
)
    n = length(x1)
    @assert length(x2) == n "Coordinate vectors must have same length"
    @assert n % 3 == 0 "Coordinate vector length must be divisible by 3"
    n_atoms = n ÷ 3

    # Default: all atoms same type
    types = isempty(atom_types) ? ones(Int, n_atoms) : atom_types
    @assert length(types) == n_atoms "atom_types length must match number of atoms"

    # Reshape into 3 x n_atoms
    pos1 = reshape(collect(Float64, x1), 3, n_atoms)
    pos2 = reshape(collect(Float64, x2), 3, n_atoms)

    unique_types = unique(types)
    max_mean_disp = 0.0

    for t in unique_types
        idx = findall(==(t), types)
        nt = length(idx)

        if nt == 1
            d = norm(pos1[:, idx[1]] - pos2[:, idx[1]])
            max_mean_disp = max(max_mean_disp, d)
            continue
        end

        # Build cost matrix: C[i,j] = ||r_i^(1) - r_j^(2)|| for atoms of type t
        C = zeros(nt, nt)
        for i in 1:nt, j in 1:nt
            C[i, j] = norm(pos1[:, idx[i]] - pos2[:, idx[j]])
        end

        # Solve assignment problem
        if nt <= 8
            # Brute force for small groups
            min_cost = _bruteforce_assignment(C)
        else
            # Greedy assignment for larger groups
            min_cost = _greedy_assignment(C)
        end

        mean_disp = min_cost / nt
        max_mean_disp = max(max_mean_disp, mean_disp)
    end

    return max_mean_disp
end

"""
Brute-force optimal assignment for small cost matrices (n ≤ 8).
Returns the minimum total cost over all permutations.
"""
function _bruteforce_assignment(C::Matrix{Float64})
    n = size(C, 1)
    if n == 1
        return C[1, 1]
    end

    # Generate permutations via Heap's algorithm inlined
    perm = collect(1:n)
    best = sum(C[i, perm[i]] for i in 1:n)

    # Iterate through all permutations
    for p in _permutations(n)
        cost = sum(C[i, p[i]] for i in 1:n)
        best = min(best, cost)
    end

    return best
end

"""
Generate all permutations of 1:n using a simple recursive approach.
"""
function _permutations(n::Int)
    if n == 1
        return [[1]]
    end
    result = Vector{Int}[]
    for p in _permutations(n - 1)
        for i in 1:n
            new_p = copy(p)
            insert!(new_p, i, n)
            push!(result, new_p)
        end
    end
    return result
end

"""
Greedy assignment for larger cost matrices. Not optimal but O(n²).
Returns the total cost of the greedy matching.
"""
function _greedy_assignment(C::Matrix{Float64})
    n = size(C, 1)
    assigned_cols = falses(n)
    total_cost = 0.0

    for i in 1:n
        best_j = 0
        best_c = Inf
        for j in 1:n
            if !assigned_cols[j] && C[i, j] < best_c
                best_c = C[i, j]
                best_j = j
            end
        end
        assigned_cols[best_j] = true
        total_cost += best_c
    end

    return total_cost
end
