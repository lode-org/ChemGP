# ==============================================================================
# Helper: Feature Computation (1/r)
# ==============================================================================

"""
    compute_inverse_distances(x_flat, frozen_flat)

Compute inverse interatomic distance features (1/r) for all Moving-Moving and
Moving-Frozen atom pairs.

Given flat coordinate vectors for moving atoms (`x_flat`) and frozen atoms
(`frozen_flat`), returns a vector of `1/r` values in canonical order:
1. Moving-Moving pairs (upper triangle: j < i)
2. Moving-Frozen pairs (all combinations)

The total number of features is `N_mov*(N_mov-1)/2 + N_mov*N_fro`.

These features are rotationally and translationally invariant by construction,
which is the key property that makes inverse distance kernels suitable for
molecular systems.

See also: [`MolInvDistSE`](@ref), [`MolInvDistMatern52`](@ref)
"""
function compute_inverse_distances(x_flat::AbstractVector, frozen_flat::AbstractVector)
    if length(x_flat) % 3 != 0
        throw(ArgumentError("Moving coordinates must be 3D."))
    end

    N_mov = div(length(x_flat), 3)
    N_fro = div(length(frozen_flat), 3)

    # Pre-allocate
    n_mm = div(N_mov * (N_mov - 1), 2)
    n_mf = N_mov * N_fro
    features = similar(x_flat, n_mm + n_mf)

    idx = 1

    # 1. Moving-Moving Pairs (Upper Triangle)
    # Matches C++ loop: for j=0..N-1, for i=j+1..N
    for j in 1:(N_mov - 1)
        xj, yj, zj = x_flat[3 * j - 2], x_flat[3 * j - 1], x_flat[3 * j]
        for i in (j + 1):N_mov
            xi, yi, zi = x_flat[3 * i - 2], x_flat[3 * i - 1], x_flat[3 * i]
            d2 = (xi - xj)^2 + (yi - yj)^2 + (zi - zj)^2
            features[idx] = 1.0 / sqrt(d2 + 1e-18)
            idx += 1
        end
    end

    # 2. Moving-Frozen Pairs
    # Matches C++ loop: for j=0..N_mov, for k=0..N_fro
    if N_fro > 0
        for j in 1:N_mov
            xj, yj, zj = x_flat[3 * j - 2], x_flat[3 * j - 1], x_flat[3 * j]
            for k in 1:N_fro
                xf, yf, zf = frozen_flat[3 * k - 2],
                frozen_flat[3 * k - 1],
                frozen_flat[3 * k]
                d2 = (xj - xf)^2 + (yj - yf)^2 + (zj - zf)^2
                features[idx] = 1.0 / sqrt(d2 + 1e-18)
                idx += 1
            end
        end
    end

    return features
end

# ==============================================================================
# Helper: Parameter Mapping (Types -> Index)
# ==============================================================================

function build_feature_map(
    N_mov::Int,
    N_fro::Int,
    mov_types::Vector{Int},
    fro_types::Vector{Int},
    pair_map::Matrix{Int},
)
    map_indices = Int[]
    sizehint!(map_indices, div(N_mov*(N_mov-1), 2) + N_mov*N_fro)

    # 1. Moving-Moving
    for j in 1:(N_mov - 1)
        for i in (j + 1):N_mov
            t1 = mov_types[j]
            t2 = mov_types[i]
            # Map types to parameter index
            push!(map_indices, pair_map[t1, t2])
        end
    end

    # 2. Moving-Frozen
    if N_fro > 0
        for j in 1:N_mov
            for k in 1:N_fro
                t1 = mov_types[j]
                t2 = fro_types[k]
                push!(map_indices, pair_map[t1, t2])
            end
        end
    end

    return map_indices
end

# ==============================================================================
# Helper: Atom-Type Pair Scheme from Atomic Numbers
# ==============================================================================

"""
    build_pair_scheme(atomic_numbers_mov; atomic_numbers_fro=Int[])

Build the pair-type mapping from atomic numbers, matching the C++ gpr_optim
convention where each unique (species_i, species_j) pair gets its own
inverse lengthscale parameter.

Only pair types that actually appear in the moving-moving or moving-frozen
feature set are assigned parameter indices. For example, HCN (one H, one C,
one N, no frozen) produces 3 pairs: H-C, H-N, C-N (no H-H, C-C, or N-N
since each species appears only once among moving atoms).

Returns a NamedTuple with fields:
- `mov_types::Vector{Int}`: 1-based type index for each moving atom
- `fro_types::Vector{Int}`: 1-based type index for each frozen atom
- `pair_map::Matrix{Int}`: symmetric matrix mapping (type_i, type_j) -> param index
- `n_params::Int`: number of unique pair-type parameters (= length of inv_lengthscales)
- `species::Vector{Int}`: sorted unique atomic numbers
"""
function build_pair_scheme(
    atomic_numbers_mov::AbstractVector{<:Integer};
    atomic_numbers_fro::AbstractVector{<:Integer}=Int[],
)
    all_species = sort(unique(vcat(atomic_numbers_mov, atomic_numbers_fro)))
    species_to_type = Dict(z => i for (i, z) in enumerate(all_species))

    mov_types = [species_to_type[z] for z in atomic_numbers_mov]
    fro_types = [species_to_type[z] for z in atomic_numbers_fro]

    n_types = length(all_species)
    N_mov = length(mov_types)
    N_fro = length(fro_types)

    # Scan which pair types actually appear in the feature set
    used_pairs = Set{Tuple{Int,Int}}()

    # Moving-Moving (upper triangle: j < i, matching compute_inverse_distances)
    for j in 1:(N_mov - 1)
        for i in (j + 1):N_mov
            t1, t2 = minmax(mov_types[j], mov_types[i])
            push!(used_pairs, (t1, t2))
        end
    end

    # Moving-Frozen
    for j in 1:N_mov
        for k in 1:N_fro
            t1, t2 = minmax(mov_types[j], fro_types[k])
            push!(used_pairs, (t1, t2))
        end
    end

    # Assign contiguous indices to used pairs (sorted for determinism)
    sorted_pairs = sort(collect(used_pairs))
    n_params = length(sorted_pairs)
    pair_to_idx = Dict(p => i for (i, p) in enumerate(sorted_pairs))

    pair_map = zeros(Int, n_types, n_types)
    for ((t1, t2), idx) in pair_to_idx
        pair_map[t1, t2] = idx
        pair_map[t2, t1] = idx  # symmetric
    end

    return (
        mov_types=mov_types,
        fro_types=fro_types,
        pair_map=pair_map,
        n_params=n_params,
        species=all_species,
    )
end
