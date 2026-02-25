"""
    MolInvDistSE{T,V,F} <: AbstractMoleculeKernel

Squared Exponential (SE) kernel on inverse interatomic distance features.

Computes `k(x, y) = σ² exp(-Σᵢ (θᵢ (fᵢ(x) - fᵢ(y)))²)` where `fᵢ` are
inverse interatomic distances (`1/rᵢⱼ`) and `θᵢ` are inverse lengthscales.

The SE kernel is infinitely differentiable, producing very smooth GP surfaces.
For rougher potentials, consider [`MolInvDistMatern52`](@ref).

# Constructors
- `MolInvDistSE(signal, inv_ls, frozen)`: Isotropic — single lengthscale for all pairs
- `MolInvDistSE(signal, inv_ls, frozen, mov_types, fro_types, pair_map)`: Type-aware —
  different lengthscales per atom-type pair

# Fields
- `signal_variance::T`: Output variance σ²
- `inv_lengthscales::V`: Inverse lengthscale(s) θ
- `frozen_coords::F`: Flat coordinates of frozen atoms (empty if none)
- `feature_params_map::Vector{Int}`: Maps each feature to its lengthscale index
  (empty for isotropic mode)

# Example
```julia
# Isotropic kernel for a 3-atom cluster (no frozen atoms)
k = MolInvDistSE(1.0, [0.5], Float64[])

# Type-aware kernel: 2 moving Cu atoms, 1 frozen H
k = MolInvDistSE(1.0, [0.5, 0.3], frozen_coords,
                 [1, 1], [2], [1 2; 2 1])
```

See also: [`MolInvDistMatern52`](@ref), [`compute_inverse_distances`](@ref)
"""
struct MolInvDistSE{T,V,F} <: AbstractMoleculeKernel
    signal_variance::T
    inv_lengthscales::V
    frozen_coords::F
    feature_params_map::Vector{Int}
end

# --- Constructors ---

# Type-Aware Constructor
function MolInvDistSE(
    signal::Real,
    inv_ls::AbstractVector,
    frozen::AbstractVector,
    mov_types::Vector{Int},
    fro_types::Vector{Int},
    pair_map::Matrix{Int},
)
    N_mov = length(mov_types)
    N_fro = div(length(frozen), 3)

    if length(fro_types) != N_fro
        throw(ArgumentError("Size mismatch: fro_types vs frozen_coords"))
    end

    # Uses build_feature_map from invdist.jl
    feat_map = build_feature_map(N_mov, N_fro, mov_types, fro_types, pair_map)

    return MolInvDistSE(signal, inv_ls, frozen, feat_map)
end

# Isotropic / Simple Constructor
function MolInvDistSE(signal::Real, inv_ls::AbstractVector, frozen::AbstractVector)
    # Use empty map to signal "Global Broadcasting" (Isotropic)
    return MolInvDistSE(signal, inv_ls, frozen, Int[])
end

# Atomic-number constructor: automatically builds pair-type scheme from
# species lists, matching the C++ gpr_optim convention.
function MolInvDistSE(
    atomic_numbers_mov::AbstractVector{<:Integer},
    frozen_coords::AbstractVector;
    atomic_numbers_fro::AbstractVector{<:Integer}=Int[],
    signal_variance::Real=1.0,
    inv_lengthscale::Real=1.0,
)
    scheme = build_pair_scheme(atomic_numbers_mov; atomic_numbers_fro)
    inv_ls = fill(Float64(inv_lengthscale), scheme.n_params)
    return MolInvDistSE(
        signal_variance, inv_ls, frozen_coords,
        scheme.mov_types, scheme.fro_types, scheme.pair_map,
    )
end

# --- Functor ---

function (k::MolInvDistSE)(x::AbstractVector, y::AbstractVector)
    fx = compute_inverse_distances(x, k.frozen_coords)
    fy = compute_inverse_distances(y, k.frozen_coords)

    d2 = zero(eltype(fx))

    # Mode 1: Type-Aware
    if !isempty(k.feature_params_map)
        @inbounds for i in eachindex(fx)
            idx = k.feature_params_map[i]
            val = (fx[i] - fy[i]) * k.inv_lengthscales[idx]
            d2 += val^2
        end
        # Mode 2: Isotropic
    elseif length(k.inv_lengthscales) == 1
        θ = k.inv_lengthscales[1]
        @inbounds for i in eachindex(fx)
            d2 += (fx[i] - fy[i])^2
        end
        d2 *= θ^2
    end

    return k.signal_variance * exp(-d2)
end
