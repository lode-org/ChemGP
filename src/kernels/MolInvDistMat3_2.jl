"""
    MolInvDistMatern32{T,V,F} <: AbstractMoleculeKernel

Matern 3/2 kernel on inverse interatomic distance features.

Computes `k(x, y) = σ² (1 + √3 d) exp(-√3 d)` where
`d = √(Σᵢ (θᵢ (fᵢ(x) - fᵢ(y)))²)` and `fᵢ` are inverse interatomic distances.

The Matern 3/2 kernel is only once differentiable (C¹), producing the roughest GP
surfaces among the standard Matern family. This is useful when the energy surface
is expected to have sharp features, such as near repulsive walls or bond-breaking
events. For smoother surfaces, prefer [`MolInvDistMatern52`](@ref) (C²) or
[`MolInvDistSE`](@ref) (C∞).

# Constructors
Same interface as [`MolInvDistSE`](@ref):
- `MolInvDistMatern32(signal, inv_ls, frozen)`: Isotropic
- `MolInvDistMatern32(signal, inv_ls, frozen, mov_types, fro_types, pair_map)`: Type-aware

See also: [`MolInvDistSE`](@ref), [`MolInvDistMatern52`](@ref), [`compute_inverse_distances`](@ref)
"""
struct MolInvDistMatern32{T,V,F} <: AbstractMoleculeKernel
    signal_variance::T
    inv_lengthscales::V
    frozen_coords::F
    feature_params_map::Vector{Int}
end

# --- Constructors ---

function MolInvDistMatern32(
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
    feat_map = build_feature_map(N_mov, N_fro, mov_types, fro_types, pair_map)
    return MolInvDistMatern32(signal, inv_ls, frozen, feat_map)
end

function MolInvDistMatern32(signal::Real, inv_ls::AbstractVector, frozen::AbstractVector)
    return MolInvDistMatern32(signal, inv_ls, frozen, Int[])
end

# Atomic-number constructor
function MolInvDistMatern32(
    atomic_numbers_mov::AbstractVector{<:Integer},
    frozen_coords::AbstractVector;
    atomic_numbers_fro::AbstractVector{<:Integer}=Int[],
    signal_variance::Real=1.0,
    inv_lengthscale::Real=1.0,
)
    scheme = build_pair_scheme(atomic_numbers_mov; atomic_numbers_fro)
    inv_ls = fill(Float64(inv_lengthscale), scheme.n_params)
    return MolInvDistMatern32(
        signal_variance, inv_ls, frozen_coords,
        scheme.mov_types, scheme.fro_types, scheme.pair_map,
    )
end

# --- Functor ---

function (k::MolInvDistMatern32)(x::AbstractVector, y::AbstractVector)
    fx = compute_inverse_distances(x, k.frozen_coords)
    fy = compute_inverse_distances(y, k.frozen_coords)

    d2 = zero(eltype(fx))

    if !isempty(k.feature_params_map)
        @inbounds for i in eachindex(fx)
            idx = k.feature_params_map[i]
            val = (fx[i] - fy[i]) * k.inv_lengthscales[idx]
            d2 += val^2
        end
    elseif length(k.inv_lengthscales) == 1
        θ = k.inv_lengthscales[1]
        @inbounds for i in eachindex(fx)
            d2 += (fx[i] - fy[i])^2
        end
        d2 *= θ^2
    end

    d = sqrt(d2 + 1e-18)
    sqrt3_d = sqrt(3) * d

    return k.signal_variance * (1 + sqrt3_d) * exp(-sqrt3_d)
end
