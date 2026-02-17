"""
    MolInvDistMatern52{T,V,F} <: AbstractMoleculeKernel

Matern 5/2 kernel on inverse interatomic distance features.

Computes `k(x, y) = σ² (1 + √5 d + 5/3 d²) exp(-√5 d)` where
`d = √(Σᵢ (θᵢ (fᵢ(x) - fᵢ(y)))²)` and `fᵢ` are inverse interatomic distances.

The Matern 5/2 kernel is twice differentiable (C²), producing rougher GP surfaces
than the SE kernel. This is often more appropriate for molecular potential energy
surfaces, which can have sharp features near repulsive walls.

# Constructors
Same interface as [`MolInvDistSE`](@ref):
- `MolInvDistMatern52(signal, inv_ls, frozen)`: Isotropic
- `MolInvDistMatern52(signal, inv_ls, frozen, mov_types, fro_types, pair_map)`: Type-aware

See also: [`MolInvDistSE`](@ref), [`compute_inverse_distances`](@ref)
"""
struct MolInvDistMatern52{T,V,F} <: AbstractMoleculeKernel
    signal_variance::T
    inv_lengthscales::V
    frozen_coords::F
    feature_params_map::Vector{Int}
end

# --- Constructors ---

function MolInvDistMatern52(
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
    return MolInvDistMatern52(signal, inv_ls, frozen, feat_map)
end

function MolInvDistMatern52(signal::Real, inv_ls::AbstractVector, frozen::AbstractVector)
    return MolInvDistMatern52(signal, inv_ls, frozen, Int[])
end

# --- Functor ---

function (k::MolInvDistMatern52)(x::AbstractVector, y::AbstractVector)
    fx = compute_inverse_distances(x, k.frozen_coords)
    fy = compute_inverse_distances(y, k.frozen_coords)

    d2 = zero(eltype(fx))

    # Note: Matern on high-dim features usually applies to the Euclidean norm of the feature vector
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
    sqrt5_d = sqrt(5) * d

    return k.signal_variance * (1 + sqrt5_d + (5/3)*d2) * exp(-sqrt5_d)
end
