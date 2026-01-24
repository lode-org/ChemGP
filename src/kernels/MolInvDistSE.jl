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
