module MolecularKernels

using LinearAlgebra
using ForwardDiff
using KernelFunctions

export MolecularKernel, kernel_blocks

# ==============================================================================
# Helper: Feature Computation (1/r)
# ==============================================================================

"""
    compute_features(x_flat, frozen_flat)

Computes 1/r features for Moving-Moving and Moving-Frozen pairs.
Order is strictly deterministic to match the parameter mapping.
"""
function compute_features(x_flat::AbstractVector, frozen_flat::AbstractVector)
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
    for j = 1:(N_mov-1)
        xj, yj, zj = x_flat[3*j-2], x_flat[3*j-1], x_flat[3*j]
        for i = (j+1):N_mov
            xi, yi, zi = x_flat[3*i-2], x_flat[3*i-1], x_flat[3*i]
            d2 = (xi - xj)^2 + (yi - yj)^2 + (zi - zj)^2
            features[idx] = 1.0 / sqrt(d2 + 1e-18)
            idx += 1
        end
    end

    # 2. Moving-Frozen Pairs
    # Matches C++ loop: for j=0..N_mov, for k=0..N_fro
    if N_fro > 0
        for j = 1:N_mov
            xj, yj, zj = x_flat[3*j-2], x_flat[3*j-1], x_flat[3*j]
            for k = 1:N_fro
                xf, yf, zf = frozen_flat[3*k-2], frozen_flat[3*k-1], frozen_flat[3*k]
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
    for j = 1:(N_mov-1)
        for i = (j+1):N_mov
            t1 = mov_types[j]
            t2 = mov_types[i]
            # Map types to parameter index
            push!(map_indices, pair_map[t1, t2])
        end
    end

    # 2. Moving-Frozen
    if N_fro > 0
        for j = 1:N_mov
            for k = 1:N_fro
                t1 = mov_types[j]
                t2 = fro_types[k]
                push!(map_indices, pair_map[t1, t2])
            end
        end
    end

    return map_indices
end

# ==============================================================================
# Kernel Definition
# ==============================================================================

struct MolecularKernel{T<:Real,V<:AbstractVector{T},F<:AbstractVector{T}} <:
       KernelFunctions.Kernel
    signal_variance::T
    inv_lengthscales::V
    frozen_coords::F

    # Metadata for Type-Awareness
    feature_params_map::Vector{Int} # Maps feature_idx -> inv_lengthscale_idx
end

"""
    MolecularKernel(signal, inv_lengthscales, frozen_coords, 
                    mov_types, fro_types, pair_map)

Constructor that builds the type-aware mapping.
- mov_types, fro_types: Vectors of atom types (integers 1..K)
- pair_map: Matrix where pair_map[t1, t2] = param_index
"""
function MolecularKernel(
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

    # Pre-calculate mapping
    feat_map = build_feature_map(N_mov, N_fro, mov_types, fro_types, pair_map)

    return MolecularKernel(signal, inv_ls, frozen, feat_map)
end

# Backward compatibility constructor (Isotropic / No Types)
function MolecularKernel(signal::Real, inv_ls::AbstractVector, frozen::AbstractVector)
    # Use empty map to signal "Global Broadcasting" (Isotropic)
    return MolecularKernel(signal, inv_ls, frozen, Int[])
end

# Functor implementation
function (k::MolecularKernel)(x::AbstractVector, y::AbstractVector)
    feat_x = compute_features(x, k.frozen_coords)
    feat_y = compute_features(y, k.frozen_coords)

    d2 = zero(eltype(feat_x))

    # Mode 1: Type-Aware Mapping (Preferred)
    if !isempty(k.feature_params_map)
        @inbounds for i in eachindex(feat_x)
            param_idx = k.feature_params_map[i]
            # Use the specific lengthscale for this pair type
            # C++ uses s2 = 1/lengthscale^2. 
            # We use inv_lengthscales = 1/lengthscale.
            # So s2 = inv_lengthscales^2.
            θ = k.inv_lengthscales[param_idx]

            # C++ Line 251: dist += 2 * s2 * diff^2
            d2 += (feat_x[i] - feat_y[i])^2 * θ^2
        end

        # Mode 2: Isotropic (Fallback)
    elseif length(k.inv_lengthscales) == 1
        θ = k.inv_lengthscales[1]
        @inbounds for i in eachindex(feat_x)
            d2 += (feat_x[i] - feat_y[i])^2
        end
        d2 *= θ^2
    else
        # Mode 3: Raw ARD (1-to-1 mapping, rare)
        @inbounds for i in eachindex(feat_x)
            d2 += (feat_x[i] - feat_y[i])^2 * k.inv_lengthscales[i]^2
        end
    end

    # C++ returns sqrt(dist) at the end of dist_at, but GP kernels usually 
    # expect the squared distance in the exponent: exp(-0.5 * dist^2)
    # The C++ code multiplies by 2 inside the sum, and divides by 2 in the exp later.
    # Julia: exp( - (d2) ) is equivalent to C++ exp( - 0.5 * (2 * d2) )
    return k.signal_variance * exp(-d2)
end

# ==============================================================================
# Derivative Block Logic
# ==============================================================================


"""
    kernel_blocks(k::MolecularKernel, x1, x2)

Computes the full block covariance for Energy and Gradients.
 with automatic differentiation.

As far as GP / Kringing etc. are concerned it is the gradients, not forces which
matter.. Which means there's a sign change.

Returns:
- k_ee: Energy-Energy (scalar)
- k_ef: Energy-Force  (1 x D)
- k_fe: Force-Energy  (D x 1)
- k_ff: Force-Force   (D x D)
"""
function kernel_blocks(k::MolecularKernel, x1::AbstractVector, x2::AbstractVector)
    # 1. Energy-Energy
    k_ee = k(x1, x2)

    # 2. Energy-Force (Gradient w.r.t x2)
    g_x2 = ForwardDiff.gradient(x -> k(x1, x), x2)

    # 3. Force-Energy (Gradient w.r.t x1)
    g_x1 = ForwardDiff.gradient(x -> k(x, x2), x1)

    # 4. Force-Force (Hessian)
    # C++ implementation includes a flipped sign: D12 *= -1
    # MATLAB implementation also has DK = -ma2./2.*D12;
    # The mixed derivative of the kernel (∂²k/∂x∂y) is the Covariance of Gradients.
    # This is naturally +ve
    # Action: DON'T FLIP to match C++ (ensure +ve definiteness)
    H_cross = ForwardDiff.jacobian(x -> ForwardDiff.gradient(y -> k(x, y), x2), x1)

    return k_ee, g_x2', g_x1, H_cross
end

end # module
