# ==============================================================================
# Helper: Feature Computation (1/r)
# ==============================================================================

"""
    compute_features(x_flat, frozen_flat)

Computes 1/r features for Moving-Moving and Moving-Frozen pairs.
Order is strictly deterministic to match the parameter mapping.
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
