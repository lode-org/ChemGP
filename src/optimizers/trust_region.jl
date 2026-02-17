# ==============================================================================
# Trust Region Utilities
# ==============================================================================
#
# In GP-guided optimization, the GP model is only reliable near its training
# data. The trust region ensures that optimization steps on the GP surface
# don't stray too far from where the model has been validated by oracle calls.
#
# Two complementary checks are used:
# 1. Distance to nearest training point (Euclidean or MAX_1D_LOG)
# 2. Interatomic distance ratio check (physical reasonability)
#
# Additionally, rigid body mode removal projects out unphysical translations
# and rotations from GP-predicted steps on molecular systems.

"""
    min_distance_to_data(x, X_train; distance_fn)

Minimum distance from configuration `x` to any training point in `X_train`.

The default distance function is Euclidean (norm). For rotationally invariant
checks, pass `distance_fn=max_1d_log_distance`.
"""
function min_distance_to_data(
    x::AbstractVector,
    X_train::Matrix{Float64};
    distance_fn::Function = (a, b) -> norm(a - b),
)
    N = size(X_train, 2)
    return minimum(distance_fn(x, X_train[:, i]) for i in 1:N)
end

"""
    check_interatomic_ratio(x_new, X_train, ratio_limit)

Check whether `x_new` is "geometrically reasonable" relative to the training
data. Returns `true` if at least one training point has all interatomic
distance ratios within bounds.

This prevents the optimizer from distorting the molecule beyond the GP's
knowledge. The check uses `max_1d_log_distance` (from gpr_optim) to compare
the maximum log-ratio of any interatomic distance pair.

The `ratio_limit` (e.g., 2/3) means no interatomic distance should change by
more than a factor of `ratio_limit` relative to the nearest training point.
"""
function check_interatomic_ratio(
    x_new::AbstractVector,
    X_train::Matrix{Float64},
    ratio_limit::Float64,
)
    threshold = abs(log(ratio_limit))
    for k in 1:size(X_train, 2)
        d = max_1d_log_distance(x_new, X_train[:, k])
        if d < threshold
            return true  # At least one training point is geometrically close
        end
    end
    return false
end

# ==============================================================================
# Rigid Body Mode Removal
# ==============================================================================

"""
    remove_rigid_body_modes!(step, x, n_atoms)

Project out rigid body translations and rotations from a step vector.

For a system of `n_atoms` atoms with flat coordinates `x = [x1,y1,z1,x2,...]`,
constructs the 6 rigid body modes (3 translational + 3 rotational about the
center of mass) and removes their components from `step` via Gram-Schmidt
orthogonalization.

Returns the norm of the removed component (useful for diagnostics: if this is
large relative to the step, the GP gradient has significant unphysical content).

Handles linear molecules gracefully: degenerate rotational modes (norm < 1e-9
after orthogonalization) are skipped automatically.

Reference: C++ gpr_optim Dimer.cpp `project_out_rot_trans_with_feedback`
"""
function remove_rigid_body_modes!(step::Vector{Float64}, x::Vector{Float64}, n_atoms::Int)
    D = 3 * n_atoms
    @assert length(step) == D
    @assert length(x) == D

    # Center of mass
    com = zeros(3)
    for i in 1:n_atoms
        com .+= @view x[(3*(i-1)+1):(3*i)]
    end
    com ./= n_atoms

    # Build 6 basis vectors: 3 translational + 3 rotational
    basis = Vector{Float64}[]

    # Translational modes: uniform displacement in x, y, z
    for d in 1:3
        t = zeros(D)
        for i in 1:n_atoms
            t[3*(i-1)+d] = 1.0
        end
        push!(basis, t)
    end

    # Rotational modes: ω × (r - com) for ω along each axis
    for ax in 1:3
        r = zeros(D)
        for i in 1:n_atoms
            pos = x[(3*(i-1)+1):(3*i)] .- com
            # Cross product: ω_ax × pos
            if ax == 1      # ω = (1,0,0): (0, -pz, py)
                r[3*(i-1)+2] = -pos[3]
                r[3*(i-1)+3] =  pos[2]
            elseif ax == 2  # ω = (0,1,0): (pz, 0, -px)
                r[3*(i-1)+1] =  pos[3]
                r[3*(i-1)+3] = -pos[1]
            else            # ω = (0,0,1): (-py, px, 0)
                r[3*(i-1)+1] = -pos[2]
                r[3*(i-1)+2] =  pos[1]
            end
        end
        push!(basis, r)
    end

    # Gram-Schmidt orthonormalization
    ortho = Vector{Float64}[]
    for v in basis
        u = copy(v)
        for ou in ortho
            u .-= dot(v, ou) .* ou
        end
        un = norm(u)
        if un > 1e-9  # Skip degenerate modes (e.g., linear molecules)
            push!(ortho, u ./ un)
        end
    end

    # Project out and return magnitude
    removed = zeros(D)
    for u in ortho
        removed .+= dot(step, u) .* u
    end
    step .-= removed

    return norm(removed)
end
