# ==============================================================================
# Shared Optimizer Step (L-BFGS with per-atom max_move)
# ==============================================================================
#
# Reusable L-BFGS step function shared by NEB and dimer. Wraps LBFGSHistory
# with previous-state tracking, angle check, and per-atom max_move clipping
# matching eOn's LBFGS.cpp logic.
#
# Reference:
#   Asgeirsson, V. et al. (2021). Nudged elastic band method for molecular
#   reactions using energy-weighted springs combined with eigenvector following.
#   J. Chem. Theory Comput., 17(8), 4929-4945.

"""
    OptimState

Mutable optimizer state wrapping an [`LBFGSHistory`](@ref) with tracking of
previous positions and gradients for L-BFGS pair updates.

# Fields
- `lbfgs`: L-BFGS circular buffer
- `prev_x`: Previous concatenated position vector (or `nothing`)
- `prev_g`: Previous concatenated gradient vector (or `nothing`)
"""
mutable struct OptimState
    lbfgs::LBFGSHistory
    prev_x::Union{Nothing,Vector{Float64}}
    prev_g::Union{Nothing,Vector{Float64}}
end

OptimState(memory::Int) = OptimState(LBFGSHistory(memory), nothing, nothing)

function reset!(s::OptimState)
    reset!(s.lbfgs)
    s.prev_x = nothing
    s.prev_g = nothing
end

"""
    optim_step!(state, x, force, max_move; n_coords_per_atom=3) -> Vector{Float64}

Compute an L-BFGS displacement vector from the current position `x` and
`force` (pointing downhill). Returns the displacement to add to `x`.

Matches eOn's LBFGS.cpp:
- Updates L-BFGS memory with (s, y) pairs
- Angle check: resets to steepest descent if step >90 deg from force
- Distance reset: if max per-atom displacement exceeds `max_move`, resets
  L-BFGS history and returns SD step clipped to `max_move`
"""
function optim_step!(
    state::OptimState,
    x::Vector{Float64},
    force::Vector{Float64},
    max_move::Float64;
    n_coords_per_atom::Int = 3,
)
    g = -force  # gradient (uphill) for L-BFGS

    # Update L-BFGS memory with previous step
    if state.prev_x !== nothing
        push_pair!(state.lbfgs, x - state.prev_x, g - state.prev_g)
    end
    state.prev_x = copy(x)
    state.prev_g = copy(g)

    # Compute L-BFGS direction (points downhill)
    direction = compute_direction(state.lbfgs, g)

    # Angle check: if step >90 deg from force, reset and use SD
    fn = norm(force)
    dn = norm(direction)
    if fn > 1e-30 && dn > 1e-30
        cos_angle = dot(direction, force) / (dn * fn)
        if cos_angle < 0
            reset!(state)
            state.prev_x = copy(x)
            state.prev_g = copy(g)
            direction = copy(force)
        end
    end

    # Distance reset: if step exceeds max_move, reset history and use SD
    # (matches eOn LBFGS.cpp:78-86, distance_reset=true by default)
    n_atoms = div(length(direction), n_coords_per_atom)
    max_disp = 0.0
    for a in 1:n_atoms
        off = (a - 1) * n_coords_per_atom
        disp = norm(@view direction[off+1:off+n_coords_per_atom])
        max_disp = max(max_disp, disp)
    end
    if max_disp > max_move
        reset!(state)
        state.prev_x = copy(x)
        state.prev_g = copy(g)
        # Fall back to force direction, clipped to max_move
        direction = copy(force)
        max_disp_sd = 0.0
        for a in 1:n_atoms
            off = (a - 1) * n_coords_per_atom
            disp = norm(@view direction[off+1:off+n_coords_per_atom])
            max_disp_sd = max(max_disp_sd, disp)
        end
        if max_disp_sd > max_move
            direction .*= max_move / max_disp_sd
        end
    end

    return direction
end
