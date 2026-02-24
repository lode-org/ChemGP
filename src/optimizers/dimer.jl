# ==============================================================================
# GP-Dimer Saddle Point Search
# ==============================================================================
#
# The dimer method finds first-order saddle points (transition states) on the
# potential energy surface. A "dimer" is a pair of configurations separated by
# a small distance along a direction vector. By rotating the dimer to align
# with the lowest curvature mode and translating with a modified force (inverted
# along the dimer direction), the algorithm converges to a saddle point.
#
# In gpr_optim, the dimer method is combined with GP regression: instead of
# calling the expensive oracle for every rotation/translation step, the GP
# surrogate provides cheap predictions. The oracle is only called when the GP
# optimization converges or the trust region is exceeded.
#
# Three rotation strategies are available:
#   :simple  — Single-step angle estimate (original pedagogical version)
#   :lbfgs   — L-BFGS direction with modified Newton angle optimization
#   :cg      — Conjugate gradient (PRP) with modified Newton angle optimization
#
# Two translation strategies:
#   :simple  — Adaptive step size based on curvature (original)
#   :lbfgs   — L-BFGS for negative curvature, fixed step for positive curvature
#
# Reference: Henkelman & Jonsson, J. Chem. Phys. 111, 7010 (1999)
# GP-Dimer: Koistinen et al., J. Chem. Theory Comput. 16, 499 (2020)
# Implementation: Goswami et al., J. Chem. Theory Comput. (2025), doi:10.1021/acs.jctc.5c00866
# Pruning: Goswami & Jónsson, ChemPhysChem (2025), doi:10.1002/cphc.202500730
# Thesis: Goswami, Efficient exploration of chemical kinetics (2025), arXiv:2510.21368

# ==============================================================================
# Types
# ==============================================================================

"""
    DimerState

Current state of the dimer in the saddle point search.
"""
mutable struct DimerState
    R::Vector{Float64}           # Midpoint coordinates (flat)
    orient::Vector{Float64}      # Dimer orientation (unit vector)
    dimer_sep::Float64           # Half-length of dimer
end

"""
    DimerConfig

Configuration parameters for the GP-dimer method.

# Rotation/translation method selection
- `rotation_method`: `:simple`, `:lbfgs`, or `:cg` (default `:lbfgs`)
- `translation_method`: `:simple` or `:lbfgs` (default `:lbfgs`)
- `lbfgs_memory`: Number of stored (s,y) pairs for L-BFGS (default 5)

# Step size controls
- `max_step`: Maximum translation step length (default 0.5)
- `step_convex`: Fixed step size in convex (positive curvature) regions (default 0.1)
"""
Base.@kwdef struct DimerConfig
    T_force_true::Float64 = 1e-3     # True convergence threshold (on oracle gradient)
    T_force_gp::Float64 = 1e-2     # GP convergence threshold (on GP gradient)
    T_angle_rot::Float64 = 1e-3     # Rotation angle convergence threshold
    trust_radius::Float64 = 0.1      # Max distance from training data
    ratio_at_limit::Float64 = 2 / 3    # Inter-atomic distance ratio limit
    max_outer_iter::Int = 50       # Max outer iterations (oracle call cycles)
    max_inner_iter::Int = 100      # Max GP steps per outer iteration
    max_rot_iter::Int = 10       # Max rotations per translation step
    alpha_trans::Float64 = 0.01     # Translation step size factor (simple mode)
    gp_train_iter::Int = 300      # Nelder-Mead iterations for GP training
    n_initial_perturb::Int = 4        # Number of initial perturbation points
    perturb_scale::Float64 = 0.15     # Scale of initial perturbations
    rotation_method::Symbol = :lbfgs   # :simple, :lbfgs, or :cg
    translation_method::Symbol = :lbfgs # :simple or :lbfgs
    lbfgs_memory::Int = 5        # L-BFGS memory depth
    max_step::Float64 = 0.5      # Max translation step length
    step_convex::Float64 = 0.1      # Step size in convex regions
    verbose::Bool = true
end

"""
    DimerResult

Result of a GP-dimer saddle point search.
"""
struct DimerResult
    state::DimerState               # Final dimer state (position + orientation)
    converged::Bool                 # Whether convergence criterion was met
    stop_reason::StopReason         # Why the optimizer terminated
    oracle_calls::Int               # Total oracle evaluations
    history::Dict{String,Vector}    # Convergence history
end

# ==============================================================================
# Internal: CG rotation state
# ==============================================================================

mutable struct CGRotationState
    F_rot_old::Vector{Float64}
    F_modrot_old::Vector{Float64}
    orient_rot_old::Vector{Float64}
    iter::Int
    max_iter::Int
end

function CGRotationState(D::Int; max_iter::Int=20)
    return CGRotationState(Float64[], Float64[], Float64[], 0, max_iter)
end

function reset_cg!(cg::CGRotationState)
    empty!(cg.F_rot_old)
    empty!(cg.F_modrot_old)
    empty!(cg.orient_rot_old)
    cg.iter = 0
end

# ==============================================================================
# Dimer Utility Functions
# ==============================================================================

"""
    dimer_images(state::DimerState) -> (R0, R1, R2)

Get positions of the three dimer image points:
- R0: midpoint
- R1: midpoint + sep * orientation
- R2: midpoint - sep * orientation
"""
function dimer_images(state::DimerState)
    R0 = state.R
    R1 = state.R + state.dimer_sep * state.orient
    R2 = state.R - state.dimer_sep * state.orient
    return R0, R1, R2
end

"""
    curvature(G0, G1, orient, dimer_sep)

Compute curvature along the dimer direction via finite differences:
    C = (G1 - G0) . orient / dimer_sep

Negative curvature indicates the dimer is aligned with a descent direction
from a saddle point (the desired mode for saddle point search).
"""
function curvature(G0, G1, orient, dimer_sep)
    return dot(G1 - G0, orient) / dimer_sep
end

"""
    rotational_force(G0, G1, orient, dimer_sep)

Force perpendicular to the dimer for rotation. This drives the dimer
to align with the lowest curvature mode.

    F_rot = (G1 - G0)/sep - [(G1 - G0)/sep . orient] * orient
"""
function rotational_force(G0, G1, orient, dimer_sep)
    G_diff = (G1 - G0) / dimer_sep
    F_rot = G_diff - dot(G_diff, orient) * orient
    return F_rot
end

"""
    translational_force(G0, orient)

Effective force for translation toward a saddle point. The gradient
component along the dimer direction is inverted, so the algorithm
climbs along the lowest curvature mode while descending in all other
directions (Henkelman & Jonsson 1999, Eq. 14).

    F_eff = -G0 + 2*(G0 . orient)*orient

Equivalently: F_eff = -G_perp + G_parallel, where G_perp is the
perpendicular gradient and G_parallel = (G0 . orient) * orient.
"""
function translational_force(G0, orient)
    F_parallel = dot(G0, orient) * orient
    return -G0 + 2 * F_parallel
end

# ==============================================================================
# Predict dimer gradients (shared helper)
# ==============================================================================

function predict_dimer_gradients(state::DimerState, model::GPModel, y_std::Float64)
    _, R1, R2 = dimer_images(state)
    pred0 = predict(model, reshape(state.R, :, 1))
    pred1 = predict(model, reshape(R1, :, 1))

    G0 = pred0[2:end] .* y_std
    G1 = pred1[2:end] .* y_std
    E0 = pred0[1] * y_std

    return G0, G1, E0
end

# ==============================================================================
# Modified Newton Rotation (Parabolic Fit)
# ==============================================================================
#
# Given a rotation direction (from L-BFGS, CG, or the raw rotational force),
# find the optimal rotation angle via a parabolic fit on the rotation plane.
#
# Algorithm (from MATLAB rotate_dimer.m):
# 1. Compute initial test angle from |F_rot| and curvature C
# 2. Evaluate GP at a trial rotation by dtheta
# 3. Fit parabola F(θ) = a1*cos(2θ) + b1*sin(2θ) using the two evaluations
# 4. Optimal angle: θ* = 0.5*atan(b1/a1), adjusted to be a minimum

"""
    rotate_dimer_newton!(state, model, F_rot_direction, config; y_std)

Single rotation step with modified Newton angle optimization (parabolic fit).

Takes a rotation direction `F_rot_direction` (from L-BFGS, CG, or raw force)
and finds the optimal rotation angle via a parabolic fit on the rotation plane.
Returns the curvature estimate after rotation, or `nothing` if no rotation was needed.
"""
function rotate_dimer_newton!(
    state::DimerState,
    model::GPModel,
    F_rot_direction::Vector{Float64},
    config::DimerConfig;
    y_std::Float64=1.0,
    verbose::Bool=false,
)
    orient = state.orient
    F_rot_norm = norm(F_rot_direction)

    if F_rot_norm < 1e-10
        verbose && println("    Newton: F_rot ~ 0, skip")
        return nothing
    end

    # Compute curvature at current orientation
    G0, G1, _ = predict_dimer_gradients(state, model, y_std)
    C0 = curvature(G0, G1, orient, state.dimer_sep)

    # Initial test angle (modified Newton estimate)
    dtheta = 0.5 * atan(0.5 * F_rot_norm / (abs(C0) + 1e-10))

    if dtheta < config.T_angle_rot
        verbose && @printf(
            "    Newton: converged (dθ = %.5f < %.5f)\n", dtheta, config.T_angle_rot
        )
        return C0
    end

    # Rotation plane basis: orient and orient_rot (perpendicular unit vector)
    orient_rot = F_rot_direction / F_rot_norm

    # Trial rotation by dtheta
    orient_trial = cos(dtheta) * orient + sin(dtheta) * orient_rot
    orient_trial ./= norm(orient_trial)

    # Evaluate GP at trial R1
    R1_trial = state.R + state.dimer_sep * orient_trial
    pred1_trial = predict(model, reshape(R1_trial, :, 1))
    G1_trial = pred1_trial[2:end] .* y_std

    # Rotational force at trial orientation
    F_rot_trial = rotational_force(G0, G1_trial, orient_trial, state.dimer_sep)

    # Project trial force onto the rotated perpendicular direction
    orient_rot_trial = -sin(dtheta) * orient + cos(dtheta) * orient_rot
    orient_rot_trial ./= norm(orient_rot_trial)
    F_dtheta = dot(F_rot_trial, orient_rot_trial)

    # F_0 is the projection of the original rotational force onto orient_rot
    F_0 = dot(F_rot_direction, orient_rot)

    # Parabolic fit: F(θ) ≈ a1*cos(2θ) + b1*sin(2θ)
    # From two evaluations: F(0) = F_0, F(dtheta) = F_dtheta
    sin2 = sin(2 * dtheta)
    cos2 = cos(2 * dtheta)

    if abs(sin2) < 1e-12
        # dtheta too small for parabolic fit, use the simple estimate
        state.orient = orient_trial
        return nothing
    end

    a1 = (F_dtheta - F_0 * cos2) / sin2
    b1 = -0.5 * F_0

    # Optimal angle minimizing the curvature (minimum of parabola)
    angle_rot = 0.5 * atan(b1 / (a1 + 1e-18))

    # Ensure we pick the minimum, not the maximum
    if a1 * cos(2 * angle_rot) + b1 * sin(2 * angle_rot) > 0
        angle_rot += π / 2
    end

    # Apply final rotation
    orient_new = cos(angle_rot) * orient + sin(angle_rot) * orient_rot
    state.orient = orient_new / norm(orient_new)

    # Estimate curvature after rotation
    C_est = C0 + a1 * (cos(2 * angle_rot) - 1) + b1 * sin(2 * angle_rot)

    verbose && @printf(
        "    Newton: dθ_test=%.4f → θ_opt=%.4f, C≈%+.3e\n", dtheta, angle_rot, C_est
    )

    return C_est
end

# ==============================================================================
# Rotation Strategies
# ==============================================================================

"""
    rotate_dimer!(state, model, config; y_std, verbose, rot_hist, cg_state)

Rotate the dimer to align with the lowest curvature mode using GP predictions.
Dispatches to the appropriate rotation method based on `config.rotation_method`.

Methods:
- `:simple` — Direct angle estimate from rotational force magnitude/curvature ratio
- `:lbfgs` — L-BFGS search direction with modified Newton angle optimization
- `:cg` — Conjugate gradient (PRP) direction with modified Newton angle optimization
"""
function rotate_dimer!(
    state::DimerState,
    model::GPModel,
    config::DimerConfig;
    y_std::Float64=1.0,
    verbose::Bool=false,
    rot_hist::Union{Nothing,LBFGSHistory}=nothing,
    cg_state::Union{Nothing,CGRotationState}=nothing,
)
    if config.rotation_method == :simple
        rotate_dimer_simple!(state, model, config; y_std, verbose)
    elseif config.rotation_method == :lbfgs
        rotate_dimer_lbfgs!(state, model, config, rot_hist; y_std, verbose)
    elseif config.rotation_method == :cg
        rotate_dimer_cg!(state, model, config, cg_state; y_std, verbose)
    else
        error("Unknown rotation method: $(config.rotation_method)")
    end
end

"""
    rotate_dimer_simple!(state, model, config; y_std, verbose)

Original simple rotation: estimate angle from |F_rot|/|C| ratio, rotate directly.
"""
function rotate_dimer_simple!(
    state::DimerState,
    model::GPModel,
    config::DimerConfig;
    y_std::Float64=1.0,
    verbose::Bool=false,
)
    for rot_iter in 1:(config.max_rot_iter)
        G0, G1, _ = predict_dimer_gradients(state, model, y_std)

        F_rot = rotational_force(G0, G1, state.orient, state.dimer_sep)
        F_rot_norm = norm(F_rot)

        if F_rot_norm < 1e-10
            verbose && println("  Rotation converged (F_rot ~ 0)")
            break
        end

        C = curvature(G0, G1, state.orient, state.dimer_sep)
        dtheta = 0.5 * atan(F_rot_norm / (abs(C) + 1e-10))

        if dtheta < config.T_angle_rot
            verbose && @printf(
                "  Rotation converged (theta = %.5f < %.5f)\n",
                dtheta,
                config.T_angle_rot
            )
            break
        end

        b1 = F_rot / F_rot_norm
        orient_new = cos(dtheta) * state.orient + sin(dtheta) * b1
        state.orient = orient_new / norm(orient_new)

        verbose && @printf(
            "  Rotation %d: theta = %.5f, |F_rot| = %.5f\n",
            rot_iter,
            dtheta,
            F_rot_norm
        )
    end
end

"""
    rotate_dimer_lbfgs!(state, model, config, rot_hist; y_std, verbose)

L-BFGS rotation: use L-BFGS to choose the rotation search direction, then
apply modified Newton angle optimization (parabolic fit) to find the optimal
angle in that direction.

The L-BFGS direction is projected perpendicular to the current dimer orientation
before being used. This is the key difference from standard unconstrained L-BFGS:
the optimization is constrained to the unit sphere.

Reference: MATLAB rot_iter_lbfgs.m
"""
function rotate_dimer_lbfgs!(
    state::DimerState,
    model::GPModel,
    config::DimerConfig,
    rot_hist::Union{Nothing,LBFGSHistory};
    y_std::Float64=1.0,
    verbose::Bool=false,
)
    hist = rot_hist === nothing ? LBFGSHistory(config.lbfgs_memory) : rot_hist
    F_rot_prev = Float64[]
    orient_prev = Float64[]

    for rot_iter in 1:(config.max_rot_iter)
        G0, G1, _ = predict_dimer_gradients(state, model, y_std)

        F_rot = rotational_force(G0, G1, state.orient, state.dimer_sep)
        F_rot_norm = norm(F_rot)

        if F_rot_norm < 1e-10
            verbose && println("  L-BFGS rotation converged (F_rot ~ 0)")
            break
        end

        # Update L-BFGS history with (s, y) pair from orientation change
        if !isempty(F_rot_prev)
            s = state.orient - orient_prev
            y = -(F_rot - F_rot_prev)  # Gradient difference (sign: minimizing)
            push_pair!(hist, s, y)
        end

        # Compute L-BFGS search direction
        search_dir = compute_direction(hist, -F_rot)

        # Project perpendicular to current orientation
        search_dir .-= dot(search_dir, state.orient) * state.orient
        sn = norm(search_dir)
        if sn < 1e-12
            verbose && println("  L-BFGS rotation: degenerate direction, using F_rot")
            search_dir = F_rot
            sn = F_rot_norm
        end
        search_dir ./= sn

        # Project the rotational force onto this search direction
        F_rot_oriented = dot(F_rot, search_dir) * search_dir

        # Save state for next iteration's L-BFGS update
        F_rot_prev = copy(F_rot)
        orient_prev = copy(state.orient)

        # Apply modified Newton angle optimization in this direction
        C_est = rotate_dimer_newton!(state, model, F_rot_oriented, config; y_std, verbose)

        if C_est !== nothing
            dtheta = acos(clamp(dot(orient_prev, state.orient), -1.0, 1.0))
            verbose && @printf(
                "  L-BFGS rot %d: |F_rot| = %.5f, Δθ = %.5f, C ≈ %+.3e\n",
                rot_iter,
                F_rot_norm,
                dtheta,
                C_est
            )
            if dtheta < config.T_angle_rot
                break
            end
        else
            break
        end
    end
end

"""
    rotate_dimer_cg!(state, model, config, cg_state; y_std, verbose)

Conjugate gradient (Polak-Ribiere-Polyak) rotation: use CG to choose the
rotation search direction, then apply modified Newton angle optimization.

Falls back to steepest descent when γ < 0 or the CG direction is worse
than the raw force.

Reference: MATLAB rot_iter_cg.m
"""
function rotate_dimer_cg!(
    state::DimerState,
    model::GPModel,
    config::DimerConfig,
    cg_state::Union{Nothing,CGRotationState};
    y_std::Float64=1.0,
    verbose::Bool=false,
)
    D = length(state.R)
    cg = cg_state === nothing ? CGRotationState(D) : cg_state

    for rot_iter in 1:(config.max_rot_iter)
        # Check CG reset
        if cg.iter >= cg.max_iter
            reset_cg!(cg)
        end
        cg.iter += 1

        G0, G1, _ = predict_dimer_gradients(state, model, y_std)

        F_rot = rotational_force(G0, G1, state.orient, state.dimer_sep)
        F_rot_norm = norm(F_rot)

        if F_rot_norm < 1e-10
            verbose && println("  CG rotation converged (F_rot ~ 0)")
            break
        end

        # PRP conjugate gradient direction
        if isempty(cg.F_rot_old)
            # First iteration: steepest descent
            F_modrot = copy(F_rot)
        else
            # Polak-Ribiere-Polyak gamma
            gamma =
                dot(F_rot - cg.F_rot_old, F_rot) / (dot(cg.F_rot_old, cg.F_rot_old) + 1e-18)

            if gamma < 0 || norm(gamma * cg.F_modrot_old) > F_rot_norm
                # Restart: steepest descent
                F_modrot = copy(F_rot)
                cg.iter = 1
            else
                # CG update using previous perpendicular direction
                F_modrot = F_rot + gamma * norm(cg.F_modrot_old) * cg.orient_rot_old
            end
        end

        # Project perpendicular to current orientation
        orient_rot = F_modrot - dot(F_modrot, state.orient) * state.orient
        orn = norm(orient_rot)
        if orn < 1e-12
            verbose && println("  CG rotation: degenerate direction")
            break
        end
        orient_rot ./= orn

        # Project force onto CG-chosen direction
        F_rot_oriented = dot(F_rot, orient_rot) * orient_rot

        # Save CG state
        cg.F_rot_old = copy(F_rot)
        cg.F_modrot_old = copy(F_modrot)

        orient_prev = copy(state.orient)

        # Apply modified Newton angle optimization
        C_est = rotate_dimer_newton!(state, model, F_rot_oriented, config; y_std, verbose)

        if C_est !== nothing
            # Update perpendicular direction for next CG step
            # The new orient_rot after rotation = the perpendicular component
            # of the old orient_rot in the new frame
            residual = orient_prev - dot(orient_prev, state.orient) * state.orient
            rn = norm(residual)
            cg.orient_rot_old = rn > 1e-12 ? residual / rn : orient_rot

            dtheta = acos(clamp(dot(orient_prev, state.orient), -1.0, 1.0))
            verbose && @printf(
                "  CG rot %d: |F_rot| = %.5f, Δθ = %.5f, C ≈ %+.3e\n",
                rot_iter,
                F_rot_norm,
                dtheta,
                C_est
            )
            if dtheta < config.T_angle_rot
                break
            end
        else
            reset_cg!(cg)
            break
        end
    end
end

# ==============================================================================
# Translation Strategies
# ==============================================================================

"""
    translate_dimer_lbfgs!(state, G0, G1, model, config, trans_hist, td; y_std, verbose)

Curvature-dependent translation using L-BFGS.

- **Negative curvature** (desired): L-BFGS step on the modified translational force
  `F_trans = G0 - 2*(G0·n̂)n̂`, with max step limit. Memory is maintained across
  inner iterations.
- **Positive curvature** (still searching): Fixed step along `-(G·n̂)*n̂ / ||...||`
  with conservative step length. L-BFGS memory is reset.

Returns `(R_new, step_taken)` or `(nothing, false)` if trust region exceeded.

Reference: MATLAB trans_iter_lbfgs.m
"""
function translate_dimer_lbfgs!(
    state::DimerState,
    G0::Vector{Float64},
    G1::Vector{Float64},
    config::DimerConfig,
    trans_hist::LBFGSHistory,
    td::TrainingData;
    F_trans_prev::Vector{Float64}=Float64[],
    y_std::Float64=1.0,
    verbose::Bool=false,
)
    orient = state.orient
    C = curvature(G0, G1, orient, state.dimer_sep)

    if C < 0
        # --- Negative curvature: L-BFGS on effective translational force ---
        F_trans = translational_force(G0, orient)
        F_norm = norm(F_trans)

        # L-BFGS expects the gradient (uphill); F_trans is the force (downhill
        # toward saddle), so the gradient is -F_trans.
        search_dir = compute_direction(trans_hist, -F_trans)

        step_len = norm(search_dir)

        if step_len > config.max_step
            search_dir .*= config.max_step / step_len
            # Reset memory on clipped step
            reset!(trans_hist)
            verbose && @printf(
                "  Trans L-BFGS: step clipped (%.4f -> %.4f), memory reset\n",
                step_len,
                config.max_step
            )
        end

        # search_dir = -H*(-F_trans) = H*F_trans points toward saddle
        R_new = state.R + search_dir

        return R_new, F_trans, C
    else
        # --- Positive curvature: simple step along dimer axis ---
        F_along = -dot(G0, orient) * orient
        fn = norm(F_along)
        if fn < 1e-12
            return state.R, zeros(length(state.R)), C
        end

        R_new = state.R + config.step_convex * (F_along / fn)

        # Reset L-BFGS memory when entering convex region
        reset!(trans_hist)

        verbose && @printf(
            "  Trans simple (convex): C = %+.3e, step = %.4f\n", C, config.step_convex
        )

        return R_new, F_along, C
    end
end

# ==============================================================================
# Main GP-Dimer Algorithm
# ==============================================================================

"""
    gp_dimer(oracle, x_init, orient_init, kernel; config, training_data, dimer_sep)

GP-dimer saddle point search.

Arguments:
- `oracle`: Function `x -> (E, G)` mapping flat coordinates to energy and gradient
- `x_init`: Starting configuration (flat coordinate vector)
- `orient_init`: Initial dimer orientation (will be normalized)
- `kernel`: Initial kernel (e.g., `MolInvDistSE(1.0, [0.5], Float64[])`)

Keyword arguments:
- `config`: `DimerConfig` with algorithm parameters
- `training_data`: Optional pre-existing `TrainingData`
- `dimer_sep`: Dimer half-separation (default 0.01)
- `on_step`: Optional callback `f(info::Dict) -> Any` called after each oracle evaluation.
  Return `:stop` to trigger early termination.

The algorithm alternates between:
1. Training the GP on accumulated oracle data
2. Inner loop: rotating dimer to lowest curvature mode and translating with
   modified force (all on the GP surface)
3. Calling the oracle when the GP converges or trust region is exceeded
4. Checking convergence on the true gradient and curvature

Rotation and translation methods are selected via `config.rotation_method` and
`config.translation_method`. See [`DimerConfig`](@ref) for options.

Returns a [`DimerResult`](@ref).
"""
function gp_dimer(
    oracle::Function,
    x_init::Vector{Float64},
    orient_init::Vector{Float64},
    kernel;
    config::DimerConfig=DimerConfig(),
    training_data::Union{Nothing,TrainingData}=nothing,
    dimer_sep::Float64=0.01,
    on_step::Union{Function,Nothing}=nothing,
)
    D = length(x_init)
    cfg = config

    # Initialize dimer state
    orient = orient_init / norm(orient_init)
    state = DimerState(copy(x_init), orient, dimer_sep)

    # Initialize training data
    td = if training_data === nothing
        TrainingData(D)
    else
        deepcopy(training_data)
    end

    # Generate initial training data if needed
    if npoints(td) == 0
        cfg.verbose && println("Generating initial training data...")

        for k in 1:(cfg.n_initial_perturb + 1)
            if k == 1
                x = copy(x_init)
            else
                perturb = (rand(D) .- 0.5) .* cfg.perturb_scale
                x = x_init + perturb
            end

            E, G = oracle(x)
            if isfinite(E) && E < 1e6
                add_point!(td, x, E, G)
                cfg.verbose && @printf("  Point %d: E = %.4f\n", k, E)
            end
        end
    end

    oracle_calls = npoints(td)

    cfg.verbose && println("="^70)
    cfg.verbose && println("GP-Dimer Saddle Point Search")
    cfg.verbose && @printf(
        "  Rotation: %s | Translation: %s\n",
        cfg.rotation_method,
        cfg.translation_method
    )
    cfg.verbose && println("="^70)
    cfg.verbose && @printf(
        "Training points: %d | Dimer sep: %.4f | Trust radius: %.3f\n\n",
        oracle_calls,
        dimer_sep,
        cfg.trust_radius
    )

    history = Dict(
        "E_true" => Float64[],
        "F_true" => Float64[],
        "curv_true" => Float64[],
        "oracle_calls" => Int[],
    )

    # Initialize rotation/translation state
    rot_hist = cfg.rotation_method == :lbfgs ? LBFGSHistory(cfg.lbfgs_memory) : nothing
    cg_state = cfg.rotation_method == :cg ? CGRotationState(D) : nothing
    trans_hist = cfg.translation_method == :lbfgs ? LBFGSHistory(cfg.lbfgs_memory) : nothing
    F_trans_prev = Float64[]

    converged = false
    stop_reason = MAX_ITERATIONS
    stagnation_count = 0
    prev_f_true = -Inf

    for outer_iter in 1:(cfg.max_outer_iter)
        cfg.verbose && println("-"^70)
        cfg.verbose &&
            @printf("OUTER ITERATION %d (Oracle calls: %d)\n", outer_iter, oracle_calls)

        # Train GP on current data
        y_gp, y_mean, y_std = normalize(td)

        model = GPModel(
            kernel, td.X, y_gp; noise_var=1e-2, grad_noise_var=1e-1, jitter=1e-3
        )

        cfg.verbose && @printf("Training GP on %d points...\n", npoints(td))
        train_model!(model; iterations=cfg.gp_train_iter)

        # Reset L-BFGS/CG state for new outer iteration (new GP model)
        rot_hist !== nothing && reset!(rot_hist)
        cg_state !== nothing && reset_cg!(cg_state)
        trans_hist !== nothing && reset!(trans_hist)
        F_trans_prev = Float64[]

        # Inner loop: optimize on GP surface
        R_prev = copy(state.R)

        for inner_iter in 1:(cfg.max_inner_iter)
            # Rotate dimer to find lowest curvature mode
            rotate_dimer!(state, model, cfg; y_std, verbose=false, rot_hist, cg_state)

            # Predict at current position
            G0, G1, E0 = predict_dimer_gradients(state, model, y_std)
            E0_pred = E0 + y_mean  # de-normalize (E0 was already scaled by y_std)

            if cfg.translation_method == :lbfgs && trans_hist !== nothing
                # L-BFGS translation
                R_new, F_trans_cur, C = translate_dimer_lbfgs!(
                    state, G0, G1, cfg, trans_hist, td; F_trans_prev, y_std, verbose=false
                )
                F_norm = norm(F_trans_cur)

                # Update L-BFGS history for translation
                if !isempty(F_trans_prev)
                    s = R_new - R_prev
                    y = -(F_trans_cur - F_trans_prev)
                    push_pair!(trans_hist, s, y)
                end
                F_trans_prev = copy(F_trans_cur)
            else
                # Simple translation (original algorithm)
                F_trans = translational_force(G0, state.orient)
                F_norm = norm(F_trans)
                C = curvature(G0, G1, state.orient, state.dimer_sep)

                if abs(C) > 1e-6
                    step_size = min(cfg.alpha_trans, 0.1 * F_norm / abs(C))
                else
                    step_size = cfg.alpha_trans
                end

                R_new = state.R + step_size * F_trans
            end

            min_dist = min_distance_to_data(state.R, td.X)

            if inner_iter % 10 == 0 || inner_iter == 1
                cfg.verbose && @printf(
                    "  GP step %3d: E = %8.4f | |F| = %.5f | C = %+.3e | d_min = %.4f\n",
                    inner_iter,
                    E0_pred,
                    F_norm,
                    C,
                    min_dist
                )
            end

            # Check GP convergence
            if F_norm < cfg.T_force_gp
                cfg.verbose && println("  Converged on GP surface!")
                break
            end

            # Check trust radius
            min_dist_new = min_distance_to_data(R_new, td.X)
            if min_dist_new > cfg.trust_radius
                cfg.verbose && @printf(
                    "  Trust radius exceeded (%.4f > %.4f)\n",
                    min_dist_new,
                    cfg.trust_radius
                )
                # Scale step to stay within trust radius
                step_vec = R_new - state.R
                scale = cfg.trust_radius / min_dist_new * 0.95
                R_prev = copy(state.R)
                state.R = state.R + scale * step_vec
                break
            end

            # Check inter-atomic distance ratios
            if !check_interatomic_ratio(R_new, td.X, cfg.ratio_at_limit)
                cfg.verbose && println("  Inter-atomic distance ratio violated")
                break
            end

            # Accept step
            R_prev = copy(state.R)
            state.R = R_new
        end

        # Call oracle at current position
        cfg.verbose && println("Calling oracle...")
        E_true, G_true = oracle(state.R)
        oracle_calls += 1

        # Also evaluate at R1 for curvature
        _, R1, _ = dimer_images(state)
        E1_true, G1_true = oracle(R1)
        oracle_calls += 1

        C_true = curvature(G_true, G1_true, state.orient, state.dimer_sep)
        F_trans_true = translational_force(G_true, state.orient)
        F_norm_true = norm(F_trans_true)

        cfg.verbose && @printf(
            "  True: E = %8.4f | |F| = %.5f | C = %+.3e\n", E_true, F_norm_true, C_true
        )

        # Force decomposition for verbose output
        if cfg.verbose
            F_par = dot(G_true, state.orient)
            F_perp_norm = norm(G_true - F_par * state.orient)
            @printf("  Force decomp: F_par=%+.5f | F_perp=%.5f\n", F_par, F_perp_norm)
        end

        # Stagnation check (force-based: detect when translational force stops changing)
        if abs(F_norm_true - prev_f_true) < 1e-10
            stagnation_count += 1
        else
            stagnation_count = 0
        end
        prev_f_true = F_norm_true
        cfg.verbose && stagnation_count > 0 &&
            @printf("  Stagnation counter: %d/3\n", stagnation_count)

        if stagnation_count >= 3
            cfg.verbose && @printf("  Outer %d: Force stagnation (|F_trans| unchanged for 3 steps). Exiting.\n", outer_iter)
            stop_reason = FORCE_STAGNATION
            break
        end

        # Store history
        push!(history["E_true"], E_true)
        push!(history["F_true"], F_norm_true)
        push!(history["curv_true"], C_true)
        push!(history["oracle_calls"], oracle_calls)

        # Add to training set
        add_point!(td, state.R, E_true, G_true)
        add_point!(td, R1, E1_true, G1_true)

        # on_step callback
        if on_step !== nothing
            step_info = Dict{String,Any}(
                "step" => outer_iter,
                "energy" => E_true,
                "force_trans" => F_norm_true,
                "curvature" => C_true,
                "oracle_calls" => oracle_calls,
                "R" => copy(state.R),
                "orient" => copy(state.orient),
                "training_points" => npoints(td),
            )
            cb_result = on_step(step_info)
            if cb_result === :stop
                cfg.verbose && println("  Stopped by on_step callback.")
                stop_reason = USER_CALLBACK
                break
            end
        end

        # Check true convergence (small force + negative curvature)
        if F_norm_true < cfg.T_force_true && C_true < 0.0
            cfg.verbose && println("\n" * "="^70)
            cfg.verbose && println("CONVERGED TO SADDLE POINT!")
            cfg.verbose && println("="^70)
            cfg.verbose && @printf("Final Energy:    %.6f\n", E_true)
            cfg.verbose && @printf("Final |F|:       %.6f\n", F_norm_true)
            cfg.verbose && @printf("Final Curvature: %+.6f\n", C_true)
            cfg.verbose && @printf("Oracle calls:    %d\n", oracle_calls)
            converged = true
            stop_reason = CONVERGED
            break
        end

        if outer_iter == cfg.max_outer_iter
            cfg.verbose && println("\nMaximum outer iterations reached")
        end

        cfg.verbose && println()
    end

    cfg.verbose && @printf("Stop reason: %s\n", stop_reason)

    return DimerResult(state, converged, stop_reason, oracle_calls, history)
end
