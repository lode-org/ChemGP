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
# Reference: Henkelman & Jonsson, J. Chem. Phys. 111, 7010 (1999)
# GP-Dimer: Koistinen et al., J. Chem. Theory Comput. 16, 499 (2020)

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
"""
Base.@kwdef struct DimerConfig
    T_force_true::Float64   = 1e-3     # True convergence threshold (on oracle gradient)
    T_force_gp::Float64     = 1e-2     # GP convergence threshold (on GP gradient)
    T_angle_rot::Float64    = 1e-3     # Rotation angle convergence threshold
    trust_radius::Float64   = 0.1      # Max distance from training data
    ratio_at_limit::Float64 = 2 / 3    # Inter-atomic distance ratio limit
    max_outer_iter::Int     = 50       # Max outer iterations (oracle call cycles)
    max_inner_iter::Int     = 100      # Max GP steps per outer iteration
    max_rot_iter::Int       = 10       # Max rotations per translation step
    alpha_trans::Float64    = 0.01     # Translation step size factor
    gp_train_iter::Int      = 300      # Nelder-Mead iterations for GP training
    n_initial_perturb::Int  = 4        # Number of initial perturbation points
    perturb_scale::Float64  = 0.15     # Scale of initial perturbations
    verbose::Bool           = true
end

"""
    DimerResult

Result of a GP-dimer saddle point search.
"""
struct DimerResult
    state::DimerState               # Final dimer state (position + orientation)
    converged::Bool                 # Whether convergence criterion was met
    oracle_calls::Int               # Total oracle evaluations
    history::Dict{String,Vector}    # Convergence history
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

Modified force for translation toward a saddle point. The component
along the dimer direction is inverted, so the algorithm climbs along
the lowest curvature mode while descending in all other directions.

    F_trans = G0 - 2*(G0 . orient)*orient
"""
function translational_force(G0, orient)
    F_parallel = dot(G0, orient) * orient
    F_perp = G0 - F_parallel
    F_trans = F_perp - F_parallel  # Invert along dimer direction
    return F_trans
end

# ==============================================================================
# Dimer Rotation
# ==============================================================================

"""
    rotate_dimer!(state, model, config)

Rotate the dimer to align with the lowest curvature mode using GP predictions.
The rotation is performed iteratively, estimating the rotational force from
GP predictions at the dimer endpoints.
"""
function rotate_dimer!(state::DimerState, model::GPModel, config::DimerConfig;
                       y_std::Float64 = 1.0, verbose::Bool = false)
    for rot_iter in 1:(config.max_rot_iter)
        # Get current images
        _, R1, R2 = dimer_images(state)

        # Predict gradients at both endpoints
        pred1 = predict(model, reshape(R1, :, 1))
        pred2 = predict(model, reshape(R2, :, 1))

        G1 = pred1[2:end] .* y_std
        G2 = pred2[2:end] .* y_std
        G0 = -(G1 + G2) / 2  # Estimate gradient at center

        # Compute rotational force
        F_rot = rotational_force(G0, G1, state.orient, state.dimer_sep)
        F_rot_norm = norm(F_rot)

        if F_rot_norm < 1e-10
            verbose && println("  Rotation converged (F_rot ~ 0)")
            break
        end

        # Estimate rotation angle
        C = curvature(G0, G1, state.orient, state.dimer_sep)
        dtheta = 0.5 * atan(F_rot_norm / (abs(C) + 1e-10))

        if dtheta < config.T_angle_rot
            verbose && @printf("  Rotation converged (theta = %.5f < %.5f)\n",
                               dtheta, config.T_angle_rot)
            break
        end

        # Rotate orientation vector
        b1 = F_rot / F_rot_norm  # Unit vector perpendicular to orient
        orient_new = cos(dtheta) * state.orient + sin(dtheta) * b1
        state.orient = orient_new / norm(orient_new)  # Renormalize

        verbose && @printf("  Rotation %d: theta = %.5f, |F_rot| = %.5f\n",
                           rot_iter, dtheta, F_rot_norm)
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

The algorithm alternates between:
1. Training the GP on accumulated oracle data
2. Inner loop: rotating dimer to lowest curvature mode and translating with
   modified force (all on the GP surface)
3. Calling the oracle when the GP converges or trust region is exceeded
4. Checking convergence on the true gradient and curvature

Returns a `DimerResult`.
"""
function gp_dimer(
    oracle::Function,
    x_init::Vector{Float64},
    orient_init::Vector{Float64},
    kernel;
    config::DimerConfig = DimerConfig(),
    training_data::Union{Nothing,TrainingData} = nothing,
    dimer_sep::Float64 = 0.01,
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
    cfg.verbose && println("="^70)
    cfg.verbose && @printf("Training points: %d | Dimer sep: %.4f | Trust radius: %.3f\n\n",
                           oracle_calls, dimer_sep, cfg.trust_radius)

    history = Dict(
        "E_true" => Float64[],
        "F_true" => Float64[],
        "curv_true" => Float64[],
        "oracle_calls" => Int[],
    )

    converged = false

    for outer_iter in 1:(cfg.max_outer_iter)
        cfg.verbose && println("-"^70)
        cfg.verbose && @printf("OUTER ITERATION %d (Oracle calls: %d)\n", outer_iter, oracle_calls)

        # Train GP on current data
        y_gp, y_mean, y_std = normalize(td)

        model = GPModel(kernel, td.X, y_gp;
                        noise_var = 1e-2,
                        grad_noise_var = 1e-1,
                        jitter = 1e-3)

        cfg.verbose && @printf("Training GP on %d points...\n", npoints(td))
        train_model!(model, iterations = cfg.gp_train_iter)

        # Inner loop: optimize on GP surface
        for inner_iter in 1:(cfg.max_inner_iter)
            # Rotate dimer to find lowest curvature mode
            rotate_dimer!(state, model, cfg; y_std = y_std, verbose = false)

            # Predict at current position and R1
            R0, R1, _ = dimer_images(state)

            pred0 = predict(model, reshape(R0, :, 1))
            pred1 = predict(model, reshape(R1, :, 1))

            E0_pred = pred0[1] * y_std + y_mean
            G0_pred = pred0[2:end] .* y_std
            G1_pred = pred1[2:end] .* y_std

            F_trans = translational_force(G0_pred, state.orient)
            F_norm = norm(F_trans)
            C = curvature(G0_pred, G1_pred, state.orient, state.dimer_sep)

            min_dist = min_distance_to_data(state.R, td.X)

            if inner_iter % 10 == 0 || inner_iter == 1
                cfg.verbose && @printf("  GP step %3d: E = %8.4f | |F| = %.5f | C = %+.3e | d_min = %.4f\n",
                                       inner_iter, E0_pred, F_norm, C, min_dist)
            end

            # Check GP convergence
            if F_norm < cfg.T_force_gp
                cfg.verbose && println("  Converged on GP surface!")
                break
            end

            # Propose translation step (adaptive based on curvature)
            if abs(C) > 1e-6
                step_size = min(cfg.alpha_trans, 0.1 * F_norm / abs(C))
            else
                step_size = cfg.alpha_trans
            end

            R_new = state.R + step_size * F_trans

            # Check trust radius
            min_dist_new = min_distance_to_data(R_new, td.X)
            if min_dist_new > cfg.trust_radius
                cfg.verbose && @printf("  Trust radius exceeded (%.4f > %.4f)\n",
                                       min_dist_new, cfg.trust_radius)
                scale = cfg.trust_radius / min_dist_new * 0.95
                state.R = state.R + scale * step_size * F_trans
                break
            end

            # Check inter-atomic distance ratios
            if !check_interatomic_ratio(R_new, td.X, cfg.ratio_at_limit)
                cfg.verbose && println("  Inter-atomic distance ratio violated")
                break
            end

            # Accept step
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

        cfg.verbose && @printf("  True: E = %8.4f | |F| = %.5f | C = %+.3e\n",
                               E_true, F_norm_true, C_true)

        # Store history
        push!(history["E_true"], E_true)
        push!(history["F_true"], F_norm_true)
        push!(history["curv_true"], C_true)
        push!(history["oracle_calls"], oracle_calls)

        # Add to training set
        add_point!(td, state.R, E_true, G_true)
        add_point!(td, R1, E1_true, G1_true)

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
            break
        end

        if outer_iter == cfg.max_outer_iter
            cfg.verbose && println("\nMaximum outer iterations reached")
        end

        cfg.verbose && println()
    end

    return DimerResult(state, converged, oracle_calls, history)
end
