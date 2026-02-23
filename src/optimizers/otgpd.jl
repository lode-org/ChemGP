# ==============================================================================
# Optimal Transport GP Dimer (OTGPD)
# ==============================================================================
#
# The full production algorithm from gpr_optim / gpr_dimer_matlab. Extends the
# basic GP-dimer (gp_dimer) with:
#
# 1. Adaptive GP convergence threshold that tightens as optimization progresses
# 2. Optional initial rotation phase on true potential (before GP interpolation)
# 3. Optional evaluation of image 1 at each outer iteration (more data per step)
# 4. Optional training data pruning to keep GP inference tractable
# 5. Enhanced stopping criteria and convergence monitoring
#
# Reference:
#   Goswami, R. et al. (2025). Efficient implementation of Gaussian process
#   regression accelerated saddle point searches with application to molecular
#   reactions. J. Chem. Theory Comput., doi:10.1021/acs.jctc.5c00866.
#
#   Goswami, R. & Jónsson, H. (2025). Adaptive pruning for increased robustness
#   and reduced computational overhead in Gaussian process accelerated saddle
#   point searches. ChemPhysChem, doi:10.1002/cphc.202500730.
#
#   Goswami, R. (2025). Efficient exploration of chemical kinetics. PhD thesis,
#   University of Iceland. arXiv:2510.21368.
#
#   Koistinen, O.-P. et al. (2020). Minimum mode saddle point searches using
#   Gaussian process surfaces. J. Chem. Theory Comput., 16, 499-509.

"""
    OTGPDConfig

Configuration for the Optimal Transport GP Dimer algorithm.

Extends [`DimerConfig`](@ref) with adaptive thresholds, initial rotation,
and data management options.

# Core convergence
- `T_dimer`: True max force threshold for final convergence
- `T_dimer_gp_init`: Initial GP convergence threshold
- `divisor_T_dimer_gp`: Adaptive divisor (>0 enables adaptive mode)
- `T_angle_rot`: Rotation angle convergence threshold

# Iteration limits
- `max_outer_iter`: Maximum outer iterations (oracle calls)
- `max_inner_iter`: Maximum GP steps per outer iteration
- `max_rot_iter`: Maximum rotations per translation step

# Dimer geometry and evaluation
- `dimer_sep`: Distance from midpoint to image 1
- `eval_image1`: Evaluate oracle at image 1 each outer iteration

# Rotation and translation
- `rotation_method`: `:lbfgs`, `:cg`, or `:simple`
- `translation_method`: `:lbfgs` or `:simple`
- `lbfgs_memory`: L-BFGS history depth
- `step_convex`: Step size for positive curvature regions
- `max_step`: Maximum translation step length

# Trust region
- `trust_radius`: Maximum distance from nearest training point
- `ratio_at_limit`: Inter-atomic distance ratio limit

# Initial rotation phase
- `initial_rotation`: Whether to perform initial rotation on true potential
- `max_initial_rot`: Maximum initial rotation iterations

# Data management
- `gp_train_iter`: GP hyperparameter optimization iterations
- `n_initial_perturb`: Number of initial perturbation points
- `perturb_scale`: Scale of initial perturbations
- `max_training_points`: Maximum training set size (0 = no pruning)
"""
Base.@kwdef struct OTGPDConfig
    # Core convergence
    T_dimer::Float64            = 0.01
    T_dimer_gp_init::Float64    = 0.001
    divisor_T_dimer_gp::Float64 = 10.0
    T_angle_rot::Float64        = 1e-3

    # Iteration limits
    max_outer_iter::Int         = 50
    max_inner_iter::Int         = 10000
    max_rot_iter::Int           = 5

    # Dimer geometry
    dimer_sep::Float64          = 0.01
    eval_image1::Bool           = true

    # Rotation/translation
    rotation_method::Symbol     = :lbfgs
    translation_method::Symbol  = :lbfgs
    lbfgs_memory::Int           = 5
    step_convex::Float64        = 0.1
    max_step::Float64           = 0.5
    alpha_trans::Float64        = 0.01

    # Trust region
    trust_radius::Float64       = 0.5
    ratio_at_limit::Float64     = 2 / 3

    # Initial rotation
    initial_rotation::Bool      = true
    max_initial_rot::Int        = 20

    # Data management
    gp_train_iter::Int          = 300
    n_initial_perturb::Int      = 4
    perturb_scale::Float64      = 0.15
    max_training_points::Int    = 0

    # FPS subset selection for hyperparameter optimization
    fps_history::Int            = 5     # subset size for hyperopt (0=use all)
    fps_latest_points::Int      = 2     # most recent points always included
    fps_metric::Symbol          = :emd  # :emd, :max_1d_log, :euclidean

    # Hyperparameter Oscillation Detection (HOD)
    use_hod::Bool               = true
    hod_monitoring_window::Int  = 5
    hod_flip_threshold::Float64 = 0.8
    hod_history_increment::Int  = 2
    hod_max_history::Int        = 30

    # Trust region metric and adaptive threshold
    trust_metric::Symbol         = :emd
    atom_types::Vector{Int}      = Int[]
    use_adaptive_threshold::Bool = false
    adaptive_t_min::Float64      = 0.15
    adaptive_delta_t::Float64    = 0.35
    adaptive_n_half::Int         = 50
    adaptive_A::Float64          = 1.3
    adaptive_floor::Float64      = 0.2

    verbose::Bool               = true
end

"""
    OTGPDResult

Result of an OTGPD saddle point search.

# Fields
- `state`: Final [`DimerState`](@ref) (position, orientation, separation)
- `converged`: Whether the algorithm converged
- `oracle_calls`: Total oracle evaluations
- `history`: Convergence history with keys:
  - `"E_true"`: True energies at oracle evaluations
  - `"F_true"`: True translational force norms
  - `"curv_true"`: True curvatures along dimer direction
  - `"oracle_calls"`: Cumulative oracle call count
  - `"T_gp"`: Adaptive GP threshold at each outer iteration
"""
struct OTGPDResult
    state::DimerState
    converged::Bool
    oracle_calls::Int
    history::Dict{String,Vector}
end

# ==============================================================================
# FPS subset selection for hyperparameter optimization
# ==============================================================================

# Thin wrapper calling shared trust utility (see distances_trust.jl)
_fps_distance_fn(metric::Symbol, atom_types::Vector{Int} = Int[]) =
    trust_distance_fn(metric, atom_types)

"""
    _select_optim_subset(td, x_current, n_select, n_latest; distance_fn)

Select a subset of training data for hyperparameter optimization using
farthest point sampling. Always includes the `n_latest` most recently
added points, then fills the rest via FPS from the remaining candidates.

Returns column indices into `td.X`.
"""
function _select_optim_subset(
    td::TrainingData,
    x_current::Vector{Float64},
    n_select::Int,
    n_latest::Int;
    distance_fn::Function = max_1d_log_distance,
)
    N = npoints(td)
    n_select <= 0 && return collect(1:N)
    n_select >= N && return collect(1:N)

    # Always include the most recent n_latest points
    n_latest = min(n_latest, N, n_select)
    latest_idx = collect((N - n_latest + 1):N)

    n_fps = n_select - n_latest
    if n_fps <= 0
        return latest_idx
    end

    # Candidates: everything except the latest points
    cand_idx = setdiff(1:N, latest_idx)
    if isempty(cand_idx)
        return latest_idx
    end

    # Build candidate and selected matrices for FPS
    cand_X = td.X[:, cand_idx]
    sel_X = td.X[:, latest_idx]

    fps_idx = farthest_point_sampling(cand_X, sel_X, n_fps;
                                       distance_fn = distance_fn)

    result = sort(vcat(latest_idx, cand_idx[fps_idx]))
    return result
end

"""
    _extract_subset(td, indices)

Create a new TrainingData from selected column indices.
"""
function _extract_subset(td::TrainingData, indices::Vector{Int})
    D = size(td.X, 1)
    sub = TrainingData(D)
    sub.X = td.X[:, indices]
    sub.energies = td.energies[indices]
    new_grads = Float64[]
    for i in indices
        s = (i - 1) * D + 1
        e = i * D
        append!(new_grads, td.gradients[s:e])
    end
    sub.gradients = new_grads
    return sub
end

# ==============================================================================
# Hyperparameter Oscillation Detection (HOD)
# ==============================================================================
#
# Monitors GP hyperparameters across outer iterations. If they oscillate
# (sign-flip in consecutive differences), the FPS subset is enlarged to
# provide more stable optimization landscape.
#
# Reference:
#   Goswami, R. & Jonsson, H. (2025). Adaptive Pruning for Increased
#   Robustness and Reduced Computational Overhead in Gaussian Process
#   Accelerated Saddle Point Searches. ChemPhysChem.

"""
    HODState

Mutable state for hyperparameter oscillation detection.
Tracks log-space hyperparameter vectors across iterations and counts
sign-flips in consecutive differences.
"""
mutable struct HODState
    history::Vector{Vector{Float64}}  # log-space hyperparams per iteration
    current_fps_history::Int          # current FPS subset size
end

HODState(initial_fps::Int) = HODState(Vector{Float64}[], initial_fps)

"""
    _extract_hyperparams(model::GPModel)

Extract GP hyperparameters in log-space for oscillation monitoring.
Returns a vector of log(signal_var), log.(inv_lengthscales), log(noise_var),
log(grad_noise_var).
"""
function _extract_hyperparams(model::GPModel)
    hp = Float64[]
    k = model.kernel
    if hasproperty(k, :signal_variance)
        push!(hp, log(max(k.signal_variance, 1e-30)))
    end
    if hasproperty(k, :inv_lengthscales)
        append!(hp, log.(max.(k.inv_lengthscales, 1e-30)))
    elseif hasproperty(k, :lengthscale)
        push!(hp, log(max(k.lengthscale, 1e-30)))
    end
    push!(hp, log(max(model.noise_var, 1e-30)))
    push!(hp, log(max(model.grad_noise_var, 1e-30)))
    return hp
end

"""
    _check_hod!(hod, model, cfg) -> Bool

Check for hyperparameter oscillation. Returns true if oscillation was
detected and the FPS subset was enlarged.
"""
function _check_hod!(hod::HODState, model::GPModel, cfg::OTGPDConfig)
    hp = _extract_hyperparams(model)
    push!(hod.history, hp)

    n = length(hod.history)
    n < 3 && return false  # need at least 3 points for 2 differences

    window = min(cfg.hod_monitoring_window, n - 2)
    n_flips = 0
    n_pairs = 0

    for i in max(n - window, 2):(n - 1)
        d1 = hod.history[i] - hod.history[i-1]
        d2 = hod.history[i+1] - hod.history[i]
        for j in eachindex(d1)
            if abs(d1[j]) > 1e-10 && abs(d2[j]) > 1e-10
                n_pairs += 1
                if sign(d1[j]) != sign(d2[j])
                    n_flips += 1
                end
            end
        end
    end

    flip_ratio = n_pairs > 0 ? n_flips / n_pairs : 0.0

    if flip_ratio > cfg.hod_flip_threshold
        new_fps = min(hod.current_fps_history + cfg.hod_history_increment,
                      cfg.hod_max_history)
        if new_fps > hod.current_fps_history
            hod.current_fps_history = new_fps
            cfg.verbose && @printf("  HOD: oscillation detected (%.0f%%), FPS history -> %d\n",
                                   flip_ratio * 100, new_fps)
            return true
        end
    end
    return false
end

# ==============================================================================
# Adaptive trust radius threshold
# ==============================================================================

# Thin wrapper calling shared trust utility (see distances_trust.jl)
function _adaptive_trust_threshold(cfg::OTGPDConfig, n_data::Int, n_atoms::Int)
    adaptive_trust_threshold(cfg.trust_radius, n_data, n_atoms;
        use_adaptive=cfg.use_adaptive_threshold, t_min=cfg.adaptive_t_min,
        delta_t=cfg.adaptive_delta_t, n_half=cfg.adaptive_n_half,
        A=cfg.adaptive_A, floor=cfg.adaptive_floor)
end

# Thin wrappers calling shared trust utilities (see distances_trust.jl)
function _trust_distance(x1::AbstractVector, x2::AbstractVector, cfg::OTGPDConfig)
    dist_fn = trust_distance_fn(cfg.trust_metric, cfg.atom_types)
    return dist_fn(x1, x2)
end

function _trust_min_distance(x::AbstractVector, td::TrainingData, cfg::OTGPDConfig)
    trust_min_distance(x, td.X, cfg.trust_metric; atom_types=cfg.atom_types)
end

# Convert OTGPD config to DimerConfig for use by rotation/translation functions
function _make_dimer_config(cfg::OTGPDConfig; T_force_gp::Float64 = 0.001)
    DimerConfig(;
        T_force_true = cfg.T_dimer,
        T_force_gp = T_force_gp,
        T_angle_rot = cfg.T_angle_rot,
        trust_radius = cfg.trust_radius,
        ratio_at_limit = cfg.ratio_at_limit,
        max_outer_iter = 1,  # Not used by rotation/translation
        max_inner_iter = 1,
        max_rot_iter = cfg.max_rot_iter,
        alpha_trans = cfg.alpha_trans,
        gp_train_iter = cfg.gp_train_iter,
        rotation_method = cfg.rotation_method,
        translation_method = cfg.translation_method,
        lbfgs_memory = cfg.lbfgs_memory,
        max_step = cfg.max_step,
        step_convex = cfg.step_convex,
        verbose = false,
    )
end

"""
    otgpd(oracle, x_init, orient_init, kernel; config, training_data) -> OTGPDResult

Optimal Transport GP Dimer saddle point search.

This is the full production algorithm that extends [`gp_dimer`](@ref) with:
- Adaptive GP convergence threshold that tightens as optimization progresses
- Optional initial rotation phase on the true potential
- Optional oracle evaluation at image 1 for additional training data
- Optional training data pruning for large datasets

# Arguments
- `oracle`: Function `x -> (E, G)` returning energy and gradient
- `x_init`: Starting configuration (flat coordinate vector)
- `orient_init`: Initial dimer orientation (will be normalized)
- `kernel`: GP kernel (e.g., `MolInvDistSE(...)`)

# Keyword arguments
- `config`: [`OTGPDConfig`](@ref) with algorithm parameters
- `training_data`: Optional pre-existing [`TrainingData`](@ref)

# Algorithm

**Phase 1 — Initial rotation** (optional): Rotate the dimer on the true
potential (not GP) to approximately align with the lowest curvature mode.
This avoids wasting GP evaluations on large rotations early in the search.

**Phase 2 — Main loop**: Alternates between GP-based optimization (inner loop)
and oracle evaluation (outer loop). The GP convergence threshold adapts:

```
T_gp = max(min(F_true_history) / divisor, T_dimer / 10)
```

As the true forces decrease, the GP threshold tightens, demanding more
precise GP optimization before querying the oracle again.

Returns an [`OTGPDResult`](@ref).

See also: [`gp_dimer`](@ref), [`DimerState`](@ref)
"""
function otgpd(
    oracle::Function,
    x_init::Vector{Float64},
    orient_init::Vector{Float64},
    kernel;
    config::OTGPDConfig = OTGPDConfig(),
    training_data::Union{Nothing,TrainingData} = nothing,
)
    D = length(x_init)
    cfg = config

    # Initialize dimer state
    orient = orient_init / norm(orient_init)
    state = DimerState(copy(x_init), orient, cfg.dimer_sep)

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

    # Evaluate at midpoint + image1 if not already in training data
    E_mid, G_mid = oracle(state.R)
    add_point!(td, state.R, E_mid, G_mid)
    oracle_calls += 1

    R1 = state.R + cfg.dimer_sep * state.orient
    E1, G1 = oracle(R1)
    add_point!(td, R1, E1, G1)
    oracle_calls += 1

    cfg.verbose && println("=" ^ 70)
    cfg.verbose && println("OTGPD — Optimal Transport GP Dimer")
    cfg.verbose && @printf("  Rotation: %s | Translation: %s\n",
                           cfg.rotation_method, cfg.translation_method)
    cfg.verbose && @printf("  Adaptive threshold: divisor = %.1f\n", cfg.divisor_T_dimer_gp)
    cfg.verbose && println("=" ^ 70)
    cfg.verbose && @printf("Training points: %d | Dimer sep: %.4f\n\n",
                           npoints(td), cfg.dimer_sep)

    history = Dict{String,Vector}(
        "E_true" => Float64[],
        "F_true" => Float64[],
        "curv_true" => Float64[],
        "oracle_calls" => Int[],
        "T_gp" => Float64[],
    )

    # Track true force history for adaptive threshold
    F_true_history = Float64[]

    # =========================================================================
    # Phase 1: Initial Rotation on True Potential
    # =========================================================================

    if cfg.initial_rotation && cfg.max_initial_rot > 0
        cfg.verbose && println("-" ^ 70)
        cfg.verbose && println("PHASE 1: Initial Rotation on True Potential")
        cfg.verbose && println("-" ^ 70)

        for init_rot in 1:cfg.max_initial_rot
            # Compute rotational force from true gradients
            _, R1_cur, _ = dimer_images(state)
            E0, G0 = oracle(state.R)
            E1, G1 = oracle(R1_cur)
            oracle_calls += 2
            add_point!(td, state.R, E0, G0)
            add_point!(td, R1_cur, E1, G1)

            F_rot = rotational_force(G0, G1, state.orient, state.dimer_sep)
            F_rot_norm = norm(F_rot)
            C = curvature(G0, G1, state.orient, state.dimer_sep)

            # Initial angle estimate
            dtheta = 0.5 * atan(0.5 * F_rot_norm / (abs(C) + 1e-10))

            cfg.verbose && @printf("  Init rot %d: |F_rot| = %.5f, C = %+.3e, dθ = %.5f\n",
                                   init_rot, F_rot_norm, C, dtheta)

            if dtheta < cfg.T_angle_rot
                cfg.verbose && println("  Initial rotation converged.")
                break
            end

            # Rotate using the rotational force direction
            orient_rot = F_rot / F_rot_norm

            # Trial rotation at dtheta
            orient_trial = cos(dtheta) * state.orient + sin(dtheta) * orient_rot
            orient_trial ./= norm(orient_trial)

            R1_trial = state.R + state.dimer_sep * orient_trial
            E1_trial, G1_trial = oracle(R1_trial)
            oracle_calls += 1
            add_point!(td, R1_trial, E1_trial, G1_trial)

            # Parabolic fit for optimal angle
            F_rot_trial = rotational_force(G0, G1_trial, orient_trial, state.dimer_sep)
            orient_rot_trial = -sin(dtheta) * state.orient + cos(dtheta) * orient_rot
            orient_rot_trial ./= norm(orient_rot_trial)

            F_dtheta = dot(F_rot_trial, orient_rot_trial)
            F_0 = dot(F_rot, orient_rot)

            sin2 = sin(2 * dtheta)
            if abs(sin2) > 1e-12
                cos2 = cos(2 * dtheta)
                a1 = (F_dtheta - F_0 * cos2) / sin2
                b1 = -0.5 * F_0

                angle_rot = 0.5 * atan(b1 / (a1 + 1e-18))
                if a1 * cos(2 * angle_rot) + b1 * sin(2 * angle_rot) > 0
                    angle_rot += π / 2
                end

                orient_new = cos(angle_rot) * state.orient + sin(angle_rot) * orient_rot
                state.orient = orient_new / norm(orient_new)
            else
                state.orient = orient_trial
            end
        end

        # Record force after initial rotation
        _, R1_final, _ = dimer_images(state)
        E0_fin, G0_fin = oracle(state.R)
        E1_fin, G1_fin = oracle(R1_final)
        oracle_calls += 2
        add_point!(td, state.R, E0_fin, G0_fin)
        add_point!(td, R1_final, E1_fin, G1_fin)

        F_trans_fin = translational_force(G0_fin, state.orient)
        F_fin = norm(F_trans_fin)
        C_fin = curvature(G0_fin, G1_fin, state.orient, state.dimer_sep)
        push!(F_true_history, F_fin)

        cfg.verbose && @printf("\n  After init rotation: |F| = %.5f, C = %+.3e\n\n",
                               F_fin, C_fin)
    end

    # =========================================================================
    # Phase 2: Main GP-Accelerated Loop
    # =========================================================================

    cfg.verbose && println("-" ^ 70)
    cfg.verbose && println("PHASE 2: GP-Accelerated Dimer Relaxation")
    cfg.verbose && println("-" ^ 70)

    # Build DimerConfig for internal rotation/translation functions
    dimer_cfg = _make_dimer_config(cfg)

    # Initialize rotation/translation state
    rot_hist = cfg.rotation_method == :lbfgs ? LBFGSHistory(cfg.lbfgs_memory) : nothing
    cg_state = cfg.rotation_method == :cg ? CGRotationState(D) : nothing
    trans_hist = cfg.translation_method == :lbfgs ? LBFGSHistory(cfg.lbfgs_memory) : nothing
    F_trans_prev = Float64[]

    converged = false
    hod_state = cfg.use_hod ? HODState(cfg.fps_history) : nothing
    n_atoms = div(D, 3)

    for outer_iter in 1:cfg.max_outer_iter
        cfg.verbose && @printf("\n--- Outer %d (oracle calls: %d, training: %d) ---\n",
                               outer_iter, oracle_calls, npoints(td))

        # Optional: prune training data
        if cfg.max_training_points > 0
            n_removed = prune_training_data!(td, state.R, cfg.max_training_points)
            if n_removed > 0
                cfg.verbose && @printf("  Pruned %d points (kept %d)\n", n_removed, npoints(td))
            end
        end

        # Train GP: two-path strategy depending on kernel type
        # FPS subset selection (shared by both paths)
        dist_fn = _fps_distance_fn(cfg.fps_metric, cfg.atom_types)
        fps_size = hod_state !== nothing ? hod_state.current_fps_history : cfg.fps_history
        if fps_size > 0 && npoints(td) > fps_size
            sub_idx = _select_optim_subset(td, state.R, fps_size,
                                            cfg.fps_latest_points;
                                            distance_fn = dist_fn)
            td_sub = _extract_subset(td, sub_idx)
            cfg.verbose && @printf("  FPS subset: %d/%d points\n",
                                   npoints(td_sub), npoints(td))
        else
            td_sub = td
        end

        n_sub = npoints(td_sub)
        barrier = min(1e-4 + 1e-3 * n_sub, 0.5)

        if kernel isa AbstractMoleculeKernel
            # Molecular kernel path: energy shift, fix_noise, warm-start
            E_ref_sub = td_sub.energies[1]
            y_sub = vcat(td_sub.energies .- E_ref_sub, td_sub.gradients)

            kern_sub = if outer_iter == 1
                init_mol_invdist_se(td_sub, kernel)
            else
                model.kernel  # warm-start from previous iteration
            end

            model_sub = GPModel(kern_sub, td_sub.X, y_sub;
                                noise_var = 1e-6, grad_noise_var = 1e-4, jitter = 1e-6)
            train_model!(model_sub; iterations = cfg.gp_train_iter,
                         fix_noise = true, verbose = cfg.verbose,
                         barrier_strength = barrier)

            # Rebuild on full data
            E_ref = td.energies[1]
            y_gp = vcat(td.energies .- E_ref, td.gradients)
            y_mean = E_ref
            y_std = 1.0
            model = GPModel(model_sub.kernel, td.X, y_gp;
                            noise_var = model_sub.noise_var,
                            grad_noise_var = model_sub.grad_noise_var,
                            jitter = model_sub.jitter)
        else
            # Generic kernel path: normalize, optimize noise
            y_sub, y_mean_sub, y_std_sub = normalize(td_sub)
            model_sub = GPModel(kernel, td_sub.X, y_sub;
                                noise_var = 1e-2,
                                grad_noise_var = 1e-1,
                                jitter = 1e-3)
            train_model!(model_sub; iterations = cfg.gp_train_iter,
                         verbose = cfg.verbose)

            # Rebuild on full data
            y_gp, y_mean, y_std = normalize(td)
            model = GPModel(model_sub.kernel, td.X, y_gp;
                            noise_var = model_sub.noise_var,
                            grad_noise_var = model_sub.grad_noise_var,
                            jitter = model_sub.jitter)
        end

        # HOD: check for hyperparameter oscillation, enlarge FPS if needed
        if hod_state !== nothing
            _check_hod!(hod_state, model_sub, cfg)
        end

        # Adaptive GP convergence threshold
        T_gp = if cfg.divisor_T_dimer_gp > 0 && !isempty(F_true_history)
            max(minimum(F_true_history) / cfg.divisor_T_dimer_gp, cfg.T_dimer / 10)
        else
            cfg.T_dimer / 10
        end

        push!(history["T_gp"], T_gp)
        cfg.verbose && @printf("  Adaptive T_gp = %.6f\n", T_gp)

        # Reset optimizer state for new GP model
        rot_hist !== nothing && reset!(rot_hist)
        cg_state !== nothing && reset_cg!(cg_state)
        trans_hist !== nothing && reset!(trans_hist)
        F_trans_prev = Float64[]

        # Inner loop: optimize on GP surface
        for inner_iter in 1:cfg.max_inner_iter
            # Rotate dimer to lowest curvature mode
            rotate_dimer!(state, model, dimer_cfg; y_std, verbose = false,
                          rot_hist, cg_state)

            # Predict at current position
            G0, G1, E0 = predict_dimer_gradients(state, model, y_std)
            E0_phys = E0 + y_mean

            if cfg.translation_method == :lbfgs && trans_hist !== nothing
                R_new, F_trans_cur, C = translate_dimer_lbfgs!(
                    state, G0, G1, dimer_cfg, trans_hist, td;
                    F_trans_prev, y_std, verbose = false,
                )
                F_norm = norm(F_trans_cur)

                if !isempty(F_trans_prev)
                    s = R_new - state.R
                    y = -(F_trans_cur - F_trans_prev)
                    push_pair!(trans_hist, s, y)
                end
                F_trans_prev = copy(F_trans_cur)
            else
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

            if inner_iter % 100 == 0 || inner_iter == 1
                min_dist = min_distance_to_data(state.R, td.X)
                cfg.verbose && @printf("  GP step %4d: E = %8.4f | |F| = %.5f | C = %+.3e | d = %.4f\n",
                                       inner_iter, E0_phys, F_norm, C, min_dist)
            end

            # Check GP convergence
            if F_norm < T_gp
                cfg.verbose && @printf("  GP converged at step %d (|F| = %.5f < T_gp = %.5f)\n",
                                       inner_iter, F_norm, T_gp)
                state.R = R_new
                break
            end

            # Trust region check (configurable metric, adaptive threshold)
            trust_thresh = _adaptive_trust_threshold(cfg, npoints(td), n_atoms)
            trust_dist = _trust_min_distance(R_new, td, cfg)
            if trust_dist > trust_thresh
                step_vec = R_new - state.R
                scale = trust_thresh / trust_dist * 0.95
                state.R = state.R + scale * step_vec
                cfg.verbose && @printf("  Trust radius at step %d (%.4f > %.4f)\n",
                                       inner_iter, trust_dist, trust_thresh)
                break
            end

            # Inter-atomic distance ratio check
            if !check_interatomic_ratio(R_new, td.X, cfg.ratio_at_limit)
                cfg.verbose && @printf("  Inter-atomic ratio violated at step %d\n", inner_iter)
                break
            end

            state.R = R_new
        end

        # Evaluate oracle at converged position
        cfg.verbose && println("  Oracle evaluation...")
        E_true, G_true = oracle(state.R)
        oracle_calls += 1
        add_point!(td, state.R, E_true, G_true)

        # Optionally evaluate image 1
        C_true = NaN
        if cfg.eval_image1
            _, R1_eval, _ = dimer_images(state)
            E1_true, G1_true = oracle(R1_eval)
            oracle_calls += 1
            add_point!(td, R1_eval, E1_true, G1_true)

            C_true = curvature(G_true, G1_true, state.orient, state.dimer_sep)
        end

        F_trans_true = translational_force(G_true, state.orient)
        F_norm_true = norm(F_trans_true)

        push!(F_true_history, F_norm_true)

        push!(history["E_true"], E_true)
        push!(history["F_true"], F_norm_true)
        push!(history["curv_true"], C_true)
        push!(history["oracle_calls"], oracle_calls)

        cfg.verbose && @printf("  True: E = %8.4f | |F| = %.5f | C = %+.3e\n",
                               E_true, F_norm_true, C_true)

        # Check convergence: small force + negative curvature
        if F_norm_true < cfg.T_dimer && (isnan(C_true) || C_true < 0.0)
            cfg.verbose && println("\n" * "=" ^ 70)
            cfg.verbose && println("OTGPD CONVERGED TO SADDLE POINT!")
            cfg.verbose && println("=" ^ 70)
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
    end

    return OTGPDResult(state, converged, oracle_calls, history)
end
