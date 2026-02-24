# ==============================================================================
# GP-Guided Minimization
# ==============================================================================
#
# The core optimization loop from gpr_optim:
#
# 1. Evaluate oracle at initial point and a few perturbations
# 2. Train GP on accumulated data (molecular kernel path: fix_noise + data-dep init)
# 3. Optimize on GP surface using L-BFGS with trust region penalty
# 4. EMD-based trust clip (post-inner, matching GP-NEB/OTGPD pattern)
# 5. Evaluate oracle at the GP-predicted minimum
# 6. Check convergence on true gradient norm
# 7. If not converged, add new data and go to step 2

"""
    MinimizationConfig

Configuration for the GP-guided minimization loop.
All fields have sensible defaults via `@kwdef`.

# Fields
- `trust_radius::Float64`: Max distance from training data
- `conv_tol::Float64`: Gradient norm convergence threshold
- `max_iter::Int`: Max outer iterations (GP-guided steps)
- `max_oracle_calls::Int`: Max oracle evaluations (0 = no cap)
- `gp_opt_tol::Float64`: Convergence tolerance for GP inner optimization
- `gp_train_iter::Int`: Nelder-Mead iterations for GP hyperparameter training
- `n_initial_perturb::Int`: Number of perturbed initial points
- `perturb_scale::Float64`: Scale of initial perturbations
- `penalty_coeff::Float64`: Soft trust region penalty coefficient
- `max_move::Float64`: Per-atom max displacement (Ang)
- `dedup_tol::Float64`: 0 = auto (conv_tol * 0.1)
- `explosion_recovery::Symbol`: :perturb_best or :reset_prev
- `max_training_points::Int`: 0 = no pruning
- `rff_features::Int`: 0 = exact GP; >0 = RFF approximation
- `trust_metric::Symbol`: :emd or :euclidean
- `atom_types::Vector{Int}`: Atom types for EMD trust metric
- `use_adaptive_threshold::Bool`: Whether to use adaptive trust radius
- `adaptive_t_min::Float64`: Minimum adaptive trust radius
- `adaptive_delta_t::Float64`: Range of adaptive trust radius
- `adaptive_n_half::Int`: Number of points for half-saturation of trust radius
- `adaptive_A::Float64`: Steepness of adaptive trust radius curve
- `adaptive_floor::Float64`: Floor for adaptive trust radius
- `verbose::Bool`: Whether to print progress
"""
Base.@kwdef struct MinimizationConfig
    trust_radius::Float64 = 0.1     # Max distance from training data
    conv_tol::Float64 = 5e-3    # Gradient norm convergence threshold
    max_iter::Int = 500     # Max outer iterations (GP-guided steps)
    max_oracle_calls::Int = 0   # 0 = no cap; >0 = hard limit on oracle evaluations
    gp_opt_tol::Float64 = 1e-2    # Convergence tolerance for GP inner optimization
    gp_train_iter::Int = 300     # Nelder-Mead iterations for GP hyperparameter training
    n_initial_perturb::Int = 4       # Number of perturbed initial points
    perturb_scale::Float64 = 0.1     # Scale of initial perturbations
    penalty_coeff::Float64 = 1e3     # Soft trust region penalty coefficient
    max_move::Float64 = 0.1          # Per-atom max displacement (Ang)
    dedup_tol::Float64 = 0.0         # 0 = auto (conv_tol * 0.1)
    explosion_recovery::Symbol = :perturb_best  # or :reset_prev
    max_training_points::Int = 0     # 0 = no pruning
    rff_features::Int = 0            # 0 = exact GP; >0 = RFF approximation
    # Trust region metric and adaptive threshold (matching GP-NEB/OTGPD)
    trust_metric::Symbol = :emd
    atom_types::Vector{Int} = Int[]
    use_adaptive_threshold::Bool = false
    adaptive_t_min::Float64 = 0.15
    adaptive_delta_t::Float64 = 0.35
    adaptive_n_half::Int = 50
    adaptive_A::Float64 = 1.3
    adaptive_floor::Float64 = 0.2
    verbose::Bool = true
end

"""
    MinimizationResult

Result of a GP-guided minimization run.
"""
struct MinimizationResult
    x_final::Vector{Float64}           # Final position
    E_final::Float64                   # Final energy
    G_final::Vector{Float64}           # Final gradient
    converged::Bool                    # Whether convergence criterion was met
    stop_reason::StopReason            # Why the optimizer terminated
    oracle_calls::Int                  # Total oracle evaluations
    trajectory::Vector{Vector{Float64}} # All evaluated configurations
    energies::Vector{Float64}          # All evaluated energies
end

"""
    gp_minimize(oracle, x_init, kernel; config, training_data)

GP-guided minimization of an oracle function.

Arguments:
- `oracle`: Function `x -> (E, G)` mapping flat coordinates to energy and gradient
- `x_init`: Starting configuration (flat coordinate vector)
- `kernel`: Initial kernel (e.g., `MolInvDistSE(1.0, [0.5], Float64[])`)

Keyword arguments:
- `config`: `MinimizationConfig` with algorithm parameters
- `training_data`: Optional pre-existing `TrainingData` to warm-start from
- `on_step`: Optional callback `f(info::Dict) -> Any` called after each oracle evaluation.
  Return `:stop` to trigger early termination.

Returns a `MinimizationResult`.
"""
function gp_minimize(
    oracle::Function,
    x_init::Vector{Float64},
    kernel;
    config::MinimizationConfig=MinimizationConfig(),
    training_data::Union{Nothing,TrainingData}=nothing,
    on_step::Union{Function,Nothing}=nothing,
)
    D = length(x_init)
    cfg = config
    is_mol = kernel isa AbstractMoleculeKernel

    # Initialize or use provided training data
    td = if training_data === nothing
        TrainingData(D)
    else
        deepcopy(training_data)
    end

    trajectory = Vector{Float64}[]
    all_energies = Float64[]

    # Step 1: Generate initial training data
    if npoints(td) == 0
        cfg.verbose && println("Generating initial training data...")

        # Evaluate at starting point
        E, G = oracle(x_init)
        add_point!(td, x_init, E, G)
        push!(trajectory, copy(x_init))
        push!(all_energies, E)
        cfg.verbose && @printf("  Initial: E = %.4f\n", E)

        # Evaluate at perturbed points
        for k in 1:(cfg.n_initial_perturb)
            perturb = (rand(D) .- 0.5) .* cfg.perturb_scale
            x_p = x_init + perturb
            E_p, G_p = oracle(x_p)

            if isfinite(E_p) && E_p < 1e6
                add_point!(td, x_p, E_p, G_p)
                push!(trajectory, copy(x_p))
                push!(all_energies, E_p)
                cfg.verbose && @printf("  Perturb %d: E = %.4f\n", k, E_p)
            end
        end
    end

    x_curr = copy(x_init)
    oracle_calls = npoints(td)

    cfg.verbose && println("\n=== Starting GP Minimization ===")
    cfg.verbose && @printf(
        "Trust radius: %.3f | Conv. tol: %.1e | Training points: %d\n\n",
        cfg.trust_radius,
        cfg.conv_tol,
        oracle_calls
    )

    converged = false
    stop_reason = MAX_ITERATIONS
    eff_dedup = cfg.dedup_tol > 0 ? cfg.dedup_tol : cfg.conv_tol * 0.1
    prev_kern = nothing  # warm-start kernel across outer iterations

    stagnation_count = 0
    prev_force = -Inf

    for outer_step in 1:(cfg.max_iter)
        if cfg.max_oracle_calls > 0 && oracle_calls >= cfg.max_oracle_calls
            cfg.verbose && @printf("Reached oracle call cap (%d). Stopping.\n", cfg.max_oracle_calls)
            stop_reason = ORACLE_CAP
            break
        end

        cfg.verbose && println("-"^60)
        cfg.verbose &&
            @printf("OUTER ITERATION %d (Oracle calls: %d)\n", outer_step, oracle_calls)

        # Step 2: Train GP on current data
        # Two paths matching GP-NEB _train_neb_gp:
        #   Molecular kernels: fix_noise=true, low noise, data-dependent init
        #   Generic kernels:   normalize, optimize all hyperparameters
        cfg.verbose && @printf("Training GP on %d points...\n", npoints(td))
        _t_train = time()

        if is_mol
            E_ref = td.energies[1]
            y_gp = vcat(td.energies .- E_ref, td.gradients)
            y_mean = E_ref
            y_std = 1.0

            kern = prev_kern === nothing ? init_mol_invdist_se(td, kernel) : prev_kern
            # Clamp inv_lengthscales and signal_variance to finite, positive ranges
            # to prevent failure when Nelder-Mead drives values to Inf/NaN.
            _eps_ph = 1e-6
            _max_ph = 1e10
            clamped_ls = [isfinite(x) ? clamp(x, _eps_ph, _max_ph) : _max_ph for x in kern.inv_lengthscales]
            clamped_sv = isfinite(kern.signal_variance) ? clamp(kern.signal_variance, _eps_ph, _max_ph) : _max_ph
            
            if clamped_ls != kern.inv_lengthscales || clamped_sv != kern.signal_variance
                kern = typeof(kern)(
                    clamped_sv,
                    clamped_ls,
                    kern.frozen_coords,
                    kern.feature_params_map,
                )
            end
            gp_model = GPModel(
                kern, td.X, y_gp; noise_var=1e-6, grad_noise_var=1e-4, jitter=1e-6
            )
            train_model!(
                gp_model; iterations=cfg.gp_train_iter, fix_noise=true, verbose=cfg.verbose
            )
            prev_kern = gp_model.kernel
        else
            y_gp, y_mean, y_std = normalize(td)
            gp_model = GPModel(
                kernel, td.X, y_gp; noise_var=1e-2, grad_noise_var=1e-1, jitter=1e-3
            )
            train_model!(gp_model; iterations=cfg.gp_train_iter, verbose=cfg.verbose)
        end

        _dt_train = time() - _t_train
        if cfg.verbose
            @printf("  GP trained in %.2f s", _dt_train)
            if gp_model.kernel isa AbstractMoleculeKernel
                k = gp_model.kernel
                @printf(" | sig_var=%.3e | inv_ls=[%s]",
                    k.signal_variance,
                    join([@sprintf("%.2e", l) for l in k.inv_lengthscales[1:min(3, length(k.inv_lengthscales))]], ", ") *
                    (length(k.inv_lengthscales) > 3 ? ", ..." : ""))
            end
            println()
        end

        # Use RFF approximation if configured (faster for high-D systems)
        model = if cfg.rff_features > 0 && kernel isa MolInvDistSE
            noise_e = is_mol ? 1e-6 : 1e-2
            noise_g = is_mol ? 1e-4 : 1e-1
            rff = build_rff(
                gp_model.kernel,
                td.X,
                y_gp,
                cfg.rff_features;
                noise_var=noise_e,
                grad_noise_var=noise_g,
            )
            cfg.verbose && @printf(
                "  RFF: %d features, %d training points\n",
                cfg.rff_features,
                npoints(td)
            )
            rff
        else
            gp_model
        end

        # Step 3: Optimize on GP surface using L-BFGS
        # Objective: GP-predicted energy + soft trust region penalty
        function gp_objective(x)
            preds = predict(model, reshape(x, :, 1))
            E = preds[1] * y_std + y_mean

            # Soft trust region penalty
            min_dist = min_distance_to_data(x, td.X)
            if min_dist > cfg.trust_radius
                E += cfg.penalty_coeff * (min_dist - cfg.trust_radius)^2
            end
            return E
        end

        function gp_gradient!(G, x)
            preds = predict(model, reshape(x, :, 1))
            G_pred = preds[2:end] .* y_std

            # Trust region penalty gradient
            min_dist = Inf
            nearest_idx = 1
            for i in 1:size(td.X, 2)
                d = norm(x - td.X[:, i])
                if d < min_dist
                    min_dist = d
                    nearest_idx = i
                end
            end

            if min_dist > cfg.trust_radius
                direction = (x - td.X[:, nearest_idx]) / (min_dist + 1e-10)
                penalty_grad =
                    2 * cfg.penalty_coeff * (min_dist - cfg.trust_radius) * direction
                G .= G_pred + penalty_grad
            else
                G .= G_pred
            end
        end

        x_prev = copy(x_curr)

        # Warm-start from best training point if it's better than current
        best_idx = argmin(td.energies)
        x_start = td.energies[best_idx] < all_energies[end] ? td.X[:, best_idx] : x_curr

        result = Optim.optimize(
            gp_objective,
            gp_gradient!,
            x_start,
            LBFGS(),
            Optim.Options(; g_tol=cfg.gp_opt_tol, iterations=100, show_trace=false),
        )

        x_curr = Optim.minimizer(result)

        # Ensure we don't get stuck: if x_curr is too close to x_prev or existing data, 
        # and we haven't converged, use a Lower Confidence Bound (LCB) acquisition
        # function to intelligently explore the GP surface.
        if norm(x_curr - x_prev) < eff_dedup || min_distance_to_data(x_curr, td.X) < eff_dedup
            cfg.verbose && println("  Stuck or duplicate point predicted - seeking improvement via LCB...")
            
            # LCB objective for minimization (mu - kappa * sigma)
            # kappa=2.0 corresponds to roughly 95% confidence
            function lcb_objective(x)
                mu_all, var_all = predict_with_variance(gp_model, reshape(x, :, 1))
                mu = mu_all[1] * y_std + y_mean
                sigma = sqrt(max(var_all[1], 1e-12)) * y_std
                return mu - 2.0 * sigma
            end

            # Optimize LCB starting from best training point
            # Use NelderMead as we don't have gradients for sigma
            best_idx = argmin(td.energies)
            res_lcb = Optim.optimize(
                lcb_objective,
                td.X[:, best_idx],
                NelderMead(),
                Optim.Options(; iterations=100, show_trace=false),
            )
            x_curr = Optim.minimizer(res_lcb)
            cfg.verbose && @printf("  LCB candidate: obj=%.4f, dist_from_prev=%.4f\n",
                Optim.minimum(res_lcb), norm(x_curr - x_prev))
        end

        # Per-atom max-move clip for 3D molecular systems
        D_curr = length(x_curr)
        n_at = div(D_curr, 3)
        if n_at >= 2 && D_curr == 3 * n_at
            x_curr = x_prev + _clip_to_max_move(x_curr - x_prev, cfg.max_move, 3)
        end

        # EMD-based trust clip (post-inner, matching GP-NEB pattern)
        n_atoms = div(D, 3)
        thresh = adaptive_trust_threshold(
            cfg.trust_radius,
            npoints(td),
            n_atoms;
            use_adaptive=cfg.use_adaptive_threshold,
            t_min=cfg.adaptive_t_min,
            delta_t=cfg.adaptive_delta_t,
            n_half=cfg.adaptive_n_half,
            A=cfg.adaptive_A,
            floor=cfg.adaptive_floor,
        )
        d_trust = trust_min_distance(x_curr, td.X, cfg.trust_metric; atom_types=cfg.atom_types)
        cfg.verbose && @printf("  Trust: d=%.4f, thresh=%.4f (%s)\n",
            d_trust, thresh, d_trust > thresh ? "CLIPPED" : "ok")
        if d_trust > thresh
            dist_fn = trust_distance_fn(cfg.trust_metric, cfg.atom_types)
            nearest_idx = argmin(
                dist_fn(x_curr, view(td.X, :, j)) for j in 1:npoints(td)
            )
            nearest = td.X[:, nearest_idx]
            disp = x_curr - nearest
            x_curr = nearest + disp * (thresh / d_trust * 0.95)
        end

        # Step 4: Call oracle at new point
        cfg.verbose && println("Calling oracle...")
        E_true, G_true = oracle(x_curr)
        oracle_calls += 1
        # Per-atom max force for molecular systems (3D per atom);
        # fall back to full norm for non-molecular (e.g. 2D) coordinates
        D_g = length(G_true)
        n_atoms_g = div(D_g, 3)
        G_norm = if n_atoms_g >= 1 && D_g == 3 * n_atoms_g
            maximum(norm(@view G_true[(3 * (a - 1) + 1):(3 * a)]) for a in 1:n_atoms_g)
        else
            norm(G_true)
        end

        cfg.verbose && @printf("  True: E = %.4f | max|F_atom| = %.5f\n", E_true, G_norm)

        # Stagnation check (force-based: detect when max force stops changing)
        if abs(G_norm - prev_force) < 1e-10
            stagnation_count += 1
        else
            stagnation_count = 0
        end
        prev_force = G_norm
        cfg.verbose && stagnation_count > 0 &&
            @printf("  Stagnation counter: %d/3\n", stagnation_count)

        if stagnation_count >= 3
            cfg.verbose && @printf("  Outer %d: Force stagnation (max|F| unchanged for 3 steps). Exiting.\n", outer_step)
            stop_reason = FORCE_STAGNATION
            break
        end

        # Explosion recovery
        if !isfinite(E_true) || E_true > 1e6
            cfg.verbose && println("  Energy exploded - recovering")
            if cfg.explosion_recovery == :perturb_best
                best_idx = argmin(td.energies)
                x_curr = td.X[:, best_idx] + (rand(D) .- 0.5) .* (cfg.perturb_scale * 0.5)
            else
                x_curr = td.X[:, end]
            end
            continue
        end

        # Always record in trajectory for convergence plots
        push!(trajectory, copy(x_curr))
        push!(all_energies, E_true)

        # Conditionally add to GP training set (deduplication)
        if min_distance_to_data(x_curr, td.X) > eff_dedup
            add_point!(td, x_curr, E_true, G_true)
        end

        # Optional training data pruning
        if cfg.max_training_points > 0
            prune_training_data!(
                td, x_curr, cfg.max_training_points; distance_fn=(a, b) -> norm(a - b)
            )
        end

        # on_step callback
        if on_step !== nothing
            step_info = Dict{String,Any}(
                "step" => outer_step,
                "energy" => E_true,
                "max_force" => G_norm,
                "oracle_calls" => oracle_calls,
                "x" => copy(x_curr),
                "training_points" => npoints(td),
            )
            cb_result = on_step(step_info)
            if cb_result === :stop
                cfg.verbose && println("  Stopped by on_step callback.")
                stop_reason = USER_CALLBACK
                break
            end
        end

        # Step 5: Check TRUE convergence
        if G_norm < cfg.conv_tol
            cfg.verbose && println("\n" * "="^60)
            cfg.verbose && println("CONVERGED!")
            cfg.verbose && @printf("Final Energy: %.6f\n", E_true)
            cfg.verbose && @printf("Final |grad|: %.6f\n", G_norm)
            cfg.verbose && @printf("Oracle calls: %d\n", oracle_calls)
            cfg.verbose && println("="^60)
            converged = true
            stop_reason = CONVERGED
            break
        end

        cfg.verbose && println()
    end

    cfg.verbose && @printf("Stop reason: %s\n", stop_reason)

    E_final, G_final = oracle(x_curr)

    return MinimizationResult(
        x_curr, E_final, G_final, converged, stop_reason, oracle_calls, trajectory, all_energies
    )
end
