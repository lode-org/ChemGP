# ==============================================================================
# GP-Guided Minimization
# ==============================================================================
#
# The core optimization loop from gpr_optim, simplified for pedagogy:
#
# 1. Evaluate oracle at initial point and a few perturbations
# 2. Train GP on accumulated data
# 3. Optimize on GP surface using L-BFGS with trust region penalty
# 4. Evaluate oracle at the GP-predicted minimum
# 5. Check convergence on true gradient norm
# 6. If not converged, add new data and go to step 2
#
# The key idea: the GP acts as a cheap surrogate for the expensive oracle.
# By optimizing on the GP surface (step 3), we find promising configurations
# without calling the oracle. The trust region ensures we don't extrapolate
# beyond where the GP is reliable.

"""
    MinimizationConfig

Configuration for the GP-guided minimization loop.
All fields have sensible defaults via `@kwdef`.
"""
Base.@kwdef struct MinimizationConfig
    trust_radius::Float64 = 0.1     # Max distance from training data
    conv_tol::Float64 = 5e-3    # Gradient norm convergence threshold
    max_iter::Int = 500     # Max outer iterations (oracle calls)
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

Returns a `MinimizationResult`.
"""
function gp_minimize(
    oracle::Function,
    x_init::Vector{Float64},
    kernel;
    config::MinimizationConfig=MinimizationConfig(),
    training_data::Union{Nothing,TrainingData}=nothing,
)
    D = length(x_init)
    cfg = config

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
    eff_dedup = cfg.dedup_tol > 0 ? cfg.dedup_tol : cfg.conv_tol * 0.1

    for outer_step in 1:(cfg.max_iter)
        cfg.verbose && println("-"^60)
        cfg.verbose &&
            @printf("OUTER ITERATION %d (Oracle calls: %d)\n", outer_step, oracle_calls)

        # Step 2: Train GP on current data
        y_gp, y_mean, y_std = normalize(td)

        gp_model = GPModel(
            kernel, td.X, y_gp; noise_var=1e-2, grad_noise_var=1e-1, jitter=1e-3
        )

        cfg.verbose && @printf("Training GP on %d points...\n", npoints(td))
        train_model!(gp_model; iterations=cfg.gp_train_iter)

        # Use RFF approximation if configured (faster for high-D systems)
        model = if cfg.rff_features > 0 && kernel isa MolInvDistSE
            rff = build_rff(
                gp_model.kernel,
                td.X,
                y_gp,
                cfg.rff_features;
                noise_var=1e-2,
                grad_noise_var=1e-1,
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

        result = Optim.optimize(
            gp_objective,
            gp_gradient!,
            x_curr,
            LBFGS(),
            Optim.Options(; g_tol=cfg.gp_opt_tol, iterations=100, show_trace=false),
        )

        x_curr = Optim.minimizer(result)

        # Hard trust clip: limit total displacement from anchor
        disp = x_curr - x_prev
        dn = norm(disp)
        if dn > cfg.trust_radius
            x_curr = x_prev + disp * (cfg.trust_radius / dn)
        end

        # Per-atom max-move clip for 3D molecular systems
        D_curr = length(x_curr)
        n_at = div(D_curr, 3)
        if n_at >= 2 && D_curr == 3 * n_at
            x_curr = x_prev + _clip_to_max_move(x_curr - x_prev, cfg.max_move, 3)
        end

        # Step 4: Call oracle at new point
        cfg.verbose && println("Calling oracle...")
        E_true, G_true = oracle(x_curr)
        oracle_calls += 1
        # Per-atom max force for molecular systems (3D per atom);
        # fall back to full norm for non-molecular (e.g. 2D) coordinates
        D_g = length(G_true)
        n_atoms = div(D_g, 3)
        G_norm = if n_atoms >= 1 && D_g == 3 * n_atoms
            maximum(norm(@view G_true[(3 * (a - 1) + 1):(3 * a)]) for a in 1:n_atoms)
        else
            norm(G_true)
        end

        cfg.verbose && @printf("  True: E = %.4f | max|F_atom| = %.5f\n", E_true, G_norm)

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

        # Step 5: Check TRUE convergence
        if G_norm < cfg.conv_tol
            cfg.verbose && println("\n" * "="^60)
            cfg.verbose && println("CONVERGED!")
            cfg.verbose && @printf("Final Energy: %.6f\n", E_true)
            cfg.verbose && @printf("Final |grad|: %.6f\n", G_norm)
            cfg.verbose && @printf("Oracle calls: %d\n", oracle_calls)
            cfg.verbose && println("="^60)
            converged = true
            break
        end

        cfg.verbose && println()
    end

    E_final, G_final = oracle(x_curr)

    return MinimizationResult(
        x_curr, E_final, G_final, converged, oracle_calls, trajectory, all_energies
    )
end
