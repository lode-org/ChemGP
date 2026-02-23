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

    for outer_step in 1:(cfg.max_iter)
        cfg.verbose && println("-"^60)
        cfg.verbose &&
            @printf("OUTER ITERATION %d (Oracle calls: %d)\n", outer_step, oracle_calls)

        # Step 2: Train GP on current data
        y_gp, y_mean, y_std = normalize(td)

        model = GPModel(
            kernel, td.X, y_gp; noise_var=1e-2, grad_noise_var=1e-1, jitter=1e-3
        )

        cfg.verbose && @printf("Training GP on %d points...\n", npoints(td))
        train_model!(model; iterations=cfg.gp_train_iter)

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

        result = Optim.optimize(
            gp_objective,
            gp_gradient!,
            x_curr,
            LBFGS(),
            Optim.Options(; g_tol=cfg.gp_opt_tol, iterations=100, show_trace=false),
        )

        x_curr = Optim.minimizer(result)

        # Step 4: Call oracle at new point
        cfg.verbose && println("Calling oracle...")
        E_true, G_true = oracle(x_curr)
        oracle_calls += 1
        # Per-atom max force (matching eOn convention)
        n_atoms = div(length(G_true), 3)
        G_norm = maximum(norm(@view G_true[(3 * (a - 1) + 1):(3 * a)]) for a in 1:n_atoms)

        cfg.verbose && @printf("  True: E = %.4f | max|F_atom| = %.5f\n", E_true, G_norm)

        # Sanity check
        if !isfinite(E_true) || E_true > 1e6
            cfg.verbose && println("  Energy exploded - resetting to previous point")
            x_curr = td.X[:, end]
            continue
        end

        # Add to training set
        add_point!(td, x_curr, E_true, G_true)
        push!(trajectory, copy(x_curr))
        push!(all_energies, E_true)

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
