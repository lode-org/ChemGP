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
- `energy_regression_tol::Float64`: Revert to best if oracle E > E_best + tol AND forces large (0 = auto)
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
- `machine_output::String`: JSONL output destination; `""` = disabled,
  `"host:port"` = TCP socket to writer, otherwise = file path
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
    energy_regression_tol::Float64 = 0.0  # 0 = auto (max(std(E)*3, 1.0))
    max_training_points::Int = 0     # 0 = no pruning
    rff_features::Int = 0            # 0 = exact GP; >0 = RFF approximation
    # Trust region metric and adaptive threshold (matching GP-NEB/OTGPD)
    # FPS subset selection for hyperparameter optimization
    fps_history::Int = 0          # subset size for hyperopt (0=use all)
    fps_latest_points::Int = 2    # most recent points always included
    fps_metric::Symbol = :emd     # :emd, :max_1d_log, :euclidean
    # Trust region metric and adaptive threshold
    trust_metric::Symbol = :emd
    atom_types::Vector{Int} = Int[]
    use_adaptive_threshold::Bool = false
    adaptive_t_min::Float64 = 0.15
    adaptive_delta_t::Float64 = 0.35
    adaptive_n_half::Int = 50
    adaptive_A::Float64 = 1.3
    adaptive_floor::Float64 = 0.2
    verbose::Bool = true
    machine_output::String = ""  # "" = disabled; "host:port" = TCP socket; else = file path
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

# JSONL formatting helpers (no JSON dependency, just @sprintf)
function _jsonl_iter(;
    i::Int, E::Float64, F::Float64, oc::Int, tp::Int, t::Float64,
    sv::Float64, ls::Vector{Float64}, td::Float64, gate::String,
)
    ls_str = join([@sprintf("%.4e", l) for l in ls], ",")
    return @sprintf(
        "{\"i\":%d,\"E\":%.6f,\"F\":%.6f,\"oc\":%d,\"tp\":%d,\"t\":%.3f,\"sv\":%.4e,\"ls\":[%s],\"td\":%.4f,\"gate\":\"%s\"}",
        i, E, F, oc, tp, t, sv, ls_str, td, gate,
    )
end

function _jsonl_summary(; status::String, oc::Int, E::Float64, F::Float64, iters::Int)
    return @sprintf(
        "{\"status\":\"%s\",\"oc\":%d,\"E\":%.6f,\"F\":%.6f,\"iters\":%d}",
        status, oc, E, F, iters,
    )
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

    # Machine-readable output sink (file or TCP socket to writer)
    _mach_io = if !isempty(cfg.machine_output)
        _mo = cfg.machine_output
        if occursin(r"^[^/].*:\d+$", _mo)
            # "host:port" -> TCP socket
            _parts = split(_mo, ":")
            _host = String(_parts[1])
            _port = parse(Int, _parts[2])
            try
                Sockets.connect(_host, _port)
            catch e
                @warn "JSONL writer connection failed, falling back to file" host=_host port=_port exception=e
                open(_mo * ".jsonl", "w")
            end
        else
            open(_mo, "w")
        end
    else
        nothing
    end

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
        # FPS subset selection for hyperparameter optimization
        fps_dist = trust_distance_fn(cfg.fps_metric, cfg.atom_types)
        if cfg.fps_history > 0 && npoints(td) > cfg.fps_history
            sub_idx = _select_optim_subset(
                td, x_curr, cfg.fps_history, cfg.fps_latest_points; distance_fn=fps_dist
            )
            td_sub = _extract_subset(td, sub_idx)
            cfg.verbose && @printf(
                "Training GP on %d points (FPS subset: %d/%d)...\n",
                npoints(td), npoints(td_sub), npoints(td)
            )
        else
            td_sub = td
            cfg.verbose && @printf("Training GP on %d points...\n", npoints(td))
        end
        _t_train = time()

        # Adaptive training iterations: full budget on cold start, 1/3 on warm start
        _train_iters = prev_kern === nothing ? cfg.gp_train_iter : max(cfg.gp_train_iter ÷ 3, 50)

        if is_mol
            E_ref_sub = td_sub.energies[1]
            y_sub = vcat(td_sub.energies .- E_ref_sub, td_sub.gradients)

            kern = prev_kern === nothing ? init_mol_invdist_se(td_sub, kernel) : prev_kern
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
            gp_sub = GPModel(
                kern, td_sub.X, y_sub; noise_var=1e-6, grad_noise_var=1e-4, jitter=1e-6
            )
            train_model!(
                gp_sub; iterations=_train_iters, fix_noise=true, verbose=cfg.verbose
            )
            prev_kern = gp_sub.kernel

            # Rebuild on full data with optimized hyperparameters
            E_ref = td.energies[1]
            y_gp = vcat(td.energies .- E_ref, td.gradients)
            y_mean = E_ref
            y_std = 1.0
            gp_model = GPModel(
                gp_sub.kernel, td.X, y_gp; noise_var=1e-6, grad_noise_var=1e-4, jitter=1e-6
            )
        else
            y_sub, y_mean_sub, y_std_sub = normalize(td_sub)
            kern = prev_kern === nothing ? kernel : prev_kern
            gp_sub = GPModel(
                kern, td_sub.X, y_sub; noise_var=1e-2, grad_noise_var=1e-1, jitter=1e-3
            )
            train_model!(gp_sub; iterations=_train_iters, verbose=cfg.verbose)
            prev_kern = gp_sub.kernel

            # Rebuild on full data
            y_gp, y_mean, y_std = normalize(td)
            gp_model = GPModel(
                gp_sub.kernel, td.X, y_gp;
                noise_var=gp_sub.noise_var, grad_noise_var=gp_sub.grad_noise_var, jitter=1e-3
            )
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
            # Variance floor check: only use LCB when GP uncertainty is meaningful.
            # When RFF variance collapses (~1e-6), LCB degenerates to pure mean
            # prediction and can drive the optimizer into uncharted bad regions.
            _var_probe = predict_with_variance(model, reshape(x_curr, :, 1))
            _sigma_probe = sqrt(max(_var_probe[2][1], 0.0)) * y_std
            if _sigma_probe > 1e-4
                cfg.verbose && println("  Stuck or duplicate point predicted - seeking improvement via LCB...")

                # LCB objective for minimization (mu - kappa * sigma)
                # kappa=2.0 corresponds to roughly 95% confidence
                function lcb_objective(x)
                    mu_all, var_all = predict_with_variance(model, reshape(x, :, 1))
                    mu = mu_all[1] * y_std + y_mean
                    sigma = sqrt(max(var_all[1], 1e-12)) * y_std
                    return mu - 2.0 * sigma
                end

                best_idx = argmin(td.energies)
                res_lcb = Optim.optimize(
                    lcb_objective,
                    td.X[:, best_idx],
                    NelderMead(),
                    Optim.Options(; iterations=100, show_trace=false),
                )
                x_lcb = Optim.minimizer(res_lcb)

                # Energy sanity check: reject if LCB candidate predicts much worse
                # energy than current best (exploration went too far)
                mu_lcb = predict(model, reshape(x_lcb, :, 1))[1] * y_std + y_mean
                if mu_lcb < td.energies[best_idx] + 5.0 * _sigma_probe
                    x_curr = x_lcb
                    cfg.verbose && @printf("  LCB candidate: obj=%.4f, dist_from_prev=%.4f\n",
                        Optim.minimum(res_lcb), norm(x_curr - x_prev))
                else
                    cfg.verbose && @printf("  LCB rejected (E_pred=%.4f too high), keeping GP minimum\n", mu_lcb)
                end
            else
                cfg.verbose && @printf("  Stuck but variance too low (sigma=%.2e), skipping LCB\n", _sigma_probe)
            end
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
            if _mach_io !== nothing
                _k_mach = is_mol ? gp_model.kernel : nothing
                _sv_mach = _k_mach !== nothing ? Float64(_k_mach.signal_variance) : 0.0
                _ls_mach = _k_mach !== nothing ? Float64.(collect(_k_mach.inv_lengthscales)) : Float64[]
                println(_mach_io, _jsonl_iter(;
                    i=outer_step, E=isfinite(E_true) ? E_true : 1e30,
                    F=isfinite(G_norm) ? G_norm : 1e30,
                    oc=oracle_calls, tp=npoints(td), t=_dt_train,
                    sv=_sv_mach, ls=_ls_mach, td=d_trust, gate="exploded",
                ))
                flush(_mach_io)
            end
            if cfg.explosion_recovery == :perturb_best
                best_idx = argmin(td.energies)
                x_curr = td.X[:, best_idx] + (rand(D) .- 0.5) .* (cfg.perturb_scale * 0.5)
            else
                x_curr = td.X[:, end]
            end
            continue
        end

        # Energy regression gate: if oracle energy is much worse than best AND
        # forces are large, the GP prediction was wrong. Add data (so GP learns)
        # but revert to best position instead of continuing from the bad point.
        _gate = "ok"
        best_idx = argmin(td.energies)
        E_best = td.energies[best_idx]
        _regress_tol = if cfg.energy_regression_tol > 0
            cfg.energy_regression_tol
        else
            npoints(td) >= 3 ? max(std(td.energies) * 3, 1.0) : Inf
        end
        if E_true > E_best + _regress_tol && G_norm > cfg.conv_tol * 10
            cfg.verbose && @printf(
                "  Energy regression gate: E=%.4f >> E_best=%.4f (delta=%.4f > tol=%.4f)\n",
                E_true, E_best, E_true - E_best, _regress_tol,
            )
            cfg.verbose && println("  Reverting to best position (data retained for GP)")
            _gate = "energy_revert"
            # Record the bad point for GP training and trajectory
            push!(trajectory, copy(x_curr))
            push!(all_energies, E_true)
            if min_distance_to_data(x_curr, td.X) > eff_dedup
                add_point!(td, x_curr, E_true, G_true)
            end
            # Emit JSONL before reverting
            if _mach_io !== nothing
                _k_mach = is_mol ? gp_model.kernel : nothing
                _sv_mach = _k_mach !== nothing ? Float64(_k_mach.signal_variance) : 0.0
                _ls_mach = _k_mach !== nothing ? Float64.(collect(_k_mach.inv_lengthscales)) : Float64[]
                println(_mach_io, _jsonl_iter(;
                    i=outer_step, E=E_true, F=G_norm, oc=oracle_calls,
                    tp=npoints(td), t=_dt_train, sv=_sv_mach, ls=_ls_mach,
                    td=d_trust, gate=_gate,
                ))
                flush(_mach_io)
            end
            x_curr = copy(td.X[:, best_idx])
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

        # Machine-readable JSONL output
        if _mach_io !== nothing
            _k_mach = is_mol ? gp_model.kernel : nothing
            _sv_mach = _k_mach !== nothing ? Float64(_k_mach.signal_variance) : 0.0
            _ls_mach = _k_mach !== nothing ? Float64.(collect(_k_mach.inv_lengthscales)) : Float64[]
            println(_mach_io, _jsonl_iter(;
                i=outer_step, E=E_true, F=G_norm, oc=oracle_calls,
                tp=npoints(td), t=_dt_train, sv=_sv_mach, ls=_ls_mach,
                td=d_trust, gate=_gate,
            ))
            flush(_mach_io)
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

    # Machine-readable summary line
    if _mach_io !== nothing
        D_g_final = length(G_final)
        n_atoms_final = div(D_g_final, 3)
        F_final = if n_atoms_final >= 1 && D_g_final == 3 * n_atoms_final
            maximum(norm(@view G_final[(3 * (a - 1) + 1):(3 * a)]) for a in 1:n_atoms_final)
        else
            norm(G_final)
        end
        println(_mach_io, _jsonl_summary(;
            status=string(stop_reason), oc=oracle_calls,
            E=E_final, F=F_final, iters=length(all_energies),
        ))
        close(_mach_io)
    end

    return MinimizationResult(
        x_curr, E_final, G_final, converged, stop_reason, oracle_calls, trajectory, all_energies
    )
end
