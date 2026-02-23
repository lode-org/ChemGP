# ==============================================================================
# NEB Optimization Methods
# ==============================================================================
#
# Three NEB variants:
# 1. neb_optimize    -- Standard NEB (oracle at every step, baseline)
# 2. gp_neb_aie      -- GP-NEB with All Images Evaluated per outer iteration
# 3. gp_neb_oie      -- GP-NEB with One Image Evaluated (max uncertainty)
#
# Reference:
#   Goswami, R. et al. (2025). Efficient implementation of Gaussian process
#   regression accelerated saddle point searches with application to molecular
#   reactions. J. Chem. Theory Comput., doi:10.1021/acs.jctc.5c00866.
#
#   Goswami, R., Gunde, M. & Jonsson, H. (2026). Enhanced climbing image nudged
#   elastic band method with Hessian eigenmode alignment. arXiv:2601.12630.
#
#   Goswami, R. (2025). Efficient exploration of chemical kinetics. PhD thesis,
#   University of Iceland. arXiv:2510.21368.
#
#   Koistinen, O.-P. et al. (2017). Nudged elastic band calculations
#   accelerated with Gaussian process regression. J. Chem. Phys., 147, 152720.

# ==============================================================================
# Shared helpers
# ==============================================================================

"""
    _init_neb_images(cfg, x_start, x_end)

Initialize NEB path images using the method specified in `cfg.initializer`.
Returns a vector of image position vectors.
"""
function _init_neb_images(cfg::NEBConfig, x_start, x_end)
    if cfg.initializer == :sidpp
        cfg.verbose && println("  Generating S-IDPP initial path...")
        sidpp_interpolation(x_start, x_end, cfg.images + 2;
                            spring_constant = cfg.spring_constant)
    elseif cfg.initializer == :idpp
        cfg.verbose && println("  Generating IDPP initial path...")
        idpp_interpolation(x_start, x_end, cfg.images + 2)
    else
        linear_interpolation(x_start, x_end, cfg.images + 2)
    end
end

"""
    _init_hessian_data(cfg, ep_oracle, x_start, x_end, D)

Generate virtual Hessian perturbation points around endpoints for GP training.
Returns `(hess_X, hess_E, hess_G, n_hess, oracle_calls)`.
"""
function _init_hessian_data(cfg::NEBConfig, ep_oracle, x_start, x_end, D)
    hess_X = Matrix{Float64}(undef, D, 0)
    hess_E = Float64[]
    hess_G = Float64[]
    n_hess = 0
    calls = 0

    if cfg.num_hess_iter > 0
        hess_pts = get_hessian_points(x_start, x_end, cfg.eps_hess)
        n_hess = length(hess_pts)
        hess_X = Matrix{Float64}(undef, D, n_hess)
        for (idx, pt) in enumerate(hess_pts)
            E, G = ep_oracle(pt)
            hess_X[:, idx] = pt
            push!(hess_E, E)
            append!(hess_G, G)
            calls += 1
        end
        cfg.verbose && @printf("  Generated %d virtual Hessian points (%d oracle calls)\n",
                               n_hess, n_hess)
    end

    return hess_X, hess_E, hess_G, n_hess, calls
end

"""
    _check_ci(cfg, ci_on, max_f, ci_f, baseline_force, iter)

Check climbing image activation using dynamic thresholding (eOn-style).
Returns `(ci_on, conv_metric, activated)` where `activated` is true if CI
just turned on this iteration.
"""
function _check_ci(cfg::NEBConfig, ci_on::Bool, max_f, ci_f, baseline_force, iter)
    activated = false

    if cfg.climbing_image && iter > 1
        # CI toggle always uses max_f over all images (matching eOn):
        # "is the band relaxed enough for climbing?" is a band-level
        # question, independent of ci_converged_only.
        new_ci = max_f < cfg.ci_trigger_rel * baseline_force ||
                 max_f < cfg.ci_activation_tol
        activated = new_ci && !ci_on
        ci_on = new_ci
    end

    # Convergence metric: CI force when ci_converged_only + CI active
    conv_metric = (ci_on && cfg.ci_converged_only) ? ci_f : max_f
    return ci_on, conv_metric, activated
end

"""
    _train_neb_gp(td, kernel, cfg, prev_kern, hess_X, hess_E, hess_G,
                  n_hess, outer_iter)

Train the GP model for NEB, handling both molecular kernels (energy shift,
fixed noise) and generic kernels (normalize, optimize noise).

When `cfg.max_gp_points > 0` and the training set exceeds that limit,
a FPS subset is selected for hyperparameter optimization, then a
[`NystromGP`](@ref) is built using all data for prediction. This caps
training cost at O(M^3) while prediction uses all N points.

Returns `(model, E_ref, y_std, kern)` where `model` is either a
`GPModel` (exact) or `NystromGP` (approximate), and `kern` is the
trained kernel for warm-starting.
"""
function _train_neb_gp(
    td::TrainingData,
    kernel,
    cfg::NEBConfig,
    prev_kern,
    hess_X::Matrix{Float64},
    hess_E::Vector{Float64},
    hess_G::Vector{Float64},
    n_hess::Int,
    outer_iter::Int,
)
    # FPS subset selection when training set exceeds max_gp_points
    fps_fn = trust_distance_fn(cfg.trust_metric, cfg.atom_types)
    td_use = _fps_subset_td(td, cfg.max_gp_points; distance_fn=fps_fn)
    use_nystrom = npoints(td_use) < npoints(td)
    if use_nystrom
        cfg.verbose && @printf("  Nystrom: %d inducing / %d total training points\n",
                               npoints(td_use), npoints(td))
    end

    if kernel isa AbstractMoleculeKernel
        E_ref = td_use.energies[1]

        use_hess = n_hess > 0 && outer_iter <= cfg.num_hess_iter
        if use_hess
            X_train = hcat(hess_X, td_use.X)
            E_train = vcat(hess_E, td_use.energies)
            G_train = vcat(hess_G, td_use.gradients)
        else
            X_train = td_use.X
            E_train = td_use.energies
            G_train = td_use.gradients
        end

        y_target = vcat(E_train .- E_ref, G_train)

        # Warm-start: reuse previous kernel, only init on first iteration
        kern = prev_kern === nothing ? init_mol_invdist_se(td_use, kernel) : prev_kern
        base = GPModel(kern, X_train, y_target;
                       noise_var = 1e-6, grad_noise_var = 1e-4, jitter = 1e-6)
        train_model!(base; iterations = cfg.gp_train_iter,
                     fix_noise = true, verbose = cfg.verbose)

        if use_nystrom
            # Build Nystrom using all data (same normalization)
            y_all = vcat(td.energies .- E_ref, td.gradients)
            model = build_nystrom(base, td.X, y_all)
        else
            model = base
        end
        return model, E_ref, 1.0, base.kernel
    else
        y_gp, E_ref, y_std = normalize(td_use)
        kern = if prev_kern !== nothing
            prev_kern
        elseif kernel isa CartesianSE
            init_cartesian_se(td_use)
        else
            kernel
        end
        base = GPModel(kern, td_use.X, y_gp;
                       noise_var = 1e-6, grad_noise_var = 1e-6, jitter = 1e-4)
        train_model!(base; iterations = cfg.gp_train_iter, verbose = cfg.verbose)

        if use_nystrom
            # Build Nystrom using all data (same normalization)
            y_all = vcat((td.energies .- E_ref) ./ y_std, td.gradients ./ y_std)
            model = build_nystrom(base, td.X, y_all)
        else
            model = base
        end
        return model, E_ref, y_std, base.kernel
    end
end

"""
    _gp_inner_relax(model, images, energies, gradients, cfg, ci_on, E_ref, y_std, gp_tol)

Relax NEB images on the GP surrogate surface using L-BFGS (or SD) with
Euclidean trust radius clipping. Returns a new vector of relaxed image positions.

The inner loop uses a simple Euclidean clip from the oracle-evaluated anchor
to prevent cumulative drift over many L-BFGS steps. EMD-based trust checking
is applied at the outer loop boundary instead (see `_emd_trust_clip!`),
where it cannot disrupt L-BFGS curvature estimates.
"""
function _gp_inner_relax(
    model,
    images::Vector{Vector{Float64}},
    energies::Vector{Float64},
    gradients::Vector{Vector{Float64}},
    cfg::NEBConfig,
    ci_on::Bool,
    E_ref::Float64,
    y_std::Float64,
    gp_tol::Float64,
)
    N = length(images)
    D = length(images[1])
    N_mov = N - 2

    gp_images = deepcopy(images)
    start_images = images  # anchor for trust radius (no mutation)
    gp_optim = cfg.optimizer == :lbfgs ? OptimState(cfg.lbfgs_memory) : nothing

    for inner_iter in 1:(cfg.max_iter)
        gp_energies = copy(energies)
        gp_gradients = deepcopy(gradients)

        for i in 2:(N - 1)
            pred = predict(model, reshape(gp_images[i], :, 1))
            gp_energies[i] = pred[1] * y_std + E_ref
            gp_gradients[i] = pred[2:end] .* y_std
        end

        gp_path = NEBPath(gp_images, gp_energies, gp_gradients, cfg.spring_constant)
        gp_forces, gp_max_f, _, _ = compute_all_neb_forces(gp_path, cfg; ci_on)

        if gp_max_f < gp_tol
            break
        end

        # Compute step (concatenated over all movable images)
        cur_x = vcat(gp_images[2:N-1]...)
        cur_force = vcat(gp_forces[2:N-1]...)

        if gp_optim !== nothing
            displacement = optim_step!(gp_optim, cur_x, cur_force, cfg.max_move;
                                       n_coords_per_atom = 3)
        else
            displacement = cfg.step_size * cur_force
            displacement = _clip_to_max_move(displacement, cfg.max_move, 3)
        end

        new_x = cur_x + displacement
        for img_idx in 1:N_mov
            offset = (img_idx - 1) * D
            candidate = new_x[offset+1:offset+D]
            # Trust radius: clip total displacement from oracle-evaluated position
            disp = candidate - start_images[img_idx + 1]
            dn = norm(disp)
            if dn > cfg.trust_radius
                candidate = start_images[img_idx + 1] + disp * (cfg.trust_radius / dn)
            end
            gp_images[img_idx + 1] = candidate
        end
    end

    return gp_images
end

"""
    _emd_trust_clip!(images, td, cfg)

Post-inner-loop EMD trust region clip. For each movable image, check the
minimum distance (using `cfg.trust_metric`) to all training data. If an
image exceeds the adaptive threshold, scale its displacement from the
nearest training point back to the boundary.

This is applied AFTER inner relaxation completes -- never inside the inner
loop -- because per-step clipping disrupts L-BFGS curvature estimates and
causes force divergence.
"""
function _emd_trust_clip!(
    images::Vector{Vector{Float64}},
    td::TrainingData,
    cfg::NEBConfig,
)
    N = length(images)
    D = length(images[1])
    n_atoms = div(D, 3)
    thresh = adaptive_trust_threshold(cfg.trust_radius, npoints(td), n_atoms;
                 use_adaptive=cfg.use_adaptive_threshold,
                 t_min=cfg.adaptive_t_min, delta_t=cfg.adaptive_delta_t,
                 n_half=cfg.adaptive_n_half, A=cfg.adaptive_A,
                 floor=cfg.adaptive_floor)

    for i in 2:(N - 1)
        d = trust_min_distance(images[i], td.X, cfg.trust_metric;
                               atom_types=cfg.atom_types)
        if d > thresh
            # Find nearest training point and scale back toward it
            dist_fn = trust_distance_fn(cfg.trust_metric, cfg.atom_types)
            nearest_idx = argmin(dist_fn(images[i], view(td.X, :, j))
                                for j in 1:npoints(td))
            nearest = td.X[:, nearest_idx]
            disp = images[i] - nearest
            images[i] = nearest + disp * (thresh / d * 0.95)
        end
    end
end

# ==============================================================================
# Parallel oracle evaluation
# ==============================================================================

# Union type for oracle: single function or pool of functions
const OracleOrPool = Union{Function, AbstractVector{<:Function}}

# Extract a single oracle from an oracle-or-pool (for endpoint evaluation etc.)
_single_oracle(oracle::Function) = oracle
_single_oracle(oracles::AbstractVector{<:Function}) = oracles[1]

"""
    _eval_images!(oracle, images, energies, gradients, indices)

Evaluate oracle at multiple images. When `oracle` is a vector of functions,
evaluations are dispatched in parallel using `Threads.@spawn`, round-robin
across the pool.

Returns the number of evaluations performed.
"""
function _eval_images!(oracle::Function, images, energies, gradients, indices)
    for i in indices
        E, G = oracle(images[i])
        energies[i] = E
        gradients[i] = G
    end
    return length(indices)
end

function _eval_images!(oracles::AbstractVector{<:Function},
                       images, energies, gradients, indices)
    n = length(indices)
    n_workers = length(oracles)
    tasks = Vector{Task}(undef, n)
    for (k, i) in enumerate(indices)
        w = ((k - 1) % n_workers) + 1
        tasks[k] = Threads.@spawn oracles[w](images[i])
    end
    for (k, i) in enumerate(indices)
        E, G = fetch(tasks[k])
        energies[i] = E
        gradients[i] = G
    end
    return n
end

# ==============================================================================
# Standard NEB
# ==============================================================================

"""
    neb_optimize(oracle, x_start, x_end; config) -> NEBResult

Standard NEB optimization (oracle-only baseline).

Evaluates the oracle at all intermediate images on every iteration. Uses
L-BFGS (default) or steepest descent for relaxation, with max_move
clipping to prevent overshoot. Supports energy-weighted springs.

`oracle` can be a single `Function` or a `Vector{Function}` for parallel
evaluation of images.
"""
function neb_optimize(
    oracle::OracleOrPool,
    x_start::Vector{Float64},
    x_end::Vector{Float64};
    config::NEBConfig = NEBConfig(),
    on_step::Union{Function,Nothing} = nothing,
)
    cfg = config
    N = cfg.images + 2
    D = length(x_start)
    N_mov = N - 2

    images = _init_neb_images(cfg, x_start, x_end)

    # Evaluate endpoints (fixed)
    ep_oracle = _single_oracle(oracle)
    E_start, G_start = ep_oracle(x_start)
    E_end, G_end = ep_oracle(x_end)
    energies = zeros(N)
    gradients = [zeros(D) for _ in 1:N]
    energies[1] = E_start
    energies[end] = E_end
    gradients[1] = G_start
    gradients[end] = G_end

    oracle_calls = 2

    # Evaluate intermediate images (parallel when oracle is a pool)
    oracle_calls += _eval_images!(oracle, images, energies, gradients, 2:(N-1))

    path = NEBPath(images, energies, gradients, cfg.spring_constant)

    history = Dict(
        "max_force" => Float64[],
        "ci_force" => Float64[],
        "oracle_calls" => Int[],
        "max_energy" => Float64[],
    )

    # Optimizer state (L-BFGS with per-atom max_move, or nothing for SD)
    optim = cfg.optimizer == :lbfgs ? OptimState(cfg.lbfgs_memory) : nothing

    ci_on = false
    converged = false
    baseline_force = 0.0

    for iter in 1:(cfg.max_iter)
        forces, max_f, ci_f, i_max = compute_all_neb_forces(path, cfg; ci_on)

        push!(history["max_force"], max_f)
        push!(history["ci_force"], ci_f)
        push!(history["oracle_calls"], oracle_calls)
        push!(history["max_energy"], maximum(energies))

        if iter == 1
            baseline_force = max_f
        end

        # Dynamic CI activation (matches eOn NudgedElasticBand.cpp:405-410)
        ci_on, conv_metric, ci_activated = _check_ci(cfg, ci_on, max_f, ci_f, baseline_force, iter)
        if ci_activated
            cfg.verbose && @printf("  Iter %d: Climbing image activated (image %d)\n", iter, i_max)
            # Recompute forces with CI enabled
            forces, max_f, ci_f, i_max = compute_all_neb_forces(path, cfg; ci_on)
            conv_metric = cfg.ci_converged_only ? ci_f : max_f
        end

        # Check convergence
        conv_check = ci_on || !cfg.climbing_image
        if conv_check && conv_metric < cfg.conv_tol
            cfg.verbose && @printf("NEB converged at iter %d: max|F| = %.5f\n", iter, conv_metric)
            converged = true
            break
        end

        if iter % 50 == 0 || iter == 1
            cfg.verbose && @printf("  Iter %3d: max|F| = %.5f | CI|F| = %.5f | E_max = %.4f\n",
                                   iter, max_f, ci_f, maximum(energies))
        end

        # Concatenate movable image positions and NEB forces
        cur_x = vcat(images[2:N-1]...)
        cur_force = vcat(forces[2:N-1]...)

        # Compute step via OptimState (L-BFGS) or steepest descent
        if optim !== nothing
            displacement = optim_step!(optim, cur_x, cur_force, cfg.max_move;
                                       n_coords_per_atom = 3)
        else
            displacement = cfg.step_size * cur_force
            displacement = _clip_to_max_move(displacement, cfg.max_move, 3)
        end

        # Update image positions
        new_x = cur_x + displacement
        for img_idx in 1:N_mov
            offset = (img_idx - 1) * D
            images[img_idx + 1] = new_x[offset+1:offset+D]
        end

        # Re-evaluate oracle (parallel when oracle is a pool)
        oracle_calls += _eval_images!(oracle, images, energies, gradients, 2:(N-1))

        path.images = images
        path.energies = energies
        path.gradients = gradients

        on_step !== nothing && on_step(path, iter)
    end

    i_max = argmax(energies[2:end-1]) + 1
    return NEBResult(path, converged, oracle_calls, i_max, history)
end

# ==============================================================================
# FPS subset selection for GP training
# ==============================================================================

"""
    _fps_subset_td(td, max_points; distance_fn=max_1d_log_distance)

Return a new TrainingData containing at most `max_points` from `td`,
selected by Farthest Point Sampling for maximum diversity.

The first two points (endpoints) are always kept. When `max_points <= 0`
or `npoints(td) <= max_points`, returns `td` unchanged (no copy).
"""
function _fps_subset_td(
    td::TrainingData,
    max_points::Int;
    distance_fn::Function = max_1d_log_distance,
)
    N = npoints(td)
    if max_points <= 0 || N <= max_points
        return td
    end

    D = size(td.X, 1)

    # Seed FPS with the first point (endpoint 1)
    seed = td.X[:, 1:1]
    # Select max_points - 1 more from all data
    fps_idx = farthest_point_sampling(td.X, seed, max_points - 1; distance_fn)
    # Combine seed + FPS selections, deduplicate and sort
    keep = sort(unique([1; fps_idx]))
    # Ensure we don't exceed max_points
    if length(keep) > max_points
        keep = keep[1:max_points]
    end

    td_sub = TrainingData(D)
    for i in keep
        s = (i - 1) * D + 1
        e = i * D
        add_point!(td_sub, td.X[:, i], td.energies[i], td.gradients[s:e])
    end
    return td_sub
end

# ==============================================================================
# GP-NEB AIE (All Images Evaluated)
# ==============================================================================

"""
    gp_neb_aie(oracle, x_start, x_end, kernel; config) -> NEBResult

GP-NEB with All Images Evaluated per outer iteration.

At each outer iteration:
1. Evaluate the oracle at all intermediate image positions
2. Train the GP on all accumulated data
3. Relax the path on the GP surface (inner loop)
4. Check convergence on true forces

This variant uses fewer oracle calls than standard NEB because the inner
relaxation (which takes many steps) operates on the cheap GP surface.

Training data is deduplicated: new oracle evaluations are only added when
the image is sufficiently far from all existing training points. This
prevents near-singular covariance matrices from accumulating almost-
identical data rows across outer iterations.
"""
function gp_neb_aie(
    oracle::OracleOrPool,
    x_start::Vector{Float64},
    x_end::Vector{Float64},
    kernel;
    config::NEBConfig = NEBConfig(),
    on_step::Union{Function,Nothing} = nothing,
)
    cfg = config
    N = cfg.images + 2
    D = length(x_start)
    dedup_tol = cfg.conv_tol * 0.1

    images = _init_neb_images(cfg, x_start, x_end)

    # Evaluate endpoints
    ep_oracle = _single_oracle(oracle)
    E_start, G_start = ep_oracle(x_start)
    E_end, G_end = ep_oracle(x_end)
    oracle_calls = 2

    # Training data accumulator
    td = TrainingData(D)
    add_point!(td, x_start, E_start, G_start)
    add_point!(td, x_end, E_end, G_end)

    # Virtual Hessian points
    hess_X, hess_E, hess_G, n_hess, hess_calls = _init_hessian_data(
        cfg, ep_oracle, x_start, x_end, D)
    oracle_calls += hess_calls

    # Evaluate all intermediate images (parallel when oracle is a pool)
    energies = zeros(N)
    gradients = [zeros(D) for _ in 1:N]
    energies[1] = E_start
    energies[end] = E_end
    gradients[1] = G_start
    gradients[end] = G_end

    oracle_calls += _eval_images!(oracle, images, energies, gradients, 2:(N-1))
    for i in 2:(N-1)
        add_point!(td, images[i], energies[i], gradients[i])
    end

    path = NEBPath(images, energies, gradients, cfg.spring_constant)

    history = Dict(
        "max_force" => Float64[],
        "ci_force" => Float64[],
        "oracle_calls" => Int[],
        "max_energy" => Float64[],
    )

    ci_on = false
    converged = false
    prev_kern = nothing
    baseline_force = 0.0

    for outer_iter in 1:(cfg.max_outer_iter)
        # Compute true forces
        forces_true, max_f_true, ci_f_true, i_max = compute_all_neb_forces(path, cfg; ci_on)

        push!(history["max_force"], max_f_true)
        push!(history["ci_force"], ci_f_true)
        push!(history["oracle_calls"], oracle_calls)
        push!(history["max_energy"], maximum(energies))

        if outer_iter == 1
            baseline_force = max_f_true
        end

        n_total = npoints(td) + (outer_iter <= cfg.num_hess_iter ? n_hess : 0)
        cfg.verbose && @printf("GP-NEB-AIE outer %d: max|F| = %.5f | CI|F| = %.5f | N_train = %d | calls = %d\n",
                               outer_iter, max_f_true, ci_f_true, n_total, oracle_calls)

        # Dynamic CI activation
        ci_on, conv_metric, ci_activated = _check_ci(
            cfg, ci_on, max_f_true, ci_f_true, baseline_force, outer_iter)
        if ci_activated
            cfg.verbose && @printf("  Climbing image activated (image %d)\n", i_max)
        end

        conv_check = ci_on || !cfg.climbing_image
        if conv_check && conv_metric < cfg.conv_tol
            cfg.verbose && println("GP-NEB-AIE converged!")
            converged = true
            break
        end

        # Train GP (FPS subset applied inside _train_neb_gp when max_gp_points > 0)
        model, E_ref, y_std, prev_kern = _train_neb_gp(
            td, kernel, cfg, prev_kern,
            hess_X, hess_E, hess_G, n_hess, outer_iter)

        # Inner loop: relax on GP surface
        gp_tol = max(min(max_f_true / 10, cfg.conv_tol), cfg.conv_tol / 10)
        images = _gp_inner_relax(model, images, energies, gradients,
                                 cfg, ci_on, E_ref, y_std, gp_tol)

        # EMD trust clip: scale back images that drifted beyond the adaptive
        # threshold from all training data. Applied AFTER inner relaxation
        # to avoid disrupting L-BFGS curvature estimates.
        _emd_trust_clip!(images, td, cfg)

        # Evaluate oracle at new positions (parallel); deduplicate training data
        oracle_calls += _eval_images!(oracle, images, energies, gradients, 2:(N-1))
        for i in 2:(N-1)
            if min_distance_to_data(images[i], td.X) > dedup_tol
                add_point!(td, images[i], energies[i], gradients[i])
            end
        end

        path.images = images
        path.energies = energies
        path.gradients = gradients

        on_step !== nothing && on_step(path, outer_iter)
    end

    i_max = argmax(energies[2:end-1]) + 1
    return NEBResult(path, converged, oracle_calls, i_max, history)
end

# GP-NEB OIE variants are in neb_oie.jl (Koistinen) and neb_oie_naive.jl (pedagogical)
