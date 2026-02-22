# ==============================================================================
# NEB Optimization Methods
# ==============================================================================
#
# Three NEB variants:
# 1. neb_optimize    — Standard NEB (oracle at every step, baseline)
# 2. gp_neb_aie      — GP-NEB with All Images Evaluated per outer iteration
# 3. gp_neb_oie      — GP-NEB with One Image Evaluated (max uncertainty)
#
# Reference:
#   Goswami, R. et al. (2025). Efficient implementation of Gaussian process
#   regression accelerated saddle point searches with application to molecular
#   reactions. J. Chem. Theory Comput., doi:10.1021/acs.jctc.5c00866.
#
#   Goswami, R., Gunde, M. & Jónsson, H. (2026). Enhanced climbing image nudged
#   elastic band method with Hessian eigenmode alignment. arXiv:2601.12630.
#
#   Goswami, R. (2025). Efficient exploration of chemical kinetics. PhD thesis,
#   University of Iceland. arXiv:2510.21368.
#
#   Koistinen, O.-P. et al. (2017). Nudged elastic band calculations
#   accelerated with Gaussian process regression. J. Chem. Phys., 147, 152720.

# ==============================================================================
# Shared GP training helper for GP-NEB (AIE and OIE)
# ==============================================================================

"""
    _train_neb_gp(td, kernel, cfg, prev_kern, hess_X, hess_E, hess_G,
                  n_hess, outer_iter)

Train the GP model for NEB, handling both molecular kernels (energy shift,
fixed noise) and generic kernels (normalize, optimize noise).

Returns `(model, E_ref, y_std, kern)` where `kern` is the trained kernel
suitable for warm-starting the next iteration.
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
    if kernel isa AbstractMoleculeKernel
        E_ref = td.energies[1]

        use_hess = n_hess > 0 && outer_iter <= cfg.num_hess_iter
        if use_hess
            X_train = hcat(hess_X, td.X)
            E_train = vcat(hess_E, td.energies)
            G_train = vcat(hess_G, td.gradients)
        else
            X_train = td.X
            E_train = td.energies
            G_train = td.gradients
        end

        y_target = vcat(E_train .- E_ref, G_train)

        # Warm-start: reuse previous kernel, only init on first iteration
        kern = prev_kern === nothing ? init_mol_invdist_se(td, kernel) : prev_kern
        model = GPModel(kern, X_train, y_target;
                        noise_var = 1e-6, grad_noise_var = 1e-4, jitter = 1e-6)
        train_model!(model; iterations = cfg.gp_train_iter,
                     fix_noise = true, verbose = cfg.verbose)
        return model, E_ref, 1.0, model.kernel
    else
        y_gp, E_ref, y_std = normalize(td)
        kern = if prev_kern !== nothing
            prev_kern
        elseif kernel isa CartesianSE
            init_cartesian_se(td)
        else
            kernel
        end
        model = GPModel(kern, td.X, y_gp;
                        noise_var = 1e-6, grad_noise_var = 1e-6, jitter = 1e-4)
        train_model!(model; iterations = cfg.gp_train_iter, verbose = cfg.verbose)
        return model, E_ref, y_std, model.kernel
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
    N = cfg.n_images
    D = length(x_start)
    N_mov = N - 2  # number of movable images

    # Initialize path
    images = if cfg.initializer == :sidpp
        cfg.verbose && println("  Generating S-IDPP initial path...")
        sidpp_interpolation(x_start, x_end, N;
                            spring_constant = cfg.spring_constant)
    elseif cfg.initializer == :idpp
        cfg.verbose && println("  Generating IDPP initial path...")
        idpp_interpolation(x_start, x_end, N)
    else
        linear_interpolation(x_start, x_end, N)
    end

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
    ci_ever = false  # track whether CI has ever been active (for logging)
    converged = false
    baseline_force = 0.0

    for iter in 1:(cfg.max_iter)
        forces, max_f, ci_f, i_max = compute_all_neb_forces(path, cfg; ci_on)

        push!(history["max_force"], max_f)
        push!(history["ci_force"], ci_f)
        push!(history["oracle_calls"], oracle_calls)
        push!(history["max_energy"], maximum(energies))

        # Record baseline force on first iteration
        if iter == 1
            baseline_force = max_f
        end

        # Dynamic CI activation (matches eOn NudgedElasticBand.cpp:405-410):
        # CI is active when convergence metric drops below relative or absolute
        # threshold. Unlike a sticky flag, CI can toggle off if forces increase.
        # The convergence metric is CI-only when ci_converged_only and CI active.
        conv_metric = (ci_on && cfg.ci_converged_only) ? ci_f : max_f
        if cfg.climbing_image && iter > 1
            ci_was_on = ci_on
            ci_on = conv_metric < cfg.ci_trigger_rel * baseline_force ||
                    conv_metric < cfg.ci_activation_tol
            if ci_on && !ci_was_on
                ci_ever = true
                cfg.verbose && @printf("  Iter %d: Climbing image activated (image %d)\n", iter, i_max)
                # Recompute forces with CI
                forces, max_f, ci_f, i_max = compute_all_neb_forces(path, cfg; ci_on)
                conv_metric = cfg.ci_converged_only ? ci_f : max_f
            end
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
            # Steepest descent with per-atom max_move clipping
            displacement = cfg.step_size * cur_force
            n_atoms = div(length(displacement), 3)
            max_disp = 0.0
            for a in 1:n_atoms
                off = (a - 1) * 3
                disp = norm(@view displacement[off+1:off+3])
                max_disp = max(max_disp, disp)
            end
            if max_disp > cfg.max_move
                displacement .*= cfg.max_move / max_disp
            end
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

        # Per-step callback (write .dat/.xyz files etc.)
        on_step !== nothing && on_step(path, iter)
    end

    i_max = argmax(energies[2:end-1]) + 1
    return NEBResult(path, converged, oracle_calls, i_max, history)
end

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
    N = cfg.n_images
    D = length(x_start)

    # Minimum distance threshold for adding training data.
    # Points closer than this to existing data are redundant and would
    # create near-singular rows in the covariance matrix.
    dedup_tol = cfg.conv_tol * 0.1

    # Initialize path
    images = if cfg.initializer == :sidpp
        cfg.verbose && println("  Generating S-IDPP initial path...")
        sidpp_interpolation(x_start, x_end, N;
                            spring_constant = cfg.spring_constant)
    elseif cfg.initializer == :idpp
        cfg.verbose && println("  Generating IDPP initial path...")
        idpp_interpolation(x_start, x_end, N)
    else
        linear_interpolation(x_start, x_end, N)
    end

    # Evaluate endpoints
    ep_oracle = _single_oracle(oracle)
    E_start, G_start = ep_oracle(x_start)
    E_end, G_end = ep_oracle(x_end)
    oracle_calls = 2

    # Training data accumulator (physical points only)
    td = TrainingData(D)
    add_point!(td, x_start, E_start, G_start)
    add_point!(td, x_end, E_end, G_end)

    # Virtual Hessian points: displace endpoints by eps_hess along each axis.
    # Evaluated once; included in GP training for first num_hess_iter iterations
    # to provide curvature info, then dropped. Ranges for kernel init always
    # use physical data only (matching MATLAB atomic_GP_NEB_AIE.m).
    hess_X = Matrix{Float64}(undef, D, 0)
    hess_E = Float64[]
    hess_G = Float64[]
    n_hess = 0

    if cfg.num_hess_iter > 0
        hess_pts = get_hessian_points(x_start, x_end, cfg.eps_hess)
        n_hess = length(hess_pts)
        hess_X = Matrix{Float64}(undef, D, n_hess)
        for (idx, pt) in enumerate(hess_pts)
            E, G = ep_oracle(pt)
            hess_X[:, idx] = pt
            push!(hess_E, E)
            append!(hess_G, G)
            oracle_calls += 1
        end
        cfg.verbose && @printf("  Generated %d virtual Hessian points (%d oracle calls)\n",
                               n_hess, n_hess)
    end

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
    prev_kern = nothing  # warm-start: reuse kernel from previous iteration
    baseline_force = 0.0

    for outer_iter in 1:(cfg.max_outer_iter)
        # Compute true forces and check convergence
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

        # Dynamic CI activation (matches eOn)
        conv_metric = (ci_on && cfg.ci_converged_only) ? ci_f_true : max_f_true
        if cfg.climbing_image && outer_iter > 1
            ci_was_on = ci_on
            ci_on = conv_metric < cfg.ci_trigger_rel * baseline_force ||
                    conv_metric < cfg.ci_activation_tol
            if ci_on && !ci_was_on
                cfg.verbose && @printf("  Climbing image activated (image %d)\n", i_max)
                conv_metric = cfg.ci_converged_only ? ci_f_true : max_f_true
            end
        end

        conv_check = ci_on || !cfg.climbing_image
        if conv_check && conv_metric < cfg.conv_tol
            cfg.verbose && println("GP-NEB-AIE converged!")
            converged = true
            break
        end

        # Train GP (shared helper handles kernel type dispatch, warm-start, noise)
        model, E_ref, y_std, prev_kern = _train_neb_gp(
            td, kernel, cfg, prev_kern,
            hess_X, hess_E, hess_G, n_hess, outer_iter)

        # Inner loop: relax on GP surface (L-BFGS/SD + trust radius)
        gp_images = deepcopy(images)
        start_images = deepcopy(images)  # anchor for trust radius
        N_mov = N - 2
        gp_optim = cfg.optimizer == :lbfgs ? OptimState(cfg.lbfgs_memory) : nothing

        for inner_iter in 1:(cfg.max_iter)
            # Predict energies and gradients from GP
            gp_energies = copy(energies)
            gp_gradients = deepcopy(gradients)

            for i in 2:(N - 1)
                pred = predict(model, reshape(gp_images[i], :, 1))
                gp_energies[i] = pred[1] * y_std + E_ref
                gp_gradients[i] = pred[2:end] .* y_std
            end

            gp_path = NEBPath(gp_images, gp_energies, gp_gradients, cfg.spring_constant)
            gp_forces, gp_max_f, _, _ = compute_all_neb_forces(gp_path, cfg; ci_on)

            # Adaptive GP convergence threshold
            gp_tol = max(min(max_f_true / 10, cfg.conv_tol), cfg.conv_tol / 10)

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

        # Update path with GP-relaxed images
        images = gp_images

        # Evaluate oracle at new positions (parallel); deduplicate training data
        n_new = _eval_images!(oracle, images, energies, gradients, 2:(N-1))
        oracle_calls += n_new
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

"""
    gp_neb_oie(oracle, x_start, x_end, kernel; config) -> NEBResult

GP-NEB with One Image Evaluated per outer iteration (uncertainty-based).

At each outer iteration:
1. Train the GP on accumulated data
2. Relax the path on the GP surface
3. Compute predictive variance at all images
4. Evaluate the oracle at the image with maximum uncertainty
5. Check convergence

This is the most oracle-efficient variant. Uses [`predict_with_variance`](@ref)
to select the most informative evaluation point.

Training data is deduplicated to prevent near-singular covariance matrices.
"""
function gp_neb_oie(
    oracle::OracleOrPool,
    x_start::Vector{Float64},
    x_end::Vector{Float64},
    kernel;
    config::NEBConfig = NEBConfig(),
    on_step::Union{Function,Nothing} = nothing,
)
    cfg = config
    N = cfg.n_images
    D = length(x_start)
    # OIE evaluates one image per iteration; use single oracle throughout
    ep_oracle = _single_oracle(oracle)

    dedup_tol = cfg.conv_tol * 0.1

    # Initialize path
    images = if cfg.initializer == :sidpp
        cfg.verbose && println("  Generating S-IDPP initial path...")
        sidpp_interpolation(x_start, x_end, N;
                            spring_constant = cfg.spring_constant)
    elseif cfg.initializer == :idpp
        cfg.verbose && println("  Generating IDPP initial path...")
        idpp_interpolation(x_start, x_end, N)
    else
        linear_interpolation(x_start, x_end, N)
    end

    # Evaluate endpoints
    E_start, G_start = ep_oracle(x_start)
    E_end, G_end = ep_oracle(x_end)
    oracle_calls = 2

    # Training data accumulator (physical points only)
    td = TrainingData(D)
    add_point!(td, x_start, E_start, G_start)
    add_point!(td, x_end, E_end, G_end)

    # Virtual Hessian points (same mechanism as AIE)
    hess_X = Matrix{Float64}(undef, D, 0)
    hess_E = Float64[]
    hess_G = Float64[]
    n_hess = 0

    if cfg.num_hess_iter > 0
        hess_pts = get_hessian_points(x_start, x_end, cfg.eps_hess)
        n_hess = length(hess_pts)
        hess_X = Matrix{Float64}(undef, D, n_hess)
        for (idx, pt) in enumerate(hess_pts)
            E, G = ep_oracle(pt)
            hess_X[:, idx] = pt
            push!(hess_E, E)
            append!(hess_G, G)
            oracle_calls += 1
        end
        cfg.verbose && @printf("  Generated %d virtual Hessian points (%d oracle calls)\n",
                               n_hess, n_hess)
    end

    # Initial evaluations: evaluate a few images to bootstrap the GP
    energies = zeros(N)
    gradients = [zeros(D) for _ in 1:N]
    energies[1] = E_start
    energies[end] = E_end
    gradients[1] = G_start
    gradients[end] = G_end
    evaluated = falses(N)
    evaluated[1] = true
    evaluated[end] = true

    # Evaluate midpoint to start
    mid = div(N, 2) + 1
    E_mid, G_mid = ep_oracle(images[mid])
    energies[mid] = E_mid
    gradients[mid] = G_mid
    add_point!(td, images[mid], E_mid, G_mid)
    evaluated[mid] = true
    oracle_calls += 1

    path = NEBPath(images, energies, gradients, cfg.spring_constant)

    history = Dict(
        "max_force" => Float64[],
        "ci_force" => Float64[],
        "oracle_calls" => Int[],
        "max_energy" => Float64[],
        "image_evaluated" => Int[],
    )

    ci_on = false
    converged = false
    prev_kern = nothing  # warm-start: reuse kernel from previous iteration
    baseline_force = 0.0

    for outer_iter in 1:(cfg.max_outer_iter)
        # Train GP (shared helper handles kernel type dispatch, warm-start, noise)
        model, E_ref, y_std, prev_kern = _train_neb_gp(
            td, kernel, cfg, prev_kern,
            hess_X, hess_E, hess_G, n_hess, outer_iter)

        # Predict at all unevaluated images
        for i in 2:(N - 1)
            pred = predict(model, reshape(images[i], :, 1))
            energies[i] = pred[1] * y_std + E_ref
            gradients[i] = pred[2:end] .* y_std
        end

        # Compute forces with current estimates
        path.energies = energies
        path.gradients = gradients
        forces, max_f, ci_f, i_max = compute_all_neb_forces(path, cfg; ci_on)

        # Select image with maximum predictive variance
        max_var = -Inf
        i_eval = 2  # Default

        for i in 2:(N - 1)
            _, var_vec = predict_with_variance(model, reshape(images[i], :, 1))
            var_E = var_vec[1]  # Energy variance (first element)
            if var_E > max_var
                max_var = var_E
                i_eval = i
            end
        end

        # If climbing image is on and CI hasn't been evaluated, prioritize it
        if ci_on && !evaluated[i_max]
            i_eval = i_max
        end

        # Evaluate oracle at selected image; deduplicate training data
        E_eval, G_eval = ep_oracle(images[i_eval])
        energies[i_eval] = E_eval
        gradients[i_eval] = G_eval
        evaluated[i_eval] = true
        oracle_calls += 1

        if min_distance_to_data(images[i_eval], td.X) > dedup_tol
            add_point!(td, images[i_eval], E_eval, G_eval)
        end

        push!(history["image_evaluated"], i_eval)

        # Recompute forces with the new accurate value
        path.energies = energies
        path.gradients = gradients
        forces, max_f, ci_f, i_max = compute_all_neb_forces(path, cfg; ci_on)

        push!(history["max_force"], max_f)
        push!(history["ci_force"], ci_f)
        push!(history["oracle_calls"], oracle_calls)
        push!(history["max_energy"], maximum(energies))

        if outer_iter == 1
            baseline_force = max_f
        end

        cfg.verbose && @printf("GP-NEB-OIE outer %d: eval image %d | max|F| = %.5f | var = %.3e | N_train = %d | calls = %d\n",
                               outer_iter, i_eval, max_f, max_var, npoints(td), oracle_calls)

        # Dynamic CI activation (matches eOn)
        conv_metric = (ci_on && cfg.ci_converged_only) ? ci_f : max_f
        if cfg.climbing_image && outer_iter > 1
            ci_was_on = ci_on
            ci_on = conv_metric < cfg.ci_trigger_rel * baseline_force ||
                    conv_metric < cfg.ci_activation_tol
            if ci_on && !ci_was_on
                cfg.verbose && @printf("  Climbing image activated (image %d)\n", i_max)
                conv_metric = cfg.ci_converged_only ? ci_f : max_f
            end
        end

        # Check convergence (only if all images evaluated or force is small enough)
        conv_check = ci_on || !cfg.climbing_image
        if conv_check && all(evaluated[2:end-1]) && conv_metric < cfg.conv_tol
            cfg.verbose && println("GP-NEB-OIE converged!")
            converged = true
            break
        end

        # Inner loop: relax on GP surface (L-BFGS/SD + trust radius)
        gp_images = deepcopy(images)
        start_images = deepcopy(images)  # anchor for trust radius
        N_mov = N - 2
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

            gp_tol = max(cfg.conv_tol / 10, max_f / 10)
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

        # Update path with relaxed images
        images = gp_images
        path.images = images

        on_step !== nothing && on_step(path, outer_iter)

        # Reset evaluation flags for moved images
        for i in 2:(N - 1)
            evaluated[i] = false
        end
    end

    i_max = argmax(energies[2:end-1]) + 1
    return NEBResult(path, converged, oracle_calls, i_max, history)
end
