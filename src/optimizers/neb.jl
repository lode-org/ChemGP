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

"""
    neb_optimize(oracle, x_start, x_end; config) -> NEBResult

Standard NEB optimization (oracle-only baseline).

Evaluates the oracle at all intermediate images on every iteration. Uses
steepest descent for relaxation. This is the reference implementation for
comparison — GP-NEB variants should use fewer oracle calls.
"""
function neb_optimize(
    oracle::Function,
    x_start::Vector{Float64},
    x_end::Vector{Float64};
    config::NEBConfig = NEBConfig(),
)
    cfg = config
    N = cfg.n_images

    # Initialize path
    images = linear_interpolation(x_start, x_end, N)

    # Evaluate endpoints (fixed)
    E_start, G_start = oracle(x_start)
    E_end, G_end = oracle(x_end)
    energies = zeros(N)
    gradients = [zeros(length(x_start)) for _ in 1:N]
    energies[1] = E_start
    energies[end] = E_end
    gradients[1] = G_start
    gradients[end] = G_end

    oracle_calls = 2

    # Evaluate intermediate images
    for i in 2:(N - 1)
        E, G = oracle(images[i])
        energies[i] = E
        gradients[i] = G
        oracle_calls += 1
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

    for iter in 1:(cfg.max_iter)
        forces, max_f, ci_f, i_max = compute_all_neb_forces(path, cfg; ci_on)

        push!(history["max_force"], max_f)
        push!(history["ci_force"], ci_f)
        push!(history["oracle_calls"], oracle_calls)
        push!(history["max_energy"], maximum(energies))

        # Activate climbing image
        if !ci_on && cfg.climbing_image && max_f < cfg.ci_activation_tol
            ci_on = true
            cfg.verbose && @printf("  Iter %d: Climbing image activated (image %d)\n", iter, i_max)
            # Recompute forces with CI
            forces, max_f, ci_f, i_max = compute_all_neb_forces(path, cfg; ci_on)
        end

        # Check convergence
        conv_check = ci_on || !cfg.climbing_image
        if conv_check && max_f < cfg.conv_tol
            cfg.verbose && @printf("NEB converged at iter %d: max|F| = %.5f\n", iter, max_f)
            converged = true
            break
        end

        if iter % 50 == 0 || iter == 1
            cfg.verbose && @printf("  Iter %3d: max|F| = %.5f | CI|F| = %.5f | E_max = %.4f\n",
                                   iter, max_f, ci_f, maximum(energies))
        end

        # Steepest descent update
        for i in 2:(N - 1)
            images[i] = images[i] + cfg.step_size * forces[i]
        end

        # Re-evaluate oracle
        for i in 2:(N - 1)
            E, G = oracle(images[i])
            energies[i] = E
            gradients[i] = G
            oracle_calls += 1
        end

        path.images = images
        path.energies = energies
        path.gradients = gradients
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
"""
function gp_neb_aie(
    oracle::Function,
    x_start::Vector{Float64},
    x_end::Vector{Float64},
    kernel;
    config::NEBConfig = NEBConfig(),
)
    cfg = config
    N = cfg.n_images
    D = length(x_start)

    # Initialize path
    images = linear_interpolation(x_start, x_end, N)

    # Evaluate endpoints
    E_start, G_start = oracle(x_start)
    E_end, G_end = oracle(x_end)
    oracle_calls = 2

    # Training data accumulator
    td = TrainingData(D)
    add_point!(td, x_start, E_start, G_start)
    add_point!(td, x_end, E_end, G_end)

    # Evaluate all intermediate images
    energies = zeros(N)
    gradients = [zeros(D) for _ in 1:N]
    energies[1] = E_start
    energies[end] = E_end
    gradients[1] = G_start
    gradients[end] = G_end

    for i in 2:(N - 1)
        E, G = oracle(images[i])
        energies[i] = E
        gradients[i] = G
        add_point!(td, images[i], E, G)
        oracle_calls += 1
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

    for outer_iter in 1:(cfg.max_outer_iter)
        # Compute true forces and check convergence
        forces_true, max_f_true, ci_f_true, i_max = compute_all_neb_forces(path, cfg; ci_on)

        push!(history["max_force"], max_f_true)
        push!(history["ci_force"], ci_f_true)
        push!(history["oracle_calls"], oracle_calls)
        push!(history["max_energy"], maximum(energies))

        cfg.verbose && @printf("GP-NEB-AIE outer %d: max|F| = %.5f | CI|F| = %.5f | calls = %d\n",
                               outer_iter, max_f_true, ci_f_true, oracle_calls)

        # Activate climbing image
        if !ci_on && cfg.climbing_image && max_f_true < cfg.ci_activation_tol
            ci_on = true
            cfg.verbose && @printf("  Climbing image activated (image %d)\n", i_max)
        end

        conv_check = ci_on || !cfg.climbing_image
        if conv_check && max_f_true < cfg.conv_tol
            cfg.verbose && println("GP-NEB-AIE converged!")
            converged = true
            break
        end

        # Train GP
        y_gp, y_mean, y_std = normalize(td)
        model = GPModel(kernel, td.X, y_gp;
                        noise_var = 1e-4, grad_noise_var = 1e-4, jitter = 1e-3)
        train_model!(model; iterations = cfg.gp_train_iter)

        # Inner loop: relax on GP surface
        gp_images = deepcopy(images)
        for inner_iter in 1:(cfg.max_iter)
            # Predict energies and gradients from GP
            gp_energies = copy(energies)
            gp_gradients = deepcopy(gradients)

            for i in 2:(N - 1)
                pred = predict(model, reshape(gp_images[i], :, 1))
                gp_energies[i] = pred[1] * y_std + y_mean
                gp_gradients[i] = pred[2:end] .* y_std
            end

            gp_path = NEBPath(gp_images, gp_energies, gp_gradients, cfg.spring_constant)
            gp_forces, gp_max_f, _, _ = compute_all_neb_forces(gp_path, cfg; ci_on)

            # Adaptive GP convergence threshold
            gp_tol = max(min(max_f_true / 10, cfg.conv_tol), cfg.conv_tol / 10)

            if gp_max_f < gp_tol
                break
            end

            # Steepest descent on GP
            for i in 2:(N - 1)
                gp_images[i] = gp_images[i] + cfg.step_size * gp_forces[i]
            end
        end

        # Update path with GP-relaxed images
        images = gp_images

        # Evaluate oracle at new positions
        for i in 2:(N - 1)
            E, G = oracle(images[i])
            energies[i] = E
            gradients[i] = G
            add_point!(td, images[i], E, G)
            oracle_calls += 1
        end

        path.images = images
        path.energies = energies
        path.gradients = gradients
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
"""
function gp_neb_oie(
    oracle::Function,
    x_start::Vector{Float64},
    x_end::Vector{Float64},
    kernel;
    config::NEBConfig = NEBConfig(),
)
    cfg = config
    N = cfg.n_images
    D = length(x_start)

    # Initialize path
    images = linear_interpolation(x_start, x_end, N)

    # Evaluate endpoints
    E_start, G_start = oracle(x_start)
    E_end, G_end = oracle(x_end)
    oracle_calls = 2

    # Training data accumulator
    td = TrainingData(D)
    add_point!(td, x_start, E_start, G_start)
    add_point!(td, x_end, E_end, G_end)

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
    E_mid, G_mid = oracle(images[mid])
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

    for outer_iter in 1:(cfg.max_outer_iter)
        # Train GP
        y_gp, y_mean, y_std = normalize(td)
        model = GPModel(kernel, td.X, y_gp;
                        noise_var = 1e-4, grad_noise_var = 1e-4, jitter = 1e-3)
        train_model!(model; iterations = cfg.gp_train_iter)

        # Predict at all unevaluated images
        for i in 2:(N - 1)
            pred = predict(model, reshape(images[i], :, 1))
            energies[i] = pred[1] * y_std + y_mean
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

        # Evaluate oracle at selected image
        E_eval, G_eval = oracle(images[i_eval])
        energies[i_eval] = E_eval
        gradients[i_eval] = G_eval
        add_point!(td, images[i_eval], E_eval, G_eval)
        evaluated[i_eval] = true
        oracle_calls += 1

        push!(history["image_evaluated"], i_eval)

        # Recompute forces with the new accurate value
        path.energies = energies
        path.gradients = gradients
        forces, max_f, ci_f, i_max = compute_all_neb_forces(path, cfg; ci_on)

        push!(history["max_force"], max_f)
        push!(history["ci_force"], ci_f)
        push!(history["oracle_calls"], oracle_calls)
        push!(history["max_energy"], maximum(energies))

        cfg.verbose && @printf("GP-NEB-OIE outer %d: eval image %d | max|F| = %.5f | var = %.3e | calls = %d\n",
                               outer_iter, i_eval, max_f, max_var, oracle_calls)

        # Activate climbing image
        if !ci_on && cfg.climbing_image && max_f < cfg.ci_activation_tol
            ci_on = true
            cfg.verbose && @printf("  Climbing image activated (image %d)\n", i_max)
        end

        # Check convergence (only if all images evaluated or force is small enough)
        conv_check = ci_on || !cfg.climbing_image
        if conv_check && all(evaluated[2:end-1]) && max_f < cfg.conv_tol
            cfg.verbose && println("GP-NEB-OIE converged!")
            converged = true
            break
        end

        # Inner loop: relax on GP surface
        gp_images = deepcopy(images)
        for inner_iter in 1:(cfg.max_iter)
            gp_energies = copy(energies)
            gp_gradients = deepcopy(gradients)

            for i in 2:(N - 1)
                pred = predict(model, reshape(gp_images[i], :, 1))
                gp_energies[i] = pred[1] * y_std + y_mean
                gp_gradients[i] = pred[2:end] .* y_std
            end

            gp_path = NEBPath(gp_images, gp_energies, gp_gradients, cfg.spring_constant)
            gp_forces, gp_max_f, _, _ = compute_all_neb_forces(gp_path, cfg; ci_on)

            gp_tol = max(cfg.conv_tol / 10, max_f / 10)
            if gp_max_f < gp_tol
                break
            end

            for i in 2:(N - 1)
                gp_images[i] = gp_images[i] + cfg.step_size * gp_forces[i]
            end
        end

        # Update path with relaxed images
        images = gp_images
        path.images = images
        # Reset evaluation flags for moved images
        for i in 2:(N - 1)
            evaluated[i] = false
        end
    end

    i_max = argmax(energies[2:end-1]) + 1
    return NEBResult(path, converged, oracle_calls, i_max, history)
end
