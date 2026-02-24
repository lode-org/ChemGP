# ==============================================================================
# GP-NEB OIE -- Naive (pedagogical) variant
# ==============================================================================
#
# Minimal OIE: train GP, relax on GP surface, pick the image with highest
# predictive variance, evaluate oracle there, repeat.
#
# No priority cascade, no convergence check phase, no early stopping.
# Good for understanding the concept and for 2D test surfaces where
# bond-stretch guards are irrelevant.
#
# For the full algorithm with all guardrails, see neb_oie.jl.

"""
    gp_neb_oie_naive(oracle, x_start, x_end, kernel; config) -> NEBResult

Naive GP-NEB with One Image Evaluated per outer iteration.

At each outer iteration:
1. Train the GP on accumulated data
2. Relax the path on the GP surface (inner loop)
3. Pick the image with the highest predictive variance
4. Evaluate the oracle there
5. Check convergence

This is the pedagogical version -- easy to understand and sufficient for
simple surfaces. For molecular systems, use [`gp_neb_oie`](@ref) which
adds early stopping, convergence check phases, and a priority cascade
for image selection (Koistinen et al. 2019).
"""
function gp_neb_oie_naive(
    oracle::OracleOrPool,
    x_start::Vector{Float64},
    x_end::Vector{Float64},
    kernel;
    config::NEBConfig=NEBConfig(),
    on_step::Union{Function,Nothing}=nothing,
)
    cfg = config
    N = cfg.images + 2
    D = length(x_start)
    ep_oracle = _single_oracle(oracle)
    dedup_tol = cfg.conv_tol * 0.1

    images = _init_neb_images(cfg, x_start, x_end)

    # Evaluate endpoints
    E_start, G_start = ep_oracle(x_start)
    E_end, G_end = ep_oracle(x_end)
    oracle_calls = 2

    # Training data accumulator
    td = TrainingData(D)
    add_point!(td, x_start, E_start, G_start)
    add_point!(td, x_end, E_end, G_end)

    # Virtual Hessian points
    hess_X, hess_E, hess_G, n_hess, hess_calls = _init_hessian_data(
        cfg, ep_oracle, x_start, x_end, D
    )
    oracle_calls += hess_calls

    # Bootstrap: evaluate midpoint
    energies = zeros(N)
    gradients = [zeros(D) for _ in 1:N]
    energies[1] = E_start
    energies[end] = E_end
    gradients[1] = G_start
    gradients[end] = G_end
    evaluated = falses(N)
    evaluated[1] = true
    evaluated[end] = true

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
    stop_reason = MAX_ITERATIONS
    prev_kern = nothing
    baseline_force = 0.0

    for outer_iter in 1:(cfg.max_outer_iter)
        # Train GP (per-bead subset when max_gp_points > 0)
        model, E_ref, y_std, prev_kern = _train_neb_gp(
            td, kernel, cfg, prev_kern, hess_X, hess_E, hess_G, n_hess, outer_iter; images
        )

        # Predict at all intermediate images
        for i in 2:(N - 1)
            pred = predict(model, reshape(images[i], :, 1))
            energies[i] = pred[1] * y_std + E_ref
            gradients[i] = pred[2:end] .* y_std
        end

        path.energies = energies
        path.gradients = gradients
        forces, max_f, ci_f, i_max = compute_all_neb_forces(path, cfg; ci_on)

        # Select image with maximum predictive variance
        max_var = -Inf
        i_eval = 2

        for i in 2:(N - 1)
            _, var_vec = predict_with_variance(model, reshape(images[i], :, 1))
            var_E = var_vec[1]
            if var_E > max_var
                max_var = var_E
                i_eval = i
            end
        end

        # Prioritize CI image if not yet evaluated
        if ci_on && !evaluated[i_max]
            i_eval = i_max
        end

        # Evaluate oracle at selected image; deduplicate
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

        cfg.verbose && @printf(
            "GP-NEB-OIE-naive %d: eval image %d | max|F| = %.5f | var = %.3e | N_train = %d | calls = %d\n",
            outer_iter,
            i_eval,
            max_f,
            max_var,
            npoints(td),
            oracle_calls
        )

        # Dynamic CI activation
        ci_on, conv_metric, ci_activated = _check_ci(
            cfg, ci_on, max_f, ci_f, baseline_force, outer_iter
        )
        if ci_activated
            cfg.verbose && @printf("  Climbing image activated (image %d)\n", i_max)
        end

        # Check convergence: CI image must be oracle-evaluated when ci_converged_only
        conv_check = ci_on || !cfg.climbing_image
        ci_verified = !cfg.ci_converged_only || evaluated[i_max]
        if conv_check && ci_verified && conv_metric < cfg.conv_tol
            cfg.verbose && println("GP-NEB-OIE-naive converged!")
            converged = true
            stop_reason = CONVERGED
            break
        end

        # Inner loop: relax on GP surface
        gp_tol = max(max_f / 10, cfg.conv_tol / 10)
        images = _gp_inner_relax(
            model, images, energies, gradients, cfg, ci_on, E_ref, y_std, gp_tol
        )

        # EMD trust clip at outer boundary
        _emd_trust_clip!(images, td, cfg)
        path.images = images

        if on_step !== nothing
            cb_result = on_step(path, outer_iter)
            if cb_result === :stop
                cfg.verbose && println("  Stopped by on_step callback.")
                stop_reason = USER_CALLBACK
                break
            end
        end

        # Reset evaluation flags for moved images
        for i in 2:(N - 1)
            evaluated[i] = false
        end
    end

    i_max = argmax(energies[2:(end - 1)]) + 1
    return NEBResult(path, converged, stop_reason, oracle_calls, i_max, history)
end
