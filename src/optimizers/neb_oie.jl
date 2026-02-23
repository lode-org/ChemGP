# ==============================================================================
# GP-NEB OIE (One Image Evaluated)
# ==============================================================================
#
# Faithful implementation of Koistinen's OIE algorithm:
#   Koistinen, O.-P. et al. (2019). Nudged elastic band calculations
#   accelerated with Gaussian process regression based on inverse
#   interatomic distances. J. Chem. Theory Comput. 15, 6738.
#
# Key differences from AIE:
# - One oracle evaluation per outer iteration (selected by priority cascade)
# - Convergence check phase: evaluate images one-by-one WITHOUT moving path
# - Early stopping: reject inner step if bond ratios or displacement exceed limits
# - All images reset to "unevaluated" after each relaxation phase
#
# The algorithm has three phases per outer iteration:
# 1. SELECT which image to evaluate (priority cascade)
# 2. DECIDE whether to relax or just evaluate more (convergence check)
# 3. RELAX on GP surface if needed, with early stopping guards

"""
    _oie_effective_ci_tol(cfg)

Effective CI convergence threshold. Returns `conv_tol` when `ci_force_tol < 0`.
"""
_oie_effective_ci_tol(cfg::NEBConfig) =
    cfg.ci_force_tol < 0 ? cfg.conv_tol : cfg.ci_force_tol

"""
    _oie_gp_tol(cfg, smallest_acc_force)

Adaptive GP inner convergence threshold (Koistinen 2019, Sec. III.B).

When `gp_tol_divisor > 0`, the threshold adapts to the smallest accurate
perpendicular gradient seen so far at any image, floored by 1/10 of the
minimum final threshold. This prevents the inner loop from over-relaxing
when the GP is still inaccurate, while tightening as data accumulates.
"""
function _oie_gp_tol(cfg::NEBConfig, smallest_acc_force::Float64)
    ci_tol = _oie_effective_ci_tol(cfg)
    floor_tol = min(cfg.conv_tol, ci_tol) / 10
    if cfg.gp_tol_divisor > 0 && isfinite(smallest_acc_force)
        return max(smallest_acc_force / cfg.gp_tol_divisor, floor_tol)
    else
        return floor_tol
    end
end

"""
    _oie_path_scale(images)

Total Euclidean length of the initial path (sum of consecutive image distances).
Used to normalize `max_step_frac`.
"""
function _oie_path_scale(images::Vector{Vector{Float64}})
    s = 0.0
    for i in 1:(length(images) - 1)
        s += norm(images[i+1] - images[i])
    end
    return s
end

"""
    _oie_check_early_stop(R_new, td, cfg, path_scale)

Check whether the proposed inner step should be rejected.

Returns `(stop, offending_image)` where `offending_image` is the 1-based
index into the *intermediate* images (2:N-1 in path indexing) that
violated a trust criterion, or 0 if no violation.

Two checks (matching Koistinen's atomic_GP_NEB_OIE2.m):

1. **Bond stretch**: For each intermediate image, the max-1D-log distance
   to the nearest training point must be less than |log(bond_stretch_limit)|.
   This catches unphysical bond distortions.

2. **Displacement**: Each intermediate image must be within
   `max_step_frac * path_scale` Euclidean distance of the nearest training
   point. This prevents the optimizer from wandering into unexplored regions.
"""
function _oie_check_early_stop(
    R_new::Vector{Vector{Float64}},
    td::TrainingData,
    cfg::NEBConfig,
    path_scale::Float64,
)
    N = length(R_new)
    N_train = npoints(td)
    log_limit = abs(log(cfg.bond_stretch_limit))
    disp_limit = cfg.max_step_frac * path_scale

    for i in 2:(N - 1)
        # Check 1: interatomic distance ratios (max-1D-log)
        if length(R_new[i]) >= 6  # need at least 2 atoms
            min_log_d = Inf
            for k in 1:N_train
                d = max_1d_log_distance(R_new[i], view(td.X, :, k))
                min_log_d = min(min_log_d, d)
            end
            if min_log_d > log_limit
                return true, i
            end
        end

        # Check 2: Euclidean displacement from nearest training point
        min_disp = Inf
        for k in 1:N_train
            d = norm(R_new[i] - view(td.X, :, k))
            min_disp = min(min_disp, d)
        end
        if min_disp > disp_limit
            return true, i
        end
    end

    return false, 0
end

"""
    _oie_inner_relax(model, images, energies, gradients, td, cfg,
                     ci_on, E_ref, y_std, gp_tol, path_scale)

Relax NEB images on the GP surface with early stopping guards.

Returns `(relaxed_images, ci_index, early_stop_image)` where:
- `relaxed_images`: image positions after relaxation
- `ci_index`: climbing image index (1-based into intermediate images, 0 if CI off)
- `early_stop_image`: path-index of the image that caused early stopping (0 if none)

The inner loop activates CI when max GP force drops below
`cfg.inner_ci_threshold` (matching Koistinen's T_CIon_gp). When CI
activates, L-BFGS curvature is reset.

After each step, the proposed positions are checked against training data
for bond stretch and displacement violations. On violation, the step is
REJECTED (positions revert) and the offending image index is returned
so the outer loop can evaluate it next.
"""
function _oie_inner_relax(
    model,
    images::Vector{Vector{Float64}},
    energies::Vector{Float64},
    gradients::Vector{Vector{Float64}},
    td::TrainingData,
    cfg::NEBConfig,
    ci_on_outer::Bool,
    E_ref::Float64,
    y_std::Float64,
    gp_tol::Float64,
    path_scale::Float64,
)
    N = length(images)
    D = length(images[1])
    N_mov = N - 2

    gp_images = deepcopy(images)
    start_images = images  # anchor for Euclidean clip
    gp_optim = cfg.optimizer == :lbfgs ? OptimState(cfg.lbfgs_memory) : nothing

    ci_on = false
    i_CI = 0
    early_stop_img = 0
    R_latest_equal = nothing  # path snapshot when CI activates

    for inner_iter in 1:(cfg.max_iter)
        gp_energies = copy(energies)
        gp_gradients = deepcopy(gradients)

        for i in 2:(N - 1)
            pred = predict(model, reshape(gp_images[i], :, 1))
            gp_energies[i] = pred[1] * y_std + E_ref
            gp_gradients[i] = pred[2:end] .* y_std
        end

        gp_path = NEBPath(gp_images, gp_energies, gp_gradients, cfg.spring_constant)
        gp_forces, gp_max_f, _, gp_i_max = compute_all_neb_forces(gp_path, cfg; ci_on)

        # CI activation during inner relaxation
        if !ci_on && cfg.inner_ci_threshold > 0 && gp_max_f < cfg.inner_ci_threshold
            R_latest_equal = deepcopy(gp_images)
            ci_on = true
            # Reset L-BFGS after CI activation (curvature changes)
            gp_optim = cfg.optimizer == :lbfgs ? OptimState(cfg.lbfgs_memory) : nothing
            # Recompute forces with CI
            gp_forces, gp_max_f, _, gp_i_max = compute_all_neb_forces(gp_path, cfg; ci_on)
            cfg.verbose && @printf("    Inner iter %d: CI activated (image %d)\n",
                                   inner_iter, gp_i_max)
        end
        i_CI = ci_on ? gp_i_max : 0

        if gp_max_f < gp_tol && inner_iter > 1
            break
        end

        # Compute step
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

        # Build candidate images with Euclidean anchor clip
        candidate_images = deepcopy(gp_images)
        for img_idx in 1:N_mov
            offset = (img_idx - 1) * D
            candidate = new_x[offset+1:offset+D]
            disp = candidate - start_images[img_idx + 1]
            dn = norm(disp)
            if dn > cfg.trust_radius
                candidate = start_images[img_idx + 1] + disp * (cfg.trust_radius / dn)
            end
            candidate_images[img_idx + 1] = candidate
        end

        # Early stopping: check bond stretch and displacement
        stop, offending = _oie_check_early_stop(candidate_images, td, cfg, path_scale)
        if stop
            early_stop_img = offending
            cfg.verbose && @printf("    Inner iter %d: early stop (image %d violated trust)\n",
                                   inner_iter, offending)
            break  # reject step, return current gp_images (not candidate)
        end

        # Accept step
        gp_images = candidate_images

        if inner_iter == cfg.max_iter
            break
        end
    end

    return gp_images, i_CI, early_stop_img
end

"""
    gp_neb_oie(oracle, x_start, x_end, kernel; config) -> NEBResult

GP-NEB with One Image Evaluated per outer iteration.

Implements the OIE algorithm of Koistinen et al. (2019):

**Outer loop** (one oracle call per iteration):
1. Select which image to evaluate using a priority cascade:
   - Priority 1: Image that caused early stopping (bond/displacement violation)
   - Priority 2: Climbing image (only during convergence check phase)
   - Priority 3: Image with highest GP predictive variance
2. Evaluate the oracle at the selected image
3. Decide whether to relax or continue checking:
   - If max perpendicular force >= `conv_tol`: start a relaxation phase
   - If max force < `conv_tol` but CI unevaluated: evaluate CI next (no relaxation)
   - If max force < `conv_tol` and CI force < `ci_force_tol`: keep checking (no relaxation)
   - If max force < `conv_tol` and CI force >= `ci_force_tol`: relax and re-check CI
4. If relaxing: run inner GP relaxation with early stopping, then reset all
   images to unevaluated

**Convergence**: When all images are evaluated with accurate forces below
both `conv_tol` and `ci_force_tol`.
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
    N = cfg.images + 2
    D = length(x_start)
    ep_oracle = _single_oracle(oracle)
    ci_tol = _oie_effective_ci_tol(cfg)

    images = _init_neb_images(cfg, x_start, x_end)
    path_scale = _oie_path_scale(images)

    # Evaluate endpoints
    E_start, G_start = ep_oracle(x_start)
    E_end, G_end = ep_oracle(x_end)
    oracle_calls = 2

    # Training data
    td = TrainingData(D)
    add_point!(td, x_start, E_start, G_start)
    add_point!(td, x_end, E_end, G_end)

    # Virtual Hessian points
    hess_X, hess_E, hess_G, n_hess, hess_calls = _init_hessian_data(
        cfg, ep_oracle, x_start, x_end, D)
    oracle_calls += hess_calls

    # Path state
    energies = zeros(N)
    gradients = [zeros(D) for _ in 1:N]
    energies[1] = E_start
    energies[end] = E_end
    gradients[1] = G_start
    gradients[end] = G_end

    # Unevaluated flags (endpoints are always "evaluated")
    uneval = falses(N)
    for i in 2:(N - 1)
        uneval[i] = true
    end

    # Priority cascade state
    eval_next_early = 0   # image index from early stopping (priority 1)
    eval_next_ci = false   # should we evaluate CI next? (priority 2)

    path = NEBPath(images, energies, gradients, cfg.spring_constant)

    history = Dict(
        "max_force" => Float64[],
        "ci_force" => Float64[],
        "oracle_calls" => Int[],
        "max_energy" => Float64[],
        "image_evaluated" => Int[],
    )

    converged = false
    prev_kern = nothing
    smallest_acc_force = Inf  # smallest accurate max|G_perp| at any image
    dedup_tol = cfg.conv_tol * 0.1

    for outer_iter in 1:(cfg.max_outer_iter)
        # =====================================================================
        # STEP 1: Select which image to evaluate (priority cascade)
        # =====================================================================
        if eval_next_early > 0
            # Priority 1: image that caused early stopping
            i_eval = eval_next_early
            eval_next_early = 0
            cfg.verbose && @printf("GP-NEB-OIE %d: evaluate early-stopped image %d\n",
                                   outer_iter, i_eval)
        elseif eval_next_ci && cfg.climbing_image
            # Priority 2: climbing image (during convergence check)
            i_max = argmax(energies[2:end-1]) + 1
            i_eval = i_max
            eval_next_ci = false
            cfg.verbose && @printf("GP-NEB-OIE %d: evaluate climbing image %d\n",
                                   outer_iter, i_eval)
        else
            # Priority 3: image with highest predictive variance
            # Need a trained GP to compute variance -- train first
            model, E_ref, y_std, prev_kern = _train_neb_gp(
                td, kernel, cfg, prev_kern,
                hess_X, hess_E, hess_G, n_hess, outer_iter)

            # Predict at unevaluated images to get variance
            max_var = -Inf
            i_eval = 0
            for i in 2:(N - 1)
                if uneval[i]
                    _, var_vec = predict_with_variance(model, reshape(images[i], :, 1))
                    var_E = var_vec[1]
                    if var_E > max_var
                        max_var = var_E
                        i_eval = i
                    end
                end
            end
            # Fallback: if no unevaluated images, pick max variance overall
            if i_eval == 0
                for i in 2:(N - 1)
                    _, var_vec = predict_with_variance(model, reshape(images[i], :, 1))
                    var_E = var_vec[1]
                    if var_E > max_var
                        max_var = var_E
                        i_eval = i
                    end
                end
            end
            cfg.verbose && @printf("GP-NEB-OIE %d: evaluate max-variance image %d (var=%.3e)\n",
                                   outer_iter, i_eval, max_var)
        end

        # =====================================================================
        # STEP 2: Evaluate oracle at selected image
        # =====================================================================
        E_eval, G_eval = ep_oracle(images[i_eval])
        energies[i_eval] = E_eval
        gradients[i_eval] = G_eval
        uneval[i_eval] = false
        oracle_calls += 1

        if min_distance_to_data(images[i_eval], td.X) > dedup_tol
            add_point!(td, images[i_eval], E_eval, G_eval)
        end

        push!(history["image_evaluated"], i_eval)

        # =====================================================================
        # Check final convergence when ALL images are evaluated
        # =====================================================================
        n_uneval = count(uneval)
        if n_uneval == 0
            forces, max_f, ci_f, i_max = compute_all_neb_forces(path, cfg; ci_on=cfg.climbing_image)
            if max_f < cfg.conv_tol && ci_f < ci_tol
                push!(history["max_force"], max_f)
                push!(history["ci_force"], ci_f)
                push!(history["oracle_calls"], oracle_calls)
                push!(history["max_energy"], maximum(energies))
                cfg.verbose && @printf("GP-NEB-OIE converged! max|F|=%.5f, CI|F|=%.5f, calls=%d\n",
                                       max_f, ci_f, oracle_calls)
                converged = true
                break
            end
        end

        # Remove Hessian points when their iteration window expires
        # (handled inside _train_neb_gp via outer_iter check)

        # =====================================================================
        # STEP 3: Train GP and update estimates at unevaluated images
        # =====================================================================
        model, E_ref, y_std, prev_kern = _train_neb_gp(
            td, kernel, cfg, prev_kern,
            hess_X, hess_E, hess_G, n_hess, outer_iter)

        # Predict at unevaluated images
        for i in 2:(N - 1)
            if uneval[i]
                pred = predict(model, reshape(images[i], :, 1))
                energies[i] = pred[1] * y_std + E_ref
                gradients[i] = pred[2:end] .* y_std
            end
        end

        path.images = images
        path.energies = energies
        path.gradients = gradients

        # Compute forces (with CI if configured)
        forces, max_f, ci_f, i_max = compute_all_neb_forces(path, cfg; ci_on=cfg.climbing_image)

        # Track smallest accurate perpendicular gradient
        # (use the force at the just-evaluated image if it's accurate)
        if !uneval[i_eval]
            img_force_norm = norm(forces[i_eval])
            smallest_acc_force = min(smallest_acc_force, img_force_norm)
        end

        push!(history["max_force"], max_f)
        push!(history["ci_force"], ci_f)
        push!(history["oracle_calls"], oracle_calls)
        push!(history["max_energy"], maximum(energies))

        n_total = npoints(td) + (outer_iter <= cfg.num_hess_iter ? n_hess : 0)
        cfg.verbose && @printf("  max|F|=%.5f CI|F|=%.5f N_train=%d calls=%d uneval=%d\n",
                               max_f, ci_f, n_total, oracle_calls, n_uneval)

        # =====================================================================
        # STEP 4: Decide whether to relax or continue convergence check
        # =====================================================================
        start_relax = false
        eval_next_ci = false

        if max_f >= cfg.conv_tol
            # Forces above threshold -- need to relax
            start_relax = true
        else
            # Forces below threshold -- convergence check phase
            if cfg.climbing_image && uneval[i_max]
                # CI not yet evaluated -- evaluate it next (no relaxation)
                eval_next_ci = true
                start_relax = false
            elseif cfg.climbing_image && ci_f >= ci_tol
                # CI evaluated but its force is too high -- relax and re-check
                start_relax = true
                eval_next_ci = true
            else
                # All looks good so far -- keep evaluating without moving
                start_relax = false
            end
        end

        # =====================================================================
        # STEP 5: Relax on GP surface (if desired)
        # =====================================================================
        if start_relax
            gp_tol = _oie_gp_tol(cfg, smallest_acc_force)
            cfg.verbose && @printf("  Starting relaxation (GP tol=%.5f)\n", gp_tol)

            images, i_CI, early_stop = _oie_inner_relax(
                model, images, energies, gradients, td, cfg,
                cfg.climbing_image, E_ref, y_std, gp_tol, path_scale)

            # EMD trust clip at outer boundary
            _emd_trust_clip!(images, td, cfg)

            path.images = images

            # Record which image caused early stopping (for priority 1 next iteration)
            if early_stop > 0
                eval_next_early = early_stop
            end

            # Reset: all intermediate images become unevaluated after relaxation
            for i in 2:(N - 1)
                uneval[i] = true
            end
            # Zero out energies/gradients for unevaluated images
            for i in 2:(N - 1)
                energies[i] = 0.0
                gradients[i] = zeros(D)
            end
            path.energies = energies
            path.gradients = gradients
        end

        on_step !== nothing && on_step(path, outer_iter)
    end

    i_max = argmax(energies[2:end-1]) + 1
    return NEBResult(path, converged, oracle_calls, i_max, history)
end
