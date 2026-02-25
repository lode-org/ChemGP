# ==============================================================================
# Covariance Matrix Assembly
# ==============================================================================
#
# The key insight of GP regression with derivative observations is that if we
# know the covariance k(x, x') between energies, we can derive the covariance
# between energies and gradients (dk/dx') and between gradients (d^2k/dxdx').
#
# For N training points in D dimensions, the full covariance matrix has
# dimension N*(1+D): N energy observations + N*D gradient observations.

# Inverse normal CDF at p=0.75: norminv(0.75, 0, 1).
# Used for data-dependent kernel initialization matching GPstuff/MATLAB.
const NORMINV_075 = 0.6744897501960817

"""
    build_full_covariance(kernel, X, noise_e, noise_g, jitter)

Assemble the full N*(1+D) x N*(1+D) covariance matrix for GP regression with
derivative observations.

For `N` training points in `D` dimensions, the matrix has block structure:

    [ K_EE + σ_n²I    K_EG          ]
    [ K_GE             K_GG + σ_g²I  ]

where `K_EE` is the N x N energy-energy block, `K_EG` and `K_GE` are the
N x ND cross-covariance blocks, and `K_GG` is the ND x ND gradient-gradient
block. The blocks are computed via [`kernel_blocks`](@ref) using automatic
differentiation.

Returns a `Symmetric` matrix.

See also: [`kernel_blocks`](@ref), [`train_model!`](@ref)
"""
function build_full_covariance(
    kernel::Kernel, X::Matrix{Float64}, noise_e::Real, noise_g::Real, jitter::Real
)
    D, N = size(X)
    TotalDim = N * (1 + D)
    K_mat = zeros(TotalDim, TotalDim)

    for i in 1:N
        xi = view(X, :, i)

        # Diagonal Blocks
        k_ee, k_ef, k_fe, k_ff = kernel_blocks(kernel, xi, xi)

        # Energy Index
        K_mat[i, i] = k_ee + noise_e + jitter

        # Gradient Indices
        s_g = N + (i-1)*D + 1
        e_g = N + i*D

        K_mat[i, s_g:e_g] = k_ef
        K_mat[s_g:e_g, i] = k_fe
        K_mat[s_g:e_g, s_g:e_g] = k_ff + (noise_g + jitter) * I

        # Off-diagonal Interactions
        for j in (i + 1):N
            xj = view(X, :, j)
            k_ee, k_ef, k_fe, k_ff = kernel_blocks(kernel, xi, xj)

            j_s = N + (j-1)*D + 1
            j_e = N + j*D

            K_mat[i, j] = k_ee
            K_mat[j, i] = k_ee

            K_mat[i, j_s:j_e] = k_ef
            K_mat[s_g:e_g, j] = k_fe

            K_mat[j, s_g:e_g] = k_fe'
            K_mat[j_s:j_e, i] = k_ef'

            K_mat[s_g:e_g, j_s:j_e] = k_ff
            K_mat[j_s:j_e, s_g:e_g] = k_ff'
        end
    end

    # Truncate near-zero entries to prevent accumulation of floating-point
    # noise that can create slightly negative eigenvalues.
    # Matches MATLAB GPstuff: C(C<eps)=0
    @inbounds for idx in eachindex(K_mat)
        if abs(K_mat[idx]) < eps(Float64)
            K_mat[idx] = 0.0
        end
    end

    return Symmetric(K_mat)
end

# ==============================================================================
# Data-dependent kernel initialization
# ==============================================================================

"""
    init_mol_invdist_se(td::TrainingData, kernel::MolInvDistSE) -> MolInvDistSE

Data-dependent initialization of MolInvDistSE hyperparameters.

Computes pairwise distances in inverse distance feature space and sets
signal_variance and inv_lengthscales from the ranges of training energies
and feature-space distances, matching the MATLAB GPstuff initialization
used in `atomic_GP_NEB_AIE.m`:

    magnSigma2  = (norminv(0.75) * range_y / 3)^2
    lengthScale = norminv(0.75) * range_x / 3

where `range_x` is the maximum pairwise distance in inverse distance
feature space (with the sqrt(2) factor from MATLAB's `dist_at`).

Preserves the kernel's `frozen_coords` and `feature_params_map`.
"""
function init_mol_invdist_se(td::TrainingData, kernel::MolInvDistSE)
    N = npoints(td)
    frozen = kernel.frozen_coords

    # Compute inverse distance features for all training points
    features = [compute_inverse_distances(view(td.X, :, i), frozen) for i in 1:N]

    # Max pairwise distance in feature space
    max_feat_dist = 0.0
    for i in 1:N
        for j in (i + 1):N
            d = norm(features[i] - features[j])
            max_feat_dist = max(max_feat_dist, d)
        end
    end

    # Energy range
    range_y = maximum(td.energies) - minimum(td.energies)
    range_y = max(range_y, 1e-10)

    # MATLAB dist_at returns sqrt(2*sum((delta_invr/l)^2)) with l=1,
    # so range_x includes a sqrt(2) factor over the raw feature distance
    range_x = sqrt(2) * max(max_feat_dist, 1e-10)

    sigma2 = (NORMINV_075 * range_y / 3)^2
    ell = NORMINV_075 * range_x / 3
    inv_ell = 1.0 / max(ell, 1e-10)

    # Preserve kernel structure: fill all lengthscale slots uniformly
    n_ls = length(kernel.inv_lengthscales)
    inv_ls = fill(inv_ell, n_ls)

    return typeof(kernel)(sigma2, inv_ls, kernel.frozen_coords, kernel.feature_params_map)
end

"""
    init_cartesian_se(td::TrainingData) -> CartesianSE

Data-dependent initialization of CartesianSE hyperparameters.

Sets signal_variance and lengthscale from the ranges of training energies
and coordinates, matching the MATLAB GPstuff initialization:

    magnSigma2 = (norminv(0.75) * range_y / 3)^2
    lengthScale = norminv(0.75) * range_x / 3

where norminv(0.75, 0, 1) = 0.6745.
"""
function init_cartesian_se(td::TrainingData)
    range_y = maximum(td.energies) - minimum(td.energies)
    range_y = max(range_y, 1e-10)

    # Max pairwise distance between training configurations
    N = npoints(td)
    range_x = 0.0
    for i in 1:N
        for j in (i + 1):N
            d = norm(td.X[:, i] - td.X[:, j])
            range_x = max(range_x, d)
        end
    end
    range_x = max(range_x, 1e-10)

    sigma2 = (NORMINV_075 * range_y / 3)^2
    ell = NORMINV_075 * range_x / 3

    return CartesianSE(sigma2, ell)
end

# ==============================================================================
# Robust Cholesky with adaptive jitter
# ==============================================================================

"""
    _robust_cholesky(K; max_attempts=8) -> Cholesky

Attempt Cholesky factorization with exponentially increasing jitter.

Starts with no jitter, then scales relative to the maximum diagonal entry.
This handles rank-deficient covariance matrices that arise from molecular
kernels where the feature space dimension is smaller than the coordinate
space (e.g., 3 inverse distances for 9D coordinates of a 3-atom system).
"""
function _robust_cholesky(K::AbstractMatrix; max_attempts::Int=8)
    L = try
        cholesky(K)
    catch
        nothing
    end
    L !== nothing && return L

    max_diag = maximum(diag(K))
    scale = max(max_diag, 1.0)
    jitter = scale * 1e-8

    for attempt in 2:max_attempts
        L = try
            cholesky(K + jitter * I)
        catch
            nothing
        end
        L !== nothing && return L
        jitter *= 10
    end

    error("Cholesky failed after $max_attempts attempts (max jitter = $(jitter / 10))")
end

# ==============================================================================
# MAP NLL + Analytical Gradient (for SCG training)
# ==============================================================================

"""
    nll_and_grad(w, X, y, frozen, feat_map, noise_e, noise_g, jitter, w_prior, prior_var)

Compute the MAP negative log-likelihood and its gradient w.r.t. log-space
hyperparameters `w = [log(sigma2); log.(inv_lengthscales)]`.

Uses analytical dK/d(log w) from `kernel_blocks_and_hypergrads` and the
standard GP gradient formula:

    grad_j = 0.5 * tr(W * dK/dw_j) + prior_grad_j

where `W = K_inv - alpha * alpha'`, `alpha = K \\ y`.

MAP prior: Gaussian in log-space centered at `w_prior` with variance `prior_var`.
"""
function nll_and_grad(
    w::Vector{Float64},
    X::Matrix{Float64},
    y::Vector{Float64},
    frozen::AbstractVector,
    feat_map::Vector{Int},
    noise_e::Float64,
    noise_g::Float64,
    jitter::Float64,
    w_prior::Vector{Float64},
    prior_var::Vector{Float64},
)
    sigma2 = exp(w[1])
    inv_ls = exp.(w[2:end])
    n_params = length(w)

    kern = MolInvDistSE(sigma2, inv_ls, frozen, feat_map)
    D, N = size(X)
    TotalDim = N * (1 + D)

    # Build covariance matrix
    K_mat = zeros(TotalDim, TotalDim)
    # Store per-pair grad contributions: dK/dw_j accumulated
    dK = [zeros(TotalDim, TotalDim) for _ in 1:n_params]

    for i in 1:N
        xi = view(X, :, i)
        s_gi = N + (i - 1) * D + 1
        e_gi = N + i * D

        # Diagonal
        (b_ee, b_ef, b_fe, b_ff), grad_b = kernel_blocks_and_hypergrads(kern, xi, xi)
        K_mat[i, i] = b_ee + noise_e + jitter
        K_mat[i, s_gi:e_gi] = vec(b_ef)
        K_mat[s_gi:e_gi, i] = b_fe
        K_mat[s_gi:e_gi, s_gi:e_gi] = b_ff + (noise_g + jitter) * I

        for j_p in 1:n_params
            (de, def_p, dfe_p, dff_p) = grad_b[j_p]
            dK[j_p][i, i] += de
            dK[j_p][i, s_gi:e_gi] .+= vec(def_p)
            dK[j_p][s_gi:e_gi, i] .+= dfe_p
            dK[j_p][s_gi:e_gi, s_gi:e_gi] .+= dff_p
        end

        # Off-diagonal
        for j in (i + 1):N
            xj = view(X, :, j)
            s_gj = N + (j - 1) * D + 1
            e_gj = N + j * D

            (b_ee, b_ef, b_fe, b_ff), grad_b = kernel_blocks_and_hypergrads(kern, xi, xj)

            K_mat[i, j] = b_ee
            K_mat[j, i] = b_ee
            K_mat[i, s_gj:e_gj] = vec(b_ef)
            K_mat[s_gi:e_gi, j] = b_fe
            K_mat[j, s_gi:e_gi] = b_fe
            K_mat[s_gj:e_gj, i] = vec(b_ef)
            K_mat[s_gi:e_gi, s_gj:e_gj] = b_ff
            K_mat[s_gj:e_gj, s_gi:e_gi] = b_ff'

            for j_p in 1:n_params
                (de, def_p, dfe_p, dff_p) = grad_b[j_p]
                dK[j_p][i, j] += de
                dK[j_p][j, i] += de
                dK[j_p][i, s_gj:e_gj] .+= vec(def_p)
                dK[j_p][s_gi:e_gi, j] .+= dfe_p
                dK[j_p][j, s_gi:e_gi] .+= dfe_p
                dK[j_p][s_gj:e_gj, i] .+= vec(def_p)
                dK[j_p][s_gi:e_gi, s_gj:e_gj] .+= dff_p
                dK[j_p][s_gj:e_gj, s_gi:e_gi] .+= dff_p'
            end
        end
    end

    # Truncate near-zero (matching build_full_covariance)
    @inbounds for idx in eachindex(K_mat)
        if abs(K_mat[idx]) < eps(Float64)
            K_mat[idx] = 0.0
        end
    end

    K_sym = Symmetric(K_mat)

    # Cholesky factorization
    L = try
        _robust_cholesky(K_sym)
    catch
        return (Inf, zeros(n_params))
    end

    alpha = L \ y
    # Standard GP NLL: 0.5*y'K^{-1}y + 0.5*logdet(K) + 0.5*n*log(2pi)
    # logdet(L::Cholesky) returns logdet(K), so multiply by 0.5
    nll = 0.5 * dot(y, alpha) + 0.5 * logdet(L) + 0.5 * TotalDim * log(2pi)

    # MAP prior contribution
    nll += 0.5 * sum((w .- w_prior) .^ 2 ./ prior_var)

    # Gradient: W = K_inv - alpha*alpha'
    # grad_j = 0.5 * tr(W * dK_j) = 0.5 * sum(W .* dK_j')
    K_inv = inv(L)
    W = K_inv - alpha * alpha'

    grad = Vector{Float64}(undef, n_params)
    for j_p in 1:n_params
        # tr(W * dK_j) = sum(W .* dK_j') = sum(W' .* dK_j) = sum(W .* dK_j)
        # since W is symmetric and we treat dK_j as symmetric too
        grad[j_p] = 0.5 * dot(W, dK[j_p])
    end

    # MAP prior gradient
    grad .+= (w .- w_prior) ./ prior_var

    return (nll, grad)
end

# ==============================================================================
# Training (ParameterHandling + Optim)
# ==============================================================================

# NOTE(rg): The AbstractMoleculeKernel path now uses SCG with MAP NLL
# and analytical gradients, matching the C++ gpr_optim / MATLAB gpstuff
# production implementations. Nelder-Mead is kept as fallback.
"""
    train_model!(model::GPModel{Tk}; iterations=1000) where {Tk<:AbstractMoleculeKernel}

Optimize GP hyperparameters by minimizing the MAP negative log-likelihood.

For `MolInvDistSE` kernels with `fix_noise=true`, uses SCG (Scaled Conjugate
Gradient) with analytical gradients and a Gaussian MAP prior in log-space,
matching the C++ gpr_optim / MATLAB gpstuff production implementations.

Falls back to Nelder-Mead if SCG does not converge or for non-SE mol kernels.

When `fix_noise=false`, uses Nelder-Mead on all parameters (noise included).

Mutates `model` in-place with the optimized kernel and noise parameters.

See also: [`build_full_covariance`](@ref), [`predict`](@ref), [`scg_optimize`](@ref)
"""
function train_model!(
    model::GPModel{Tk};
    iterations=1000,
    fix_noise::Bool=false,
    verbose::Bool=true,
    barrier_strength::Float64=0.0,
) where {Tk<:AbstractMoleculeKernel}
    frozen = model.kernel.frozen_coords
    feat_map = model.kernel.feature_params_map

    # --- SCG path: MolInvDistSE with fix_noise=true ---
    if Tk === MolInvDistSE && fix_noise
        # Pack to log-space (bypass ParameterHandling for mol kernel path)
        w0 = vcat(
            [log(max(Float64(model.kernel.signal_variance), 1e-30))],
            [log(max(Float64(l), 1e-30)) for l in model.kernel.inv_lengthscales],
        )
        w_prior = copy(w0)  # MAP center = data-dependent init values
        # Gaussian MAP prior in log-space: sigma2 gets s2=2.0, lengthscales
        # scaled by the number of features (distances) per pair type.
        # Pair types with many distances (e.g. C-H with 8 pairs) get s2=0.5;
        # pair types with few distances (e.g. C-C with 1 pair) get tighter
        # s2~0.15 to prevent collapse from underdetermined optimization.
        n_ls = length(model.kernel.inv_lengthscales)
        n_feat_per_param = zeros(Int, n_ls)
        if !isempty(feat_map)
            for p_idx in feat_map
                n_feat_per_param[p_idx] += 1
            end
        else
            fill!(n_feat_per_param, max(1, length(model.kernel.inv_lengthscales)))
        end
        ls_prior_var = [0.5 * clamp(n_feat_per_param[p] / 3, 0.3, 1.0) for p in 1:n_ls]
        prior_var = vcat([2.0], ls_prior_var)

        noise_e = model.noise_var
        noise_g = model.grad_noise_var
        jit = model.jitter

        # Capture references for closure
        X_ref = model.X
        y_ref = model.y

        function _scg_fg!(f_ref, g_vec, w)
            f_val, g_val = nll_and_grad(
                w, X_ref, y_ref, frozen, feat_map,
                noise_e, noise_g, jit, w_prior, prior_var,
            )
            f_ref[] = f_val
            g_vec .= g_val
        end

        w_best, f_best, converged = scg_optimize(
            _scg_fg!, w0;
            max_iter=iterations, tol_f=1e-4, verbose=verbose,
        )

        if converged || f_best < Inf
            sigma2_opt = exp(w_best[1])
            inv_ls_opt = exp.(w_best[2:end])
            model.kernel = Tk(sigma2_opt, inv_ls_opt, frozen, feat_map)
            verbose && @printf("SCG Training Complete. Final MAP NLL: %.4f\n", f_best)
            return model
        end

        # SCG failed: fall through to Nelder-Mead
        verbose && println("SCG did not converge, falling back to Nelder-Mead...")
    end

    # --- Nelder-Mead fallback (all mol kernel types, or fix_noise=false) ---
    raw_initial_params = if fix_noise
        (
            signal_var=positive(model.kernel.signal_variance),
            inv_lengthscales=positive(model.kernel.inv_lengthscales),
        )
    else
        (
            signal_var=positive(model.kernel.signal_variance),
            inv_lengthscales=positive(model.kernel.inv_lengthscales),
            noise=positive(model.noise_var),
            grad_noise=positive(model.grad_noise_var),
        )
    end

    flat_initial_params, unflatten = ParameterHandling.value_flatten(raw_initial_params)

    fixed_noise = model.noise_var
    fixed_grad_noise = model.grad_noise_var

    function objective(params::NamedTuple)
        k = Tk(params.signal_var, params.inv_lengthscales, frozen, feat_map)
        noise = fix_noise ? fixed_noise : params.noise
        grad_noise = fix_noise ? fixed_grad_noise : params.grad_noise

        K = build_full_covariance(k, model.X, noise, grad_noise, model.jitter)
        L = try
            _robust_cholesky(K)
        catch
            return Inf
        end

        nll = 0.5 * dot(model.y, L \ model.y) + 0.5 * logdet(L) + 0.5 * length(model.y) * log(2pi)

        if barrier_strength > 0.0
            log_sv = log(max(params.signal_var, 1e-30))
            max_log_sv = log(max(params.signal_var, 1.0))
            barrier = max_log_sv - log_sv
            nll -= barrier_strength * log(max(barrier, 1e-30))
        end

        return nll
    end

    res = Optim.optimize(
        objective ∘ unflatten,
        flat_initial_params,
        NelderMead(),
        Optim.Options(; iterations=iterations, show_trace=verbose && !fix_noise),
    )

    best_params = unflatten(Optim.minimizer(res))
    model.kernel = Tk(
        best_params.signal_var, best_params.inv_lengthscales, frozen, feat_map
    )
    if !fix_noise
        model.noise_var = best_params.noise
        model.grad_noise_var = best_params.grad_noise
    end

    verbose && @printf("NM Training Complete. Final NLL: %.4f\n", Optim.minimum(res))
    return model
end

"""
    train_model!(model::GPModel{Tk}; iterations=1000) where {Tk<:Kernel}

Fallback training for generic KernelFunctions.jl kernels (e.g., `SqExponentialKernel`).

Since we cannot generically introspect kernel hyperparameters, this method keeps
the kernel fixed and optimizes only the noise variances by minimizing the negative
log marginal likelihood.

This is used by GP-NEB on non-molecular surfaces (Muller-Brown 2D).
"""
function train_model!(
    model::GPModel{Tk}; iterations=1000, verbose::Bool=true
) where {Tk<:Kernel}
    kernel = model.kernel

    raw_initial_params = (
        noise=positive(model.noise_var), grad_noise=positive(model.grad_noise_var)
    )

    flat_initial_params, unflatten = ParameterHandling.value_flatten(raw_initial_params)

    function objective(params::NamedTuple)
        K = build_full_covariance(
            kernel, model.X, params.noise, params.grad_noise, model.jitter
        )
        L = try
            _robust_cholesky(K)
        catch
            return Inf
        end
        return 0.5 * dot(model.y, L \ model.y) + 0.5 * logdet(L) + 0.5 * length(model.y) * log(2π)
    end

    res = Optim.optimize(
        objective ∘ unflatten,
        flat_initial_params,
        NelderMead(),
        Optim.Options(; iterations=iterations, show_trace=false),
    )

    best_params = unflatten(Optim.minimizer(res))
    model.noise_var = best_params.noise
    model.grad_noise_var = best_params.grad_noise

    return model
end

"""
    train_model!(model::GPModel{CartesianSE}; iterations=1000)

Hyperparameter optimization for CartesianSE kernel.

Optimizes signal_variance, lengthscale, noise_var, and grad_noise_var
by minimizing the negative log marginal likelihood via Nelder-Mead.

Matches the MATLAB GPstuff `gp_optim` behavior for `gpcf_sexp`.
"""
function train_model!(
    model::GPModel{CartesianSE{T}}; iterations=1000, verbose::Bool=true
) where {T}
    raw_initial_params = (
        signal_var=positive(model.kernel.signal_variance),
        lengthscale=positive(model.kernel.lengthscale),
        noise=positive(model.noise_var),
        grad_noise=positive(model.grad_noise_var),
    )

    flat_initial_params, unflatten = ParameterHandling.value_flatten(raw_initial_params)

    function objective(params::NamedTuple)
        k = CartesianSE(params.signal_var, params.lengthscale)
        K = build_full_covariance(k, model.X, params.noise, params.grad_noise, model.jitter)
        L = try
            _robust_cholesky(K)
        catch
            return Inf
        end
        return 0.5 * dot(model.y, L \ model.y) + 0.5 * logdet(L) + 0.5 * length(model.y) * log(2π)
    end

    res = Optim.optimize(
        objective ∘ unflatten,
        flat_initial_params,
        NelderMead(),
        Optim.Options(; iterations=iterations, show_trace=false),
    )

    best_params = unflatten(Optim.minimizer(res))
    model.kernel = CartesianSE(best_params.signal_var, best_params.lengthscale)
    model.noise_var = best_params.noise
    model.grad_noise_var = best_params.grad_noise

    verbose && @printf(
        "CartesianSE training: NLL = %.4f, sigma2 = %.4f, ell = %.4f\n",
        Optim.minimum(res),
        best_params.signal_var,
        best_params.lengthscale
    )
    return model
end

# ==============================================================================
# Prediction (Mean)
# ==============================================================================

"""
    predict(model::GPModel, X_test::Matrix{Float64})

Compute the GP posterior mean at test points.

For each test point, returns predicted energy and gradient in a single vector
laid out as `[E1, G1_1, ..., G1_D, E2, G2_1, ..., G2_D, ...]` with length
`N_test * (1 + D)`.

Predictions are in the **normalized** space of `model.y`. To recover physical
units, scale by `y_std` and shift energies by `y_mean` (the values returned
by [`normalize`](@ref)).

Uses Cholesky factorization with adaptive jitter for numerical stability.

See also: [`predict_with_variance`](@ref), [`train_model!`](@ref)
"""
function predict(model::GPModel, X_test::Matrix{Float64})
    D_test, N_test = size(X_test)
    D, N_train = size(model.X)

    K_train = build_full_covariance(
        model.kernel, model.X, model.noise_var, model.grad_noise_var, model.jitter
    )

    L = _robust_cholesky(K_train)
    alpha = L \ model.y

    dim_test_block = 1 + D
    K_star = zeros(N_test * dim_test_block, length(model.y))

    for i in 1:N_test
        xt = view(X_test, :, i)
        r_e = (i-1)*dim_test_block + 1
        r_g = (r_e + 1):(r_e + D)

        for j in 1:N_train
            xtrain = view(model.X, :, j)
            c_e = j
            c_g = (N_train + (j - 1) * D + 1):(N_train + j * D)

            k_ee, k_ef, k_fe, k_ff = kernel_blocks(model.kernel, xt, xtrain)

            K_star[r_e, c_e] = k_ee
            K_star[r_e, c_g] = k_ef
            K_star[r_g, c_e] = k_fe
            K_star[r_g, c_g] = k_ff
        end
    end

    return K_star * alpha
end

# ==============================================================================
# Prediction with Variance (Mean + Uncertainty)
# ==============================================================================
#
# The standard GP posterior provides not just a mean prediction but also
# uncertainty estimates. This is the key advantage of GP-guided optimization:
# the model knows where it is uncertain, allowing intelligent exploration.
#
# Posterior:
#   mu_*  = K_* K^{-1} y
#   var_* = diag(K_** - K_* K^{-1} K_*^T)

"""
    predict_with_variance(model::GPModel, X_test::Matrix{Float64})

Returns `(mean, variance)` where:
- `mean`: Vector of predictions [E1, G1_1, ..., G1_D, E2, ...] (same as `predict`)
- `variance`: Vector of same length, diagonal of the predictive covariance

Variance at training points is approximately the noise variance.
Variance far from training data approaches the prior (signal) variance.
"""
function predict_with_variance(model::GPModel, X_test::Matrix{Float64})
    D_test, N_test = size(X_test)
    D, N_train = size(model.X)

    K_train = build_full_covariance(
        model.kernel, model.X, model.noise_var, model.grad_noise_var, model.jitter
    )

    L = _robust_cholesky(K_train)

    alpha = L \ model.y

    dim_test_block = 1 + D
    n_out = N_test * dim_test_block
    K_star = zeros(n_out, length(model.y))

    # Build K_* (cross-covariance between test and training)
    for i in 1:N_test
        xt = view(X_test, :, i)
        r_e = (i-1)*dim_test_block + 1
        r_g = (r_e + 1):(r_e + D)

        for j in 1:N_train
            xtrain = view(model.X, :, j)
            c_e = j
            c_g = (N_train + (j - 1) * D + 1):(N_train + j * D)

            k_ee, k_ef, k_fe, k_ff = kernel_blocks(model.kernel, xt, xtrain)

            K_star[r_e, c_e] = k_ee
            K_star[r_e, c_g] = k_ef
            K_star[r_g, c_e] = k_fe
            K_star[r_g, c_g] = k_ff
        end
    end

    # Predictive mean: mu_* = K_* alpha
    mu = K_star * alpha

    # Predictive variance: diag(K_** - K_* K^{-1} K_*^T)
    # V = L^{-1} K_*^T, then var = diag(K_**) - sum(V .^ 2, dims=1)
    V = L.L \ K_star'

    variance = zeros(n_out)

    # Compute prior variance (diagonal of K_**)
    for i in 1:N_test
        xt = view(X_test, :, i)
        k_ee, _, _, k_ff = kernel_blocks(model.kernel, xt, xt)

        r_e = (i-1)*dim_test_block + 1
        r_g = (r_e + 1):(r_e + D)

        variance[r_e] = k_ee
        for d in 1:D
            variance[r_g[d]] = k_ff[d, d]
        end
    end

    # Subtract explained variance
    for idx in 1:n_out
        variance[idx] -= dot(V[:, idx], V[:, idx])
        variance[idx] = max(variance[idx], 0.0)  # Numerical floor
    end

    return mu, variance
end
