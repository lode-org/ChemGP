# ==============================================================================
# GP Model Container
# ==============================================================================

mutable struct GPModel{Tk<:Kernel}
    kernel::Tk
    X::AbstractMatrix{Float64}       # Inputs (D x N)
    y::Vector{Float64}       # Targets [Energies; Gradients]
    noise_var::Float64       # Energy noise variance (σ_n^2)
    grad_noise_var::Float64  # Gradient noise variance (σ_g^2)
    jitter::Float64          # Stability jitter
end

function GPModel(kernel, X, y; noise_var = 1e-6, grad_noise_var = 1e-6, jitter = 1e-6)
    return GPModel(kernel, Matrix(X), y, noise_var, grad_noise_var, jitter)
end

# ==============================================================================
# Covariance Matrix Assembly
# ==============================================================================

function build_full_covariance(
    kernel::Kernel,
    X::Matrix{Float64},
    noise_e::Real,
    noise_g::Real,
    jitter::Real,
)
    D, N = size(X)
    TotalDim = N * (1 + D)
    K_mat = zeros(TotalDim, TotalDim)

    for i = 1:N
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
        for j = (i+1):N
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

    return Symmetric(K_mat)
end

# ==============================================================================
# Training (ParameterHandling + Optim)
# ==============================================================================

function train_model!(model::GPModel{Tk}; iterations = 1000) where {Tk}
    # 1. Define Initial Parameters (Structured)
    # Warm start from current model values as the starting point.
    # ParameterHandling.positive ensures they stay > 0 during optimization.
    raw_initial_params = (
        signal_var = positive(model.kernel.signal_variance),
        inv_lengthscales = positive(model.kernel.inv_lengthscales),
        noise = positive(model.noise_var),
        grad_noise = positive(model.grad_noise_var),
    )

    # 2. Flatten parameters into a vector for Optim.jl
    # 'unflatten' is a function that converts the vector back to the NamedTuple
    flat_initial_params, unflatten = ParameterHandling.value_flatten(raw_initial_params)

    # Preserve structural info not being optimized
    frozen = model.kernel.frozen_coords
    feat_map = model.kernel.feature_params_map

    # 3. Define Objective Function
    function objective(params::NamedTuple)
        # Reconstruct kernel from structured parameters
        k = Tk(params.signal_var, params.inv_lengthscales, frozen, feat_map)

        # Build Covariance
        K = build_full_covariance(k, model.X, params.noise, params.grad_noise, model.jitter)

        # Compute NLL
        L = try
            cholesky(K)
        catch
            return Inf # Signal failure to Optim
        end

        return 0.5 * dot(model.y, L \ model.y) + logdet(L) + 0.5 * length(model.y) * log(2π)
    end

    # 4. Run Optimization
    # We use NelderMead because build_full_covariance uses mutation (not Zygote-compatible).
    # 'objective ∘ unflatten' composes the functions so Optim sees a vector input.
    res = Optim.optimize(
        objective ∘ unflatten,
        flat_initial_params,
        NelderMead(),
        Optim.Options(iterations = iterations, show_trace = true),
    )

    # 5. Update Model with Best Parameters
    best_params = unflatten(Optim.minimizer(res))

    model.kernel =
        Tk(best_params.signal_var, best_params.inv_lengthscales, frozen, feat_map)
    model.noise_var = best_params.noise
    model.grad_noise_var = best_params.grad_noise

    @printf("Optimization Complete. Final NLL: %.4f\n", Optim.minimum(res))
    return model
end

# ==============================================================================
# Prediction
# ==============================================================================

function predict(model::GPModel, X_test::Matrix{Float64})
    D_test, N_test = size(X_test)
    D, N_train = size(model.X)

    K_train = build_full_covariance(
        model.kernel,
        model.X,
        model.noise_var,
        model.grad_noise_var,
        model.jitter,
    )

    L = nothing
    jitter_add = 0.0
    for attempt = 1:5
        try
            L = cholesky(K_train + jitter_add * I)
            break
        catch
            jitter_add = (attempt == 1) ? 1e-6 : jitter_add * 10
        end
    end
    if L === nothing
        ;
        error("Singular matrix");
    end

    alpha = L \ model.y

    dim_test_block = 1 + D
    K_star = zeros(N_test * dim_test_block, length(model.y))

    for i = 1:N_test
        xt = view(X_test, :, i)
        r_e = (i-1)*dim_test_block + 1
        r_g = (r_e+1):(r_e+D)

        for j = 1:N_train
            xtrain = view(model.X, :, j)
            c_e = j
            c_g = (N_train+(j-1)*D+1):(N_train+j*D)

            k_ee, k_ef, k_fe, k_ff = kernel_blocks(model.kernel, xt, xtrain)

            K_star[r_e, c_e] = k_ee
            K_star[r_e, c_g] = k_ef
            K_star[r_g, c_e] = k_fe
            K_star[r_g, c_g] = k_ff
        end
    end

    return K_star * alpha
end
