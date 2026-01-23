# ==============================================================================
# GP Model Container
# ==============================================================================

mutable struct GPModel{Tk<:Kernel}
    kernel::Tk
    X::Matrix{Float64}       # Inputs (D x N)
    y::Vector{Float64}       # Targets [Energies; Gradients]
    noise_var::Float64       # Energy noise variance (σ_n^2)
    grad_noise_var::Float64  # Gradient noise variance (σ_g^2)
    jitter::Float64          # Stability jitter
end

function GPModel(kernel, X, y; noise_var = 1e-6, grad_noise_var = 1e-6, jitter = 1e-6)
    return GPModel(kernel, X, y, noise_var, grad_noise_var, jitter)
end

# ==============================================================================
# Generic Derivative Block Logic (ForwardDiff + KernelFunctions)
# ==============================================================================

function kernel_val(k::Kernel, x1, x2)
    return k(x1, x2)
end


"""
    kernel_blocks(k::MolecularKernel, x1, x2)

Computes the full block covariance for Energy and Forces.
Uses automatic differentiation.

Returns:
- k_ee: Energy-Energy (scalar)
- k_ef: Energy-Force  (1 x D)
- k_fe: Force-Energy  (D x 1)
- k_ff: Force-Force   (D x D)
"""
function kernel_blocks(k::Kernel, x1::AbstractVector, x2::AbstractVector)
    # 1. Energy-Energy
    k_ee = kernel_val(k, x1, x2)

    # 2. Force(x2) - Energy(x1): d/dx' k(x, x')
    g_x2 = ForwardDiff.gradient(x -> kernel_val(k, x1, x), x2)

    # 3. Force(x1) - Energy(x2): d/dx k(x, x')
    g_x1 = ForwardDiff.gradient(x -> kernel_val(k, x, x2), x1)

    # 4. Force(x1) - Force(x2): d2/dx dx' k(x, x')
    H_cross =
        ForwardDiff.jacobian(x -> ForwardDiff.gradient(y -> kernel_val(k, x, y), x2), x1)

    # ==========================================================================
    # PHYSICS CONVENTION
    # ==========================================================================
    # Forces are Negative Gradients: F = -dV/dx
    # Cov(F, F) = Cov(-dE/dx, -dE/dx') = d2k/dxdx' (Negatives cancel)
    # Cov(E, F) = Cov(E, -dE/dx')      = -dk/dx'   (One negative remains)
    # ==========================================================================
    k_ef = -g_x2'  # Transpose to 1xD Row Vector
    k_fe = -g_x1   # Dx1 Col Vector
    k_ff = H_cross # DxD Matrix

    return k_ee, k_ef, k_fe, k_ff
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

function train_model!(model::GPModel; iterations = 1000)
    D = size(model.X, 1)

    # 1. Initialize Parameters using Inverse Lengthscales
    # We use 'positive' to ensure 1/ℓ > 0
    initial_theta = (
        signal_var = positive(1.0),
        inv_lengthscales = positive(ones(D)),
        noise = positive(model.noise_var),
        grad_noise = positive(model.grad_noise_var),
    )

    flat_theta, unflatten = flatten(initial_theta)

    # 2. Define Cost Function
    function cost(theta_flat)
        # Unwrap parameters and get values
        params_wrapped = unflatten(theta_flat)
        params = ParameterHandling.value(params_wrapped)

        # Construct our CUSTOM KERNEL
        k_custom = MolecularKernel(params.signal_var, params.inv_lengthscales)

        # Build Matrix
        K = build_full_covariance(
            k_custom,
            model.X,
            params.noise,
            params.grad_noise,
            model.jitter,
        )

        L = try
            cholesky(K)
        catch
            return 1e10
        end

        alpha = L \ model.y
        nll = 0.5 * dot(model.y, alpha) + logdet(L) + 0.5 * length(model.y) * log(2π)

        return (isnan(nll) || isinf(nll)) ? 1e10 : nll
    end

    if cost(flat_theta) >= 1e10
        @warn "Initial parameters unstable. Boosting jitter."
        model.jitter = 1e-4
    end

    res = optimize(
        cost,
        flat_theta,
        NelderMead(),
        Optim.Options(iterations = iterations, show_trace = true),
    )

    # 3. Update Model
    final_params_wrapped = unflatten(Optim.minimizer(res))
    final_params = ParameterHandling.value(final_params_wrapped)

    model.kernel = MolecularKernel(final_params.signal_var, final_params.inv_lengthscales)
    model.noise_var = final_params.noise
    model.grad_noise_var = final_params.grad_noise

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
