# ==============================================================================
# Abstract Types & Structs
# ==============================================================================

abstract type AbstractGPKernel end

"""
    SquaredExpKernel
    Uses ANALYTICAL derivatives. Fastest, best for production.
"""
struct SquaredExpKernel <: AbstractGPKernel
    log_signal_var::Float64
    log_lengthscales::Vector{Float64}
end
SquaredExpKernel(var::Real, ells::Vector{<:Real}) = 
    SquaredExpKernel(log(var), log.(ells))

"""
    SquaredExpADKernel
    Uses FORWARDDIFF (Automatic Differentiation). 
    Slower, but guarantees correctness and easy to modify.
"""
struct SquaredExpADKernel <: AbstractGPKernel
    log_signal_var::Float64
    log_lengthscales::Vector{Float64}
end
SquaredExpADKernel(var::Real, ells::Vector{<:Real}) = 
    SquaredExpADKernel(log(var), log.(ells))

mutable struct GPModel
    kernel::AbstractGPKernel
    X::Matrix{Float64}
    y::Vector{Float64}
    log_noise_var::Float64
    log_grad_var::Float64
    jitter::Float64
end

GPModel(k, X, y, ln, lg) = GPModel(k, X, y, ln, lg, 1e-6)

# ==============================================================================
# 1. Analytical Implementation (Fast)
# ==============================================================================

function kernel_blocks(k::SquaredExpKernel, x1::AbstractVector, x2::AbstractVector)
    D = length(x1)
    σ_f2 = exp(k.log_signal_var)
    ℓ = exp.(k.log_lengthscales)
    
    diff = x1 .- x2
    scaled_diff = diff ./ ℓ
    r2 = dot(scaled_diff, scaled_diff)
    
    # Base Kernel (Energy-Energy)
    k_ee = σ_f2 * exp(-0.5 * r2)
    
    # Gradient factors
    inv_ell2 = 1.0 ./ (ℓ.^2)
    grad_factor = -diff .* inv_ell2
    
    # Energy-Force / Force-Energy
    k_ef = k_ee .* grad_factor
    k_fe = -k_ef
    
    # Force-Force (Hessian)
    # Analytic Form: K * ( 1/l^2 * I - (x-x')^2 / l^4 )
    H_outer = grad_factor * grad_factor' 
    H_kron = Diagonal(inv_ell2)
    
    k_ff = k_ee .* (H_kron .- H_outer)
    
    return k_ee, k_ef, k_fe, k_ff
end

# ==============================================================================
# 2. ForwardDiff Implementation (Flexible)
# ==============================================================================

# Helper: Basic scalar kernel definition (needed for AD)
function kernel_val(k::SquaredExpADKernel, x1::AbstractVector, x2::AbstractVector)
    σ_f2 = exp(k.log_signal_var)
    ℓ = exp.(k.log_lengthscales)
    
    # Must use broadcasting with generic types for Dual numbers
    diff = (x1 .- x2) ./ ℓ
    r2 = dot(diff, diff)
    
    return σ_f2 * exp(-0.5 * r2)
end

function kernel_blocks(k::SquaredExpADKernel, x1::AbstractVector, x2::AbstractVector)
    # 1. Energy-Energy
    k_ee = kernel_val(k, x1, x2)
    
    # 2. Gradient w.r.t x2 (Force on 2)
    g_x2 = ForwardDiff.gradient(x -> kernel_val(k, x1, x), x2)
    
    # 3. Gradient w.r.t x1 (Force on 1)
    g_x1 = ForwardDiff.gradient(x -> kernel_val(k, x, x2), x1)
    
    # 4. Hessian (Force-Force)
    # d/dx1 ( d/dx2 k )
    H_cross = ForwardDiff.jacobian(x -> ForwardDiff.gradient(y -> kernel_val(k, x, y), x2), x1)
    
    # Apply Physics Conventions (Forces are negative gradients)
    # Cov(E, F)  = -dk/dx'
    # Cov(F, E)  = -dk/dx
    # Cov(F, F)  = d2k/dxdx'
    
    k_ef = -g_x2'  # Row vector
    k_fe = -g_x1   # Col vector
    k_ff = H_cross # Matrix
    
    return k_ee, k_ef, k_fe, k_ff
end

# ==============================================================================
# Generic GP Logic (Works for ANY kernel type)
# ==============================================================================

function build_full_covariance(model::GPModel; apply_noise=true)
    D, N = size(model.X)
    TotalDim = N * (1 + D)
    K_mat = zeros(TotalDim, TotalDim)
    
    σ_n2 = exp(model.log_noise_var)
    σ_g2 = exp(model.log_grad_var)
    
    # Dispatch occurs here based on model.kernel type
    kern = model.kernel
    X = model.X
    
    for i in 1:N
        xi = view(X, :, i)
        
        # Diagonal Blocks
        k_ee, k_ef, k_fe, k_ff = kernel_blocks(kern, xi, xi)
        
        K_mat[i, i] = k_ee + (apply_noise ? σ_n2 : 0.0) + model.jitter
        
        s_g = N + (i-1)*D + 1
        e_g = N + i*D
        
        K_mat[i, s_g:e_g] = k_ef
        K_mat[s_g:e_g, i] = k_fe
        K_mat[s_g:e_g, s_g:e_g] = k_ff + ((apply_noise ? σ_g2 : 0.0) + model.jitter) * I
        
        # Interactions
        for j in (i+1):N
            xj = view(X, :, j)
            k_ee, k_ef, k_fe, k_ff = kernel_blocks(kern, xi, xj)
            
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

function negative_log_likelihood(params::Vector{Float64}, model_struct::GPModel)
    D, N = size(model_struct.X)
    
    log_sf2 = params[1]
    log_ell = params[2:D+1]
    log_sn2 = params[D+2]
    log_sg2 = params[D+3]
    
    # Reconstruct the correct kernel type using typeof
    KType = typeof(model_struct.kernel)
    temp_kern = KType(log_sf2, log_ell)
    
    temp_model = GPModel(temp_kern, model_struct.X, model_struct.y, log_sn2, log_sg2, model_struct.jitter)
    
    K = build_full_covariance(temp_model)
    
    L = try
        cholesky(K)
    catch
        return 1e10
    end
    
    y = model_struct.y
    alpha = L \ y
    nll = 0.5 * dot(y, alpha) + logdet(L) + 0.5 * length(y) * log(2π)
    
    if isnan(nll) || isinf(nll)
        return 1e10
    end
    return nll
end

function train_model!(model::GPModel; iterations=1000)
    D = size(model.X, 1)
    
    p0 = [
        model.kernel.log_signal_var;
        model.kernel.log_lengthscales;
        model.log_noise_var;
        model.log_grad_var
    ]
    
    func = p -> negative_log_likelihood(p, model)
    
    if func(p0) >= 1e10
        @warn "Initial parameters unstable. Boosting jitter to 1e-4."
        model.jitter = 1e-4
    end

    res = optimize(func, p0, NelderMead(), Optim.Options(iterations=iterations, show_trace=true))
    
    p_opt = Optim.minimizer(res)
    
    # Update Kernel preserving type
    KType = typeof(model.kernel)
    model.kernel = KType(p_opt[1], p_opt[2:D+1])
    
    model.log_noise_var = p_opt[D+2]
    model.log_grad_var  = p_opt[D+3]
    
    @printf("Optimization Complete. Final NLL: %.4f\n", Optim.minimum(res))
    return model
end

function predict(model::GPModel, X_test::Matrix{Float64})
    D_test, N_test = size(X_test)
    D, N_train = size(model.X)
    
    K_train = build_full_covariance(model; apply_noise=true)
    
    L = nothing
    jitter_add = 0.0
    for attempt in 1:5
        try
            L = cholesky(K_train + jitter_add * I)
            break
        catch
            jitter_add = (attempt == 1) ? 1e-6 : jitter_add * 10
        end
    end
    
    if L === nothing; error("Singular matrix"); end
    
    alpha = L \ model.y
    
    dim_test_block = 1 + D
    K_star = zeros(N_test * dim_test_block, length(model.y))
    
    for i in 1:N_test
        xt = view(X_test, :, i)
        r_e = (i-1)*dim_test_block + 1
        r_g = r_e+1 : r_e+D
        
        for j in 1:N_train
            xtrain = view(model.X, :, j)
            c_e = j
            c_g = N_train + (j-1)*D + 1 : N_train + j*D
            
            k_ee, k_ef, k_fe, k_ff = kernel_blocks(model.kernel, xt, xtrain)
            
            K_star[r_e, c_e] = k_ee
            K_star[r_e, c_g] = k_ef
            K_star[r_g, c_e] = k_fe
            K_star[r_g, c_g] = k_ff
        end
    end
    
    return K_star * alpha
end
