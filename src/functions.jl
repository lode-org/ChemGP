# ==============================================================================
# Struct Definitions
# ==============================================================================

struct SquaredExpKernel
    log_signal_var::Float64           # log(sigma_f^2)
    log_lengthscales::Vector{Float64} # log(ell) - ARD vector
end

# Constructor for easier initialization
SquaredExpKernel(var::Real, ells::Vector{<:Real}) = 
    SquaredExpKernel(log(var), log.(ells))

mutable struct GPModel
    kernel::SquaredExpKernel
    X::Matrix{Float64}       # Inputs (D x N)
    y::Vector{Float64}       # Targets [Energies; Gradients]
    log_noise_var::Float64   # Energy noise variance
    log_grad_var::Float64    # Gradient noise variance
    jitter::Float64          # Stability jitter (nugget)
end

# Default constructor with 1e-6 jitter
GPModel(k, X, y, ln, lg) = GPModel(k, X, y, ln, lg, 1e-6)

# ==============================================================================
# Kernel Logic (Energy, Force, and Hessian blocks)
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
    # dK/dx = - (x-x')/l^2 * K
    k_ef = k_ee .* grad_factor
    k_fe = -k_ef
    
    # Force-Force (Hessian: d2K / dx dx')
    # Standard RBF result: K * ( 1/l^2 - (x-x')^2 / l^4 )
    H_outer = grad_factor * grad_factor' # (x-x')^2 / l^4
    H_kron = Diagonal(inv_ell2)          # 1/l^2
    
    k_ff = k_ee .* (H_kron .- H_outer)
    
    return k_ee, k_ef, k_fe, k_ff
end

# ==============================================================================
# Covariance Matrix Assembly
# ==============================================================================

function build_full_covariance(model::GPModel; apply_noise=true)
    D, N = size(model.X)
    TotalDim = N * (1 + D)
    K_mat = zeros(TotalDim, TotalDim)
    
    σ_n2 = exp(model.log_noise_var)
    σ_g2 = exp(model.log_grad_var)
    
    X = model.X
    kern = model.kernel
    
    for i in 1:N
        xi = view(X, :, i)
        
        # Diagonal Blocks
        k_ee, k_ef, k_fe, k_ff = kernel_blocks(kern, xi, xi)
        
        # Energy diagonal index
        K_mat[i, i] = k_ee + (apply_noise ? σ_n2 : 0.0) + model.jitter
        
        # Gradient diagonal indices
        s_g = N + (i-1)*D + 1
        e_g = N + i*D
        
        K_mat[i, s_g:e_g] = k_ef
        K_mat[s_g:e_g, i] = k_fe
        
        # Force-Force diagonal block
        K_mat[s_g:e_g, s_g:e_g] = k_ff + ((apply_noise ? σ_g2 : 0.0) + model.jitter) * I
        
        # Off-diagonal Interactions
        for j in (i+1):N
            xj = view(X, :, j)
            k_ee, k_ef, k_fe, k_ff = kernel_blocks(kern, xi, xj)
            
            j_s = N + (j-1)*D + 1
            j_e = N + j*D
            
            # Fill symmetric blocks
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
# Training (Optimization)
# ==============================================================================

function negative_log_likelihood(params::Vector{Float64}, model_struct::GPModel)
    D, N = size(model_struct.X)
    
    # Unpack params
    log_sf2 = params[1]
    log_ell = params[2:D+1]
    log_sn2 = params[D+2]
    log_sg2 = params[D+3]
    
    temp_kern = SquaredExpKernel(log_sf2, log_ell)
    # Pass jitter to temp model
    temp_model = GPModel(temp_kern, model_struct.X, model_struct.y, log_sn2, log_sg2, model_struct.jitter)
    
    K = build_full_covariance(temp_model)
    
    # Robust Cholesky
    L = try
        cholesky(K)
    catch
        # Return finite but large penalty instead of Inf to allow optimizer to retreat
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
    
    # Check initial stability
    initial_nll = func(p0)
    if initial_nll >= 1e10
        @warn "Initial parameters unstable (NLL large). Boosting jitter to 1e-4."
        model.jitter = 1e-4
    end

    res = optimize(func, p0, NelderMead(), Optim.Options(iterations=iterations, show_trace=true))
    
    p_opt = Optim.minimizer(res)
    model.kernel = SquaredExpKernel(p_opt[1], p_opt[2:D+1])
    model.log_noise_var = p_opt[D+2]
    model.log_grad_var  = p_opt[D+3]
    
    @printf("Optimization Complete. Final NLL: %.4f\n", Optim.minimum(res))
    return model
end

# ==============================================================================
# Prediction
# ==============================================================================

function predict(model::GPModel, X_test::Matrix{Float64})
    D_test, N_test = size(X_test)
    D, N_train = size(model.X)
    
    K_train = build_full_covariance(model; apply_noise=true)
    
    # Robust Cholesky with retry
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
    
    if L === nothing
        error("Covariance matrix singular even with extra jitter.")
    end
    
    alpha = L \ model.y
    
    # Build K_star
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
