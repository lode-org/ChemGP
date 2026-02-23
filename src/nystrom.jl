# ==============================================================================
# Nystrom GP Approximation
# ==============================================================================
#
# When the training set grows large, exact GP inference (O(N^3)) becomes
# expensive. The Nystrom approximation selects M << N inducing points and
# approximates the full covariance:
#
#   K_NN ~ K_NM K_MM^{-1} K_MN
#
# Training (hyperparameter optimization) uses the M-point subset.
# Prediction uses all N points via the Woodbury identity, so no data
# is discarded.
#
# This is the "Deterministic Training Conditional" (DTC) approximation
# (Seeger et al. 2003, Quinonero-Candela & Rasmussen 2005).

"""
    _blocked_cross_covariance(kernel, X1, X2)

Build rectangular cross-covariance between two sets of points using
**blocked** row and column layout:

    rows:  [E1, ..., EN1, G1_1, ..., G1_D, G2_1, ..., GN1_D]
    cols:  [E1, ..., EN2, G1_1, ..., G1_D, G2_1, ..., GN2_D]

This layout matches `build_full_covariance` and is used internally by
`build_nystrom` for the K_NM matrix.

Returns a matrix of size `N1*(D+1) x N2*(D+1)`.
"""
function _blocked_cross_covariance(
    kernel,
    X1::AbstractMatrix{Float64},
    X2::AbstractMatrix{Float64},
)
    D, N1 = size(X1)
    _, N2 = size(X2)
    K = zeros(N1 * (D + 1), N2 * (D + 1))

    for i in 1:N1
        xi = view(X1, :, i)
        r_e = i
        r_g = (N1 + (i-1)*D + 1):(N1 + i*D)

        for j in 1:N2
            xj = view(X2, :, j)
            c_e = j
            c_g = (N2 + (j-1)*D + 1):(N2 + j*D)

            k_ee, k_ef, k_fe, k_ff = kernel_blocks(kernel, xi, xj)

            K[r_e, c_e] = k_ee
            K[r_e, c_g] = k_ef
            K[r_g, c_e] = k_fe
            K[r_g, c_g] = k_ff
        end
    end
    return K
end

"""
    NystromGP

Nystrom (DTC) approximation to a GP model.

Hyperparameters are from `base` (trained on M inducing points).
Predictions use all N training points via the Woodbury identity,
so no data is discarded.

# Fields
- `base`: GPModel trained on inducing points (kernel, noise parameters)
- `X_all`: All N training point positions (D x N)
- `y_all`: All N targets [E1..EN, G1..GN*D] (blocked layout)
- `alpha`: Precomputed Nystrom weights (length N*(D+1))
- `L_MM`: Cholesky of K_MM (inducing point covariance)
"""
struct NystromGP{Tk}
    base::GPModel{Tk}
    X_all::Matrix{Float64}
    y_all::Vector{Float64}
    alpha::Vector{Float64}
    L_MM::LinearAlgebra.Cholesky{Float64, Matrix{Float64}}
end

"""
    build_nystrom(base_model, X_all, y_all) -> NystromGP

Build a Nystrom GP from a base model (trained on inducing subset) and
the full training data.

The Woodbury identity gives the Nystrom weights:

    alpha = Lambda^{-1} (y - K_NM * inner^{-1} * K_MN * Lambda^{-1} * y)

where `inner = K_MM + K_MN * Lambda^{-1} * K_NM` and `Lambda` is the
diagonal noise matrix.

`y_all` must be in **blocked** layout: `[E1, ..., EN, G1_1, ..., GN_D]`,
matching `build_full_covariance`. This is the concatenation of
`(energies .- E_ref)` and `gradients`.
"""
function build_nystrom(
    base_model::GPModel,
    X_all::Matrix{Float64},
    y_all::Vector{Float64},
)
    D, N = size(X_all)
    M = size(base_model.X, 2)
    dim_block = D + 1
    n_block = N * dim_block

    # K_MM (inducing covariance, with noise)
    K_MM = build_full_covariance(
        base_model.kernel, base_model.X,
        base_model.noise_var, base_model.grad_noise_var, base_model.jitter)
    L_MM = _robust_cholesky(K_MM)

    # K_NM (cross-covariance: all x inducing, blocked layout)
    K_NM = _blocked_cross_covariance(base_model.kernel, X_all, base_model.X)

    # Diagonal noise Lambda
    lambda = zeros(n_block)
    nv = base_model.noise_var + base_model.jitter
    gv = base_model.grad_noise_var + base_model.jitter
    for i in 1:N
        lambda[i] = nv
    end
    for i in 1:(N * D)
        lambda[N + i] = gv
    end
    lambda_inv = 1.0 ./ lambda

    # Woodbury: (Q + Lambda)^{-1} y
    # where Q = K_NM K_MM^{-1} K_MN
    # alpha = Lambda^{-1} y - Lambda^{-1} K_NM inner^{-1} K_MN Lambda^{-1} y
    # inner = K_MM + K_MN Lambda^{-1} K_NM

    li_y = lambda_inv .* y_all                             # N_block
    li_K_NM = lambda_inv .* K_NM                           # N_block x M_block

    inner = Symmetric(K_MM + K_NM' * li_K_NM)               # M_block x M_block
    L_inner = _robust_cholesky(inner)

    v = L_inner.L \ (K_NM' * li_y)                         # M_block
    w = L_inner.L' \ v                                      # M_block

    alpha = li_y - li_K_NM * w                              # N_block

    return NystromGP(base_model, X_all, y_all, alpha, L_MM)
end

"""
    predict(nys::NystromGP, X_test) -> Vector

Nystrom DTC prediction: mu_* = K_*N alpha.

Since alpha is precomputed via Woodbury, prediction costs O(N) per test
point (no Cholesky solve). The output layout matches `predict(::GPModel)`:
`[E1, G1_1, ..., G1_D, E2, G2_1, ..., G2_D, ...]`.
"""
function predict(nys::NystromGP, X_test::Matrix{Float64})
    D, N_train = size(nys.X_all)
    _, N_test = size(X_test)
    dim_block = D + 1

    # Build K_*N with interleaved test rows, blocked training columns
    # (same layout as GPModel.predict's K_star)
    K_star = zeros(N_test * dim_block, N_train * dim_block)

    for i in 1:N_test
        xt = view(X_test, :, i)
        r_e = (i - 1) * dim_block + 1
        r_g = (r_e + 1):(r_e + D)

        for j in 1:N_train
            xj = view(nys.X_all, :, j)
            c_e = j
            c_g = (N_train + (j - 1) * D + 1):(N_train + j * D)

            k_ee, k_ef, k_fe, k_ff = kernel_blocks(nys.base.kernel, xt, xj)

            K_star[r_e, c_e] = k_ee
            K_star[r_e, c_g] = k_ef
            K_star[r_g, c_e] = k_fe
            K_star[r_g, c_g] = k_ff
        end
    end

    return K_star * nys.alpha
end

"""
    predict_with_variance(nys::NystromGP, X_test) -> (mean, variance)

Nystrom prediction with DTC variance estimate.

Mean: mu_* = K_*N alpha (uses all N training points).

Variance uses the DTC formula:
    var_* = K_** - K_*M K_MM^{-1} K_M*
which is the prior variance minus the variance explained by the M
inducing points. This is cheaper than exact GP variance (O(M^2) vs
O(N^2) per test point) but slightly overestimates uncertainty since
it only accounts for inducing-point information.

Output layout matches `predict_with_variance(::GPModel)`.
"""
function predict_with_variance(nys::NystromGP, X_test::Matrix{Float64})
    D = size(X_test, 1)
    N_test = size(X_test, 2)
    M = size(nys.base.X, 2)
    dim_block = D + 1

    mu = predict(nys, X_test)

    # Build K_*M with interleaved test rows, blocked inducing columns
    # (inducing column layout matches K_MM / L_MM)
    K_test_M = zeros(N_test * dim_block, M * dim_block)

    for i in 1:N_test
        xt = view(X_test, :, i)
        r_e = (i - 1) * dim_block + 1
        r_g = (r_e + 1):(r_e + D)

        for j in 1:M
            xj = view(nys.base.X, :, j)
            c_e = j
            c_g = (M + (j - 1) * D + 1):(M + j * D)

            k_ee, k_ef, k_fe, k_ff = kernel_blocks(nys.base.kernel, xt, xj)

            K_test_M[r_e, c_e] = k_ee
            K_test_M[r_e, c_g] = k_ef
            K_test_M[r_g, c_e] = k_fe
            K_test_M[r_g, c_g] = k_ff
        end
    end

    # V = L_MM^{-1} K_M* -- columns indexed by interleaved test output
    V = nys.L_MM.L \ K_test_M'

    n_out = N_test * dim_block
    variance = zeros(n_out)

    # Prior variance (diagonal of K_**)
    for i in 1:N_test
        xt = view(X_test, :, i)
        k_ee, _, _, k_ff = kernel_blocks(nys.base.kernel, xt, xt)

        r_e = (i - 1) * dim_block + 1
        r_g = (r_e + 1):(r_e + D)

        variance[r_e] = k_ee
        for d in 1:D
            variance[r_g[d]] = k_ff[d, d]
        end
    end

    # Subtract explained variance
    for idx in 1:n_out
        variance[idx] -= dot(V[:, idx], V[:, idx])
        variance[idx] = max(variance[idx], 0.0)
    end

    return mu, variance
end
