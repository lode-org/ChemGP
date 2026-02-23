# ==============================================================================
# Random Fourier Features (RFF) for MolInvDistSE
# ==============================================================================
#
# MolInvDistSE is an SE kernel in inverse-distance feature space:
#   k(x,y) = sigma^2 * exp(-theta^2 * ||phi(x) - phi(y)||^2)
#
# By Bochner's theorem, this has a Gaussian spectral density:
#   p(w) = N(0, 2*theta^2 * I)
#
# RFF approximates the kernel with D_rff random features:
#   k(x,y) ~ z(x)^T z(y)
# where z(x) = sigma * sqrt(2/D_rff) * cos(W * phi(x) + b).
#
# Training is Bayesian linear regression: O(N * D * D_rff + D_rff^3),
# which replaces the exact GP cost of O((N*(D+1))^3). For N=400 and
# D_rff=200 this is a ~1000x speedup.
#
# Gradient observations are incorporated via the Jacobian of z w.r.t. x,
# computed analytically using the chain rule through the inverse distance
# feature map.
#
# References:
#   Rahimi & Recht (2007), "Random Features for Large-Scale Kernel Machines"
#   Lazaro-Gredilla et al. (2010), "Sparse Spectrum GP Regression"

"""
    RFFModel

Random Fourier Features approximation to the MolInvDistSE GP.

Hyperparameters (sigma, theta) come from a base kernel trained on
a subset. Training and prediction use all N data points via the
D_rff-dimensional feature approximation.

# Fields
- `W`: (D_rff, d_feat) frequency matrix sampled from N(0, 2*theta^2*I)
- `b`: (D_rff,) random phase offsets in [0, 2pi)
- `c`: sigma * sqrt(2/D_rff) scaling constant
- `frozen`: flat coordinates of frozen atoms
- `alpha`: (D_rff,) Bayesian linear regression weights
- `A_chol`: Cholesky of the regularized Gram matrix (for variance)
"""
struct RFFModel
    W::Matrix{Float64}
    b::Vector{Float64}
    c::Float64
    frozen::Vector{Float64}
    alpha::Vector{Float64}
    A_chol::LinearAlgebra.Cholesky{Float64, Matrix{Float64}}
end

"""
    _rff_features(W, b, c, frozen, x) -> (z, J_z)

Compute RFF feature vector z(x) and its Jacobian J_z = dz/dx.

Returns:
- `z`: (D_rff,) feature vector
- `J_z`: (D_rff, D) Jacobian matrix
"""
function _rff_features(
    W::Matrix{Float64},
    b::Vector{Float64},
    c::Float64,
    frozen::Vector{Float64},
    x::AbstractVector{Float64},
)
    phi = compute_inverse_distances(x, frozen)
    u = W * phi .+ b
    z = c .* cos.(u)

    # Jacobian of inverse distances w.r.t. Cartesian coordinates
    J_phi = ForwardDiff.jacobian(
        x_ -> compute_inverse_distances(x_, frozen), collect(x))
    # J_phi: (d_feat, D), W: (D_rff, d_feat)
    # J_z = -c * sin(u) .* (W * J_phi), element-wise broadcast on columns
    sin_u = sin.(u)
    J_z = (-c) .* sin_u .* (W * J_phi)  # (D_rff, D)

    return z, J_z
end

"""
    build_rff(kernel, X_train, y_train, D_rff; noise_var, grad_noise_var) -> RFFModel

Build an RFF model from a trained MolInvDistSE kernel and all training data.

The frequency matrix W is sampled from the spectral density of the
trained kernel. Training solves a Bayesian linear regression problem
in the D_rff-dimensional feature space, cost O(N*D*D_rff + D_rff^3).

`y_train` must be in blocked layout: [E1..EN, G1_1..GN_D], matching
the output of `build_full_covariance`.
"""
function build_rff(
    kernel::MolInvDistSE,
    X_train::Matrix{Float64},
    y_train::Vector{Float64},
    D_rff::Int;
    noise_var::Float64 = 1e-6,
    grad_noise_var::Float64 = 1e-4,
)
    D, N = size(X_train)
    frozen = Vector{Float64}(kernel.frozen_coords)
    n_atoms = div(D, 3)
    n_frozen = div(length(frozen), 3)
    d_feat = div(n_atoms * (n_atoms - 1), 2) + n_atoms * n_frozen

    # Sample frequencies from spectral density N(0, 2*theta^2 * I)
    theta = kernel.inv_lengthscales
    W = zeros(D_rff, d_feat)
    if isempty(kernel.feature_params_map) && length(theta) == 1
        W .= randn(D_rff, d_feat) .* (sqrt(2.0) * theta[1])
    else
        for f in 1:d_feat
            idx = isempty(kernel.feature_params_map) ? 1 : kernel.feature_params_map[f]
            W[:, f] .= randn(D_rff) .* (sqrt(2.0) * theta[idx])
        end
    end

    b = rand(D_rff) .* (2.0 * pi)
    sigma = sqrt(kernel.signal_variance)
    c = sigma * sqrt(2.0 / D_rff)

    # Build design matrix Z in blocked layout: [energy rows; gradient rows]
    n_obs = N * (1 + D)
    Z = zeros(n_obs, D_rff)

    for i in 1:N
        xi = view(X_train, :, i)
        z, J_z = _rff_features(W, b, c, frozen, xi)

        # Energy row i
        Z[i, :] = z

        # Gradient rows: N + (i-1)*D + d for each Cartesian component d
        for d in 1:D
            Z[N + (i - 1) * D + d, :] = J_z[:, d]
        end
    end

    # Noise precision vector
    prec = zeros(n_obs)
    prec[1:N] .= 1.0 / noise_var
    prec[N+1:end] .= 1.0 / grad_noise_var

    # Bayesian linear regression: A = Z^T diag(prec) Z + I
    ZtP = Z' .* prec'   # (D_rff, n_obs)
    A = Symmetric(ZtP * Z + I(D_rff))
    A_chol = cholesky(A)

    alpha = A_chol \ (ZtP * y_train)

    return RFFModel(W, b, c, frozen, alpha, A_chol)
end

"""
    predict(rff::RFFModel, X_test) -> Vector

RFF prediction with interleaved output layout:
[E1, G1_1, ..., G1_D, E2, G2_1, ..., G2_D, ...].

Cost: O(D_rff * D) per test point.
"""
function predict(rff::RFFModel, X_test::Matrix{Float64})
    D, N_test = size(X_test)
    dim_block = D + 1
    result = zeros(N_test * dim_block)

    for i in 1:N_test
        xi = view(X_test, :, i)
        z, J_z = _rff_features(rff.W, rff.b, rff.c, rff.frozen, xi)

        offset = (i - 1) * dim_block
        result[offset + 1] = dot(z, rff.alpha)
        result[offset+2 : offset+dim_block] = J_z' * rff.alpha
    end

    return result
end

"""
    predict_with_variance(rff::RFFModel, X_test) -> (mean, variance)

RFF prediction with Bayesian linear regression variance.

Energy variance: var(E) = z^T A^{-1} z
Gradient variance: var(G_d) = J_z[:, d]^T A^{-1} J_z[:, d]

Output layout matches `predict_with_variance(::GPModel)`.
"""
function predict_with_variance(rff::RFFModel, X_test::Matrix{Float64})
    D, N_test = size(X_test)
    dim_block = D + 1
    mu = zeros(N_test * dim_block)
    var = zeros(N_test * dim_block)

    for i in 1:N_test
        xi = view(X_test, :, i)
        z, J_z = _rff_features(rff.W, rff.b, rff.c, rff.frozen, xi)

        offset = (i - 1) * dim_block

        # Energy mean + variance
        mu[offset + 1] = dot(z, rff.alpha)
        v = rff.A_chol.L \ z
        var[offset + 1] = dot(v, v)

        # Gradient mean + variance
        G = J_z' * rff.alpha
        for d in 1:D
            mu[offset + 1 + d] = G[d]
            v_d = rff.A_chol.L \ view(J_z, :, d)
            var[offset + 1 + d] = dot(v_d, v_d)
        end
    end

    return mu, var
end
