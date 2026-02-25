# ==============================================================================
# Derivative Block Logic
# ==============================================================================

"""
    kernel_blocks(k::Kernel, x1, x2)

Computes the full block covariance for Energy and Gradients
with automatic differentiation.

Returns:
- k_ee: Energy-Energy (scalar)
- k_ef: Energy-Force  (1 x D)
- k_fe: Force-Energy  (D x 1)
- k_ff: Force-Force   (D x D)
"""
function kernel_blocks(k::Kernel, x1::AbstractVector, x2::AbstractVector)
    k_ee = k(x1, x2)
    g_x2 = ForwardDiff.gradient(x -> k(x1, x), x2)
    g_x1 = ForwardDiff.gradient(x -> k(x, x2), x1)
    H_cross = ForwardDiff.jacobian(x -> ForwardDiff.gradient(y -> k(x, y), x2), x1)
    return k_ee, g_x2', g_x1, H_cross
end

# ==============================================================================
# Analytical kernel blocks for MolInvDistSE
# ==============================================================================
#
# CRITICAL: Do NOT use nested ForwardDiff for molecular kernels.
#
# The generic kernel_blocks above uses Dual{Dual{Float64}} to compute
# d^2k/(dx1 dx2) through compute_inverse_distances (1/sqrt(d2)) and the SE
# exponential.  This produces ~1e-8 numerical noise in gradient-gradient
# cross-blocks -- the same magnitude as observation noise (sigma2=1e-8),
# making the assembled covariance matrix non-positive-definite with 10+
# training points.  The MATLAB (gpcf_sexpat.m ginput/ginput2/ginput4) and
# C++ (gpr_optim SexpatCF.cpp + Gradient.cpp) implementations use fully
# analytical derivatives and never hit this issue.
#
# Fix: fully analytical computation matching C++ gpr_optim:
#
#   1. Inverse distance features f = 1/r  (analytical)
#   2. Jacobian J[pair,dim] = d(1/r_ij)/dx_dim  (analytical, no AD)
#   3. SE kernel derivatives in feature space  (analytical)
#   4. Chain rule:  K_FF = J1^T * H_feat * J2
#
# Algebraically equivalent to the C++ formulation:
#   K_FF(a,b) = k * (-0.5*D12_ab + 0.25*D1_a*D2_b)
# where D1,D2,D12 are derivatives of dist^2 in inverse-distance space.
# Verified: expanding D1 = 4*sum(theta^2*r*J1), D2 = -4*sum(theta^2*r*J2),
# D12 = -4*sum(theta^2*J1*J2) yields the same J1^T*H_feat*J2 formula.

"""
    _invdist_jacobian(x_flat, frozen_flat) -> (features, J)

Compute inverse distance features AND their analytical Jacobian w.r.t.
`x_flat` in a single pass.  No automatic differentiation.

For pair (j,i) with j < i, feature = 1/r_{ji}:
    d(1/r)/d(x_i_a) = -(x_i_a - x_j_a) / r^3
    d(1/r)/d(x_j_a) =  (x_i_a - x_j_a) / r^3

Returns `features` (length nf) and `J` (nf x D) where D = length(x_flat).

Matches C++ gpr_optim Gradient.cpp calculateGradientBetweenMovingAtoms.
"""
function _invdist_jacobian(
    x_flat::AbstractVector{Float64}, frozen_flat::AbstractVector{Float64}
)
    N_mov = div(length(x_flat), 3)
    N_fro = div(length(frozen_flat), 3)
    D = length(x_flat)
    n_mm = div(N_mov * (N_mov - 1), 2)
    n_mf = N_mov * N_fro
    nf = n_mm + n_mf

    features = Vector{Float64}(undef, nf)
    J = zeros(nf, D)

    idx = 1

    # Moving-Moving pairs (upper triangle: j < i, matching compute_inverse_distances)
    for j in 1:(N_mov - 1)
        xj = (3 * j - 2, 3 * j - 1, 3 * j)  # coordinate indices for atom j
        for i in (j + 1):N_mov
            xi = (3 * i - 2, 3 * i - 1, 3 * i)

            dx = x_flat[xi[1]] - x_flat[xj[1]]
            dy = x_flat[xi[2]] - x_flat[xj[2]]
            dz = x_flat[xi[3]] - x_flat[xj[3]]
            r2 = dx^2 + dy^2 + dz^2
            r = sqrt(r2)
            invr = 1.0 / r
            invr3 = invr * invr * invr

            features[idx] = invr

            # d(1/r)/d(x_i_a) = -(x_i_a - x_j_a)/r^3
            # d(1/r)/d(x_j_a) =  (x_i_a - x_j_a)/r^3
            J[idx, xi[1]] = -dx * invr3
            J[idx, xi[2]] = -dy * invr3
            J[idx, xi[3]] = -dz * invr3
            J[idx, xj[1]] = dx * invr3
            J[idx, xj[2]] = dy * invr3
            J[idx, xj[3]] = dz * invr3

            idx += 1
        end
    end

    # Moving-Frozen pairs
    if N_fro > 0
        for j in 1:N_mov
            xj = (3 * j - 2, 3 * j - 1, 3 * j)
            for fk in 1:N_fro
                xf1 = frozen_flat[3 * fk - 2]
                xf2 = frozen_flat[3 * fk - 1]
                xf3 = frozen_flat[3 * fk]

                dx = x_flat[xj[1]] - xf1
                dy = x_flat[xj[2]] - xf2
                dz = x_flat[xj[3]] - xf3
                r2 = dx^2 + dy^2 + dz^2
                r = sqrt(r2)
                invr = 1.0 / r
                invr3 = invr * invr * invr

                features[idx] = invr

                # Only moving atom j has non-zero derivatives
                J[idx, xj[1]] = -dx * invr3
                J[idx, xj[2]] = -dy * invr3
                J[idx, xj[3]] = -dz * invr3

                idx += 1
            end
        end
    end

    return features, J
end

function kernel_blocks(k::MolInvDistSE, x1::AbstractVector, x2::AbstractVector)
    frozen = k.frozen_coords

    # Fully analytical: features + Jacobians in one pass, zero AD
    f1, J1 = _invdist_jacobian(collect(Float64, x1), collect(Float64, frozen))
    f2, J2 = _invdist_jacobian(collect(Float64, x2), collect(Float64, frozen))

    nf = length(f1)

    # Per-feature theta^2 = inv_lengthscale^2
    theta2 = Vector{Float64}(undef, nf)
    if !isempty(k.feature_params_map)
        @inbounds for i in 1:nf
            theta2[i] = k.inv_lengthscales[k.feature_params_map[i]]^2
        end
    else
        t2 = k.inv_lengthscales[1]^2
        fill!(theta2, t2)
    end

    # SE kernel in feature space: k = sigma2 * exp(-sum theta_i^2 * r_i^2)
    r = f1 .- f2
    d2 = zero(Float64)
    @inbounds for i in 1:nf
        d2 += theta2[i] * r[i]^2
    end
    kval = Float64(k.signal_variance) * exp(-d2)

    # --- Energy-Energy ---
    k_ee = kval

    # --- Feature-space derivatives (analytical SE) ---
    #   dk/df2_j =  2 * kval * theta_j^2 * r_j
    #   dk/df1_i = -2 * kval * theta_i^2 * r_i
    dk_df2 = Vector{Float64}(undef, nf)
    dk_df1 = Vector{Float64}(undef, nf)
    @inbounds for i in 1:nf
        v = 2.0 * kval * theta2[i] * r[i]
        dk_df2[i] = v
        dk_df1[i] = -v
    end

    # --- Feature-space Hessian d^2 k / (df1_i df2_j) ---
    #   H[i,j] = 2*kval*(theta_i^2 * delta_ij - 2*(theta_i^2*r_i)*(theta_j^2*r_j))
    # = 2*kval*(Theta - 2*u*u^T)  where Theta=diag(theta^2), u=theta^2.*r
    u = theta2 .* r
    H_feat = Matrix{Float64}(undef, nf, nf)
    @inbounds for i in 1:nf
        H_feat[i, i] = 2.0 * kval * (theta2[i] - 2.0 * u[i]^2)
        for j in (i + 1):nf
            val = -4.0 * kval * u[i] * u[j]
            H_feat[i, j] = val
            H_feat[j, i] = val
        end
    end

    # --- Chain rule: feature-space -> Cartesian ---
    k_ef = (dk_df2' * J2)       # (1 x nf) * (nf x D) -> 1 x D
    k_fe = J1' * dk_df1          # (D x nf) * (nf x 1) -> D-vector
    k_ff = J1' * H_feat * J2     # (D x nf) * (nf x nf) * (nf x D) -> D x D

    return k_ee, k_ef, k_fe, k_ff
end

# ==============================================================================
# Analytical kernel blocks AND hyperparameter gradients for MolInvDistSE
# ==============================================================================
#
# Returns both the kernel blocks and dK/d(log w) for all hyperparameters.
# The log-space parametrization w = [log(sigma2); log.(inv_lengthscales)]
# matches C++ gp_pak / MATLAB gp_pak conventions.
#
# For k = sigma2 * exp(-d2), d2 = sum theta2[map[i]] * delta_f[i]^2:
#
#   dk/d(log sigma2) = k  (all blocks scale linearly)
#   dk/d(log theta_p): chain rule through S_p = 2*theta2_p * sum_{map[i]=p} r[i]^2

"""
    kernel_blocks_and_hypergrads(k::MolInvDistSE, x1, x2)

Compute kernel blocks AND their gradients w.r.t. log-space hyperparameters.

Returns `(blocks, grad_blocks)` where:
- `blocks = (k_ee, k_ef, k_fe, k_ff)` -- same as `kernel_blocks`
- `grad_blocks::Vector{NTuple{4,...}}` -- one tuple per log-parameter
  `[log(sigma2), log(theta_1), ..., log(theta_P)]`
"""
function kernel_blocks_and_hypergrads(k::MolInvDistSE, x1::AbstractVector, x2::AbstractVector)
    frozen = k.frozen_coords

    f1, J1 = _invdist_jacobian(collect(Float64, x1), collect(Float64, frozen))
    f2, J2 = _invdist_jacobian(collect(Float64, x2), collect(Float64, frozen))

    nf = length(f1)
    D = length(x1)
    n_ls = length(k.inv_lengthscales)
    n_params = 1 + n_ls  # [log(sigma2), log(theta_1), ..., log(theta_P)]

    has_map = !isempty(k.feature_params_map)

    # Per-feature theta^2
    theta2 = Vector{Float64}(undef, nf)
    # feature -> param index (1-based into inv_lengthscales)
    fmap = Vector{Int}(undef, nf)
    if has_map
        @inbounds for i in 1:nf
            fmap[i] = k.feature_params_map[i]
            theta2[i] = k.inv_lengthscales[fmap[i]]^2
        end
    else
        t2 = k.inv_lengthscales[1]^2
        fill!(theta2, t2)
        fill!(fmap, 1)
    end

    # Feature-space residuals
    r = f1 .- f2
    d2 = zero(Float64)
    @inbounds for i in 1:nf
        d2 += theta2[i] * r[i]^2
    end
    sigma2 = Float64(k.signal_variance)
    kval = sigma2 * exp(-d2)

    # --- Forward blocks (same as kernel_blocks) ---
    k_ee = kval
    u = theta2 .* r  # u[i] = theta2[i] * r[i]

    dk_df2 = Vector{Float64}(undef, nf)
    dk_df1 = Vector{Float64}(undef, nf)
    @inbounds for i in 1:nf
        v = 2.0 * kval * theta2[i] * r[i]
        dk_df2[i] = v
        dk_df1[i] = -v
    end

    H_feat = Matrix{Float64}(undef, nf, nf)
    @inbounds for i in 1:nf
        H_feat[i, i] = 2.0 * kval * (theta2[i] - 2.0 * u[i]^2)
        for j in (i + 1):nf
            val = -4.0 * kval * u[i] * u[j]
            H_feat[i, j] = val
            H_feat[j, i] = val
        end
    end

    k_ef = dk_df2' * J2       # 1 x D
    k_fe = J1' * dk_df1        # D vector
    k_ff = J1' * H_feat * J2   # D x D

    blocks = (k_ee, k_ef, k_fe, k_ff)

    # --- Hyperparameter gradients in log-space ---

    # S_p = 2*theta2_p * sum_{i:fmap[i]=p} r[i]^2  (partial d2 / d(log theta_p))
    S = zeros(n_ls)
    @inbounds for i in 1:nf
        p = fmap[i]
        S[p] += r[i]^2
    end
    @inbounds for p in 1:n_ls
        S[p] *= 2.0 * k.inv_lengthscales[p]^2
    end

    grad_blocks = Vector{NTuple{4,Any}}(undef, n_params)

    # --- d/d(log sigma2): all blocks scale by 1 (dk/d(log s2) = k) ---
    grad_blocks[1] = (k_ee, k_ef, k_fe, k_ff)

    # --- d/d(log theta_p) for each lengthscale parameter ---
    for p in 1:n_ls
        Sp = S[p]
        theta2_p = k.inv_lengthscales[p]^2

        # EE: dk_ee/d(log theta_p) = -k_ee * S_p
        dk_ee_p = -kval * Sp

        # Feature-space derivatives for EF/FE
        # d(dk/df2[l])/d(log theta_p) =
        #   2*kval*r[l] * (delta_{fmap[l],p}*2*theta2_p - theta2[fmap[l]]*Sp)
        ddk_df2_p = Vector{Float64}(undef, nf)
        ddk_df1_p = Vector{Float64}(undef, nf)
        @inbounds for l in 1:nf
            coeff = (fmap[l] == p ? 2.0 * theta2_p : 0.0) - theta2[l] * Sp
            v = 2.0 * kval * r[l] * coeff
            ddk_df2_p[l] = v
            ddk_df1_p[l] = -v
        end

        dk_ef_p = ddk_df2_p' * J2    # 1 x D
        dk_fe_p = J1' * ddk_df1_p     # D vector

        # FF: d(H_feat[l,m])/d(log theta_p)
        # H_feat[l,m] = 2*kval*(theta2[l]*delta_lm - 2*u[l]*u[m])
        # d/d(log theta_p) = -Sp * H_feat[l,m]       [from d(kval)/d(log tp)]
        #   + 2*kval*(delta_lm * delta_{fmap[l],p} * 2*theta2_p
        #             - 2*(du_l*u[m] + u[l]*du_m))   [from d(theta2,u)/d(log tp)]
        # du_l = d(u[l])/d(log theta_p) = delta_{fmap[l],p} * 2*theta2_p * r[l]
        # (Pure theta2 derivative only; kval change is in -Sp*H_feat term.)
        du = Vector{Float64}(undef, nf)
        @inbounds for l in 1:nf
            du[l] = (fmap[l] == p ? 2.0 * theta2_p : 0.0) * r[l]
        end

        dH_feat = Matrix{Float64}(undef, nf, nf)
        @inbounds for l in 1:nf
            diag_term = (fmap[l] == p ? 2.0 * theta2_p : 0.0)
            dH_feat[l, l] = -Sp * H_feat[l, l] + 2.0 * kval * (diag_term - 2.0 * 2.0 * du[l] * u[l])
            for m in (l + 1):nf
                val = -Sp * H_feat[l, m] + 2.0 * kval * (-2.0 * (du[l] * u[m] + u[l] * du[m]))
                dH_feat[l, m] = val
                dH_feat[m, l] = val
            end
        end

        dk_ff_p = J1' * dH_feat * J2  # D x D

        grad_blocks[1 + p] = (dk_ee_p, dk_ef_p, dk_fe_p, dk_ff_p)
    end

    return blocks, grad_blocks
end
