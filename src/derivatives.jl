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
