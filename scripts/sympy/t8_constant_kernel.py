#!/usr/bin/env python3
"""Derive constant kernel properties and RFF treatment.

Tutorial T8 (Constant Kernel): verifies that the constant kernel
sigma_c^2 contributes only to the energy-energy block, with all
derivative blocks being zero.

The full kernel for molecular systems (Koistinen et al., JCTC):
  k_total(x,y) = sigma_c^2 + sigma_m^2 * exp(-sum_p theta_p^2 * r_p^2)

The constant term sigma_c^2:
  - Adds a baseline energy-energy correlation (all geometries share
    a common energy offset)
  - Has zero derivatives w.r.t. coordinates (dk_c/dx = 0)
  - Therefore contributes only to K_EE block, not K_EF, K_FE, or K_FF
  - Is NOT optimized by SCG (fixed hyperparameter)

RFF treatment:
  - An extra basis function phi_c = sqrt(sigma_c^2) with zero Jacobian
  - Appended to the RFF feature vector: z_extended = [z_rff; phi_c]
  - d_eff = d_rff + 1 for the Gram matrix
"""
import sympy as sp

print("=" * 60)
print("Constant kernel: derivative verification")
print("=" * 60)

# 2D system for concrete examples
D = 2
x = sp.symbols([f"x{i}" for i in range(D)])
y = sp.symbols([f"y{i}" for i in range(D)])
sigma_c2 = sp.Symbol("sigma_c2", positive=True)

# Constant kernel
k_c = sigma_c2

print(f"\nk_c(x,y) = sigma_c^2 = {k_c}")
print(f"Note: k_c does not depend on x or y")

# All derivatives are zero
print("\n--- Derivative blocks ---")
for d in range(D):
    dk_dy = sp.diff(k_c, y[d])
    dk_dx = sp.diff(k_c, x[d])
    print(f"dk_c/dy_{d} = {dk_dy}  (k_ef contribution)")
    print(f"dk_c/dx_{d} = {dk_dx}  (k_fe contribution)")

print()
for di in range(D):
    for dj in range(D):
        d2k = sp.diff(k_c, x[di], y[dj])
        print(f"d^2k_c/(dx_{di} dy_{dj}) = {d2k}  (k_ff contribution)")

print("""
All derivatives are zero. The constant kernel contributes only to
the energy-energy (EE) block of the covariance matrix.
""")

# Full kernel structure
print("=" * 60)
print("Full covariance matrix structure with constant kernel")
print("=" * 60)

sigma_m2 = sp.Symbol("sigma_m2", positive=True)
theta = sp.Symbol("theta", positive=True)

r2 = sum((xi - yi) ** 2 for xi, yi in zip(x, y))
k_se = sigma_m2 * sp.exp(-theta**2 * r2)
k_total = sigma_c2 + k_se

print(f"\nk_total(x,y) = sigma_c^2 + sigma_m^2 * exp(-theta^2 * ||x-y||^2)")
print(f"\nCovariance matrix blocks:")
print(f"  K_EE[i,j] = k_total(x_i, x_j) + delta_ij * noise_e")
print(f"            = sigma_c^2 + k_se(x_i, x_j) + delta_ij * noise_e")
print(f"  K_EF[i,j,d] = dk_se/dy_d(x_i, x_j)  (no sigma_c^2 contribution)")
print(f"  K_FE[i,j,d] = dk_se/dx_d(x_i, x_j)  (no sigma_c^2 contribution)")
print(f"  K_FF[i,j,di,dj] = d^2k_se/(dx_di dy_dj)  (no sigma_c^2 contribution)")

# NLL and hyperparameter training
print("\n" + "=" * 60)
print("Hyperparameter training with constant kernel")
print("=" * 60)
print("""
sigma_c^2 is a FIXED hyperparameter, not optimized by SCG.
Only sigma_m^2 and theta_p are optimized.

Rationale: sigma_c^2 absorbs the mean energy level. For molecular
systems, the energy scale varies by orders of magnitude across different
molecules, so a fixed sigma_c^2 = 1.0 works as a regularizer. For 2D
test surfaces, sigma_c^2 = 0.0 (no constant kernel) is appropriate
because the energy scale is already handled by sigma_m^2.

The upper barrier on sigma_m^2 (max_log_sigma2 = ln(2)) prevents
sigma_m^2 from growing too large. With sigma_c^2 > 0, there is no
lower barrier: sigma_m^2 can shrink toward zero if the data supports
it, though this would degrade gradient predictions.
""")

# RFF treatment
print("=" * 60)
print("RFF treatment of constant kernel")
print("=" * 60)
print("""
The constant kernel adds one extra basis function to the RFF model:

  phi_c = sqrt(sigma_c^2)

with zero Jacobian (no coordinate dependence).

Extended feature vector:
  z_ext(x) = [z_1(x), z_2(x), ..., z_{D_rff}(x), phi_c]

where z_j(x) = c * cos(w_j^T phi(x) + b_j) are the standard RFF features
and phi_c is constant across all x.

Design matrix row for point i:
  Energy row:   [...z_ext(x_i)...]  (length D_rff + 1)
  Gradient row: [...dz/dx_d(x_i)..., 0]  (phi_c has zero derivative)

The effective dimension is d_eff = D_rff + 1, adding one column to the
design matrix and one row/column to the Gram matrix (D_rff+1 x D_rff+1).
""")

# Verify: phi_c contribution to kernel approximation
print("--- Verification ---")
print("""
For the extended features:
  z_ext(x)^T z_ext(y) = z_rff(x)^T z_rff(y) + phi_c^2
                       ~ k_se(x,y) + sigma_c^2
                       = k_total(x,y)

The constant feature contributes exactly sigma_c^2 to the energy-energy
inner product, reproducing the constant kernel term.

For gradient predictions:
  J_z_ext = [J_z_rff; 0]  (last row is zero)

So the gradient prediction uses only the RFF features, not phi_c.
This matches the exact GP behavior (sigma_c^2 contributes only to EE).
""")

# Numerical example
import numpy as np

print("=" * 60)
print("Numerical example")
print("=" * 60)

sigma_c2_val = 1.0
sigma_m2_val = 0.5
theta_val = 1.2

x_val = np.array([0.3, 0.7])
y_val = np.array([0.5, 1.0])
r2_val = np.sum((x_val - y_val)**2)

k_se_val = sigma_m2_val * np.exp(-theta_val**2 * r2_val)
k_total_val = sigma_c2_val + k_se_val

print(f"\nsigma_c^2 = {sigma_c2_val}, sigma_m^2 = {sigma_m2_val}, theta = {theta_val}")
print(f"x = {x_val}, y = {y_val}")
print(f"||x-y||^2 = {r2_val:.6f}")
print(f"k_se = {k_se_val:.6f}")
print(f"k_total = k_se + sigma_c^2 = {k_total_val:.6f}")

# RFF approximation with constant feature
np.random.seed(42)
d_rff = 500
W = np.random.randn(d_rff, 2) * np.sqrt(2) * theta_val
b = np.random.uniform(0, 2 * np.pi, d_rff)
c = np.sqrt(2 * sigma_m2_val / d_rff)

zx = np.append(c * np.cos(W @ x_val + b), np.sqrt(sigma_c2_val))
zy = np.append(c * np.cos(W @ y_val + b), np.sqrt(sigma_c2_val))

k_rff = zx @ zy
print(f"\nRFF approximation (D_rff={d_rff}): {k_rff:.6f}")
print(f"True k_total:                      {k_total_val:.6f}")
print(f"Error:                             {abs(k_rff - k_total_val):.6f}")

# Correspondence to Rust code:
print("\n--- Rust correspondence ---")
print("covariance.rs:34  const_sigma2 added to EE diagonal")
print("covariance.rs:54  const_sigma2 added to EE off-diagonal")
print("rff.rs:88-96      phi_c = sqrt(const_sigma2), zero Jacobian row")
print("predict.rs:136    const_sigma2 in k_star EE entries")
print("predict.rs:219    const_sigma2 in prior variance")
print("nll.rs:51,85-86   const_sigma2 in NLL covariance matrix")
