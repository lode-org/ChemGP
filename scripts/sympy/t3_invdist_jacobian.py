#!/usr/bin/env python3
"""Derive the Jacobian df_ij/dx for inverse distance features.

Tutorial T3 (Molecular Kernels): verifies the chain rule used in
kernel.rs::molinvdist_kernel_blocks_and_hypergrads to transform
kernel derivatives from feature space to coordinate space.

The MolInvDistSE kernel maps Cartesian coordinates to inverse
interatomic distance features:
  f_ij(x) = 1 / r_ij,  r_ij = ||x_i - x_j||

The SE kernel operates in feature space:
  k(x,y) = sigma^2 * exp(-sum_p theta_p^2 * (f_p(x) - f_p(y))^2)

Energy-force cross-covariance requires the chain rule:
  dk/dy_d = sum_l (dk/df_l(y)) * (df_l/dy_d)
          = J_y^T @ (dk/df)

where J_y is the Jacobian of features w.r.t. coordinates of y.

This script derives J analytically and verifies against numerical
finite differences for a 3-atom collinear system (like LEPS).
"""
import numpy as np
import sympy as sp

# 3-atom system in 1D (collinear, like LEPS H+H2)
# Coordinates: x_A, x_B, x_C
x_A, x_B, x_C = sp.symbols("x_A x_B x_C", real=True)
coords = [x_A, x_B, x_C]

# Interatomic distances (3 atoms -> 3 pairs)
r_AB = sp.sqrt((x_A - x_B) ** 2)
r_AC = sp.sqrt((x_A - x_C) ** 2)
r_BC = sp.sqrt((x_B - x_C) ** 2)

# Inverse distance features
f_AB = 1 / r_AB
f_AC = 1 / r_AC
f_BC = 1 / r_BC
features = [f_AB, f_AC, f_BC]
pair_labels = ["AB", "AC", "BC"]

print("=" * 60)
print("Inverse distance Jacobian for 3-atom collinear system")
print("=" * 60)

# Jacobian J[l, d] = df_l / dx_d
# l indexes features (pairs), d indexes coordinates (atoms)
n_feat = len(features)
n_coord = len(coords)
J = sp.zeros(n_feat, n_coord)

print("\nAnalytical Jacobian df/dx:")
for l in range(n_feat):
    for d in range(n_coord):
        J[l, d] = sp.diff(features[l], coords[d])
        simplified = sp.simplify(J[l, d])
        print(f"  df_{pair_labels[l]}/dx_{coords[d]} = {simplified}")

# General formula for df_ij/dx_k where f_ij = 1/r_ij:
#   df_ij/dx_k = -1/r_ij^2 * dr_ij/dx_k
#              = -1/r_ij^2 * (x_k - x_other) / r_ij   if k in {i,j}
#              = 0                                       otherwise
# Simplified:
#   df_ij/dx_i = -(x_i - x_j) / r_ij^3
#   df_ij/dx_j = -(x_j - x_i) / r_ij^3 = (x_i - x_j) / r_ij^3
#   df_ij/dx_k = 0  (k != i, k != j)
print("\n--- General formula ---")
print("df_ij/dx_i = -(x_i - x_j) / r_ij^3")
print("df_ij/dx_j = +(x_i - x_j) / r_ij^3")
print("df_ij/dx_k = 0  (k not in {i,j})")

# Chain rule for k_ef
print("\n" + "=" * 60)
print("Chain rule: kernel derivatives in coordinate space")
print("=" * 60)

# SE kernel in feature space with pair-type length scales
sigma2 = sp.Symbol("sigma2", positive=True)
theta_AB, theta_AC, theta_BC = sp.symbols("theta_AB theta_AC theta_BC", positive=True)
thetas = [theta_AB, theta_AC, theta_BC]

# Second set of features for point y
y_A, y_B, y_C = sp.symbols("y_A y_B y_C", real=True)
y_coords = [y_A, y_B, y_C]

ry_AB = sp.sqrt((y_A - y_B) ** 2)
ry_AC = sp.sqrt((y_A - y_C) ** 2)
ry_BC = sp.sqrt((y_B - y_C) ** 2)
fy = [1 / ry_AB, 1 / ry_AC, 1 / ry_BC]

# Feature differences
r_feat = [features[l] - fy[l] for l in range(n_feat)]

# d2 = sum_p theta_p^2 * (f_p(x) - f_p(y))^2
d2 = sum(thetas[l] ** 2 * r_feat[l] ** 2 for l in range(n_feat))
kval = sigma2 * sp.exp(-d2)

# dk/df_l(y) (derivative w.r.t. feature l of y)
# d/df_l(y) of exp(-sum theta_p^2 * (f_p(x) - f_p(y))^2)
#   = 2 * theta_l^2 * (f_l(x) - f_l(y)) * kval
dk_df_y = [2 * thetas[l] ** 2 * r_feat[l] * kval for l in range(n_feat)]

print("\ndk/df_l(y) = 2 * theta_l^2 * (f_l(x) - f_l(y)) * kval")

# k_ef[d] = sum_l dk/df_l(y) * df_l/dy_d = J_y^T @ dk_df
# where J_y is the Jacobian at y
J_y = sp.zeros(n_feat, n_coord)
for l in range(n_feat):
    for d in range(n_coord):
        J_y[l, d] = sp.diff(fy[l], y_coords[d])

print("\nk_ef[d] = sum_l (dk/df_l) * J_y[l,d]")
print("This is the mat_t_vec operation in kernel.rs")

# Numerical verification
print("\n" + "=" * 60)
print("Numerical verification (collinear H+H2)")
print("=" * 60)

# Concrete values: H+H2 reactant-like geometry
vals = {
    x_A: 0.0, x_B: 0.74, x_C: 3.0,
    y_A: 0.1, y_B: 0.80, y_C: 2.9,
    sigma2: 1.0, theta_AB: 1.0, theta_AC: 1.0, theta_BC: 1.0,
}

print(f"\nx = [{float(vals[x_A])}, {float(vals[x_B])}, {float(vals[x_C])}]")
print(f"y = [{float(vals[y_A])}, {float(vals[y_B])}, {float(vals[y_C])}]")

kval_num = float(kval.subs(vals))
print(f"\nkval = {kval_num:.8f}")

print("\nJacobian J_y (numerical):")
for l in range(n_feat):
    row = [float(J_y[l, d].subs(vals)) for d in range(n_coord)]
    print(f"  {pair_labels[l]}: {row}")

# Verify: direct dk/dy_d vs chain rule J^T @ dk_df
print("\nDirect dk/dy_d vs chain rule J_y^T @ dk_df:")
for d in range(n_coord):
    direct = float(sp.diff(kval, y_coords[d]).subs(vals))
    chain = sum(float(dk_df_y[l].subs(vals)) * float(J_y[l, d].subs(vals))
                for l in range(n_feat))
    print(f"  y_{y_coords[d]}: direct={direct:.8f}, chain={chain:.8f}, "
          f"match={abs(direct - chain) < 1e-10}")

# Correspondence to Rust code:
#   kernel.rs, molinvdist_kernel_blocks_and_hypergrads
#   invdist.rs, compute_inverse_distances (features)
#   invdist.rs, inverse_distance_jacobian (J matrix)
#   kernel.rs: dk_df = 2 * theta2[l] * r[l] * kval (per feature)
#   kernel.rs: k_ef = mat_t_vec(J_y, dk_df)  (J^T @ dk_df)
print("\n--- Rust correspondence ---")
print("invdist.rs: compute_inverse_distances -> features f_l")
print("invdist.rs: inverse_distance_jacobian -> J[l,d]")
print("kernel.rs: dk_df[l] = 2 * theta2[l] * r[l] * kval")
print("kernel.rs: k_ef = mat_t_vec(J, dk_df)  (J^T @ dk_df)")
print("kernel.rs: k_ff = J^T @ H_feat @ J  (jt_h_j)")
