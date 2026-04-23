#!/usr/bin/env python3
"""Derive EE, EG, GE, GG covariance blocks for the CartesianSE kernel.

Tutorial T1 (GP Basics): verifies the analytical kernel block expressions
used in kernel.rs::cartesian_kernel_blocks_and_hypergrads.

Kernel: k(x,y) = sigma^2 * exp(-theta^2 * ||x - y||^2)

The four blocks arise from differentiating the energy-energy kernel:
  k_ee(x,y) = k(x,y)
  k_ef(x,y) = dk/dy       (energy at x, gradient at y)
  k_fe(x,y) = dk/dx       (gradient at x, energy at y)
  k_ff(x,y) = d^2k/dxdy   (gradient at x, gradient at y)

ChemGP's kernel layer is written in energy/gradient form. Atomic forces are
introduced later through F = -∇V at the oracle and optimizer interfaces.
"""
import sympy as sp

# Dimension (symbolic, but we expand for d=2 concretely)
D = 2

# Coordinates
x = sp.symbols([f"x{i}" for i in range(D)])
y = sp.symbols([f"y{i}" for i in range(D)])

# Hyperparameters
sigma2, theta = sp.symbols("sigma2 theta", positive=True)

# Squared distance
r2 = sum((xi - yi) ** 2 for xi, yi in zip(x, y))

# Kernel
k = sigma2 * sp.exp(-theta**2 * r2)

print("=" * 60)
print("CartesianSE kernel blocks (D=2)")
print("=" * 60)

# k_ee
k_ee = k
print(f"\nk_ee = {sp.simplify(k_ee)}")

# k_ef[d] = d k / d y_d   (energy-gradient cross-covariance)
k_ef = [sp.diff(k, y[d]) for d in range(D)]
print("\nk_ef:")
for d in range(D):
    print(f"  [{d}] = {sp.simplify(k_ef[d])}")

# k_fe[d] = d k / d x_d   (gradient-energy cross-covariance)
k_fe = [sp.diff(k, x[d]) for d in range(D)]
print("\nk_fe:")
for d in range(D):
    print(f"  [{d}] = {sp.simplify(k_fe[d])}")

# k_ff[di, dj] = d^2k/(dx_di dy_dj)
print("\nk_ff:")
k_ff = sp.zeros(D, D)
for di in range(D):
    for dj in range(D):
        k_ff[di, dj] = sp.diff(k, x[di], y[dj])
        simplified = sp.simplify(k_ff[di, dj])
        print(f"  [{di},{dj}] = {simplified}")

# Verify structure: k_ef = -k_fe (antisymmetry from r = x - y)
print("\n--- Verification ---")
for d in range(D):
    check = sp.simplify(k_ef[d] + k_fe[d])
    print(f"k_ef[{d}] + k_fe[{d}] = {check}  (should be 0)")

# Verify k_ff diagonal and off-diagonal structure
# Diagonal: k_ff[d,d] = 2*theta^2 * k * (1 - 2*theta^2*(x_d - y_d)^2)
# Off-diag: k_ff[di,dj] = -4*theta^4 * (x_di - y_di)*(x_dj - y_dj) * k
print("\nDiagonal structure check:")
for d in range(D):
    expected = 2 * theta**2 * k * (1 - 2 * theta**2 * (x[d] - y[d]) ** 2)
    check = sp.simplify(k_ff[d, d] - expected)
    print(f"  k_ff[{d},{d}] matches 2*theta^2*k*(1 - 2*theta^2*r_d^2): {check == 0}")

print("\nOff-diagonal structure check:")
for di in range(D):
    for dj in range(D):
        if di != dj:
            expected = -4 * theta**4 * (x[di] - y[di]) * (x[dj] - y[dj]) * k
            check = sp.simplify(k_ff[di, dj] - expected)
            print(f"  k_ff[{di},{dj}] matches -4*theta^4*r_di*r_dj*k: {check == 0}")

# Correspondence to Rust code:
#   kernel.rs, cartesian_kernel_blocks_and_hypergrads
#   kval = sigma2 * exp(-theta^2 * d2)
#   k_ef[d] = +2 * theta^2 * r[d] * kval   (where r = x - y)
#   k_fe[d] = -2 * theta^2 * r[d] * kval
#   k_ff[d,d] = 2 * theta^2 * kval * (1 - 2*theta^2*r[d]^2)
#   k_ff[di,dj] = -4 * theta^4 * r[di]*r[dj] * kval   (di != dj)
print("\n--- Closed-form summary ---")
print("Let r_d = x_d - y_d, kval = sigma^2 * exp(-theta^2 * sum(r_d^2))")
print("k_ee = kval")
print("k_ef[d] = +2 * theta^2 * r_d * kval")
print("k_fe[d] = -2 * theta^2 * r_d * kval")
print("k_ff[d,d] = 2 * theta^2 * kval * (1 - 2*theta^2*r_d^2)")
print("k_ff[di,dj] = -4 * theta^4 * r_di * r_dj * kval  (di != dj)")
