#!/usr/bin/env python3
"""Validate prior-mean residualization identities for ChemGP.

Tutorial T10 (Prior Mean): verifies the algebra used by ChemGP when a
non-zero prior mean is subtracted from training targets before GP fitting
and added back at prediction time.

Posterior with prior mean m(x):
  mu_post(x*) = m(x*) + k(x*, X) K^{-1} (y - m(X))

Gradient-aware targets follow the same residualization pattern:
  y_full = [E(X) - m_E(X); G(X) - m_G(X)]

The core identity is simple but important: once the prior mean is
accounted for explicitly, the GP only models the residual surface.
"""

import sympy as sp

print("=" * 60)
print("Prior-mean residualization identity")
print("=" * 60)

# Generic symbolic GP objects
kx = sp.MatrixSymbol("k_xX", 1, 3)
Kinv = sp.MatrixSymbol("K_inv", 3, 3)
y = sp.MatrixSymbol("y", 3, 1)
mX = sp.MatrixSymbol("m_X", 3, 1)
mx = sp.Symbol("m_x")

posterior_with_prior = mx + (kx * Kinv * (y - mX))[0, 0]
posterior_residual = mx + (kx * Kinv * y)[0, 0] - (kx * Kinv * mX)[0, 0]

print("\nPosterior with explicit prior:")
print("  mu(x*) = m(x*) + k(x*,X) K^{-1} (y - m(X))")

print("\nExpanded residual form:")
print("  mu(x*) = m(x*) + k(x*,X) K^{-1} y - k(x*,X) K^{-1} m(X)")

print("\nSymbolic check:")
print(sp.simplify(posterior_with_prior - posterior_residual))

assert sp.simplify(posterior_with_prior - posterior_residual) == 0

print("\nIdentity verified: fitting the GP to residuals and then adding")
print("the prior back during prediction is algebraically exact.")

# Diagonal quadratic prior example
print("\n" + "=" * 60)
print("Quadratic prior example")
print("=" * 60)

x1, x2 = sp.symbols("x1 x2", real=True)
c1, c2 = sp.symbols("c1 c2", real=True)
k1, k2 = sp.symbols("k1 k2", positive=True)
e0 = sp.Symbol("e0", real=True)

m = e0 + sp.Rational(1, 2) * k1 * (x1 - c1) ** 2 + sp.Rational(1, 2) * k2 * (x2 - c2) ** 2
g1 = sp.diff(m, x1)
g2 = sp.diff(m, x2)

print(f"m(x)   = {sp.expand(m)}")
print(f"dm/dx1 = {sp.expand(g1)}")
print(f"dm/dx2 = {sp.expand(g2)}")

print("\nThis matches the ChemGP `Quadratic` prior implementation:")
print("  E(x) = e0 + 0.5 * sum_i k_i (x_i - c_i)^2")
print("  G_i  = k_i (x_i - c_i)")

print("\n--- Rust correspondence ---")
print("prior_mean.rs: residualize_training_data() subtracts prior energy/gradient")
print("predict.rs: build_pred_model_with_prior() fits residual targets")
print("predict.rs: apply_prior_to_prediction() adds the prior back to the mean")

# Taylor-diagonal prior example
print("\n" + "=" * 60)
print("Taylor-diagonal prior example")
print("=" * 60)

g01, g02 = sp.symbols("g01 g02", real=True)

taylor_m = (
    e0
    + g01 * (x1 - c1)
    + g02 * (x2 - c2)
    + sp.Rational(1, 2) * k1 * (x1 - c1) ** 2
    + sp.Rational(1, 2) * k2 * (x2 - c2) ** 2
)
taylor_g1 = sp.diff(taylor_m, x1)
taylor_g2 = sp.diff(taylor_m, x2)

print(f"m_taylor(x)   = {sp.expand(taylor_m)}")
print(f"dm_taylor/dx1 = {sp.expand(taylor_g1)}")
print(f"dm_taylor/dx2 = {sp.expand(taylor_g2)}")

expected_g1 = g01 + k1 * (x1 - c1)
expected_g2 = g02 + k2 * (x2 - c2)

assert sp.simplify(taylor_g1 - expected_g1) == 0
assert sp.simplify(taylor_g2 - expected_g2) == 0

print("\nIdentity verified for the ChemGP `TaylorDiagonal` prior:")
print("  E(x) = e0 + g0·(x-c) + 0.5 * sum_i k_i (x_i - c_i)^2")
print("  G_i  = g0_i + k_i (x_i - c_i)")
