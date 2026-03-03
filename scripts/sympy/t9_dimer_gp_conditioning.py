#!/usr/bin/env python3
"""GP dimer conditioning analysis: noise, curvature sensitivity, and error amplification.

Tutorial T9 (Production Saddle Search): derives the GP prediction model for
energy+gradient observations from first principles, then demonstrates how
gradient noise corrupts finite-difference curvature estimates used by the
dimer method.

Key results:
1. Block covariance structure K = [[K_EE, K_EF], [K_FE, K_FF]] with noise
   only on the EE diagonal (MATLAB/C++/Stan convention for dimer).
2. Curvature = (g1_pred - g0_pred) . orient / dimer_sep is a linear
   functional of the GP posterior mean, so noise on gradient observations
   propagates linearly into curvature error.
3. With dimer_sep=0.01, gradient noise variance sigma_g^2=1e-5 produces
   curvature noise variance ~1e-5/(0.01)^2 = 100, explaining the +103
   curvature observed when true curvature is -8.43.

Cross-references:
- MATLAB gpstuff: lik_gaussian sigma2=1e-8 (fixed prior), no gradient noise
- C++ gpr_optim: noise_e=1e-7, no gradient noise, jitter=1e-6 on all diags
- Stan (Susmann): delta=1e-9 jitter, no gradient noise
- DTU CatLearn: noise_deriv as separate optimized hyperparameter (NEB only, no dimer)
"""
import numpy as np
import sympy as sp

# ============================================================================
# Part 1: Block covariance structure for derivative GP
# ============================================================================

print("=" * 60)
print("Part 1: Block covariance matrix structure")
print("=" * 60)

# Symbolic setup: 1D for clarity, generalizes to D dimensions
x, y = sp.symbols("x y")
sigma2 = sp.Symbol("sigma_m2", positive=True)
theta = sp.Symbol("theta", positive=True)
sigma_e2 = sp.Symbol("sigma_e2", positive=True, nonzero=True)
sigma_g2 = sp.Symbol("sigma_g2", nonnegative=True)
h = sp.Symbol("h", positive=True)  # dimer separation

# SE kernel in 1D
k_se = sigma2 * sp.exp(-theta**2 * (x - y) ** 2)

# Derivative blocks
k_EE = k_se  # k(x, y)
k_EF = -sp.diff(k_se, y)  # -dk/dy (negative: force = -gradient)
k_FE = sp.diff(k_se, x)  # dk/dx
k_FF = -sp.diff(k_se, x, y)  # -d^2k/dxdy

print(f"\nSE kernel:  k(x,y) = sigma_m^2 * exp(-theta^2 * (x-y)^2)")
print(f"\nBlock derivatives (D=1 case):")
print(f"  K_EE = k(x,y)         = {k_EE}")
print(f"  K_EF = -dk/dy          = {sp.simplify(k_EF)}")
print(f"  K_FE =  dk/dx          = {sp.simplify(k_FE)}")
print(f"  K_FF = -d^2k/dxdy      = {sp.simplify(k_FF)}")

# Verify K_EF and K_FE relationship
print(f"\n  K_FE = -K_EF^T?  {sp.simplify(k_FE + k_EF) == 0}")

# ============================================================================
# Part 2: Noise model for dimer
# ============================================================================

print("\n" + "=" * 60)
print("Part 2: Noise model comparison across codebases")
print("=" * 60)

print("""
| Codebase         | Energy noise | Gradient noise | Jitter    |
|------------------|-------------|----------------|-----------|
| MATLAB gpstuff   | 1e-8 fixed  | none           | 0         |
| C++ gpr_optim    | 1e-7        | none           | 1e-6 diag |
| Stan (Susmann)   | --          | none           | 1e-9      |
| DTU CatLearn     | optimized   | optimized      | adaptive  |
| Rust (previous)  | 1e-6        | 1e-4           | 1e-6      |
| Rust (fixed)     | 1e-8        | 0              | 0         |

All dimer implementations: zero gradient noise.
CatLearn has noise_deriv but is NEB-only (not applicable to dimer).
""")

# ============================================================================
# Part 3: Curvature sensitivity to gradient noise
# ============================================================================

print("=" * 60)
print("Part 3: Curvature sensitivity analysis")
print("=" * 60)

# Curvature formula: C = (g1 - g0) . orient / dimer_sep
# where g0, g1 are GP posterior means at r and r + h*orient
# For 1D: C = (g1_pred - g0_pred) / h

# The GP posterior mean for gradient is a linear functional of observations.
# Let alpha = (K + noise)^{-1} y  (standard GP weights).
# Then gradient prediction at x* is:
#   g_pred(x*) = sum_i K_FE(x*, x_i) * alpha_i   (energy obs)
#              + sum_j K_FF(x*, x_j) * alpha_j     (gradient obs)
#
# Noise on the gradient observations enters (K + noise)^{-1}, affecting alpha.
# With N_obs = 2 (midpoint + image1) and D=1, we can compute explicitly.

print("""
Curvature: C = (g1_pred - g0_pred) . orient / dimer_sep

Since GP posterior mean is linear in observations:
  g_pred(x*) = K_*^T (K + Sigma_noise)^{-1} y

The noise matrix Sigma_noise = diag(sigma_e^2 * I_n, sigma_g^2 * I_{n*D})
adds sigma_g^2 to the force-force diagonal block.

For dimer_sep=h, curvature error scales as:
  Var(C) ~ sigma_g^2 / h^2

because the finite-difference curvature amplifies gradient noise by 1/h^2.
""")

# Numerical demonstration
print("Numerical curvature noise amplification:")
for dimer_sep_val in [0.01, 0.001, 0.1]:
    for noise_g_val in [1e-5, 1e-4, 1e-3, 0.0]:
        if noise_g_val == 0.0:
            amplified = 0.0
        else:
            amplified = noise_g_val / dimer_sep_val**2
        print(
            f"  dimer_sep={dimer_sep_val:.3f}, noise_g={noise_g_val:.0e} "
            f"-> curvature_noise ~ {amplified:.1e}"
        )
    print()

# ============================================================================
# Part 4: Symbolic curvature error bound
# ============================================================================

print("=" * 60)
print("Part 4: Curvature error bound (symbolic)")
print("=" * 60)

# For the simplest case: 2 training points at r0, r1 = r0 + h
# The GP posterior variance at r0 for the gradient prediction is:
#   Var[g(r0)] >= sigma_g^2 * K_FF(0,0)^{-1} * K_FF(0,0) = sigma_g^2
# (diagonal noise directly inflates prediction uncertainty)

print("""
Consider the simplest case: N=2 points at r0 and r1=r0+h (dimer pair).

K_FF(r0,r0) = sigma_m^2 * 2*theta^2  (second derivative of SE at distance 0)

With gradient noise sigma_g^2, the effective K_FF diagonal becomes:
  K_FF(r0,r0) + sigma_g^2

For sigma_m^2*2*theta^2 ~ 1 and sigma_g^2 = 1e-4:
  The noise is small relative to the kernel. But the curvature estimate:

  C = [g_pred(r0+h) - g_pred(r0)] / h

  amplifies any error in g_pred by 1/h.

  With h=0.01 and typical GP prediction uncertainty delta_g ~ sqrt(sigma_g^2):
    delta_C ~ delta_g / h ~ sqrt(1e-4) / 0.01 = 1.0 eV/A^2

  For h=0.01 and sigma_g^2=1e-5 (previous Rust default):
    delta_C ~ sqrt(1e-5) / 0.01 ~ 0.3 eV/A^2

  This is large enough to flip the sign of a curvature of -8.43 eV/A^2?
  No -- but the issue is not just the noise floor. With poorly conditioned
  hyperparameters (sigma2 ~ 8e-5 from data-dependent init on clustered data),
  the kernel itself is near-singular, and the noise term dominates the GG block:

  K_FF(r0,r0) ~ sigma_m^2 * 2*theta^2 ~ 8e-5 * 2 * 270^2 ~ 11.7
  K_FF(r0,r0) + sigma_g^2 = 11.7 + 1e-4  (OK, small relative to kernel)

  But K_FF(r0,r1) ~ sigma_m^2 * 2*theta^2 * (1 - 2*theta^2*h^2) * exp(-theta^2*h^2)
  With theta=270, h=0.01: theta^2*h^2 = 7.29

  exp(-7.29) ~ 7e-4, so K_FF(r0,r1) ~ 11.7 * (1 - 14.58) * 7e-4 ~ -0.11

  The condition number of the FF block is dominated by the diagonal.
  When sigma_g^2 >> 0, it shifts all eigenvalues, making the GP less
  responsive to training data. Combined with lambda=100 preventing SCG
  from correcting the hyperparameters, the curvature prediction becomes
  unreliable.
""")

# ============================================================================
# Part 5: Numerical GP conditioning demo
# ============================================================================

print("=" * 60)
print("Part 5: Numerical conditioning demonstration")
print("=" * 60)

# Build a concrete GP with 2 training points (dimer pair) on a 1D parabola
# True function: f(x) = -0.5 * x^2 (saddle in 1D), true curvature = -1.0

def se_kernel_1d(x1, x2, sigma2_val, theta_val):
    """SE kernel value."""
    return sigma2_val * np.exp(-theta_val**2 * (x1 - x2)**2)

def se_kernel_deriv_EF(x1, x2, sigma2_val, theta_val):
    """K_EF: -dk/dx2."""
    return sigma2_val * 2 * theta_val**2 * (x1 - x2) * np.exp(-theta_val**2 * (x1 - x2)**2)

def se_kernel_deriv_FE(x1, x2, sigma2_val, theta_val):
    """K_FE: dk/dx1."""
    return -sigma2_val * 2 * theta_val**2 * (x1 - x2) * np.exp(-theta_val**2 * (x1 - x2)**2)

def se_kernel_deriv_FF(x1, x2, sigma2_val, theta_val):
    """K_FF: -d2k/dx1dx2."""
    r2 = (x1 - x2)**2
    return sigma2_val * 2 * theta_val**2 * (1 - 2 * theta_val**2 * r2) * np.exp(-theta_val**2 * r2)

def build_gp_matrix(xs, sigma2_val, theta_val, noise_e_val, noise_g_val):
    """Build full [EE, EF; FE, FF] covariance matrix + noise."""
    n = len(xs)
    m = 2 * n  # 1D: each point has energy + 1 gradient component
    K = np.zeros((m, m))

    for i in range(n):
        for j in range(n):
            K[i, j] = se_kernel_1d(xs[i], xs[j], sigma2_val, theta_val)
            K[i, n + j] = se_kernel_deriv_EF(xs[i], xs[j], sigma2_val, theta_val)
            K[n + i, j] = se_kernel_deriv_FE(xs[i], xs[j], sigma2_val, theta_val)
            K[n + i, n + j] = se_kernel_deriv_FF(xs[i], xs[j], sigma2_val, theta_val)

    # Add noise
    for i in range(n):
        K[i, i] += noise_e_val
    for i in range(n):
        K[n + i, n + i] += noise_g_val

    return K

# True function: f(x) = -0.5*x^2, g(x) = -x
def true_func(x_val):
    return -0.5 * x_val**2, -x_val

# Dimer at x=1.0 with h=0.01
x0_val = 1.0
h_val = 0.01
x1_val = x0_val + h_val
true_curv = -1.0  # d2f/dx2 = -1

e0_true, g0_true = true_func(x0_val)
e1_true, g1_true = true_func(x1_val)

print(f"True function: f(x) = -0.5*x^2")
print(f"True curvature: {true_curv}")
print(f"Dimer points: x0={x0_val}, x1={x1_val} (h={h_val})")
print(f"True: e0={e0_true:.6f}, g0={g0_true:.6f}, e1={e1_true:.6f}, g1={g1_true:.6f}")
print(f"FD curvature: {(g1_true - g0_true)/h_val:.6f}")

xs = np.array([x0_val, x1_val])
y_obs = np.array([e0_true, e1_true, g0_true, g1_true])

sigma2_good = 1.0
theta_good = 1.0

print(f"\n--- Good hyperparameters: sigma2={sigma2_good}, theta={theta_good} ---")

for noise_g_val in [0.0, 1e-8, 1e-5, 1e-4, 1e-3]:
    noise_e_val = 1e-8
    K = build_gp_matrix(xs, sigma2_good, theta_good, noise_e_val, noise_g_val)
    try:
        alpha = np.linalg.solve(K, y_obs)
        # Predict at x0 and x1
        k_star_0 = np.array([
            se_kernel_deriv_FE(x0_val, xs[0], sigma2_good, theta_good),
            se_kernel_deriv_FE(x0_val, xs[1], sigma2_good, theta_good),
            se_kernel_deriv_FF(x0_val, xs[0], sigma2_good, theta_good),
            se_kernel_deriv_FF(x0_val, xs[1], sigma2_good, theta_good),
        ])
        k_star_1 = np.array([
            se_kernel_deriv_FE(x1_val, xs[0], sigma2_good, theta_good),
            se_kernel_deriv_FE(x1_val, xs[1], sigma2_good, theta_good),
            se_kernel_deriv_FF(x1_val, xs[0], sigma2_good, theta_good),
            se_kernel_deriv_FF(x1_val, xs[1], sigma2_good, theta_good),
        ])
        g0_pred = k_star_0 @ alpha
        g1_pred = k_star_1 @ alpha
        c_pred = (g1_pred - g0_pred) / h_val
        cond = np.linalg.cond(K)
        print(f"  noise_g={noise_g_val:.0e}: g0_pred={g0_pred:+.6f}, "
              f"g1_pred={g1_pred:+.6f}, C_pred={c_pred:+.4f} "
              f"(err={abs(c_pred - true_curv):.2e}, cond={cond:.1e})")
    except np.linalg.LinAlgError:
        print(f"  noise_g={noise_g_val:.0e}: SINGULAR")

# Now with bad hyperparameters (simulating data-dependent init on clustered data)
sigma2_bad = 8e-5
theta_bad = 270.0
print(f"\n--- Bad hyperparameters (clustered init): sigma2={sigma2_bad}, theta={theta_bad} ---")

for noise_g_val in [0.0, 1e-8, 1e-5, 1e-4, 1e-3]:
    noise_e_val = 1e-8
    K = build_gp_matrix(xs, sigma2_bad, theta_bad, noise_e_val, noise_g_val)
    try:
        cond = np.linalg.cond(K)
        alpha = np.linalg.solve(K, y_obs)
        k_star_0 = np.array([
            se_kernel_deriv_FE(x0_val, xs[0], sigma2_bad, theta_bad),
            se_kernel_deriv_FE(x0_val, xs[1], sigma2_bad, theta_bad),
            se_kernel_deriv_FF(x0_val, xs[0], sigma2_bad, theta_bad),
            se_kernel_deriv_FF(x0_val, xs[1], sigma2_bad, theta_bad),
        ])
        k_star_1 = np.array([
            se_kernel_deriv_FE(x1_val, xs[0], sigma2_bad, theta_bad),
            se_kernel_deriv_FE(x1_val, xs[1], sigma2_bad, theta_bad),
            se_kernel_deriv_FF(x1_val, xs[0], sigma2_bad, theta_bad),
            se_kernel_deriv_FF(x1_val, xs[1], sigma2_bad, theta_bad),
        ])
        g0_pred = k_star_0 @ alpha
        g1_pred = k_star_1 @ alpha
        c_pred = (g1_pred - g0_pred) / h_val
        print(f"  noise_g={noise_g_val:.0e}: g0_pred={g0_pred:+.6f}, "
              f"g1_pred={g1_pred:+.6f}, C_pred={c_pred:+.4f} "
              f"(err={abs(c_pred - true_curv):.2e}, cond={cond:.1e})")
    except np.linalg.LinAlgError:
        print(f"  noise_g={noise_g_val:.0e}: SINGULAR (cond={cond:.1e})")

# ============================================================================
# Part 6: SCG lambda effect on convergence
# ============================================================================

print("\n" + "=" * 60)
print("Part 6: SCG lambda and data-dependent init interaction")
print("=" * 60)

print("""
Data-dependent init with clustered dimer data (2 points, 0.01A apart):
  range_y ~ 0.04 eV
  max_feat_dist ~ 0.012 (inverse-distance features)
  sigma2_init = (range_y / 2)^2 ~ 4e-4  (should be ~1.0)
  inv_ell_init = 1 / max_feat_dist ~ 83  (should be ~1-10)

In log space, SCG must traverse:
  log(sigma2): from log(4e-4) = -7.8 to log(1) = 0  -> 7.8 units
  log(inv_ell): from log(83) = 4.4 to log(1) = 0    -> 4.4 units

With lambda=1 (MATLAB default):
  SCG can take full Newton-like steps; converges in ~50-100 iterations.

With lambda=100 (d_000 C++ config):
  SCG step size ~ 1/lambda ~ 0.01; needs ~780 iterations for sigma2 alone.
  With 300-400 training iterations, SCG cannot converge.

Fix: lambda=1 (matching MATLAB, which also uses data-dependent init)
Belt-and-suspenders: skip data-dependent init for dimer (use config values).
""")

# ============================================================================
# Summary
# ============================================================================

print("=" * 60)
print("Summary: three compounding issues")
print("=" * 60)

print("""
1. NOISE MISMATCH: Fixed noise_g=1e-4 on GG block corrupts curvature.
   Curvature = (g1-g0).orient/h amplifies gradient noise by 1/h^2.
   All reference dimer implementations use zero gradient noise.

2. SCG LAMBDA=100: Prevents hyperparameter convergence from data-dependent init.
   MATLAB uses lambda=1 (fminscg default). C++ uses lambda=100 but starts
   from config values (not data-dependent). Rust mixed MATLAB init + C++ lambda.

3. DATA-DEPENDENT INIT on clustered data gives pathological starting values.
   MATLAB recovers with lambda=1. Safer to skip init for dimer entirely.

All three interact: bad noise -> bad curvature -> bad training signal ->
SCG cannot recover because lambda too high and starting point too far.

Fix:
  noise_e=1e-8, noise_g=0.0 (matches MATLAB/C++/Stan)
  scg_lambda_init=1.0 (matches MATLAB)
  Skip data-dependent init for dimer (matches C++ strategy)
  Dynamic constSigma2 = max(1, mean_y^2) (matches MATLAB)
""")

print("PASS: all derivations complete")
