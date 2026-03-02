#!/usr/bin/env python3
"""Show E[z(x)^T z(y)] = k(x,y) for RFF approximation of the SE kernel.

Tutorial T5 (Scalability: FPS + RFF): verifies Bochner's theorem and the
random Fourier feature approximation used in rff.rs::build_rff.

Random Fourier Features (Rahimi & Recht, 2007):
  For a stationary kernel k(x,y) = k(x-y), Bochner's theorem states
  that k can be written as the Fourier transform of a spectral density:
    k(r) = integral p(w) exp(i w^T r) dw

  For the SE kernel k(r) = sigma^2 * exp(-theta^2 * ||r||^2), the
  spectral density is Gaussian:
    p(w) = (1/(2*pi*theta^2))^{D/2} * exp(-||w||^2 / (4*theta^2))

  The RFF approximation samples D_rff frequencies from p(w) and
  constructs features:
    z(x) = sqrt(2*sigma^2/D_rff) * cos(W x + b)

  where W ~ N(0, 2*theta^2 I) and b ~ Uniform(0, 2*pi).
  Then E[z(x)^T z(y)] = k(x,y).

This script verifies the expectation analytically (via sympy) and
numerically (via Monte Carlo).
"""
import numpy as np
import sympy as sp

print("=" * 60)
print("RFF expectation proof for SE kernel")
print("=" * 60)

# Symbolic proof for 1D SE kernel
print("\n--- Symbolic proof (1D) ---")
w_sym = sp.Symbol("w", real=True)
r_sym = sp.Symbol("r", real=True)
theta_sym = sp.Symbol("theta", positive=True)
sigma2_sym = sp.Symbol("sigma2", positive=True)

# SE kernel
k_se = sigma2_sym * sp.exp(-theta_sym**2 * r_sym**2)

# Spectral density of SE kernel: Gaussian with variance 2*theta^2
# p(w) = (1/sqrt(4*pi*theta^2)) * exp(-w^2 / (4*theta^2))
p_w = sp.exp(-w_sym**2 / (4 * theta_sym**2)) / sp.sqrt(4 * sp.pi * theta_sym**2)

# Verify: integral of p(w) * exp(i*w*r) dw should give k(r)/sigma^2
# (the sigma^2 factor is absorbed into the RFF scaling)
fourier = sp.integrate(p_w * sp.exp(sp.I * w_sym * r_sym), (w_sym, -sp.oo, sp.oo))
fourier_simplified = sp.simplify(fourier)
print(f"\nFourier transform of p(w): {fourier_simplified}")
print(f"k(r)/sigma^2:              {sp.simplify(k_se / sigma2_sym)}")
print(f"Match: {sp.simplify(fourier_simplified - k_se / sigma2_sym) == 0}")

# RFF feature construction
print("\n--- RFF feature construction ---")
print("""
Given D_rff sampled frequencies w_j ~ N(0, 2*theta^2) and phases b_j ~ U(0, 2*pi):

  z_j(x) = c * cos(w_j * x + b_j),  c = sigma * sqrt(2/D_rff)

Then:
  E[z(x)^T z(y)] = sum_j E[z_j(x) * z_j(y)]
                  = D_rff * c^2 * E[cos(w*x + b) * cos(w*y + b)]
                  = 2*sigma^2 * E_b[E_w[cos(w*x + b) * cos(w*y + b)]]

Using cos(a)cos(b) = 0.5*(cos(a-b) + cos(a+b)):
  E_b[cos(w*x+b)*cos(w*y+b)] = 0.5*E_b[cos(w*(x-y)) + cos(w*(x+y)+2b)]
                               = 0.5*cos(w*(x-y))  (second term integrates to 0)

So:
  E[z_j(x)*z_j(y)] = c^2 * 0.5 * E_w[cos(w*(x-y))]
  D_rff * E[...] = 2*sigma^2 * 0.5 * E_w[cos(w*r)]
                  = sigma^2 * E_w[cos(w*r)]
                  = sigma^2 * integral p(w) cos(w*r) dw
                  = k(r)

The last step follows from Bochner's theorem (real part of the characteristic
function of p(w)).
""")

# Numerical Monte Carlo verification
print("=" * 60)
print("Monte Carlo verification")
print("=" * 60)

np.random.seed(42)

sigma2 = 1.0
theta = 1.5
x_test = 0.3
y_test = 1.2
r = x_test - y_test

k_true = sigma2 * np.exp(-theta**2 * r**2)

for d_rff in [10, 50, 100, 500, 1000, 5000]:
    n_trials = 1000
    estimates = []
    for _ in range(n_trials):
        # Sample frequencies from N(0, 2*theta^2)
        W = np.random.randn(d_rff) * np.sqrt(2) * theta
        b = np.random.uniform(0, 2 * np.pi, d_rff)
        c = np.sqrt(2 * sigma2 / d_rff)
        zx = c * np.cos(W * x_test + b)
        zy = c * np.cos(W * y_test + b)
        estimates.append(zx @ zy)

    mean_est = np.mean(estimates)
    std_est = np.std(estimates)
    print(f"D_rff={d_rff:5d}: E[z(x)^T z(y)] = {mean_est:.6f} +/- {std_est:.4f}  "
          f"(true = {k_true:.6f})")

# Multi-dimensional verification (D=9, like LEPS 3-atom)
print("\n--- Multi-D verification (D=9) ---")
D = 9
x_md = np.random.randn(D) * 0.5
y_md = np.random.randn(D) * 0.5
r_md = x_md - y_md
k_true_md = sigma2 * np.exp(-theta**2 * np.sum(r_md**2))

for d_rff in [50, 200, 500]:
    n_trials = 500
    estimates = []
    for _ in range(n_trials):
        W = np.random.randn(d_rff, D) * np.sqrt(2) * theta
        b = np.random.uniform(0, 2 * np.pi, d_rff)
        c = np.sqrt(2 * sigma2 / d_rff)
        zx = c * np.cos(W @ x_md + b)
        zy = c * np.cos(W @ y_md + b)
        estimates.append(zx @ zy)

    mean_est = np.mean(estimates)
    std_est = np.std(estimates)
    print(f"D_rff={d_rff:4d}: E[z^T z] = {mean_est:.6f} +/- {std_est:.4f}  "
          f"(true = {k_true_md:.6f})")

# Gradient RFF (force predictions)
print("\n--- Force prediction via RFF Jacobian ---")
print("""
For force predictions, the RFF model needs dz/dx:
  dz_j/dx_d = -c * sin(w_j^T x + b_j) * w_j_d

In matrix form: J_z = -c * diag(sin(Wx+b)) * W
Or equivalently: J_z[j,d] = -c * sin(u_j) * W[j,d]

For the MolInvDistSE kernel, the RFF operates in feature space:
  u = W * phi(x) + b
  z = c * cos(u)
  J_z = -c * diag(sin(u)) * W * J_phi

where J_phi is the inverse distance Jacobian from t3.
This chain rule is implemented in rff.rs::build_rff (lines ~70-90).
""")

# Correspondence to Rust code:
print("--- Rust correspondence ---")
print("rff.rs:34-96    Feature and Jacobian computation")
print("rff.rs:100-150  Frequency sampling from N(0, 2*theta^2*I)")
print("rff.rs:152-184  Design matrix with noise precision weighting")
print("rff.rs:195-227  RFF Gram solve: A = Z^T P Z + I")
