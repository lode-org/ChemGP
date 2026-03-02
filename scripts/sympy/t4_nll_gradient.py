#!/usr/bin/env python3
"""Derive NLL gradient d/d(log theta_p) for MAP hyperparameter training.

Tutorial T4 (Hyperparameter Training): verifies the analytical gradient
used in nll.rs::nll_and_grad.

The MAP negative log-likelihood:
  NLL = 0.5 * y^T K^{-1} y + 0.5 * log|K| + 0.5 * N * log(2*pi) + prior

Gradient w.r.t. hyperparameter theta_j:
  dNLL/d(theta_j) = 0.5 * tr(W * dK/d(theta_j))

where W = K^{-1} - alpha * alpha^T, alpha = K^{-1} y.

In log-space (w_j = log(theta_j)):
  dNLL/d(w_j) = theta_j * dNLL/d(theta_j)
              = 0.5 * tr(W * dK/d(w_j))

where dK/d(w_j) = theta_j * dK/d(theta_j) absorbs the chain rule.
The kernel_blocks_and_hypergrads method returns dK/d(w_j) directly.

This script verifies the gradient formula with a tiny 2-point GP
and compares analytical vs finite-difference gradients.
"""
import numpy as np
import sympy as sp

print("=" * 60)
print("NLL gradient derivation (log-space)")
print("=" * 60)

# Symbolic derivation of the trace formula
print("\n--- Symbolic derivation ---")
print("""
Given:
  NLL = 0.5 * y^T K^{-1} y + 0.5 * log|K| + const

Let alpha = K^{-1} y. Then:

  d/d(theta) [y^T K^{-1} y]
    = -y^T K^{-1} (dK/dtheta) K^{-1} y
    = -alpha^T (dK/dtheta) alpha
    = -tr(alpha alpha^T dK/dtheta)

  d/d(theta) [log|K|]
    = tr(K^{-1} dK/dtheta)

Combining:
  dNLL/dtheta = 0.5 * tr((K^{-1} - alpha alpha^T) dK/dtheta)
              = 0.5 * tr(W dK/dtheta)

For log-space w = log(theta), chain rule gives:
  dNLL/dw = theta * dNLL/dtheta = 0.5 * tr(W * theta * dK/dtheta)

The Rust code computes theta * dK/dtheta in kernel_blocks_and_hypergrads
(the grad_blocks output), so the NLL gradient is just 0.5 * tr(W * dk[j]).
""")

# Numerical verification with a 2-point, 1D GP (energy-only for clarity)
print("=" * 60)
print("Numerical verification: 2-point energy-only GP")
print("=" * 60)

# Hyperparameters
log_sigma2 = 0.0  # sigma2 = 1.0
log_theta = 0.5   # theta = exp(0.5) ~ 1.65
w = np.array([log_sigma2, log_theta])

sigma2 = np.exp(w[0])
theta = np.exp(w[1])

# Training data: 2 points in 1D (energy-only for simplicity)
x_train = np.array([0.0, 1.0])
y_train = np.array([0.5, -0.3])
n = 2
noise = 0.01

# Build K
def se_kernel(x1, x2, s2, th):
    return s2 * np.exp(-th**2 * (x1 - x2)**2)

def build_K(s2, th):
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = se_kernel(x_train[i], x_train[j], s2, th)
        K[i, i] += noise
    return K

def nll(w_):
    s2 = np.exp(w_[0])
    th = np.exp(w_[1])
    K = build_K(s2, th)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(K, y_train)
    data_fit = 0.5 * y_train @ alpha
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    complexity = 0.5 * log_det
    constant = 0.5 * n * np.log(2 * np.pi)
    return data_fit + complexity + constant

# Analytical gradient
K = build_K(sigma2, theta)
alpha = np.linalg.solve(K, y_train)
K_inv = np.linalg.inv(K)
W = K_inv - np.outer(alpha, alpha)

# dK/d(log sigma2): for SE kernel, dK/d(log sigma2) = K - noise*I
dK_log_sigma2 = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        dK_log_sigma2[i, j] = se_kernel(x_train[i], x_train[j], sigma2, theta)

# dK/d(log theta): theta * dK/dtheta
# dK/dtheta = -2*theta*(x_i-x_j)^2 * k(x_i,x_j)
# dK/d(log theta) = theta * dK/dtheta = -2*theta^2*(x_i-x_j)^2 * k(x_i,x_j)
dK_log_theta = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        r2 = (x_train[i] - x_train[j])**2
        kij = se_kernel(x_train[i], x_train[j], sigma2, theta)
        dK_log_theta[i, j] = -2 * theta**2 * r2 * kij

grad_analytical = np.array([
    0.5 * np.trace(W @ dK_log_sigma2),
    0.5 * np.trace(W @ dK_log_theta),
])

# Finite-difference gradient
eps = 1e-5
grad_fd = np.zeros(2)
for j in range(2):
    w_plus = w.copy()
    w_plus[j] += eps
    w_minus = w.copy()
    w_minus[j] -= eps
    grad_fd[j] = (nll(w_plus) - nll(w_minus)) / (2 * eps)

print(f"\nw = [{w[0]:.4f}, {w[1]:.4f}]")
print(f"NLL = {nll(w):.6f}")
print(f"\nAnalytical gradient: [{grad_analytical[0]:.8f}, {grad_analytical[1]:.8f}]")
print(f"Finite-diff gradient: [{grad_fd[0]:.8f}, {grad_fd[1]:.8f}]")
print(f"Max absolute error: {np.max(np.abs(grad_analytical - grad_fd)):.2e}")
print(f"Match: {np.allclose(grad_analytical, grad_fd, atol=1e-6)}")

# MAP prior contribution
print("\n--- MAP prior ---")
print("""
MAP adds a Gaussian prior on each log-hyperparameter:
  prior_penalty = 0.5 * sum_j (w_j - w_prior_j)^2 / prior_var_j

Gradient contribution:
  d(prior)/d(w_j) = (w_j - w_prior_j) / prior_var_j

This is added directly to grad[j] in nll.rs (line 199).
""")

# Log-barrier contribution
print("--- Log-barrier ---")
print("""
Upper barrier on sigma^2 (matching C++ gpr_optim):
  barrier = -strength * ln(max_log_sigma2 - w[0])

where max_log_sigma2 = ln(2), strength = min(1e-4 + 1e-3*n, 0.5).

Gradient:
  d(barrier)/d(w[0]) = +strength / (max_log_sigma2 - w[0])

This prevents sigma^2 from growing unboundedly.
See nll.rs lines 165-175, 202-203.
""")

# Correspondence to Rust code:
print("--- Rust correspondence ---")
print("nll.rs:140-158  NLL computation (data_fit + complexity + constant)")
print("nll.rs:160-163  MAP prior contribution")
print("nll.rs:165-175  Upper barrier on sigma^2")
print("nll.rs:177-186  W matrix construction")
print("nll.rs:188-200  Gradient via tr(W * dK)")
print("nll.rs:202-203  Barrier gradient")
