#!/usr/bin/env python3
"""Derive LCB scoring for GP-accelerated optimization.

Tutorial T7 (NEB + GP): verifies the Lower Confidence Bound acquisition
function used for image selection in OIE and exploration in minimization.

Standard LCB (minimization):
  LCB(x) = mu(x) - kappa * sigma(x)

Force-weighted LCB (NEB OIE):
  score(i) = |F_perp(i)| + kappa * sigma_perp(i)

where:
  F_perp = F - (F . tau) * tau     (force perpendicular to path tangent)
  sigma_perp = sqrt(sum_d var_d * (delta_d - (delta . tau) * tau_d)^2)
             ~ projection of gradient variance onto perpendicular subspace

The OIE scoring selects the image that is both far from the MEP (large
|F_perp|) and uncertain (large sigma_perp).
"""
import numpy as np
import sympy as sp

print("=" * 60)
print("LCB acquisition function derivation")
print("=" * 60)

# Standard LCB for minimization
print("\n--- Standard LCB (minimization) ---")
print("""
The GP posterior at point x provides:
  mu(x)    = k(x,X) K^{-1} y          (predicted energy)
  sigma(x) = sqrt(k(x,x) - k(x,X) K^{-1} k(X,x))  (uncertainty)

Lower Confidence Bound:
  LCB(x) = mu(x) - kappa * sigma(x)

Interpretation: LCB is a pessimistic estimate of the energy. Points where
the GP predicts low energy AND has low uncertainty score well. Points where
the GP predicts low energy but has high uncertainty are penalized.

kappa controls the exploration-exploitation tradeoff:
  kappa = 0: pure exploitation (trust GP mean)
  kappa > 0: exploration (prefer uncertain regions)

In ChemGP minimization (minimize.rs), kappa is fixed at 1.0 and only
activates when sigma > 1e-4 (avoid noise in well-sampled regions).
""")

# Force-weighted LCB for NEB OIE
print("--- Force-weighted LCB (NEB OIE) ---")

# Symbolic derivation with D=2 for clarity
D = 2
F = sp.symbols([f"F{d}" for d in range(D)])
tau = sp.symbols([f"tau{d}" for d in range(D)])
var = sp.symbols([f"var{d}" for d in range(D)], positive=True)
kappa = sp.Symbol("kappa", positive=True)

# Perpendicular force: F_perp = F - (F . tau) * tau
F_dot_tau = sum(F[d] * tau[d] for d in range(D))
F_perp = [F[d] - F_dot_tau * tau[d] for d in range(D)]
F_perp_norm = sp.sqrt(sum(fp**2 for fp in F_perp))

print(f"\nF_perp[d] = F[d] - (F . tau) * tau[d]")
print(f"|F_perp| = sqrt(sum(F_perp[d]^2))")

# Perpendicular variance
# For each gradient component d, the variance var_d is the GP uncertainty.
# The perpendicular projection of the variance vector:
#   sigma_perp = sqrt(sum_d var_d * P_d^2)
# where P_d = delta_d - (delta . tau) * tau_d projects onto the perp subspace.
# In practice, for isotropic variance: sigma_perp ~ sigma * sqrt(1 - (tau . tau_var)^2)
# But the actual implementation uses per-component variance.

print("\nPerpendicular variance (simplified):")
print("  sigma_perp^2 = sum_d var_d * (1 - tau_d^2)")
print("  (assumes variance is diagonal and projection is onto D-1 dims)")

# Full score
print("\nOIE image score:")
print("  score(i) = |F_perp(i)| + kappa * sigma_perp(i)")
print("""
Priority cascade (neb_oie.rs):
  1. Early-stop image (force increased after last oracle) -> immediate re-eval
  2. Climbing image (if CI convergence pending) -> priority
  3. LCB score on UNEVALUATED images only -> select highest score
""")

# Numerical example
print("\n" + "=" * 60)
print("Numerical example: 5-image NEB band")
print("=" * 60)

np.random.seed(42)
n_images = 5
D = 3  # 3D for illustration

# Simulated NEB forces and uncertainties
tangents = np.random.randn(n_images, D)
tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

forces = np.random.randn(n_images, D) * 0.5
variances = np.abs(np.random.randn(n_images, D)) * 0.1

kap = 2.0
evaluated = {0, 2, 4}  # images 1, 3 not yet evaluated

print(f"\nkappa = {kap}")
print(f"Evaluated images: {evaluated}")
print(f"\n{'Image':>5} {'|F_perp|':>10} {'sigma_perp':>10} {'score':>10} {'status':>12}")
print("-" * 55)

for i in range(n_images):
    # Perpendicular force
    f = forces[i]
    t = tangents[i]
    f_dot_t = np.dot(f, t)
    f_perp = f - f_dot_t * t
    f_perp_norm = np.linalg.norm(f_perp)

    # Perpendicular variance
    v = variances[i]
    # Project each variance component
    sigma_perp = np.sqrt(np.sum(v * (1 - t**2)))

    score = f_perp_norm + kap * sigma_perp
    status = "evaluated" if i in evaluated else "CANDIDATE"
    print(f"{i:5d} {f_perp_norm:10.4f} {sigma_perp:10.4f} {score:10.4f} {status:>12}")

# Select best unevaluated
scores = []
for i in range(n_images):
    if i not in evaluated:
        f = forces[i]
        t = tangents[i]
        f_perp = f - np.dot(f, t) * t
        v = variances[i]
        sigma_perp = np.sqrt(np.sum(v * (1 - t**2)))
        scores.append((i, np.linalg.norm(f_perp) + kap * sigma_perp))

best = max(scores, key=lambda x: x[1])
print(f"\nSelected image: {best[0]} (score = {best[1]:.4f})")

print("""
The selected image has the highest combination of NEB force magnitude
and GP uncertainty among unevaluated images. Evaluating it with the
oracle provides the most information for improving the NEB band.
""")

# Correspondence to Rust code:
print("--- Rust correspondence ---")
print("neb_oie.rs: LCB acquisition in select_image_to_evaluate()")
print("neb_oie.rs: Priority cascade (early-stop > CI > LCB)")
print("minimize.rs: Standard LCB in inner optimization (sigma gate)")
