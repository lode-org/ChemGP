# Algorithm Pseudocode

Pseudocode for all GP-guided optimization methods in ChemGP.

## GP Minimization ([`gp_minimize`](@ref))

```
Input: oracle, x_init, kernel, config
Output: MinimizationResult

1. Generate initial training data (perturbations around x_init)
2. for outer_iter = 1:max_outer_iter
   a. Train GP on accumulated data
   b. Inner loop (on GP surface):
      - Predict gradient at current x
      - Take gradient descent step
      - Check: |G_gp| < T_gp → break
      - Check: trust radius exceeded → scale step, break
   c. Evaluate oracle at new position
   d. Add to training data
   e. Check: |G_true| < T_force → CONVERGED
```

## GP-Dimer ([`gp_dimer`](@ref))

```
Input: oracle, x_init, orient_init, kernel, config
Output: DimerResult

1. Initialize DimerState(R, orient, dimer_sep)
2. Generate initial training data
3. for outer_iter = 1:max_outer_iter
   a. Train GP on accumulated data
   b. Reset L-BFGS/CG state
   c. Inner loop (on GP surface):
      - Rotate: align with lowest curvature mode
        (:lbfgs/:cg → search direction + Newton angle opt)
        (:simple → direct angle estimate)
      - Translate: modified force F_trans = G - 2(G·n̂)n̂
        Negative curvature: L-BFGS step
        Positive curvature: fixed step along -G·n̂
      - Check: |F_trans| < T_gp → break
      - Check: trust radius → break
   d. Oracle at converged position (+ image 1)
   e. Check: |F_true| < T_dimer AND C < 0 → CONVERGED
```

## OTGPD ([`otgpd`](@ref))

```
Input: oracle, x_init, orient_init, kernel, config
Output: OTGPDResult

Phase 1 — Initial Rotation (optional):
  for rot = 1:max_initial_rot
    Evaluate oracle at midpoint + image 1
    Compute F_rot, curvature C
    Modified Newton angle optimization (parabolic fit):
      dθ = 0.5 * atan(|F_rot|/|C|)
      Trial rotation, evaluate oracle
      Fit F(θ) = a₁cos2θ + b₁sin2θ
      θ* = 0.5 * atan(b₁/a₁)
    Check: dθ < T_angle → break

Phase 2 — Main GP Loop:
  for outer_iter = 1:max_outer_iter
    Prune training data (if max_training_points > 0)
    Train GP
    Adaptive threshold: T_gp = max(min(F_history)/divisor, T/10)
    Inner loop (same as gp_dimer inner loop)
    Oracle evaluation (+ image 1 if eval_image1)
    Check: |F_true| < T_dimer AND C < 0 → CONVERGED
```

## Standard NEB ([`neb_optimize`](@ref))

```
Input: oracle, x_start, x_end, config
Output: NEBResult

1. Linear interpolation → N images
2. Evaluate oracle at all images
3. for iter = 1:max_iter
   a. Compute tangent at each image (improved method):
      - Monotonic increase: τ = R_{i+1} - R_i
      - Monotonic decrease: τ = R_i - R_{i-1}
      - Extremum: energy-weighted average
   b. NEB force at each image:
      - Standard: F = -G⊥ + F_spring∥
      - Climbing image: F = -G + 2(G·τ̂)τ̂
   c. Steepest descent update
   d. Re-evaluate oracle at all images
   e. Check: max|F| < conv_tol → CONVERGED
```

## GP-NEB-AIE ([`gp_neb_aie`](@ref))

```
Input: oracle, x_start, x_end, kernel, config
Output: NEBResult

1. Linear interpolation, evaluate oracle at all images
2. for outer_iter = 1:max_outer_iter
   a. Compute true NEB forces, check convergence
   b. Train GP on accumulated data
   c. Inner loop: relax path on GP surface
      - GP-predict E, G at all intermediate images
      - Compute NEB forces from GP predictions
      - Steepest descent on GP
   d. Evaluate oracle at ALL new image positions
   e. Add to training data
```

## GP-NEB-OIE ([`gp_neb_oie`](@ref))

```
Input: oracle, x_start, x_end, kernel, config
Output: NEBResult

1. Linear interpolation, evaluate midpoint
2. for outer_iter = 1:max_outer_iter
   a. Train GP on accumulated data
   b. Predict E, G at all images from GP
   c. Compute predictive variance at all images
   d. Select image i* = argmax(variance_E)
   e. Evaluate oracle at image i*
   f. Inner loop: relax path on GP surface
   g. Check convergence (all images evaluated + max|F| < tol)
```

## Modified Newton Rotation

```
Input: state, model, F_rot_direction, config
Output: curvature_estimate

1. C₀ = curvature at current orientation
2. dθ = 0.5 * atan(|F_rot|/|C₀|)
3. Trial rotation: orient_trial = cos(dθ)·orient + sin(dθ)·F̂_rot
4. Evaluate GP at trial image 1 position
5. Parabolic fit:
   F₀ = F_rot·F̂_rot  (projection at θ=0)
   F_dθ = F_rot_trial·F̂_rot_trial  (projection at θ=dθ)
   a₁ = (F_dθ - F₀·cos2dθ) / sin2dθ
   b₁ = -F₀/2
6. θ* = 0.5 * atan(b₁/a₁), adjust for minimum
7. Apply rotation: orient_new = cos(θ*)·orient + sin(θ*)·F̂_rot
```

## L-BFGS Two-Loop Recursion

```
Input: history (s,y pairs), gradient g
Output: search direction d

1. q = g
2. for i = m, m-1, ..., 1:   (newest to oldest)
   α_i = (s_i·q) / (y_i·s_i)
   q = q - α_i·y_i
3. γ = (s_m·y_m) / (y_m·y_m)  (Rayleigh quotient scaling)
4. d = γ·q
5. for i = 1, 2, ..., m:      (oldest to newest)
   β = (y_i·d) / (y_i·s_i)
   d = d + (α_i - β)·s_i
6. return d
```
