# Trust Regions

The GP model is only reliable near its training data. Trust region management
ensures that optimization steps don't stray into regions where the GP is
uncalibrated.

## Distance Metrics

ChemGP provides two distance metrics for measuring how "far" a configuration
is from training data:

### Euclidean Distance

The default in [`min_distance_to_data`](@ref):

```math
d(x, x') = \|x - x'\|_2
```

Simple and fast, but **not** rotationally invariant. Two configurations that
differ only by a rotation will have nonzero Euclidean distance.

### MAX\_1D\_LOG Distance

The primary metric from gpr_optim, implemented in [`max_1d_log_distance`](@ref):

```math
d(x_1, x_2) = \max_k \left|\log\frac{r^{(1)}_k}{r^{(2)}_k}\right|
```

where ``r^{(1)}_k`` and ``r^{(2)}_k`` are corresponding interatomic distances.

Properties:
- Rotationally and translationally invariant
- Sensitive to large relative changes in any single atom pair
- Zero when configurations are identical (up to rigid body motion)

## Soft Trust Region Penalty

In [`gp_minimize`](@ref), the trust region is enforced as a soft quadratic penalty
on the GP objective:

```math
f(x) = \mu_{\text{GP}}(x) + \lambda \max(0, d_{\min}(x) - r)^2
```

where ``d_{\min}(x)`` is the distance to the nearest training point, ``r`` is the
trust radius, and ``\lambda`` is the penalty coefficient.

The gradient of the penalty pushes the optimizer back toward the training data
when it tries to step outside the trust region.

## Interatomic Distance Ratio Check

The [`check_interatomic_ratio`](@ref) function provides a complementary physical
sanity check. It verifies that no interatomic distance has changed by more than
a factor of `ratio_limit` relative to the nearest training point:

```julia
# Reject if any r_ij changes by more than 50%
check_interatomic_ratio(x_new, X_train, 2/3)
```

This prevents the optimizer from distorting the molecule beyond recognition,
which the distance-based trust region alone might not catch.

## Tuning the Trust Radius

| Symptom | Adjustment |
|:--------|:-----------|
| Too many oracle calls | Increase `trust_radius` |
| Optimizer diverges or energy explodes | Decrease `trust_radius` |
| Optimizer oscillates near boundary | Increase `penalty_coeff` |
| Progress is very slow | Decrease `penalty_coeff` |

A typical starting value is `trust_radius = 0.1` with `penalty_coeff = 1e3`.
For systems with stiffer potentials (e.g., covalent bonds), a smaller trust
radius (0.05) may be needed. For softer systems (e.g., van der Waals clusters),
larger values (0.2–0.3) may be appropriate.
