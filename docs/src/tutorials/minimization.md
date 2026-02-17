# GP-Guided Minimization

This tutorial walks through the full GP-guided geometry optimization algorithm
implemented in [`gp_minimize`](@ref).

## Algorithm Overview

The key idea is to use a GP surrogate as a cheap stand-in for the expensive oracle
(e.g., DFT). The algorithm alternates between optimizing on the GP surface and
validating with the true oracle:

1. **Sample**: Evaluate the oracle at the initial point and a few perturbations
2. **Train**: Fit the GP to accumulated data by optimizing hyperparameters
3. **Optimize**: Find the minimum on the GP surface using L-BFGS
4. **Validate**: Evaluate the oracle at the GP-predicted minimum
5. **Converge**: Check if the true gradient norm is below threshold
6. If not converged, go to step 2

## Step-by-Step Walkthrough

### Initial Sampling

The first step generates initial training data by evaluating the oracle at the
starting configuration and several random perturbations:

```julia
config = MinimizationConfig(n_initial_perturb = 4, perturb_scale = 0.1)
```

Each evaluation provides both energy and gradient, so 5 oracle calls give
`5 * (1 + D)` observations for the GP.

### GP Training

At each outer iteration, the GP is retrained on all accumulated data. The
hyperparameters (signal variance, lengthscales, noise variances) are optimized
by minimizing the negative log marginal likelihood via Nelder-Mead:

```julia
train_model!(model, iterations = 300)
```

See [`train_model!`](@ref) for details on the optimization.

### GP Surface Optimization

The optimizer minimizes the GP-predicted energy plus a soft trust region penalty:

```math
f(x) = \mu_{\text{GP}}(x) + \lambda \max(0, d(x) - r_{\text{trust}})^2
```

where ``d(x)`` is the distance to the nearest training point and ``r_{\text{trust}}``
is the trust radius. This ensures the optimizer doesn't wander into regions where
the GP has never been validated.

The optimization uses L-BFGS with analytical gradients from the GP posterior.

### Oracle Validation and Convergence

After each GP optimization, the oracle is called at the predicted minimum.
Convergence is checked on the **true** gradient norm:

```julia
config = MinimizationConfig(conv_tol = 5e-3)  # ||∇E|| < 5e-3
```

This is critical: convergence is always measured on oracle values, not GP
predictions.

## Configuration

The [`MinimizationConfig`](@ref) struct controls all algorithm parameters:

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `trust_radius` | 0.1 | Max distance from training data |
| `conv_tol` | 5e-3 | Gradient norm convergence threshold |
| `max_iter` | 500 | Max outer iterations (oracle calls) |
| `gp_train_iter` | 300 | Nelder-Mead iterations for hyperparameters |
| `n_initial_perturb` | 4 | Number of initial perturbation samples |
| `perturb_scale` | 0.1 | Scale of perturbations |
| `penalty_coeff` | 1e3 | Trust region penalty strength |
| `verbose` | true | Print progress |

## Comparison to Direct L-BFGS

Direct L-BFGS on the oracle minimizes each step using oracle evaluations.
GP-guided optimization uses the GP surrogate for most of the work:

- **Direct L-BFGS**: Needs many oracle calls for line searches
- **GP-guided**: Each outer iteration requires only 1 oracle call (the validation step)

The savings are proportional to how expensive the oracle is. For DFT calculations
that take minutes per evaluation, reducing from hundreds to tens of oracle calls
is a major practical improvement.

## Next Steps

- [GP-Dimer Saddle Point Search](@ref): Extend to saddle point search
- [Trust Regions](@ref): Details on trust region management
