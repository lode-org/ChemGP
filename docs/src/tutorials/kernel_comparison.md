# Kernel Comparison

This tutorial compares the three molecular kernel families available in ChemGP
and helps you choose the right one for your problem.

## The Matern Family

All three kernels operate on inverse interatomic distance features ``f_i = 1/r_i``,
ensuring rotational and translational invariance. They differ in their smoothness:

| Kernel | Formula | Smoothness | Differentiability |
|:-------|:--------|:-----------|:------------------|
| [`MolInvDistMatern32`](@ref) | ``\sigma^2 (1 + \sqrt{3}d) e^{-\sqrt{3}d}`` | C¹ | Once |
| [`MolInvDistMatern52`](@ref) | ``\sigma^2 (1 + \sqrt{5}d + \frac{5}{3}d^2) e^{-\sqrt{5}d}`` | C² | Twice |
| [`MolInvDistSE`](@ref) | ``\sigma^2 e^{-d^2/2}`` | C∞ | Infinitely |

where ``d = \sqrt{\sum_i (\theta_i (f_i(x) - f_i(y)))^2}`` is the weighted
Euclidean distance in feature space.

## When to Use Each

### Matern 3/2: Rough Surfaces
The Matern 3/2 kernel produces the roughest GP surfaces. Use it when:
- The potential energy surface has sharp features (repulsive walls, bond-breaking)
- You need the GP to capture rapid changes without over-smoothing
- Working with highly reactive or strained molecular configurations

### Matern 5/2: Balanced (Recommended Default)
The Matern 5/2 kernel is twice differentiable, matching the smoothness typically
expected for molecular PES. Use it when:
- You want a good default for general molecular optimization
- The surface has moderate features but is not infinitely smooth
- Following the approach of [Koistinen et al. (2017)](@ref references)

### Squared Exponential: Very Smooth Surfaces
The SE kernel assumes infinite differentiability. Use it when:
- The surface is known to be very smooth (e.g., noble gas clusters)
- You are far from repulsive walls and bond-breaking regions
- You want maximum smoothness in the GP predictions

## Example: Comparing on a Lennard-Jones Dimer

```julia
using ChemGP

# Generate training data: 2-atom LJ at various separations
td = TrainingData(6)  # 2 atoms × 3 coords
for r in 1.0:0.2:2.5
    x = [0.0, 0.0, 0.0, r, 0.0, 0.0]
    E, G = lj_energy_gradient(x)
    add_point!(td, x, E, G)
end

y_gp, y_mean, y_std = normalize(td)

# Train three models
kernels = [
    ("Matern 3/2", MolInvDistMatern32(1.0, [1.0], Float64[])),
    ("Matern 5/2", MolInvDistMatern52(1.0, [1.0], Float64[])),
    ("SE",         MolInvDistSE(1.0, [1.0], Float64[])),
]

for (name, k) in kernels
    model = GPModel(k, td.X, y_gp; noise_var=1e-4, grad_noise_var=1e-4)
    train_model!(model; iterations=200)

    # Predict at an unseen point
    x_test = reshape([0.0, 0.0, 0.0, 1.35, 0.0, 0.0], :, 1)
    pred = predict(model, x_test)
    E_pred = pred[1] * y_std + y_mean
    println("$name: E(r=1.35) = $(round(E_pred, digits=4))")
end
```

All three kernels should give similar predictions near training data. The differences
become apparent at extrapolation points farther from the training set.

## Smoothness and Derivative Observations

Since ChemGP uses derivative observations (gradients from the oracle), the kernel's
differentiability class directly affects the GP posterior:

- **Matern 3/2 (C¹)**: Gradient predictions are continuous but their derivatives
  (Hessians) are not. This is fine for gradient-based optimization but may cause
  issues for methods needing smooth Hessians.
- **Matern 5/2 (C²)**: Both energy and gradient predictions are smooth. Hessian
  predictions exist and are continuous. Good for L-BFGS and dimer methods.
- **SE (C∞)**: All derivatives exist and are smooth. Most suitable when the
  surrogate itself will be heavily differentiated.

## Next Steps

- [Kernel Design](@ref): Learn about type-aware kernels and composition
- [GP-Guided Minimization](@ref): Apply kernels in optimization
- [Dimer Method](@ref): See how kernel choice affects saddle point search
