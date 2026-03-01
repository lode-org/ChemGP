# GP Basics

## Gaussian Process Regression

A GP models an unknown function as a distribution over functions,
characterized by a mean function and a kernel (covariance function).
For potential energy surfaces, the GP jointly models energies and
gradients, providing both predictions and uncertainty estimates.

Given training data `{(x_i, E_i, G_i)}` (coordinates, energy,
gradient), the GP posterior at a new point provides:

- Predicted energy and gradient (posterior mean)
- Uncertainty estimates (posterior variance)

## Kernels

ChemGP provides two kernel types via the `Kernel` enum:

### MolInvDistSE (molecular systems)

Maps Cartesian coordinates to inverse interatomic distances, then
applies a squared exponential in feature space:

```
f_ij(x) = 1 / ||x_i - x_j||
k(x, y) = sigma^2 * exp(-sum_p theta_p^2 * ||f_p(x) - f_p(y)||^2)
```

This provides rotational and translational invariance by construction,
with pair-type-specific length scales (H-H, H-C, etc.).

```rust
use chemgp_core::kernel::{Kernel, MolInvDistSE};
let kernel = Kernel::MolInvDist(MolInvDistSE::isotropic(1.0, 1.0, vec![]));
```

### CartesianSE (arbitrary surfaces)

Operates directly on raw coordinates:

```
k(x, y) = sigma^2 * exp(-theta^2 * ||x - y||^2)
```

Two hyperparameters: signal variance and a single inverse length scale.
Use for analytical surfaces (Muller-Brown, model potentials) or systems
where rotational invariance is not needed.

```rust
use chemgp_core::kernel::{Kernel, CartesianSE};
let kernel = Kernel::Cartesian(CartesianSE::new(100.0, 2.0));
```

## Training via MAP-NLL

Hyperparameters (signal variance, length scales) are optimized by
minimizing the negative log-likelihood with a MAP prior, using the
Scaled Conjugate Gradient (SCG) optimizer. Data-dependent initialization
(`init_kernel`) estimates starting hyperparameters from the training data
ranges.

## Kernel Blocks

For each pair of training points, the kernel computes four blocks:

- `k_ee`: energy-energy covariance (scalar)
- `k_ef` / `k_fe`: energy-gradient cross-covariance (D-vectors)
- `k_ff`: gradient-gradient covariance (D x D matrix)

The full covariance for N points with D-dimensional gradients is
N*(1+D) x N*(1+D). This grows quickly, which is why FPS subset
selection and RFF approximation are essential for scalability.
