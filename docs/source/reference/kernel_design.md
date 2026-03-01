# Kernel Design

## The Kernel Enum

ChemGP uses a `Kernel` enum to dispatch between two kernel types:

```rust
pub enum Kernel {
    MolInvDist(MolInvDistSE),
    Cartesian(CartesianSE),
}
```

All optimizers, training, prediction, and RFF code accept `&Kernel`
and dispatch through unified methods (`kernel_blocks`,
`kernel_blocks_and_hypergrads`, `with_params`, etc.).

## MolInvDistSE

The `MolInvDistSE` kernel maps Cartesian coordinates to inverse
interatomic distances, then applies a squared exponential kernel in
feature space. This provides:

- Rotational and translational invariance by construction
- Pair-type-specific length scales (e.g., H-H vs H-C distances)
- Efficient derivative blocks (EE, EF, FE, FF) via the chain rule

The feature transformation is:

```
f_ij(x) = 1 / ||x_i - x_j||
```

with Jacobian computed analytically. The kernel is:

```
k(x, y) = sigma^2 * exp(-sum_p theta_p^2 * ||f_p(x) - f_p(y)||^2)
```

where the sum runs over pair types `p`, each with its own inverse
length scale `theta_p`.

Use this kernel for molecular systems where invariance matters.

## CartesianSE

The `CartesianSE` kernel operates directly on raw coordinates:

```
k(x, y) = sigma^2 * exp(-theta^2 * ||x - y||^2)
```

Two hyperparameters: signal variance `sigma^2` and a single inverse
length scale `theta`. The features are simply the coordinates
themselves (identity Jacobian), so the kernel blocks simplify to:

- `k_ee = sigma^2 * exp(-theta^2 * d^2)`
- `k_ef[d] = 2 * theta^2 * r_d * k_ee`
- `k_fe[d] = -k_ef[d]`
- `k_ff[d1, d2] = 2 * theta^2 * k_ee * (delta(d1,d2) - 2 * theta^2 * r_d1 * r_d2)`

where `r_d = x_d - y_d` and `d = ||x - y||`.

Use this kernel for 2D/3D analytical surfaces (Muller-Brown, model
potentials) or systems where rotational invariance is not needed.

## Kernel Blocks

For two points, the kernel computes four blocks:

- `k_ee`: energy-energy covariance (scalar)
- `k_ef` / `k_fe`: energy-gradient cross-covariance (D-vectors)
- `k_ff`: gradient-gradient covariance (D x D matrix)

The full covariance for N points with D-dimensional gradients is
N*(1+D) x N*(1+D).

## Hyperparameter Gradients

Analytical gradients of the NLL with respect to log-space
hyperparameters (`log(sigma2)`, `log(theta_p)`) enable efficient
SCG optimization. Both kernel types provide `kernel_blocks_and_hypergrads`
returning the blocks and their derivatives simultaneously.

For MolInvDistSE, the FF block hyperparameter gradient uses the
PURE theta^2 derivative (not including the kval chain-rule contribution).

## Data-Dependent Initialization

`init_kernel()` dispatches to kernel-specific initialization:

- `MolInvDistSE`: computes features, estimates signal variance from
  energy range, length scales from feature distances (GPstuff approach)
- `CartesianSE`: estimates sigma from energy range, theta from
  max pairwise coordinate distance

Both use `NORMINV_075` scaling for the 75th percentile of a normal
distribution, following the MATLAB GPstuff initialization strategy.

## RFF Feature Modes

The RFF approximation also dispatches on kernel type via `FeatureMode`:

- `InverseDistances`: computes inverse distance features + finite
  difference Jacobian (for MolInvDistSE)
- `Cartesian`: uses raw coordinates with identity Jacobian (for
  CartesianSE)
