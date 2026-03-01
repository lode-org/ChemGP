# Kernel Design

```{note}
Detailed derivations are in the internal notes. This page summarizes
the key design decisions.
```

## MolInvDistSE

The `MolInvDistSE` kernel maps Cartesian coordinates to inverse
interatomic distances, then applies a squared exponential kernel in
feature space. This provides:

- Rotational and translational invariance by construction
- Pair-type-specific length scales (e.g., H-H vs H-C distances)
- Efficient derivative blocks (EE, EF, FE, FF) via the chain rule

## Kernel Blocks

For two training points, the kernel computes four blocks:

- `k_ee`: energy-energy covariance
- `k_ef` / `k_fe`: energy-gradient cross-covariance
- `k_ff`: gradient-gradient covariance (D x D matrix)

The full covariance for N points with D-dimensional gradients is
N*(1+D) x N*(1+D).

## Hyperparameter Gradients

Analytical gradients of the NLL with respect to log-space
hyperparameters (`log(sigma2)`, `log(theta_p)`) enable efficient
SCG optimization. The FF block hyperparameter gradient uses the
PURE theta^2 derivative (not including the kval chain-rule contribution).
