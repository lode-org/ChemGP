# Molecular Kernels

This tutorial explains the design of rotation-invariant molecular kernels in
ChemGP and how to choose between the available options.

## The Problem: Rotation Invariance

Molecular potential energy surfaces must be invariant to rigid body motions
(translations and rotations). A kernel operating directly on Cartesian coordinates
``k(x, x')`` would assign different covariances to rotated copies of the same
configuration, which is physically wrong.

## The Solution: Inverse Distance Features

ChemGP maps Cartesian coordinates to **inverse interatomic distances** before
computing the kernel. For ``N`` atoms, the feature vector consists of ``1/r_{ij}``
for all pairs:

```math
f(x) = \left(\frac{1}{r_{12}}, \frac{1}{r_{13}}, \ldots, \frac{1}{r_{N-1,N}}\right)
```

These features are automatically:
- **Rotationally invariant**: rotating the molecule doesn't change any ``r_{ij}``
- **Translationally invariant**: translating the molecule doesn't change any ``r_{ij}``
- **Smooth** (away from ``r = 0``): suitable for GP regression

The feature computation is implemented in [`compute_inverse_distances`](@ref),
which handles both moving-moving and moving-frozen atom pairs.

## Squared Exponential (SE) Kernel

The [`MolInvDistSE`](@ref) kernel computes:

```math
k(x, x') = \sigma^2 \exp\left(-\sum_i \theta_i^2 (f_i(x) - f_i(x'))^2\right)
```

where ``\sigma^2`` is the signal variance and ``\theta_i`` are inverse lengthscales.

Properties:
- Infinitely differentiable (``C^\infty``)
- Produces very smooth GP surfaces
- May over-smooth sharp features (e.g., repulsive walls)

```julia
k = MolInvDistSE(1.0, [0.5], Float64[])  # Isotropic, no frozen atoms
```

## Matern 5/2 Kernel

The [`MolInvDistMatern52`](@ref) kernel computes:

```math
k(x, x') = \sigma^2 \left(1 + \sqrt{5}\,d + \frac{5}{3}\,d^2\right) \exp(-\sqrt{5}\,d)
```

where ``d = \sqrt{\sum_i \theta_i^2 (f_i(x) - f_i(x'))^2}``.

Properties:
- Twice differentiable (``C^2``), which is the minimum needed for gradient observations
- Produces rougher GP surfaces than SE
- Often more appropriate for molecular PES with sharp repulsive regions

```julia
k = MolInvDistMatern52(1.0, [0.5], Float64[])
```

## Isotropic vs Type-Aware Modes

**Isotropic mode** uses a single lengthscale for all atom pairs:
```julia
k = MolInvDistSE(1.0, [0.5], Float64[])
```

**Type-aware mode** assigns different lengthscales to different atom-type pairs.
For example, in a Cu-H system, Cu-Cu, Cu-H, and H-H pairs can have independent
lengthscales:

```julia
# Define types: 2 moving Cu atoms (type 1), 1 frozen H atom (type 2)
mov_types = [1, 1]
fro_types = [2]
pair_map = [1 2; 2 3]  # Cu-Cu → param 1, Cu-H → param 2, H-H → param 3

k = MolInvDistSE(1.0, [0.5, 0.3, 0.4], frozen_coords, mov_types, fro_types, pair_map)
```

## Kernel Composition

Use [`MolSumKernel`](@ref) to combine a molecular kernel with an [`OffsetKernel`](@ref):

```julia
k_mol = MolInvDistSE(1.0, [0.5], Float64[])
k_total = MolSumKernel(k_mol, OffsetKernel(1.0))
```

The constant kernel provides a DC offset, allowing the GP to model nonzero mean
energy far from training data. This mirrors the standard `SexpatCF + ConstantCF`
setup in gpr_optim.

## Next Steps

- [Kernel Design](@ref): Practical advice on choosing and tuning kernels
- [GP Basics: Derivative Observations](@ref): Theory of the derivative observation covariance structure
