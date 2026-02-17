# GP Basics: Derivative Observations

This tutorial explains the core theory behind Gaussian process regression with
derivative observations, which is the foundation of ChemGP.

## Why Derivative Observations?

In molecular simulation, each oracle call (e.g., DFT calculation) typically
returns both the energy ``E`` and the gradient ``\nabla E`` (or equivalently,
the forces ``F = -\nabla E``). A standard GP would only use the energy, but the
gradient contains rich information about the local shape of the potential energy
surface.

For a system of ``N_{\text{atoms}}`` atoms in 3D, each oracle call gives:
- 1 energy value
- ``D = 3 N_{\text{atoms}}`` gradient components

That's ``1 + D`` observations from a single expensive calculation.

## Covariance Block Structure

Given a kernel ``k(x, x')``, the GP's covariance function naturally extends
to derivatives. If ``E \sim \mathcal{GP}(0, k)``, then:

```math
\text{Cov}(E(x), E(x')) = k(x, x')
```
```math
\text{Cov}(E(x), \frac{\partial E}{\partial x'_j}) = \frac{\partial k}{\partial x'_j}
```
```math
\text{Cov}(\frac{\partial E}{\partial x_i}, \frac{\partial E}{\partial x'_j}) = \frac{\partial^2 k}{\partial x_i \partial x'_j}
```

For ``N`` training points in ``D`` dimensions, the full covariance matrix has
dimension ``N(1+D) \times N(1+D)``:

```math
K = \begin{pmatrix}
K_{EE} + \sigma_n^2 I & K_{EG} \\
K_{GE} & K_{GG} + \sigma_g^2 I
\end{pmatrix}
```

where:
- ``K_{EE}`` is ``N \times N`` (energy-energy)
- ``K_{EG}`` is ``N \times ND`` (energy-gradient)
- ``K_{GE}`` is ``ND \times N`` (gradient-energy)
- ``K_{GG}`` is ``ND \times ND`` (gradient-gradient)

In ChemGP, these blocks are computed in [`build_full_covariance`](@ref) using
automatic differentiation via ForwardDiff. The individual blocks for a pair of
points are computed by [`kernel_blocks`](@ref).

## Why This Matters

Consider a 13-atom Lennard-Jones cluster (``D = 39``):
- Without gradients: ``N`` observations per ``N`` oracle calls
- With gradients: ``40N`` observations per ``N`` oracle calls

This 40x information gain per oracle call is why GP-guided optimization can
converge in far fewer oracle calls than direct optimization methods, which is
critical when each oracle call is an expensive quantum chemistry calculation.

## Normalization

Before training, the target vector is normalized to zero mean and unit variance:

```math
\tilde{E}_i = \frac{E_i - \bar{E}}{\sigma_E}, \qquad
\tilde{G}_i = \frac{G_i}{\sigma_E}
```

The gradient scaling uses only ``\sigma_E`` (not its own statistics) because
gradients are derivatives of the energy. This is handled by [`normalize`](@ref).

## Prediction

The posterior mean at test point ``x_*`` is:

```math
\mu_* = K_* K^{-1} y
```

where ``K_*`` is the cross-covariance between the test point and the training set,
and ``y`` is the full target vector ``[\tilde{E}_1, \ldots, \tilde{E}_N, \tilde{G}_1, \ldots]``.

The posterior variance (uncertainty) is:

```math
\sigma_*^2 = K_{**} - K_* K^{-1} K_*^T
```

This is computed by [`predict_with_variance`](@ref) and is the key advantage of
GP-guided optimization: the model knows where it is uncertain, enabling
intelligent exploration of the configuration space.

## Further Reading

- [Molecular Kernels](@ref): How to define the kernel ``k`` for molecular systems
- [References](@ref references): Rasmussen & Williams (2006), Solak et al. (2003)
