# GP Basics

```{note}
This tutorial will cover the fundamentals of Gaussian Process regression
as applied to potential energy surfaces. Content to be added with MyST
markdown and Python binding examples.
```

## Gaussian Process Regression

A GP models an unknown function as a distribution over functions,
characterized by a mean function and a kernel (covariance function).
For potential energy surfaces, the GP jointly models energies and
gradients, providing both predictions and uncertainty estimates.

## The MolInvDistSE Kernel

ChemGP uses the `MolInvDistSE` kernel, which operates on inverse
interatomic distances rather than Cartesian coordinates. This provides
built-in rotational and translational invariance without data
augmentation.

## Training via MAP-NLL

Hyperparameters (signal variance, length scales) are optimized by
minimizing the negative log-likelihood with a MAP prior, using the
Scaled Conjugate Gradient (SCG) optimizer.
