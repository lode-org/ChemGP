"""
    AbstractMoleculeKernel <: KernelFunctions.Kernel

Abstract supertype for molecular kernels that operate on inverse interatomic
distance features.

Subtypes must be callable as `k(x, y)` where `x` and `y` are flat coordinate
vectors, and must store `signal_variance`, `inv_lengthscales`, `frozen_coords`,
and `feature_params_map` fields for compatibility with [`train_model!`](@ref).

# Subtypes
- [`MolInvDistSE`](@ref): Squared exponential kernel on inverse distances
- [`MolInvDistMatern52`](@ref): Matern 5/2 kernel on inverse distances
"""
abstract type AbstractMoleculeKernel <: KernelFunctions.Kernel end
