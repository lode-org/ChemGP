# ==============================================================================
# Cartesian Squared Exponential Kernel
# ==============================================================================
#
# Standard isotropic SE kernel operating directly on Cartesian coordinates.
# Unlike MolInvDistSE which maps to inverse distance features (rotationally
# invariant, but rank-deficient gradient blocks for small molecules),
# CartesianSE uses raw coordinate differences and produces full-rank
# gradient-gradient covariance blocks.
#
# This is the kernel used in Koistinen et al. (2017) for GP-NEB and
# GP-dimer on molecular systems. The MATLAB reference (gpcf_sexp in
# GPstuff) uses the same formulation.
#
# Reference:
#   Koistinen, O.-P. et al. (2017). Nudged elastic band calculations
#   accelerated with Gaussian process regression. J. Chem. Phys., 147, 152720.

"""
    CartesianSE{T} <: Kernel

Isotropic squared exponential kernel on Cartesian coordinates:

    k(x, y) = signal_variance * exp(-||x - y||^2 / (2 * lengthscale^2))

Unlike [`MolInvDistSE`](@ref), this kernel operates directly on flat coordinate
vectors without mapping to inverse distance features. The gradient-gradient
covariance block is full-rank in the coordinate dimension, avoiding the
conditioning issues that arise when the number of features is smaller than
the number of coordinates (e.g., 3 inverse distances for 9D coords of a
3-atom system).

For GP-NEB where all images share the same reference frame (no rotation),
rotational invariance is not needed, making this kernel the natural choice.

Matches `gpcf_sexp` in the MATLAB GPstuff reference implementation.

# Fields
- `signal_variance::T`: Output variance (amplitude squared)
- `lengthscale::T`: Characteristic length scale

# Example
```julia
k = CartesianSE(1.0, 0.5)   # sigma^2 = 1.0, ell = 0.5
k([1.0, 0.0], [0.0, 1.0])   # evaluates the kernel
```

See also: [`MolInvDistSE`](@ref), [`OffsetKernel`](@ref)
"""
struct CartesianSE{T<:Real} <: Kernel
    signal_variance::T
    lengthscale::T
end

CartesianSE() = CartesianSE(1.0, 1.0)

function (k::CartesianSE)(x::AbstractVector, y::AbstractVector)
    d2 = sum((xi - yi)^2 for (xi, yi) in zip(x, y))
    return k.signal_variance * exp(-d2 / (2 * k.lengthscale^2))
end
