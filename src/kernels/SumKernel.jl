# ==============================================================================
# Additive (Sum) Kernel
# ==============================================================================
#
# Combines two kernels additively: k(x,y) = k1(x,y) + k2(x,y).
#
# This mirrors the additive kernel composition in gpr_optim where
# SexpatCF + ConstantCF is the standard setup. The SE kernel captures
# local correlations through inverse distances, while the constant kernel
# provides a global baseline.
#
# ForwardDiff differentiates through the sum naturally since
# d/dx (k1 + k2) = dk1/dx + dk2/dx.
#
# Note: train_model! only knows how to optimize MolInvDistSE / MolInvDistMatern52
# hyperparameters directly. For MolSumKernel, train the inner kernel first, then
# compose. This is a pedagogical simplification; production code (like gpr_optim)
# optimizes all hyperparameters jointly.

"""
    MolSumKernel{K1<:Kernel, K2<:Kernel} <: Kernel

Additive kernel composition: `k(x, y) = k1(x, y) + k2(x, y)`.

This mirrors the additive kernel composition in gpr_optim where
`SexpatCF + ConstantCF` is the standard setup. ForwardDiff differentiates
through the sum naturally since `d/dx (k1 + k2) = dk1/dx + dk2/dx`.

!!! note "Training limitation"
    [`train_model!`](@ref) only optimizes hyperparameters for
    `MolInvDistSE` / `MolInvDistMatern52` directly. For `MolSumKernel`,
    train the inner molecular kernel first, then compose. Production code
    (like gpr_optim) optimizes all hyperparameters jointly.

# Example
```julia
k_se = MolInvDistSE(1.0, [0.5], Float64[])
k_total = MolSumKernel(k_se, OffsetKernel(1.0))
```

See also: [`OffsetKernel`](@ref), [`MolInvDistSE`](@ref)
"""
struct MolSumKernel{K1<:Kernel, K2<:Kernel} <: Kernel
    k1::K1
    k2::K2
end

function (k::MolSumKernel)(x::AbstractVector, y::AbstractVector)
    return k.k1(x, y) + k.k2(x, y)
end

# Note: KernelFunctions already defines +(::Kernel, ::Kernel) returning a
# KernelSum. Use MolSumKernel(k1, k2) directly for explicit composition.
