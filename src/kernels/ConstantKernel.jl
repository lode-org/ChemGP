# ==============================================================================
# Offset (Constant) Kernel
# ==============================================================================
#
# A constant kernel k(x, y) = c for all x, y.
# Named OffsetKernel to avoid collision with KernelFunctions.ConstantKernel.
#
# When added to a molecular kernel (via MolSumKernel), this acts as a "DC
# offset" that allows the GP to model a nonzero mean energy far from training
# data. This mirrors the ConstantCF in gpr_optim.
#
# Since this returns a constant, ForwardDiff automatically produces zero
# derivatives for the energy-gradient and gradient-gradient blocks.

"""
    OffsetKernel{T<:Real} <: Kernel

Constant kernel `k(x, y) = c` for all `x, y`.

Named `OffsetKernel` to avoid collision with `KernelFunctions.ConstantKernel`.

When composed with a molecular kernel via [`MolSumKernel`](@ref), this acts as a
DC offset allowing the GP to model a nonzero mean energy far from training data.
This mirrors the `ConstantCF` in gpr_optim.

Since the kernel returns a constant, ForwardDiff automatically produces zero
derivatives for the energy-gradient and gradient-gradient blocks in
[`kernel_blocks`](@ref).

# Example
```julia
k_offset = OffsetKernel(1.0)
k_total = MolSumKernel(MolInvDistSE(1.0, [0.5], Float64[]), k_offset)
```

See also: [`MolSumKernel`](@ref)
"""
struct OffsetKernel{T<:Real} <: Kernel
    constant::T
end

OffsetKernel() = OffsetKernel(1.0)

function (k::OffsetKernel)(x::AbstractVector, y::AbstractVector)
    return k.constant
end
