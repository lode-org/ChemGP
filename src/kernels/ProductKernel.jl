# ==============================================================================
# Multiplicative (Product) Kernel
# ==============================================================================
#
# Combines two kernels multiplicatively: k(x,y) = k1(x,y) * k2(x,y).
#
# Product kernels are useful when one kernel captures a global envelope
# (e.g., a Matern kernel for overall smoothness) and another captures
# local features (e.g., periodic or distance-based structure).
#
# ForwardDiff differentiates through the product via the chain rule:
# d/dx (k1 * k2) = k1 * dk2/dx + dk1/dx * k2.

"""
    MolProductKernel{K1<:Kernel, K2<:Kernel} <: Kernel

Multiplicative kernel composition: `k(x, y) = k1(x, y) * k2(x, y)`.

ForwardDiff handles the product rule automatically for derivative
observations: `∂(k1·k2)/∂x = k1·∂k2/∂x + ∂k1/∂x·k2`.

# Example
```julia
k_se = MolInvDistSE(1.0, [0.5], Float64[])
k_offset = OffsetKernel(2.0)
k_prod = MolProductKernel(k_se, k_offset)  # Scales SE by constant factor
```

See also: [`MolSumKernel`](@ref), [`OffsetKernel`](@ref)
"""
struct MolProductKernel{K1<:Kernel,K2<:Kernel} <: Kernel
    k1::K1
    k2::K2
end

function (k::MolProductKernel)(x::AbstractVector, y::AbstractVector)
    return k.k1(x, y) * k.k2(x, y)
end
