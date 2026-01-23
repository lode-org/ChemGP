# ==============================================================================
# Custom Kernel Definition
# ==============================================================================

"""
    MolecularKernel
    
    A specialized Squared Exponential kernel that operates directly on 
    inverse lengthscales (θ = 1/ℓ).
    
    k(x, y) = σ² * exp( -0.5 * sum( (x_i - y_i)² * θ_i² ) )
    
    This subtypes KernelFunctions.Kernel, making it compatible with the 
    entire KernelFunctions.jl ecosystem.
"""
struct MolecularKernel{T<:Real,V<:AbstractVector{T}} <: KernelFunctions.Kernel
    signal_variance::T
    inv_lengthscales::V
end

# Functor implementation (Required for KernelFunctions.jl API)
# This is where the "Custom Distance" logic lives.
function (k::MolecularKernel)(x::AbstractVector, y::AbstractVector)
    # Explicit loop for weighted squared distance
    # This avoids allocating a temporary 'diff' vector, often faster for small D
    d2 = zero(eltype(x))
    @inbounds for i in eachindex(x)
        val = (x[i] - y[i]) * k.inv_lengthscales[i]
        d2 += val^2
    end
    return k.signal_variance * exp(-0.5 * d2)
end
