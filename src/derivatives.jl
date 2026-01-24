# ==============================================================================
# Derivative Block Logic
# ==============================================================================


"""
    kernel_blocks(k::kernel::Kernel, x1, x2)

Computes the full block covariance for Energy and Gradients.
 with automatic differentiation.

As far as GP / Kringing etc. are concerned it is the gradients, not forces which
matter.. Which means there's a sign change.

Returns:
- k_ee: Energy-Energy (scalar)
- k_ef: Energy-Force  (1 x D)
- k_fe: Force-Energy  (D x 1)
- k_ff: Force-Force   (D x D)
"""
function kernel_blocks(k::Kernel, x1::AbstractVector, x2::AbstractVector)
    # 1. Energy-Energy
    k_ee = k(x1, x2)

    # 2. Energy-Force (Gradient w.r.t x2)
    g_x2 = ForwardDiff.gradient(x -> k(x1, x), x2)

    # 3. Force-Energy (Gradient w.r.t x1)
    g_x1 = ForwardDiff.gradient(x -> k(x, x2), x1)

    # 4. Force-Force (Hessian)
    # C++ implementation includes a flipped sign: D12 *= -1
    # MATLAB implementation also has DK = -ma2./2.*D12;
    # The mixed derivative of the kernel (∂²k/∂x∂y) is the Covariance of Gradients.
    # This is naturally +ve
    # Action: DON'T FLIP to match C++ (ensure +ve definiteness)
    H_cross = ForwardDiff.jacobian(x -> ForwardDiff.gradient(y -> k(x, y), x2), x1)

    return k_ee, g_x2', g_x1, H_cross
end
