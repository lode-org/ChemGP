# ==============================================================================
# Muller-Brown Potential
# ==============================================================================
#
# The Muller-Brown potential is a standard 2D test surface for optimization
# algorithms. It has 3 local minima and 2 saddle points, making it ideal for
# testing NEB (nudged elastic band) and dimer methods.
#
# Unlike the molecular oracles (LJ, RPC), this operates on 2D coordinates
# directly, so molecular kernels (inverse distance features) are not applicable.
# Use KernelFunctions.SqExponentialKernel or similar for GP regression on this
# surface.
#
# Reference:
#   Müller, K. & Brown, L. D. (1979). Location of saddle points and minimum
#   energy paths by a constrained simplex optimization procedure.
#   Theoretica Chimica Acta, 53, 75-93.

"""
    MULLER_BROWN_A

Parameter vectors for the Muller-Brown potential (4 Gaussian terms).
"""
const MULLER_BROWN_A = [-200.0, -100.0, -170.0, 15.0]
const MULLER_BROWN_a = [-1.0, -1.0, -6.5, 0.7]
const MULLER_BROWN_b = [0.0, 0.0, 11.0, 0.6]
const MULLER_BROWN_c = [-10.0, -10.0, -6.5, 0.7]
const MULLER_BROWN_x0 = [1.0, 0.0, -0.5, -1.0]
const MULLER_BROWN_y0 = [0.0, 0.5, 1.5, 1.0]

"""
    MULLER_BROWN_MINIMA

Approximate locations of the three local minima of the Muller-Brown surface.

- Minimum A ≈ (-0.558, 1.442), E ≈ -146.7
- Minimum B ≈ (0.623, 0.028), E ≈ -108.2
- Minimum C ≈ (-0.050, 0.467), E ≈ -80.8
"""
const MULLER_BROWN_MINIMA = [
    [-0.558, 1.442],   # Deepest minimum (A)
    [0.623, 0.028],    # Second minimum (B)
    [-0.050, 0.467],   # Shallowest minimum (C)
]

"""
    MULLER_BROWN_SADDLES

Approximate locations of the two saddle points of the Muller-Brown surface.

- Saddle 1 ≈ (-0.822, 0.624), E ≈ -40.7 (between A and C)
- Saddle 2 ≈ (0.212, 0.293), E ≈ -72.2 (between B and C)
"""
const MULLER_BROWN_SADDLES = [
    [-0.822, 0.624],   # Higher saddle (between A and C)
    [0.212, 0.293],    # Lower saddle (between B and C)
]

"""
    muller_brown_energy_gradient(xy::AbstractVector) -> (E, G)

Evaluate the Muller-Brown potential and its gradient at point `xy = [x, y]`.

Returns a tuple `(E, G)` where `E` is the scalar energy and `G` is a 2-element
gradient vector `[∂E/∂x, ∂E/∂y]`.

The potential is a sum of four Gaussian terms:

```math
V(x,y) = \\sum_{k=1}^{4} A_k \\exp\\bigl[a_k(x - x_k^0)^2 + b_k(x - x_k^0)(y - y_k^0) + c_k(y - y_k^0)^2\\bigr]
```

# Example
```julia
E, G = muller_brown_energy_gradient([-0.558, 1.442])
# E ≈ -146.7 (near deepest minimum)
```
"""
function muller_brown_energy_gradient(xy::AbstractVector)
    x, y = xy[1], xy[2]

    E = 0.0
    dEdx = 0.0
    dEdy = 0.0

    @inbounds for k in 1:4
        A = MULLER_BROWN_A[k]
        ak = MULLER_BROWN_a[k]
        bk = MULLER_BROWN_b[k]
        ck = MULLER_BROWN_c[k]
        xk = MULLER_BROWN_x0[k]
        yk = MULLER_BROWN_y0[k]

        dx = x - xk
        dy = y - yk
        exponent = ak * dx^2 + bk * dx * dy + ck * dy^2
        term = A * exp(exponent)

        E += term
        dEdx += term * (2 * ak * dx + bk * dy)
        dEdy += term * (bk * dx + 2 * ck * dy)
    end

    return E, [dEdx, dEdy]
end
