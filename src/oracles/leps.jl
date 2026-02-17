# ==============================================================================
# LEPS Potential (London-Eyring-Polanyi-Sato)
# ==============================================================================
#
# The LEPS potential is a 3-atom collinear reaction surface commonly used
# for testing saddle point search and NEB methods. It describes the reaction
# A + BC -> AB + C using Morse-type pair potentials with Sato corrections.
#
# Unlike Muller-Brown (2D cartesian), LEPS operates on molecular coordinates
# (3 atoms × 3D = 9 coordinates), so molecular kernels (inverse distance
# features) can be used directly.
#
# Reference:
#   Henkelman, G. & Jónsson, H. (1999). A dimer method for finding saddle
#   points on high dimensional potential energy surfaces using only first
#   derivatives. J. Chem. Phys., 111, 7010.
#
#   The parameterization follows cstein/neb (GitHub) and Henkelman's test cases.

# --- Parameters ---

const LEPS_ALPHA = 1.942       # Morse range parameter (Å⁻¹)
const LEPS_R_E   = 0.742       # Equilibrium bond length (Å)
const LEPS_D_AB  = 4.746       # Dissociation energy A-B (eV)
const LEPS_D_BC  = 4.746       # Dissociation energy B-C (eV)
const LEPS_D_AC  = 3.445       # Dissociation energy A-C (eV)
const LEPS_S_AB  = 0.05        # Sato parameter A-B
const LEPS_S_BC  = 0.30        # Sato parameter B-C
const LEPS_S_AC  = 0.05        # Sato parameter A-C

# --- Helper functions ---

"""
Coulomb integral Q(r, D) for LEPS potential.
Returns (Q, dQ/dr).
"""
function _leps_Q(r::Float64, D::Float64)
    v = LEPS_ALPHA * (r - LEPS_R_E)
    exp_v = exp(-v)
    exp_2v = exp(-2v)
    Q = 0.5 * D * (1.5 * exp_2v - exp_v)
    dQ = 0.5 * D * LEPS_ALPHA * (-3.0 * exp_2v + exp_v)
    return Q, dQ
end

"""
Exchange integral J(r, D) for LEPS potential.
Returns (J, dJ/dr).
"""
function _leps_J(r::Float64, D::Float64)
    v = LEPS_ALPHA * (r - LEPS_R_E)
    exp_v = exp(-v)
    exp_2v = exp(-2v)
    J = 0.25 * D * (exp_2v - 6.0 * exp_v)
    dJ = 0.25 * D * LEPS_ALPHA * (6.0 * exp_v - 2.0 * exp_2v)
    return J, dJ
end

"""
    leps_energy_gradient(x::AbstractVector) -> (E, G)

Evaluate the LEPS potential and gradient for a 3-atom system.

The input `x` is a 9-element flat coordinate vector `[x₁,y₁,z₁, x₂,y₂,z₂, x₃,y₃,z₃]`
representing atoms A, B, C. The energy depends only on the three pairwise distances
`r_AB`, `r_BC`, `r_AC`.

Returns `(E, G)` where `E` is the energy in eV and `G` is the 9-element gradient.

# Example
```julia
# Collinear A-B-C with r_AB=0.742, r_BC=3.0 (reactant: A bonded to B, C far away)
x = [0.0, 0.0, 0.0,  0.742, 0.0, 0.0,  3.742, 0.0, 0.0]
E, G = leps_energy_gradient(x)
```

See also: [`leps_energy_gradient_2d`](@ref)
"""
function leps_energy_gradient(x::AbstractVector)
    # Extract atom positions
    rA = @view x[1:3]
    rB = @view x[4:6]
    rC = @view x[7:9]

    # Pairwise displacement vectors and distances
    dAB = rB - rA
    dBC = rC - rB
    dAC = rC - rA

    r_AB = norm(dAB)
    r_BC = norm(dBC)
    r_AC = norm(dAC)

    # Unit vectors (for gradient chain rule)
    uAB = r_AB > 1e-12 ? dAB / r_AB : zeros(3)
    uBC = r_BC > 1e-12 ? dBC / r_BC : zeros(3)
    uAC = r_AC > 1e-12 ? dAC / r_AC : zeros(3)

    # Sato scaling factors
    opAB = 1.0 / (1.0 + LEPS_S_AB)
    opBC = 1.0 / (1.0 + LEPS_S_BC)
    opAC = 1.0 / (1.0 + LEPS_S_AC)

    # Coulomb and exchange integrals
    Q_AB, dQ_AB = _leps_Q(r_AB, LEPS_D_AB)
    Q_BC, dQ_BC = _leps_Q(r_BC, LEPS_D_BC)
    Q_AC, dQ_AC = _leps_Q(r_AC, LEPS_D_AC)

    J_AB, dJ_AB = _leps_J(r_AB, LEPS_D_AB)
    J_BC, dJ_BC = _leps_J(r_BC, LEPS_D_BC)
    J_AC, dJ_AC = _leps_J(r_AC, LEPS_D_AC)

    # Scaled integrals
    Qs = Q_AB * opAB + Q_BC * opBC + Q_AC * opAC
    jAB = J_AB * opAB
    jBC = J_BC * opBC
    jAC = J_AC * opAC

    Js = jAB^2 + jBC^2 + jAC^2 - jAB * jBC - jBC * jAC - jAB * jAC
    sqrtJ = sqrt(max(Js, 1e-30))

    E = Qs - sqrtJ

    # --- Gradient ---
    # dE/dr_ij = dQs/dr_ij - (1/2sqrtJ) * dJs/dr_ij

    # dQs/dr_AB, etc.
    dQs_dAB = dQ_AB * opAB
    dQs_dBC = dQ_BC * opBC
    dQs_dAC = dQ_AC * opAC

    # dJs/dr_AB via chain rule on the quadratic form
    djAB = dJ_AB * opAB
    djBC = dJ_BC * opBC
    djAC = dJ_AC * opAC

    dJs_dAB = djAB * (2 * jAB - jBC - jAC)
    dJs_dBC = djBC * (2 * jBC - jAB - jAC)
    dJs_dAC = djAC * (2 * jAC - jBC - jAB)

    # dE/dr_ij
    dE_dAB = dQs_dAB - 0.5 / sqrtJ * dJs_dAB
    dE_dBC = dQs_dBC - 0.5 / sqrtJ * dJs_dBC
    dE_dAC = dQs_dAC - 0.5 / sqrtJ * dJs_dAC

    # Map dr_ij to atom gradients via chain rule: dr_AB/dxA = -uAB, dr_AB/dxB = +uAB
    G = zeros(9)

    # Atom A (indices 1:3)
    G[1:3] = -dE_dAB * uAB - dE_dAC * uAC

    # Atom B (indices 4:6)
    G[4:6] = dE_dAB * uAB - dE_dBC * uBC

    # Atom C (indices 7:9)
    G[7:9] = dE_dBC * uBC + dE_dAC * uAC

    return E, G
end

"""
    leps_energy_gradient_2d(rAB_rBC::AbstractVector) -> (E, G)

Evaluate the LEPS potential in reduced 2D coordinates `[r_AB, r_BC]`.

For the collinear arrangement A--B--C, `r_AC = r_AB + r_BC`. This 2D form
is useful for visualization and testing NEB on the LEPS surface without
the overhead of 9D molecular coordinates.

Returns `(E, G)` where `G = [∂E/∂r_AB, ∂E/∂r_BC]`.
"""
function leps_energy_gradient_2d(rAB_rBC::AbstractVector)
    r_AB = rAB_rBC[1]
    r_BC = rAB_rBC[2]
    r_AC = r_AB + r_BC

    opAB = 1.0 / (1.0 + LEPS_S_AB)
    opBC = 1.0 / (1.0 + LEPS_S_BC)
    opAC = 1.0 / (1.0 + LEPS_S_AC)

    Q_AB, dQ_AB = _leps_Q(r_AB, LEPS_D_AB)
    Q_BC, dQ_BC = _leps_Q(r_BC, LEPS_D_BC)
    Q_AC, dQ_AC = _leps_Q(r_AC, LEPS_D_AC)

    J_AB, dJ_AB = _leps_J(r_AB, LEPS_D_AB)
    J_BC, dJ_BC = _leps_J(r_BC, LEPS_D_BC)
    J_AC, dJ_AC = _leps_J(r_AC, LEPS_D_AC)

    Qs = Q_AB * opAB + Q_BC * opBC + Q_AC * opAC
    jAB = J_AB * opAB
    jBC = J_BC * opBC
    jAC = J_AC * opAC

    Js = jAB^2 + jBC^2 + jAC^2 - jAB * jBC - jBC * jAC - jAB * jAC
    sqrtJ = sqrt(max(Js, 1e-30))

    E = Qs - sqrtJ

    djAB = dJ_AB * opAB
    djBC = dJ_BC * opBC
    djAC = dJ_AC * opAC

    dJs_dAB = djAB * (2 * jAB - jBC - jAC)
    dJs_dBC = djBC * (2 * jBC - jAB - jAC)
    dJs_dAC = djAC * (2 * jAC - jBC - jAB)

    dE_dAB = dQ_AB * opAB - 0.5 / sqrtJ * dJs_dAB
    dE_dBC = dQ_BC * opBC - 0.5 / sqrtJ * dJs_dBC
    dE_dAC = dQ_AC * opAC - 0.5 / sqrtJ * dJs_dAC

    # Chain rule: r_AC = r_AB + r_BC, so dr_AC/dr_AB = 1, dr_AC/dr_BC = 1
    G_rAB = dE_dAB + dE_dAC
    G_rBC = dE_dBC + dE_dAC

    return E, [G_rAB, G_rBC]
end

# --- Known stationary points (approximate, for the asymmetric parameterization) ---

"""
    LEPS_REACTANT

Approximate reactant geometry: A far from B-C at equilibrium.
Collinear: A at origin, B at r_e, C at r_e + 3.0.
"""
const LEPS_REACTANT = [0.0, 0.0, 0.0,  LEPS_R_E, 0.0, 0.0,  LEPS_R_E + 3.0, 0.0, 0.0]

"""
    LEPS_PRODUCT

Approximate product geometry: A-B at equilibrium, C far away.
Collinear: A at origin, B at 3.0, C at 3.0 + r_e.
"""
const LEPS_PRODUCT = [0.0, 0.0, 0.0,  3.0, 0.0, 0.0,  3.0 + LEPS_R_E, 0.0, 0.0]
