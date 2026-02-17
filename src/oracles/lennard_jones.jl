# ==============================================================================
# Lennard-Jones Oracle
# ==============================================================================
#
# The "oracle" is the expensive function that the GP is trying to replace.
# In real applications, this would be a DFT or ab initio calculation.
# Here we use the Lennard-Jones potential as a pedagogical stand-in.
#
# All functions operate on flat coordinate vectors [x1,y1,z1, x2,y2,z2, ...]
# rather than AtomsBase types, keeping the core library dependency-free.

"""
    lj_energy_gradient(x::AbstractVector{<:Real}; epsilon=1.0, sigma=1.0)

Lennard-Jones energy and gradient for a flat coordinate vector.

The LJ potential for a pair of atoms at distance r is:
    V(r) = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]

Returns `(E, G)` where `E` is the total energy and `G = dE/dx` is the
gradient (not the force; the force is `-G`).

This is the "oracle" in the GP-guided optimization loop. In real
applications, each call corresponds to an expensive quantum chemistry
calculation.
"""
function lj_energy_gradient(x::AbstractVector{<:Real}; epsilon = 1.0, sigma = 1.0)
    N = div(length(x), 3)
    E = 0.0
    G = zeros(length(x))

    for i in 1:N
        for j in (i+1):N
            # Displacement vector
            dx = x[3(j-1)+1] - x[3(i-1)+1]
            dy = x[3(j-1)+2] - x[3(i-1)+2]
            dz = x[3(j-1)+3] - x[3(i-1)+3]
            r2 = dx^2 + dy^2 + dz^2
            r = sqrt(r2)

            sr6 = (sigma / r)^6
            sr12 = sr6^2

            E += 4epsilon * (sr12 - sr6)

            # dE/dr * dr/d(rij) = force magnitude per unit distance
            f_over_r = (24epsilon / r2) * (2sr12 - sr6)

            # Gradient = dE/dx (opposite sign to force on atom i)
            for d in 1:3
                rij_d = x[3(j-1)+d] - x[3(i-1)+d]
                G[3(i-1)+d] -= f_over_r * rij_d
                G[3(j-1)+d] += f_over_r * rij_d
            end
        end
    end

    return E, G
end

"""
    random_cluster(N_atoms::Int; min_dist=1.5, max_attempts=10000)

Generate a random cluster of `N_atoms` atoms with no pair closer than `min_dist`.
Returns a flat coordinate vector of length `3*N_atoms`.

The first atom is placed at the origin. Subsequent atoms are placed at random
positions within a sphere, rejecting any placement that would create a pair
distance shorter than `min_dist`.
"""
function random_cluster(N_atoms::Int; min_dist = 1.5, max_attempts = 10000)
    coords = zeros(3 * N_atoms)
    # First atom at origin (already zeros)

    for i in 2:N_atoms
        placed = false
        for _ in 1:max_attempts
            r = 2.0 * N_atoms^(1 / 3) * rand()^(1 / 3)
            theta = 2pi * rand()
            phi = acos(2rand() - 1)
            candidate = [r * sin(phi) * cos(theta), r * sin(phi) * sin(theta), r * cos(phi)]

            too_close = false
            for j in 1:(i-1)
                xj = @view coords[(3j-2):(3j)]
                if sqrt(sum((candidate .- xj) .^ 2)) < min_dist
                    too_close = true
                    break
                end
            end

            if !too_close
                coords[(3i-2):(3i)] = candidate
                placed = true
                break
            end
        end
        !placed && error("Could not place atom $i after $max_attempts attempts")
    end

    return coords
end
