# ==============================================================================
# IDPP and S-IDPP Path Interpolation
# ==============================================================================
#
# Image Dependent Pair Potential (IDPP) and Sequential IDPP (S-IDPP) methods
# for generating initial NEB paths. These produce much better initial guesses
# than linear interpolation by ensuring pairwise distances vary smoothly.
#
# Reference:
#   Smidstrup, S., Pedersen, A., Stokbro, K. & Jonsson, H. (2014).
#   Improved initial guess for minimum energy path calculations.
#   J. Chem. Phys., 140, 214106.
#
#   Schmerwitz, Y. L. A., Gunde, M., Goswami, R. & Jonsson, H. (2024).
#   Improved initialization of the optimized path of NEB transition state
#   searches. arXiv:2407.16810v2.

# ==============================================================================
# Per-image IDPP
# ==============================================================================

"""
    idpp_interpolation(x_start, x_end, n_images; kwargs...) -> Vector{Vector{Float64}}

Create an IDPP-interpolated path of `n_images` images (including fixed
endpoints). Each intermediate image is independently optimized so that its
pairwise distances match a linearly interpolated target.

Matches eOn's `idppPath` in `NEBInitialPaths.cpp`.
"""
function idpp_interpolation(
    x_start::Vector{Float64},
    x_end::Vector{Float64},
    n_images::Int;
    n_coords_per_atom::Int = 3,
    max_iter::Int = 5000,
    max_move::Float64 = 0.1,
    force_tol::Float64 = 0.001,
    lbfgs_memory::Int = 20,
)
    images = linear_interpolation(x_start, x_end, n_images)
    n_atoms = div(length(x_start), n_coords_per_atom)

    d_start = _pairwise_distances(x_start, n_atoms, n_coords_per_atom)
    d_end = _pairwise_distances(x_end, n_atoms, n_coords_per_atom)

    for img_idx in 2:(n_images - 1)
        xi = (img_idx - 1) / (n_images - 1)
        d_target = (1 - xi) * d_start + xi * d_end

        optim = OptimState(lbfgs_memory)
        x = copy(images[img_idx])

        for iter in 1:max_iter
            _, force = _idpp_energy_force(x, d_target, n_atoms, n_coords_per_atom)
            _max_atom_force(force, n_atoms, n_coords_per_atom) < force_tol && break
            disp = optim_step!(optim, x, force, max_move;
                               n_coords_per_atom)
            x .+= disp
        end

        images[img_idx] = x
    end

    return images
end

# ==============================================================================
# S-IDPP (Sequential IDPP)
# ==============================================================================

"""
    sidpp_interpolation(x_start, x_end, n_images; kwargs...) -> Vector{Vector{Float64}}

Create an S-IDPP path. Grows the path incrementally from both endpoints,
running collective IDPP-NEB relaxation after each growth step, followed by
a final full-path relaxation.

Matches eOn's `sidppPath` in `NEBInitialPaths.cpp`.
"""
function sidpp_interpolation(
    x_start::Vector{Float64},
    x_end::Vector{Float64},
    n_images::Int;
    n_coords_per_atom::Int = 3,
    max_iter::Int = 5000,
    max_move::Float64 = 0.1,
    force_tol::Float64 = 0.001,
    lbfgs_memory::Int = 20,
    spring_constant::Float64 = 5.0,
    growth_alpha::Float64 = 0.33,
)
    n_atoms = div(length(x_start), n_coords_per_atom)

    d_init = _pairwise_distances(x_start, n_atoms, n_coords_per_atom)
    d_final = _pairwise_distances(x_end, n_atoms, n_coords_per_atom)

    # Start with [reactant, product]
    path = [copy(x_start), copy(x_end)]

    n_target = n_images - 2
    n_left = 0
    n_right = 0
    n_intermediate = 0

    # Growth loop: alternate left and right
    while n_intermediate < n_target
        if n_intermediate < n_target
            frontier = path[n_left + 1]
            next = path[n_left + 2]
            insert!(path, n_left + 2,
                    (1 - growth_alpha) * frontier + growth_alpha * next)
            n_left += 1
            n_intermediate += 1
        end

        if n_intermediate < n_target
            right_idx = length(path) - n_right
            frontier = path[right_idx]
            prev = path[right_idx - 1]
            insert!(path, right_idx,
                    (1 - growth_alpha) * frontier + growth_alpha * prev)
            n_right += 1
            n_intermediate += 1
        end

        _relax_collective_idpp!(
            path, d_init, d_final, n_atoms, n_coords_per_atom;
            max_iter, max_move, force_tol, lbfgs_memory, spring_constant,
        )
    end

    # Final full-path relaxation
    _relax_collective_idpp!(
        path, d_init, d_final, n_atoms, n_coords_per_atom;
        max_iter = 500, max_move, force_tol, lbfgs_memory, spring_constant,
    )

    return path
end

# ==============================================================================
# Collective IDPP-NEB relaxation (shared by SIDPP growth steps)
# ==============================================================================

function _relax_collective_idpp!(
    path::Vector{Vector{Float64}},
    d_init::Matrix{Float64},
    d_final::Matrix{Float64},
    n_atoms::Int,
    n_coords::Int;
    max_iter::Int = 5000,
    max_move::Float64 = 0.1,
    force_tol::Float64 = 0.001,
    lbfgs_memory::Int = 20,
    spring_constant::Float64 = 5.0,
)
    n_images = length(path)
    n_mov = n_images - 2
    n_mov == 0 && return
    D = length(path[1])

    optim = OptimState(lbfgs_memory)

    for iter in 1:max_iter
        forces = _collective_idpp_forces(
            path, d_init, d_final, n_atoms, n_coords, spring_constant,
        )

        cur_force = vcat(forces[2:end-1]...)
        _max_atom_force(cur_force, div(length(cur_force), n_coords), n_coords) < force_tol && break

        cur_x = vcat(path[2:end-1]...)
        disp = optim_step!(optim, cur_x, cur_force, max_move;
                           n_coords_per_atom = n_coords)

        new_x = cur_x + disp
        for i in 1:n_mov
            off = (i - 1) * D
            path[i + 1] = new_x[off+1:off+D]
        end
    end
end

# Perpendicular IDPP forces + spring forces parallel to tangent.
function _collective_idpp_forces(
    path::Vector{Vector{Float64}},
    d_init::Matrix{Float64},
    d_final::Matrix{Float64},
    n_atoms::Int,
    n_coords::Int,
    k_spring::Float64,
)
    n_images = length(path)
    D = length(path[1])
    forces = [zeros(D) for _ in 1:n_images]

    for i in 2:(n_images - 1)
        xi = (i - 1) / (n_images - 1)
        d_target = (1 - xi) * d_init + xi * d_final

        _, f_idpp = _idpp_energy_force(path[i], d_target, n_atoms, n_coords)

        # Simple tangent: (next - prev), normalized
        tau = path[i + 1] - path[i - 1]
        tn = norm(tau)
        tau = tn > 1e-18 ? tau / tn : zeros(D)

        # Perpendicular IDPP + parallel spring
        f_perp = f_idpp - dot(f_idpp, tau) * tau
        d_next = norm(path[i + 1] - path[i])
        d_prev = norm(path[i] - path[i - 1])
        f_spring = k_spring * (d_next - d_prev) * tau

        forces[i] = f_perp + f_spring
    end

    return forces
end

# ==============================================================================
# IDPP primitives
# ==============================================================================

function _pairwise_distances(x::Vector{Float64}, n_atoms::Int, n_coords::Int)
    d = zeros(n_atoms, n_atoms)
    for i in 1:n_atoms
        off_i = (i - 1) * n_coords
        for j in (i + 1):n_atoms
            off_j = (j - 1) * n_coords
            r = norm(@view(x[off_i+1:off_i+n_coords]) - @view(x[off_j+1:off_j+n_coords]))
            d[i, j] = r
            d[j, i] = r
        end
    end
    return d
end

# E = 0.5 * sum_{i<j} (1/r^4) * (r - d_target)^2
function _idpp_energy_force(
    x::Vector{Float64},
    d_target::Matrix{Float64},
    n_atoms::Int,
    n_coords::Int,
)
    energy = 0.0
    force = zeros(length(x))

    for i in 1:n_atoms
        off_i = (i - 1) * n_coords
        for j in (i + 1):n_atoms
            off_j = (j - 1) * n_coords

            dr = x[off_i+1:off_i+n_coords] - x[off_j+1:off_j+n_coords]
            r = norm(dr)
            r = max(r, 1e-4)

            diff = r - d_target[i, j]
            r4 = r^4

            energy += 0.5 * diff^2 / r4

            # dE/dr = diff * (1 - 2*diff/r) / r^4
            dEdr = diff * (1.0 - 2.0 * diff / r) / r4
            f_pair = (-dEdr / r) .* dr

            force[off_i+1:off_i+n_coords] .+= f_pair
            force[off_j+1:off_j+n_coords] .-= f_pair
        end
    end

    return energy, force
end

# Max per-atom force norm
function _max_atom_force(force::Vector{Float64}, n_atoms::Int, n_coords::Int)
    max_f = 0.0
    for a in 1:n_atoms
        off = (a - 1) * n_coords
        max_f = max(max_f, norm(@view force[off+1:off+n_coords]))
    end
    return max_f
end
