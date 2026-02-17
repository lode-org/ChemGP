# ==============================================================================
# NEB Path Utilities
# ==============================================================================
#
# Path interpolation, tangent estimation, and NEB force computation.
#
# The improved tangent estimate follows Henkelman & Jonsson (2000),
# which handles extrema along the path without the kinks of simple
# bisection tangents.
#
# Reference:
#   Goswami, R., Gunde, M. & Jónsson, H. (2026). Enhanced climbing image nudged
#   elastic band method with Hessian eigenmode alignment. arXiv:2601.12630.
#
#   Goswami, R. (2025). Efficient exploration of chemical kinetics. PhD thesis,
#   University of Iceland. arXiv:2510.21368.
#
#   Henkelman, G., Uberuaga, B. P. & Jonsson, H. (2000).
#   A climbing image nudged elastic band method for finding saddle points
#   and minimum energy paths. J. Chem. Phys., 113, 9901-9904.

"""
    linear_interpolation(x_start, x_end, n_images) -> Vector{Vector{Float64}}

Create a linearly interpolated path of `n_images` images (including fixed
endpoints) between `x_start` and `x_end`.
"""
function linear_interpolation(
    x_start::Vector{Float64},
    x_end::Vector{Float64},
    n_images::Int,
)
    images = Vector{Float64}[]
    for i in 0:(n_images - 1)
        t = i / (n_images - 1)
        push!(images, (1 - t) * x_start + t * x_end)
    end
    return images
end

"""
    path_tangent(images, energies, i) -> Vector{Float64}

Compute the unit tangent vector at image `i` using the improved tangent
estimate (Henkelman & Jonsson 2000).

Uses energy-weighted bisection at local extrema to avoid kinks in the
tangent that cause path oscillations.
"""
function path_tangent(
    images::Vector{Vector{Float64}},
    energies::Vector{Float64},
    i::Int,
)
    N = length(images)
    @assert 2 <= i <= N - 1 "Tangent only defined for intermediate images"

    tau_plus = images[i+1] - images[i]
    tau_minus = images[i] - images[i-1]

    E_prev = energies[i-1]
    E_curr = energies[i]
    E_next = energies[i+1]

    if E_prev < E_curr < E_next
        # Monotonic increase: use forward tangent
        tau = tau_plus
    elseif E_prev > E_curr > E_next
        # Monotonic decrease: use backward tangent
        tau = tau_minus
    else
        # Local extremum: energy-weighted bisection
        dE_max = max(abs(E_next - E_curr), abs(E_prev - E_curr))
        dE_min = min(abs(E_next - E_curr), abs(E_prev - E_curr))

        if E_prev < E_next
            tau = dE_max * tau_plus + dE_min * tau_minus
        else
            tau = dE_min * tau_plus + dE_max * tau_minus
        end
    end

    tn = norm(tau)
    return tn > 1e-18 ? tau / tn : tau_plus / (norm(tau_plus) + 1e-18)
end

"""
    spring_force(images, i, k_spring, tangent) -> Vector{Float64}

Compute the spring force parallel to the tangent at image `i`.

    F_spring = k * (||R_{i+1} - R_i|| - ||R_i - R_{i-1}||) * tangent
"""
function spring_force(
    images::Vector{Vector{Float64}},
    i::Int,
    k_spring::Float64,
    tangent::Vector{Float64},
)
    d_next = norm(images[i+1] - images[i])
    d_prev = norm(images[i] - images[i-1])
    return k_spring * (d_next - d_prev) * tangent
end

"""
    neb_force(gradient, spring_f, tangent; climbing, is_highest) -> Vector{Float64}

Compute the full NEB force at an image.

Standard NEB:
    F = -G_perp + F_spring_parallel

where `G_perp = G - (G·τ)τ` is the gradient perpendicular to the path tangent.

Climbing image (if `climbing=true` and `is_highest=true`):
    F = -G + 2(G·τ)τ

The climbing image moves uphill along the tangent and downhill perpendicular,
converging to the exact saddle point.
"""
function neb_force(
    gradient::Vector{Float64},
    spring_f::Vector{Float64},
    tangent::Vector{Float64};
    climbing::Bool = false,
    is_highest::Bool = false,
)
    if climbing && is_highest
        # Climbing image: invert parallel component of gradient
        return -gradient + 2 * dot(gradient, tangent) * tangent
    else
        # Standard NEB: perpendicular gradient + parallel spring
        G_perp = gradient - dot(gradient, tangent) * tangent
        return -G_perp + spring_f
    end
end

"""
    compute_all_neb_forces(path, config; ci_on) -> (forces, max_force_norm, ci_force_norm)

Compute NEB forces at all intermediate images. Returns the force vectors,
the maximum force norm among all images, and the climbing image force norm.
"""
function compute_all_neb_forces(path::NEBPath, config::NEBConfig; ci_on::Bool = false)
    N = length(path.images)
    forces = [zeros(length(path.images[1])) for _ in 1:N]

    i_max = argmax(path.energies[2:end-1]) + 1  # Index among all images

    max_f_norm = 0.0
    ci_f_norm = 0.0

    for i in 2:(N - 1)
        tau = path_tangent(path.images, path.energies, i)
        f_spring = spring_force(path.images, i, path.spring_constant, tau)

        is_highest = (i == i_max)
        f = neb_force(path.gradients[i], f_spring, tau;
                      climbing = ci_on && config.climbing_image,
                      is_highest = is_highest)

        forces[i] = f
        fn = norm(f)
        max_f_norm = max(max_f_norm, fn)

        if is_highest
            ci_f_norm = fn
        end
    end

    return forces, max_f_norm, ci_f_norm, i_max
end
