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
#   Goswami, R., Gunde, M. & Jonsson, H. (2026). Enhanced climbing image nudged
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
    x_start::Vector{Float64}, x_end::Vector{Float64}, n_images::Int
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
function path_tangent(images::Vector{Vector{Float64}}, energies::Vector{Float64}, i::Int)
    N = length(images)
    @assert 2 <= i <= N - 1 "Tangent only defined for intermediate images"

    tau_plus = images[i + 1] - images[i]
    tau_minus = images[i] - images[i - 1]

    E_prev = energies[i - 1]
    E_curr = energies[i]
    E_next = energies[i + 1]

    if E_prev < E_curr < E_next
        tau = tau_plus
    elseif E_prev > E_curr > E_next
        tau = tau_minus
    else
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
    images::Vector{Vector{Float64}}, i::Int, k_spring::Float64, tangent::Vector{Float64}
)
    d_next = norm(images[i + 1] - images[i])
    d_prev = norm(images[i] - images[i - 1])
    return k_spring * (d_next - d_prev) * tangent
end

"""
    energy_weighted_k(energies, i_lo, i_hi, k_min, k_max) -> Float64

Compute the energy-weighted spring constant for the spring connecting
images `i_lo` and `i_hi`. Each spring gets its own k based on the higher
energy of its two endpoints.

Reference: Asgeirsson, V. et al. (2021). J. Chem. Theory Comput., 17(8),
4929-4945.
"""
function energy_weighted_k(
    energies::Vector{Float64}, i_lo::Int, i_hi::Int, k_min::Float64, k_max::Float64
)
    E_ref = max(energies[1], energies[end])
    E_max = maximum(energies)
    dE = E_max - E_ref
    if dE < 1e-18
        return k_max
    end
    E_spring = max(energies[i_lo], energies[i_hi])
    if E_spring > E_ref
        return k_max - (k_max - k_min) * (E_max - E_spring) / dE
    else
        return k_min
    end
end

"""
    neb_force(gradient, spring_f, tangent; climbing, is_highest) -> Vector{Float64}

Compute the full NEB force at an image.

Standard NEB: F = -G_perp + F_spring_parallel
Climbing image: F = -G + 2(G . tau) tau
"""
function neb_force(
    gradient::Vector{Float64},
    spring_f::Vector{Float64},
    tangent::Vector{Float64};
    climbing::Bool=false,
    is_highest::Bool=false,
)
    if climbing && is_highest
        return -gradient + 2 * dot(gradient, tangent) * tangent
    else
        G_perp = gradient - dot(gradient, tangent) * tangent
        return -G_perp + spring_f
    end
end

"""
    get_hessian_points(x_start, x_end, epsilon) -> Vector{Vector{Float64}}

Generate 2*D "virtual Hessian" points around both endpoints by displacing
+epsilon along each coordinate axis.

The oracle is evaluated at these points and they are included in GP training
data for the first `num_hess_iter` outer iterations, providing curvature
information that improves GP conditioning.

Reference: Koistinen, O.-P. et al. (2017). J. Chem. Phys. 147, 152720.
Implementation follows MATLAB GPR/aux/get_hessian_points.m.
"""
function get_hessian_points(
    x_start::Vector{Float64}, x_end::Vector{Float64}, epsilon::Float64
)
    D = length(x_start)
    points = Vector{Vector{Float64}}(undef, 2 * D)
    for d in 1:D
        p1 = copy(x_start)
        p1[d] += epsilon
        points[d] = p1

        p2 = copy(x_end)
        p2[d] += epsilon
        points[D + d] = p2
    end
    return points
end

"""
    compute_all_neb_forces(path, config; ci_on) -> (forces, max_f, ci_f, i_max)

Compute NEB forces at all intermediate images of `path`.

When `ci_on=true` and `config.climbing_image=true`, the highest-energy image
uses the climbing image force formula (Henkelman et al. 2000, Eq. 5):
`F_CI = -G + 2(G . tau) tau`, which inverts the gradient component along
the path tangent to drive the image toward the true saddle point.

When `config.energy_weighted=true`, each spring gets an energy-dependent
constant k_i interpolated between `ew_k_min` and `ew_k_max` based on the
higher energy of the two connected images (Asgeirsson et al. 2021).

# Returns
- `forces`: Vector of force vectors at each image (endpoints are zero)
- `max_f`: Maximum force norm across all intermediate images
- `ci_f`: Force norm at the highest-energy image (the CI candidate)
- `i_max`: Index of the highest-energy intermediate image
"""
function compute_all_neb_forces(path::NEBPath, config::NEBConfig; ci_on::Bool=false)
    N = length(path.images)
    forces = [zeros(length(path.images[1])) for _ in 1:N]

    i_max = argmax(path.energies[2:(end - 1)]) + 1

    max_f_norm = 0.0
    ci_f_norm = 0.0

    for i in 2:(N - 1)
        tau = path_tangent(path.images, path.energies, i)

        if config.energy_weighted
            k_prev = energy_weighted_k(
                path.energies, i - 1, i, config.ew_k_min, config.ew_k_max
            )
            k_next = energy_weighted_k(
                path.energies, i, i + 1, config.ew_k_min, config.ew_k_max
            )
            d_next = norm(path.images[i + 1] - path.images[i])
            d_prev = norm(path.images[i] - path.images[i - 1])
            f_spring = (k_next * d_next - k_prev * d_prev) * tau
        else
            f_spring = spring_force(path.images, i, path.spring_constant, tau)
        end

        is_highest = (i == i_max)
        f = neb_force(
            path.gradients[i],
            f_spring,
            tau;
            climbing=ci_on && config.climbing_image,
            is_highest=is_highest,
        )

        forces[i] = f
        # Per-atom max force for molecular systems (3D per atom);
        # fall back to full norm for non-molecular (e.g. 2D) coordinates
        D_img = length(f)
        n_atoms = div(D_img, 3)
        fn = if n_atoms >= 1 && D_img == 3 * n_atoms
            maximum(norm(@view f[(3 * (a - 1) + 1):(3 * a)]) for a in 1:n_atoms)
        else
            norm(f)
        end
        max_f_norm = max(max_f_norm, fn)

        if is_highest
            ci_f_norm = fn
        end
    end

    return forces, max_f_norm, ci_f_norm, i_max
end
