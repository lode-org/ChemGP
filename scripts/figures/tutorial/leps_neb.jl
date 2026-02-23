# LEPS-2: NEB path on LEPS surface
#
# Runs NEB in full 9D on the LEPS surface (3 atoms x 3 coords), then
# projects each image to 2D (r_AB, r_BC) and overlays on the 2D contour.

using ChemGP
using LinearAlgebra
include(joinpath(@__DIR__, "common.jl"))

# --- Grid evaluation (2D background) ---
r_AB_range = range(0.5, 4.0; length = 200)
r_BC_range = range(0.5, 4.0; length = 200)
E = eval_grid(leps_energy_gradient_2d, r_AB_range, r_BC_range)
E_clipped = clamp.(E, -5.0, 5.0)

# --- NEB optimization in 9D ---
config = NEBConfig(;
    n_images = 9,
    spring_constant = 5.0,
    climbing_image = true,
    max_iter = 500,
    conv_tol = 0.05,
    step_size = 0.005,
)

result = neb_optimize(
    leps_energy_gradient, Float64.(LEPS_REACTANT), Float64.(LEPS_PRODUCT);
    config = config,
)

println("NEB converged: $(result.converged)")
println("Oracle calls: $(result.oracle_calls)")
println("Max energy image: $(result.max_energy_image)")

# Project 9D images to 2D (r_AB, r_BC)
path_rAB = Float64[]
path_rBC = Float64[]
for img in result.path.images
    rA = img[1:3]
    rB = img[4:6]
    rC = img[7:9]
    push!(path_rAB, norm(rB - rA))
    push!(path_rBC, norm(rC - rB))
end

# --- Plot ---
set_theme!(PUBLICATION_THEME)

fig = Figure(; size = (504, 440))
ax = Axis(fig[1, 1];
    xlabel = L"$r_\mathrm{AB}$ (\AA)",
    ylabel = L"$r_\mathrm{BC}$ (\AA)",
    aspect = DataAspect())

# Background contour
cf = contourf!(ax, collect(r_AB_range), collect(r_BC_range), E_clipped;
    levels = range(-5.0, 5.0; length = 30),
    colormap = ENERGY_COLORMAP)
contour!(ax, collect(r_AB_range), collect(r_BC_range), E_clipped;
    levels = range(-5.0, 5.0; step = 0.5),
    color = :black, linewidth = 0.3)

# NEB path
lines!(ax, path_rAB, path_rBC;
    color = :white, linewidth = 2.0)
scatter!(ax, path_rAB, path_rBC;
    marker = :circle, markersize = 8,
    color = RUHI.coral, strokecolor = :white, strokewidth = 1.0)

# Number images
for i in eachindex(path_rAB)
    text!(ax, path_rAB[i] + 0.08, path_rBC[i] + 0.08;
        text = string(i), fontsize = 8, color = :white)
end

# Mark endpoints
scatter!(ax, [path_rAB[1]], [path_rBC[1]];
    marker = :star5, markersize = 12,
    color = RUHI.sunshine, strokecolor = :white, strokewidth = 1.0)
scatter!(ax, [path_rAB[end]], [path_rBC[end]];
    marker = :star5, markersize = 12,
    color = RUHI.sunshine, strokecolor = :white, strokewidth = 1.0)

Colorbar(fig[1, 2], cf; label = L"$E$ (eV)")

save_figure(fig, "leps_neb")
