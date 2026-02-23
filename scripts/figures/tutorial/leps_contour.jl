# LEPS-1: 2D PES contour
#
# Evaluates the LEPS potential on a 2D grid of (r_AB, r_BC) and plots
# a filled contour. Reactant valley (small r_AB, large r_BC) and product
# valley (large r_AB, small r_BC) are labeled. Energy in eV, clipped
# to a reasonable range.

using ChemGP
include(joinpath(@__DIR__, "common.jl"))

# --- Grid evaluation ---
r_AB_range = range(0.5, 4.0; length = 200)
r_BC_range = range(0.5, 4.0; length = 200)
E = eval_grid(leps_energy_gradient_2d, r_AB_range, r_BC_range)

# Clip energy (LEPS diverges at small r values)
E_clipped = clamp.(E, -5.0, 5.0)

# --- Plot ---
set_theme!(PUBLICATION_THEME)

fig = Figure(; size = (504, 440))
ax = Axis(fig[1, 1];
    xlabel = L"$r_\mathrm{AB}$ (\AA)",
    ylabel = L"$r_\mathrm{BC}$ (\AA)",
    aspect = DataAspect())

# Filled contour
cf = contourf!(ax, collect(r_AB_range), collect(r_BC_range), E_clipped;
    levels = range(-5.0, 5.0; length = 30),
    colormap = ENERGY_COLORMAP)
contour!(ax, collect(r_AB_range), collect(r_BC_range), E_clipped;
    levels = range(-5.0, 5.0; step = 0.5),
    color = :black, linewidth = 0.3)

# Mark reactant and product valleys
text!(ax, 0.8, 3.3; text = "Reactant\n(A + BC)", fontsize = 10, color = :white,
    align = (:center, :center))
text!(ax, 3.3, 0.8; text = "Product\n(AB + C)", fontsize = 10, color = :white,
    align = (:center, :center))

# Approximate saddle point region (near r_AB ~ r_BC ~ 1.0)
scatter!(ax, [1.0], [1.0];
    marker = :xcross, markersize = 12,
    color = :white, strokecolor = :black, strokewidth = 1.5)
text!(ax, 1.15, 1.15; text = "TS", fontsize = 10, color = :white)

Colorbar(fig[1, 2], cf; label = L"$E$ (eV)")

save_figure(fig, "leps_contour")
