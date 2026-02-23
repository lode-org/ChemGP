# MB-1: Muller-Brown PES contour with stationary points
#
# Generates a filled contour plot of the Muller-Brown surface with minima
# marked as filled circles and saddle points as crosses. Energy range
# clipped to [-200, 50] to avoid Gaussian tail blow-up.

using ChemGP
include(joinpath(@__DIR__, "common.jl"))

# --- Grid evaluation ---
x_range = range(-1.5, 1.2; length = 200)
y_range = range(-0.5, 2.0; length = 200)
E = eval_grid(muller_brown_energy_gradient, x_range, y_range)

# Clip energy for visualization
E_clipped = clamp.(E, -200, 50)

# --- Plot ---
set_theme!(PUBLICATION_THEME)

fig = Figure(; size = (504, 400))
ax = Axis(fig[1, 1]; xlabel = L"$x$", ylabel = L"$y$", aspect = DataAspect())

# Filled contour
cf = contourf!(ax, collect(x_range), collect(y_range), E_clipped;
    levels = range(-200, 50; length = 25),
    colormap = ENERGY_COLORMAP)

# Contour lines for depth cues
contour!(ax, collect(x_range), collect(y_range), E_clipped;
    levels = range(-200, 50; step = 25),
    color = :black, linewidth = 0.3)

# Stationary points
min_x = [m[1] for m in MULLER_BROWN_MINIMA]
min_y = [m[2] for m in MULLER_BROWN_MINIMA]
scatter!(ax, min_x, min_y;
    marker = :circle, markersize = 10,
    color = :white, strokecolor = :black, strokewidth = 1.5)

# Labels for minima
for (i, label) in enumerate(["A", "B", "C"])
    text!(ax, min_x[i] + 0.06, min_y[i] + 0.06; text = label,
        fontsize = 11, color = :white)
end

sad_x = [s[1] for s in MULLER_BROWN_SADDLES]
sad_y = [s[2] for s in MULLER_BROWN_SADDLES]
scatter!(ax, sad_x, sad_y;
    marker = :xcross, markersize = 12,
    color = :white, strokecolor = :black, strokewidth = 1.5)

# Labels for saddles
for (i, label) in enumerate(["S1", "S2"])
    text!(ax, sad_x[i] + 0.06, sad_y[i] + 0.06; text = label,
        fontsize = 11, color = :white)
end

Colorbar(fig[1, 2], cf; label = L"$E$ (a.u.)")

save_figure(fig, "mb_pes")
