# MB-2: NEB path on Muller-Brown surface
#
# Runs standard NEB between minimum A (deepest) and minimum B (second),
# overlays the converged path on the PES contour. The path passes through
# saddle S2 and minimum C.

using ChemGP
include(joinpath(@__DIR__, "common.jl"))

# --- Grid evaluation (background contour) ---
x_range = range(-1.5, 1.2; length=200)
y_range = range(-0.5, 2.0; length=200)
E = eval_grid(muller_brown_energy_gradient, x_range, y_range)
E_clipped = clamp.(E, -200, 50)

# --- NEB optimization ---
x_start = Float64.(MULLER_BROWN_MINIMA[1])  # A: deepest
x_end = Float64.(MULLER_BROWN_MINIMA[2])    # B: second

config = NEBConfig(;
    n_images=11,
    spring_constant=10.0,
    climbing_image=true,
    max_iter=500,
    conv_tol=0.1,
    step_size=1e-4,
)

result = neb_optimize(muller_brown_energy_gradient, x_start, x_end; config=config)

println("NEB converged: $(result.converged)")
println("Oracle calls: $(result.oracle_calls)")
println("Max energy image: $(result.max_energy_image)")

# Extract path coordinates
path_x = [img[1] for img in result.path.images]
path_y = [img[2] for img in result.path.images]

# --- Plot ---
set_theme!(PUBLICATION_THEME)

fig = Figure(; size=(504, 400))
ax = Axis(fig[1, 1]; xlabel=L"$x$", ylabel=L"$y$", aspect=DataAspect())

# Background contour
cf = contourf!(
    ax,
    collect(x_range),
    collect(y_range),
    E_clipped;
    levels=range(-200, 50; length=25),
    colormap=ENERGY_COLORMAP,
)
contour!(
    ax,
    collect(x_range),
    collect(y_range),
    E_clipped;
    levels=range(-200, 50; step=25),
    color=:black,
    linewidth=0.3,
)

# NEB path
lines!(ax, path_x, path_y; color=:white, linewidth=2.0)
scatter!(
    ax,
    path_x,
    path_y;
    marker=:circle,
    markersize=8,
    color=RUHI.coral,
    strokecolor=:white,
    strokewidth=1.0,
)

# Number the images
for i in eachindex(path_x)
    text!(ax, path_x[i] + 0.04, path_y[i] + 0.04; text=string(i), fontsize=8, color=:white)
end

# Mark minima
min_x = [m[1] for m in MULLER_BROWN_MINIMA]
min_y = [m[2] for m in MULLER_BROWN_MINIMA]
scatter!(
    ax,
    min_x,
    min_y;
    marker=:star5,
    markersize=12,
    color=RUHI.sunshine,
    strokecolor=:black,
    strokewidth=1.0,
)

Colorbar(fig[1, 2], cf; label=L"$E$ (a.u.)")

save_figure(fig, "mb_neb")
