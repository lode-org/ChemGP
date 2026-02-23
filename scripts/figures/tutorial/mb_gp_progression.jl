# MB-3: GP surrogate quality progression (2x2 panel)
#
# Shows how the GP approximation to the Muller-Brown surface improves with
# increasing training data. Four panels: N = 5, 15, 30, 50 training points
# sampled via Latin hypercube. Each panel shows GP mean contour with training
# points overlaid as black dots.

using ChemGP
using KernelFunctions
using Random
using LinearAlgebra
include(joinpath(@__DIR__, "common.jl"))

Random.seed!(42)

# --- Grid evaluation (true surface) ---
x_range = range(-1.5, 1.2; length=100)
y_range = range(-0.5, 2.0; length=100)
E_true = eval_grid(muller_brown_energy_gradient, x_range, y_range)
E_true_clipped = clamp.(E_true, -200, 50)

# --- Sample training points via Latin hypercube in domain ---
function latin_hypercube_2d(n, x_lo, x_hi, y_lo, y_hi; rng=Random.GLOBAL_RNG)
    xs = Float64[]
    ys = Float64[]
    dx = (x_hi - x_lo) / n
    dy = (y_hi - y_lo) / n
    for i in 1:n
        push!(xs, x_lo + (i - 1 + rand(rng)) * dx)
        push!(ys, y_lo + (i - 1 + rand(rng)) * dy)
    end
    # Shuffle y values to break correlation
    shuffle!(rng, ys)
    return xs, ys
end

# --- Build GP model for a given set of training points ---
function build_gp(xs, ys)
    D = 2
    td = TrainingData(D)
    for i in eachindex(xs)
        E, G = muller_brown_energy_gradient([xs[i], ys[i]])
        add_point!(td, [xs[i], ys[i]], E, G)
    end
    y_full, y_mean, y_std = ChemGP.normalize(td)
    kernel = 1.0 * with_lengthscale(SqExponentialKernel(), 0.3)
    model = GPModel(kernel, td.X, y_full)
    train_model!(model; iterations=300)
    return model, y_mean, y_std
end

# --- Plot ---
set_theme!(PUBLICATION_THEME)

n_points_list = [5, 15, 30, 50]
fig = Figure(; size=(700, 600))

for (idx, n_pts) in enumerate(n_points_list)
    row = (idx - 1) ÷ 2 + 1
    col = (idx - 1) % 2 + 1

    ax = Axis(
        fig[row, col];
        xlabel=col == 1 && row == 2 ? L"$x$" : "",
        ylabel=row == 1 && col == 1 ? L"$y$" : "",
        title=L"$N = %$n_pts$",
        aspect=DataAspect(),
    )

    # Sample points
    xs, ys = latin_hypercube_2d(n_pts, -1.5, 1.2, -0.5, 2.0; rng=MersenneTwister(42))

    # Build GP
    model, y_mean, y_std = build_gp(xs, ys)

    # Predict on grid
    E_gp, _ = gp_predict_grid(model, x_range, y_range, y_mean, y_std)
    E_gp_clipped = clamp.(E_gp, -200, 50)

    # GP mean contour
    contourf!(
        ax,
        collect(x_range),
        collect(y_range),
        E_gp_clipped;
        levels=range(-200, 50; length=25),
        colormap=ENERGY_COLORMAP,
    )
    contour!(
        ax,
        collect(x_range),
        collect(y_range),
        E_gp_clipped;
        levels=range(-200, 50; step=25),
        color=:black,
        linewidth=0.3,
    )

    # Training points
    scatter!(
        ax,
        xs,
        ys;
        marker=:circle,
        markersize=5,
        color=:black,
        strokecolor=:white,
        strokewidth=0.5,
    )
end

# Shared colorbar
Colorbar(fig[1:2, 3]; colormap=ENERGY_COLORMAP, limits=(-200, 50), label=L"$E$ (a.u.)")

save_figure(fig, "mb_gp_progression")
