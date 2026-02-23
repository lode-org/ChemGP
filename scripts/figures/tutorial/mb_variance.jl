# MB-4: GP variance heatmap
#
# After ~20 training points seeded near minimum A and the A-C saddle region,
# shows the predictive variance across the Muller-Brown domain. The
# max-variance point is annotated with an arrow.

using ChemGP
using KernelFunctions
using Random
using LinearAlgebra
include(joinpath(@__DIR__, "common.jl"))

Random.seed!(42)

# --- Sample ~20 training points near the A-to-C region ---
D = 2
td = TrainingData(D)

# Cluster points near minimum A and the A-C saddle
centers = [
    MULLER_BROWN_MINIMA[1],  # A: [-0.558, 1.442]
    MULLER_BROWN_SADDLES[1], # S1: [-0.822, 0.624]
    MULLER_BROWN_MINIMA[3],  # C: [-0.050, 0.467]
]

rng = MersenneTwister(42)
for center in centers
    for _ in 1:7
        pt = center .+ 0.2 * randn(rng, 2)
        # Clamp to domain
        pt[1] = clamp(pt[1], -1.5, 1.2)
        pt[2] = clamp(pt[2], -0.5, 2.0)
        E, G = muller_brown_energy_gradient(pt)
        add_point!(td, pt, E, G)
    end
end

println("Training points: $(npoints(td))")

# --- Build GP model ---
y_full, y_mean, y_std = ChemGP.normalize(td)
kernel = 1.0 * with_lengthscale(SqExponentialKernel(), 0.3)
model = GPModel(kernel, td.X, y_full)
train_model!(model; iterations = 300)

# --- Predict variance on grid ---
x_range = range(-1.5, 1.2; length = 100)
y_range = range(-0.5, 2.0; length = 100)

_, E_var = gp_predict_grid(model, x_range, y_range, y_mean, y_std)

# Find max variance location
max_idx = argmax(E_var)
max_i, max_j = max_idx[1], max_idx[2]
max_var_x = collect(x_range)[max_j]
max_var_y = collect(y_range)[max_i]
max_var_val = E_var[max_idx]

println("Max variance at ($max_var_x, $max_var_y) = $max_var_val")

# --- Plot ---
set_theme!(PUBLICATION_THEME)

fig = Figure(; size = (504, 400))
ax = Axis(fig[1, 1]; xlabel = L"$x$", ylabel = L"$y$", aspect = DataAspect())

# Variance heatmap
hm = heatmap!(ax, collect(x_range), collect(y_range), E_var;
    colormap = VARIANCE_COLORMAP)

# Training points
train_x = [td.X[1, i] for i in 1:npoints(td)]
train_y = [td.X[2, i] for i in 1:npoints(td)]
scatter!(ax, train_x, train_y;
    marker = :circle, markersize = 5,
    color = :black, strokecolor = :white, strokewidth = 0.5)

# Max variance annotation
scatter!(ax, [max_var_x], [max_var_y];
    marker = :diamond, markersize = 12,
    color = RUHI.coral, strokecolor = :white, strokewidth = 1.5)
text!(ax, max_var_x + 0.08, max_var_y + 0.08;
    text = L"$\max \sigma^2$", fontsize = 9, color = RUHI.coral)

Colorbar(fig[1, 2], hm; label = L"$\sigma^2_E$")

save_figure(fig, "mb_variance")
