# MB-7: Trust region illustration
#
# Builds a GP from ~10 training points clustered in [-0.5, 0.5] x [0.3, 0.8].
# Shows a 1D slice (y=0.5) with true surface, GP mean, confidence band, and
# trust radius boundary. Beyond the boundary, the GP diverges. A hypothetical
# step outside is marked with a red X, and a fallback oracle point with a
# green check.

using ChemGP
using KernelFunctions
using Random
using LinearAlgebra
include(joinpath(@__DIR__, "common.jl"))

Random.seed!(42)

# --- Training points clustered in a small region ---
D = 2
td = TrainingData(D)

rng = MersenneTwister(42)
for _ in 1:10
    x = -0.5 + 1.0 * rand(rng)    # [-0.5, 0.5]
    y = 0.3 + 0.5 * rand(rng)     # [0.3, 0.8]
    E, G = muller_brown_energy_gradient([x, y])
    add_point!(td, [x, y], E, G)
end

println("Training points: $(npoints(td))")

# --- Build GP model ---
y_full, y_mean, y_std = ChemGP.normalize(td)
kernel = 1.0 * with_lengthscale(SqExponentialKernel(), 0.3)
model = GPModel(kernel, td.X, y_full)
train_model!(model; iterations = 300)

# --- 1D slice at y=0.5 ---
y_slice = 0.5
x_slice = range(-1.5, 1.2; length = 300)

E_true = Float64[]
E_pred = Float64[]
E_std_vals = Float64[]
dists = Float64[]

for x in x_slice
    e_true, _ = muller_brown_energy_gradient([x, y_slice])
    push!(E_true, e_true)

    X_test = reshape([x, y_slice], 2, 1)
    mu, var = predict_with_variance(model, X_test)
    push!(E_pred, mu[1] * y_std + y_mean)
    push!(E_std_vals, sqrt(max(var[1], 0.0)) * y_std)

    d = min_distance_to_data([x, y_slice], td.X)
    push!(dists, d)
end

# Trust radius: use the model's default or a representative value
trust_r = 0.3

# Find trust boundary crossings
in_trust = dists .<= trust_r
boundary_idx = findall(diff(in_trust) .!= 0)

# --- Plot ---
set_theme!(PUBLICATION_THEME)

fig = Figure(; size = (504, 350))
ax = Axis(fig[1, 1]; xlabel = L"$x$", ylabel = L"$E$ (a.u.)")

# Confidence band
band!(ax, collect(x_slice),
    E_pred .- 2 .* E_std_vals, E_pred .+ 2 .* E_std_vals;
    color = (RUHI.sky, 0.25))

# True surface
lines!(ax, collect(x_slice), E_true;
    color = :black, linewidth = 1.0, linestyle = :dash, label = "True surface")

# GP mean
lines!(ax, collect(x_slice), E_pred;
    color = RUHI.teal, linewidth = 1.5, label = "GP mean")

# Trust boundary vertical lines
for idx in boundary_idx
    vlines!(ax, [collect(x_slice)[idx]];
        color = RUHI.magenta, linewidth = 1.0, linestyle = :dot)
end

# Training points projected to slice (show their x-coordinates)
for i in 1:npoints(td)
    tx = td.X[1, i]
    ty = td.X[2, i]
    if abs(ty - y_slice) < 0.3  # nearby points
        e_at_slice, _ = muller_brown_energy_gradient([tx, y_slice])
        scatter!(ax, [tx], [e_at_slice];
            marker = :circle, markersize = 6,
            color = :black)
    end
end

# Hypothetical bad step outside trust region
x_bad = 1.0
e_bad_pred = E_pred[argmin(abs.(collect(x_slice) .- x_bad))]
scatter!(ax, [x_bad], [e_bad_pred];
    marker = :xcross, markersize = 14,
    color = RUHI.coral, strokewidth = 2)

# Fallback oracle evaluation
e_bad_true, _ = muller_brown_energy_gradient([x_bad, y_slice])
scatter!(ax, [x_bad], [e_bad_true];
    marker = :star5, markersize = 12,
    color = RUHI.teal)

# Annotations
text!(ax, x_bad + 0.05, e_bad_pred + 10;
    text = "GP step", fontsize = 9, color = RUHI.coral)
text!(ax, x_bad + 0.05, e_bad_true + 10;
    text = "Oracle fallback", fontsize = 9, color = RUHI.teal)

# Trust region label
if !isempty(boundary_idx)
    bx = collect(x_slice)[boundary_idx[end]]
    text!(ax, bx + 0.03, -50;
        text = "trust\nboundary", fontsize = 8, color = RUHI.magenta)
end

ylims!(ax, -250, 100)

Legend(fig[2, 1],
    [LineElement(; color = :black, linestyle = :dash),
        LineElement(; color = RUHI.teal, linewidth = 1.5),
        PolyElement(; color = (RUHI.sky, 0.25))],
    ["True surface", "GP mean", L"$\pm 2\sigma$"];
    orientation = :horizontal, tellwidth = false, tellheight = true)

save_figure(fig, "mb_trust_region")
