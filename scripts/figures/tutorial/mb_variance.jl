# MB-4: GP variance overlaid on Muller-Brown PES
#
# After ~20 training points seeded near minimum A and the A-C saddle region,
# shows the predictive variance overlaid on the MB surface contour.
# High-variance regions are indicated by diagonal hatching lines,
# while low-variance (well-sampled) regions show the PES clearly.
# Stationary points (minima, saddles) are marked for reference.

using ChemGP
using KernelFunctions
using Random
using LinearAlgebra
using Statistics
using LaTeXStrings
include(joinpath(@__DIR__, "common.jl"))

Random.seed!(42)

# --- Sample ~20 training points near the A-to-C region ---
D = 2
td = TrainingData(D)

# Cluster points near minimum A and the A-C saddle
centers = [
    MULLER_BROWN_MINIMA[1],  # A
    MULLER_BROWN_SADDLES[1], # S1
    MULLER_BROWN_MINIMA[3],  # C
]

rng = MersenneTwister(42)
for center in centers
    for _ in 1:7
        pt = center .+ 0.2 * randn(rng, 2)
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
train_model!(model; iterations=300)

# --- Evaluate PES and variance on grid ---
x_range = range(-1.5, 1.2; length=150)
y_range = range(-0.5, 2.0; length=150)

E_pes = eval_grid(muller_brown_energy_gradient, x_range, y_range)
E_clipped = clamp.(E_pes, -200, 50)

_, E_var = gp_predict_grid(model, x_range, y_range, y_mean, y_std)

# Find max variance in interior (skip boundary to avoid edge artifacts)
xs = collect(x_range)
ys = collect(y_range)
margin = 10
interior_var = E_var[margin:(end - margin), margin:(end - margin)]
int_idx = argmax(interior_var)
max_var_x = xs[margin - 1 + int_idx[1]]
max_var_y = ys[margin - 1 + int_idx[2]]
max_var_val = interior_var[int_idx]

# Variance thresholds for hatching density
var_clip = quantile(vec(E_var), 0.95)

println("Max interior variance at ($max_var_x, $max_var_y) = $max_var_val")
println("95th percentile variance: $var_clip")

# --- Generate hatching line segments for high-variance regions ---
# Diagonal lines at 45 degrees, drawn only where variance exceeds threshold.
# Denser hatching = higher variance (two thresholds: medium and high).
function make_hatch_segments(xs, ys, var_grid, threshold; spacing=0.08)
    segs_x = Float64[]
    segs_y = Float64[]
    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)
    diag = (xmax - xmin) + (ymax - ymin)

    # Diagonal lines from bottom-left to top-right
    for offset in range(-diag, diag; step=spacing)
        # Line: y = x + offset, clipped to domain
        x0 = max(xmin, ymin - offset)
        x1 = min(xmax, ymax - offset)
        x0 >= x1 && continue

        # Sample along line, keep segments inside threshold
        n_samples = 80
        line_xs = range(x0, x1; length=n_samples)
        in_region = false
        seg_start_x = 0.0

        for lx in line_xs
            ly = lx + offset
            # Interpolate variance at (lx, ly)
            ix = searchsortedlast(xs, lx)
            iy = searchsortedlast(ys, ly)
            ix = clamp(ix, 1, length(xs) - 1)
            iy = clamp(iy, 1, length(ys) - 1)
            v = var_grid[ix, iy]

            if v >= threshold && !in_region
                in_region = true
                seg_start_x = lx
            elseif v < threshold && in_region
                in_region = false
                push!(segs_x, seg_start_x)
                push!(segs_y, seg_start_x + offset)
                push!(segs_x, lx)
                push!(segs_y, lx + offset)
            end
        end
        if in_region
            push!(segs_x, seg_start_x)
            push!(segs_y, seg_start_x + offset)
            push!(segs_x, line_xs[end])
            push!(segs_y, line_xs[end] + offset)
        end
    end
    return segs_x, segs_y
end

# --- Plot ---
set_theme!(PUBLICATION_THEME)

using Colors

fig = Figure(; size=(504, 400))
ax = Axis(fig[1, 1]; xlabel=L"$x$", ylabel=L"$y$", aspect=DataAspect())

# Base layer: MB PES filled contour
contourf!(
    ax,
    xs,
    ys,
    E_clipped;
    levels=range(-200, 50; length=25),
    colormap=ENERGY_COLORMAP,
)
contour!(ax, xs, ys, E_clipped; levels=range(-200, 50; step=25), color=:black, linewidth=0.3)

# Semi-transparent grey overlay for variance (subtle shading)
E_var_clamped = clamp.(E_var, 0, var_clip)
var_cmap = [RGBAf(0.15, 0.15, 0.15, a) for a in range(0.0, 0.35; length=256)]
hm = heatmap!(ax, xs, ys, E_var_clamped; colormap=var_cmap, colorrange=(0, var_clip))

# Hatching: medium variance (sparse lines)
med_thresh = 0.25 * var_clip
hx_m, hy_m = make_hatch_segments(xs, ys, E_var, med_thresh; spacing=0.10)
if length(hx_m) >= 2
    linesegments!(
        ax,
        [Point2f(hx_m[i], hy_m[i]) for i in eachindex(hx_m)];
        color=RGBAf(0.1, 0.1, 0.1, 0.3),
        linewidth=0.4,
    )
end

# Hatching: high variance (denser lines, perpendicular)
high_thresh = 0.55 * var_clip
hx_h, hy_h = make_hatch_segments(xs, ys, E_var, high_thresh; spacing=0.06)
if length(hx_h) >= 2
    linesegments!(
        ax,
        [Point2f(hx_h[i], hy_h[i]) for i in eachindex(hx_h)];
        color=RGBAf(0.1, 0.1, 0.1, 0.5),
        linewidth=0.5,
    )
end

# Variance contour lines
var_levels = [0.2 * var_clip, 0.5 * var_clip, 0.8 * var_clip]
contour!(ax, xs, ys, E_var_clamped; levels=var_levels, color=:grey30, linewidth=0.8, linestyle=:dot)

# Stationary points: minima as filled circles, saddles as crosses
min_x = [m[1] for m in MULLER_BROWN_MINIMA]
min_y = [m[2] for m in MULLER_BROWN_MINIMA]
scatter!(
    ax,
    min_x,
    min_y;
    marker=:circle,
    markersize=10,
    color=:white,
    strokecolor=:black,
    strokewidth=1.5,
)
for (i, (label, ha)) in enumerate(zip(["A", "B", "C"], [0.08, 0.08, 0.08]))
    text!(
        ax,
        min_x[i] + ha,
        min_y[i] + 0.08;
        text=label,
        fontsize=12,
        font=:bold,
        color=:white,
        strokecolor=:black,
        strokewidth=0.8,
    )
end

sad_x = [s[1] for s in MULLER_BROWN_SADDLES]
sad_y = [s[2] for s in MULLER_BROWN_SADDLES]
scatter!(
    ax,
    sad_x,
    sad_y;
    marker=:xcross,
    markersize=12,
    color=:white,
    strokecolor=:black,
    strokewidth=1.5,
)
for (i, label) in enumerate(["S1", "S2"])
    text!(
        ax,
        sad_x[i] + 0.08,
        sad_y[i] + 0.08;
        text=label,
        fontsize=12,
        font=:bold,
        color=:white,
        strokecolor=:black,
        strokewidth=0.8,
    )
end

# Training points
train_x = [td.X[1, i] for i in 1:npoints(td)]
train_y = [td.X[2, i] for i in 1:npoints(td)]
scatter!(
    ax,
    train_x,
    train_y;
    marker=:circle,
    markersize=5,
    color=:black,
    strokecolor=:white,
    strokewidth=0.5,
)

# Max variance annotation (interior)
scatter!(
    ax,
    [max_var_x],
    [max_var_y];
    marker=:diamond,
    markersize=12,
    color=RUHI.sunshine,
    strokecolor=:black,
    strokewidth=1.5,
)
text!(
    ax,
    max_var_x + 0.08,
    max_var_y - 0.12;
    text=L"$\max \sigma^2$",
    fontsize=9,
    color=RUHI.sunshine,
    strokecolor=:black,
    strokewidth=0.5,
)

Colorbar(fig[1, 2], hm; label=L"$\sigma^2_E$")

save_figure(fig, "mb_variance")
