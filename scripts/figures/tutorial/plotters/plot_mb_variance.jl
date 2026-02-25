# MB-VARIANCE plotter: GP variance overlaid on Muller-Brown PES
#
# Reads energy grid, variance grid, training points, minima, saddles,
# and metadata from HDF5. Draws MB contourf, semi-transparent variance
# overlay, hatching for high-variance regions, stationary point markers,
# training points, max variance diamond.
#
# Output: mb_variance.pdf

include(joinpath(@__DIR__, "common_plot.jl"))
using LaTeXStrings
using Colors

# --- Hatching helper (purely visual, stays in plotter) ---

"""Generate diagonal hatch line segments where variance exceeds threshold."""
function make_hatch_segments(xs, ys, var_grid, threshold; spacing=0.08)
    segs_x = Float64[]
    segs_y = Float64[]
    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)
    diag = (xmax - xmin) + (ymax - ymin)

    for offset in range(-diag, diag; step=spacing)
        x0 = max(xmin, ymin - offset)
        x1 = min(xmax, ymax - offset)
        x0 >= x1 && continue

        n_samples = 80
        line_xs = range(x0, x1; length=n_samples)
        in_region = false
        seg_start_x = 0.0

        for lx in line_xs
            ly = lx + offset
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

function main()
    hp = get_h5_path(; fallback_stem="mb_variance")
    if !isfile(hp)
        error("HDF5 not found: $hp -- run the generator first")
    end

    E, x_range, y_range = h5_read_grid(hp, "energy")
    E_var, _, _ = h5_read_grid(hp, "variance")
    training = h5_read_points(hp, "training")
    minima = h5_read_points(hp, "minima")
    saddles = h5_read_points(hp, "saddles")
    meta = h5_read_metadata(hp)

    max_var_x = meta["max_var_x"]
    max_var_y = meta["max_var_y"]
    var_clip = meta["var_clip"]

    E_clipped = clamp.(E, -200, 50)
    xs = collect(x_range)
    ys = collect(y_range)

    set_theme!(PUBLICATION_THEME)

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
    contour!(ax, xs, ys, E_clipped;
        levels=range(-200, 50; step=25), color=:black, linewidth=0.3)

    # Semi-transparent grey overlay for variance
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

    # Hatching: high variance (denser lines)
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
    contour!(ax, xs, ys, E_var_clamped;
        levels=var_levels, color=:grey30, linewidth=0.8, linestyle=:dot)

    # Stationary points: minima as filled circles, saddles as crosses
    min_x = minima["x"]
    min_y = minima["y"]
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

    sad_x = saddles["x"]
    sad_y = saddles["y"]
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
    scatter!(
        ax,
        training["x"],
        training["y"];
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
end

main()
