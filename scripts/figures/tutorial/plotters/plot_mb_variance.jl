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
using Statistics

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

    # Single crosshatch for high-variance regions (>= 75th percentile)
    high_thresh = Float64(quantile(vec(E_var[E_var .> 0]), 0.75))
    hx_h, hy_h = make_hatch_segments(xs, ys, E_var, high_thresh; spacing=0.06)
    if length(hx_h) >= 2
        linesegments!(
            ax,
            [Point2f(hx_h[i], hy_h[i]) for i in eachindex(hx_h)];
            color=RGBAf(1.0, 1.0, 1.0, 0.45),
            linewidth=0.5,
        )
    end

    # Magenta boundary contour at the high-variance threshold
    contour!(ax, xs, ys, E_var;
        levels=[high_thresh], color=RUHI.magenta, linewidth=1.2)

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

    # Legend
    Legend(
        fig[2, 1],
        [
            MarkerElement(; marker=:circle, markersize=8, color=:black, strokecolor=:white, strokewidth=0.5),
            MarkerElement(; marker=:circle, markersize=8, color=:white, strokecolor=:black, strokewidth=1.5),
            MarkerElement(; marker=:xcross, markersize=10, color=:white, strokecolor=:black, strokewidth=1.5),
            LineElement(; color=RUHI.magenta, linewidth=1.2),
        ],
        ["Training points", "Minima", "Saddles", L"High $\sigma^2$ boundary"];
        orientation=:horizontal,
        tellwidth=false,
        tellheight=true,
    )

    save_figure(fig, "mb_variance")
end

main()
