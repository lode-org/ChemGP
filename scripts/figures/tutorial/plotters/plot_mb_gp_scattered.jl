# MB-GP-SCATTERED plotter: GP quality with uniformly scattered training data
#
# Same layout as plot_mb_gp.jl but reads from mb_gp_scattered.h5.
# Shows how random spatial coverage improves GP surrogate quality
# compared to clustered training data.
#
# Output: mb_gp_progression_scattered.pdf

include(joinpath(@__DIR__, "common_plot.jl"))
using LaTeXStrings

function main()
    hp = get_h5_path(; fallback_stem="mb_gp_scattered")
    if !isfile(hp)
        error("HDF5 not found: $hp -- run the generator first")
    end

    _, x_range, y_range = h5_read_grid(hp, "true_energy")

    set_theme!(PUBLICATION_THEME)

    n_points_list = [3, 8, 15, 30]
    fig = Figure(; size=(700, 600))

    for (idx, n_pts) in enumerate(n_points_list)
        row = div(idx - 1, 2) + 1
        col = (idx - 1) % 2 + 1

        ax = Axis(
            fig[row, col];
            xlabel=col == 1 && row == 2 ? L"$x$" : "",
            ylabel=row == 1 && col == 1 ? L"$y$" : "",
            title=L"$N = %$n_pts$",
            aspect=DataAspect(),
        )

        E_gp, _, _ = h5_read_grid(hp, "gp_mean_N$(n_pts)")
        E_gp_clipped = clamp.(E_gp, -200, 50)

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

        pts = h5_read_points(hp, "train_N$(n_pts)")
        scatter!(
            ax,
            pts["x"],
            pts["y"];
            marker=:circle,
            markersize=5,
            color=:black,
            strokecolor=:white,
            strokewidth=0.5,
        )
    end

    Colorbar(fig[1:2, 3]; colormap=ENERGY_COLORMAP, limits=(-200, 50), label=L"$E$ (a.u.)")

    save_figure(fig, "mb_gp_progression_scattered")
end

main()
