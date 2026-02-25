# MB-NEB plotter: NEB path on Muller-Brown surface
#
# Reads grid, path, and minima from HDF5. Draws contourf of clamped energy,
# overlays NEB path as white line + coral circles, numbers images, marks
# minima as yellow stars.
#
# Output: mb_neb.pdf

include(joinpath(@__DIR__, "common_plot.jl"))
using LaTeXStrings

function main()
    hp = get_h5_path(; fallback_stem="mb_neb")
    if !isfile(hp)
        error("HDF5 not found: $hp -- run the generator first")
    end

    E, x_range, y_range = h5_read_grid(hp, "energy")
    neb = h5_read_path(hp, "neb")
    minima = h5_read_points(hp, "minima")

    path_x = neb["x"]
    path_y = neb["y"]
    min_x = minima["x"]
    min_y = minima["y"]

    E_clipped = clamp.(E, -200, 50)

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
        text!(ax, path_x[i] + 0.04, path_y[i] + 0.04;
            text=string(i), fontsize=8, color=:white)
    end

    # Mark minima
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
end

main()
