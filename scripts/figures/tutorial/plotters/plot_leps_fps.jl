# LEPS-FPS plotter: farthest point sampling visualization
#
# Reads selected + pruned PCA-projected points from HDF5. Scatter plot
# with pruned as gray circles, selected as teal diamonds. Legend.
#
# Output: leps_fps.pdf

include(joinpath(@__DIR__, "common_plot.jl"))
using LaTeXStrings

function main()
    hp = get_h5_path(; fallback_stem="leps_fps")
    if !isfile(hp)
        error("HDF5 not found: $hp -- run the generator first")
    end

    selected = h5_read_points(hp, "selected")
    pruned = h5_read_points(hp, "pruned")

    set_theme!(PUBLICATION_THEME)

    fig = Figure(; size=(504, 400))
    ax = Axis(fig[1, 1]; xlabel=L"$\mathrm{PC}_1$", ylabel=L"$\mathrm{PC}_2$")

    # Pruned points (gray)
    scatter!(
        ax,
        pruned["pc1"],
        pruned["pc2"];
        marker=:circle,
        markersize=8,
        color=(:gray70, 0.6),
        strokecolor=:gray50,
        strokewidth=0.5,
        label="Pruned",
    )

    # Selected points (colored)
    scatter!(
        ax,
        selected["pc1"],
        selected["pc2"];
        marker=:diamond,
        markersize=10,
        color=RUHI.teal,
        strokecolor=:white,
        strokewidth=1.0,
        label="FPS selected",
    )

    Legend(fig[2, 1], ax; orientation=:horizontal, tellwidth=false, tellheight=true)

    save_figure(fig, "leps_fps")
end

main()
