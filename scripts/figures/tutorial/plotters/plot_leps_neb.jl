# LEPS-NEB plotter: NEB path on LEPS surface
#
# Reads grid, path, and endpoints from HDF5. Contourf + NEB path +
# numbered images + star endpoints.
#
# Output: leps_neb.pdf

include(joinpath(@__DIR__, "common_plot.jl"))
using LaTeXStrings

function main()
    hp = get_h5_path(; fallback_stem="leps_neb")
    if !isfile(hp)
        error("HDF5 not found: $hp -- run the generator first")
    end

    E, r_AB_range, r_BC_range = h5_read_grid(hp, "energy")
    neb = h5_read_path(hp, "neb")
    endpoints = h5_read_points(hp, "endpoints")

    path_rAB = neb["rAB"]
    path_rBC = neb["rBC"]
    E_clipped = clamp.(E, -5.0, 5.0)

    set_theme!(PUBLICATION_THEME)

    fig = Figure(; size=(504, 440))
    ax = Axis(
        fig[1, 1];
        xlabel=L"$r_\mathrm{AB}$ (\AA)",
        ylabel=L"$r_\mathrm{BC}$ (\AA)",
        aspect=DataAspect(),
    )

    # Background contour
    cf = contourf!(
        ax,
        collect(r_AB_range),
        collect(r_BC_range),
        E_clipped;
        levels=range(-5.0, 5.0; length=30),
        colormap=ENERGY_COLORMAP,
    )
    contour!(
        ax,
        collect(r_AB_range),
        collect(r_BC_range),
        E_clipped;
        levels=range(-5.0, 5.0; step=0.5),
        color=:black,
        linewidth=0.3,
    )

    # NEB path
    lines!(ax, path_rAB, path_rBC; color=:white, linewidth=2.0)
    scatter!(
        ax,
        path_rAB,
        path_rBC;
        marker=:circle,
        markersize=8,
        color=RUHI.coral,
        strokecolor=:white,
        strokewidth=1.0,
    )

    # Number images
    for i in eachindex(path_rAB)
        text!(ax, path_rAB[i] + 0.08, path_rBC[i] + 0.08;
            text=string(i), fontsize=8, color=:white)
    end

    # Mark endpoints
    ep_rAB = endpoints["rAB"]
    ep_rBC = endpoints["rBC"]
    scatter!(
        ax,
        ep_rAB,
        ep_rBC;
        marker=:star5,
        markersize=12,
        color=RUHI.sunshine,
        strokecolor=:white,
        strokewidth=1.0,
    )

    Colorbar(fig[1, 2], cf; label=L"$E$ (eV)")

    save_figure(fig, "leps_neb")
end

main()
