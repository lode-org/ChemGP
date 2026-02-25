# HCN-NEB plotter: energy profile for HCN -> HNC isomerization
#
# Reads profile data from HDF5, plots image index vs delta E.
#
# Output: hcn_neb_profile.pdf

include(joinpath(@__DIR__, "common_plot.jl"))
using DataFrames

function main()
    hp = get_h5_path(; fallback_stem="hcn_neb")
    if !isfile(hp)
        error("HDF5 not found: $hp -- run the generator first")
    end

    tbl = h5_read_table(hp, "table")
    df = DataFrame(
        image=tbl["image"],
        energy=tbl["energy"],
        method=tbl["method"],
    )

    n_methods = length(unique(df.method))
    palette_slice = RUHI_CYCLE[1:min(n_methods, length(RUHI_CYCLE))]

    set_theme!(PUBLICATION_THEME)

    if n_methods > 1
        plt =
            data(df) *
            mapping(:image, :energy; color=:method) *
            (visual(Lines; linewidth=1.5) + visual(Scatter; markersize=6))

        fg = draw(
            plt,
            scales(; Color=(; palette=palette_slice));
            axis=(xlabel="Image index", ylabel=L"$\Delta E$ (eV)"),
            figure=(size=(504, 350),),
            legend=(position=:top,),
        )
    else
        fig = Figure(; size=(504, 350))
        ax = Axis(fig[1, 1]; xlabel="Image index", ylabel=L"$\Delta E$ (eV)")
        lines!(ax, df.image, df.energy;
            linewidth=1.5, color=RUHI.teal, label=df.method[1])
        scatter!(ax, df.image, df.energy; markersize=6, color=RUHI.teal)
        Legend(fig[0, 1], ax)
        fg = fig
    end

    save_figure(fg, "hcn_neb_profile")
end

main()
