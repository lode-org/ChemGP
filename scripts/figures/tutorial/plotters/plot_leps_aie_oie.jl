# LEPS-NEB plotter: Standard NEB vs GP-NEB AIE/OIE convergence
#
# Reads convergence data from HDF5, plots max force vs oracle calls.
#
# Output: leps_aie_oie.pdf

include(joinpath(@__DIR__, "common_plot.jl"))
using DataFrames

function main()
    hp = get_h5_path(; fallback_stem="leps_aie_oie")
    if !isfile(hp)
        error("HDF5 not found: $hp -- run the generator first")
    end

    tbl = h5_read_table(hp, "table")
    df = DataFrame(
        oracle_calls=tbl["oracle_calls"],
        max_force=tbl["max_force"],
        method=tbl["method"],
    )

    n_methods = length(unique(df.method))
    palette_slice = RUHI_CYCLE[1:min(n_methods, length(RUHI_CYCLE))]

    set_theme!(PUBLICATION_THEME)

    plt =
        data(df) *
        mapping(:oracle_calls, :max_force; color=:method) *
        visual(Lines; linewidth=1.5)

    fg = draw(
        plt,
        scales(; Color=(; palette=palette_slice));
        axis=(xlabel="Oracle calls", ylabel=L"$|F|_\mathrm{max}$ (eV/\AA)", yscale=log10),
        figure=(size=(504, 350),),
    )

    # Convergence threshold
    hlines!(current_axis(), [0.1]; color=:gray, linewidth=0.8, linestyle=:dash)

    save_figure(fg, "leps_aie_oie")
end

main()
