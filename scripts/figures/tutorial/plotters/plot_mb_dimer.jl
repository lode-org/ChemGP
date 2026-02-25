# MB-DIMER plotter: GP-dimer vs classical dimer convergence
#
# Reads convergence data from HDF5, plots force norm vs oracle calls.
#
# Output: mb_dimer_convergence.pdf

include(joinpath(@__DIR__, "common_plot.jl"))
using DataFrames

function main()
    hp = get_h5_path(; fallback_stem="mb_dimer")
    if !isfile(hp)
        error("HDF5 not found: $hp -- run the generator first")
    end

    tbl = h5_read_table(hp, "table")
    df = DataFrame(
        oracle_calls=tbl["oracle_calls"],
        force_norm=tbl["force_norm"],
        method=tbl["method"],
    )

    set_theme!(PUBLICATION_THEME)

    plt =
        data(df) *
        mapping(:oracle_calls, :force_norm; color=:method) *
        visual(Lines; linewidth=1.5)

    fg = draw(
        plt,
        scales(; Color=(; palette=RUHI_CYCLE));
        axis=(xlabel="Oracle calls", ylabel=L"$|F|_{\mathrm{true}}$", yscale=log10),
        figure=(size=(504, 350),),
    )

    # Convergence threshold
    hlines!(current_axis(), [1e-3]; color=:gray, linewidth=0.8, linestyle=:dash)

    save_figure(fg, "mb_dimer_convergence")
end

main()
