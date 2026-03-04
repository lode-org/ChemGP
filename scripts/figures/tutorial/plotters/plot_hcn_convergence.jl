# HCN-CONV plotter: convergence comparison for HCN -> HNC isomerization
#
# Reads convergence data from HDF5, plots max force vs oracle calls.
#
# Output: hcn_convergence.pdf

include(joinpath(@__DIR__, "common_plot.jl"))
using DataFrames

function main()
    hp = get_h5_path(; fallback_stem="hcn_convergence")
    if !isfile(hp)
        error("HDF5 not found: $hp -- run the generator first")
    end

    tbl = h5_read_table(hp, "table")
    # Use ci_force (climbing image force) when available; fall back to max_force
    force_key = haskey(tbl, "ci_force") ? "ci_force" : "max_force"
    df_raw = DataFrame(
        oracle_calls=tbl["oracle_calls"],
        max_force=tbl[force_key],
        method=tbl["method"],
    )

    # Truncate each method at first crossing below the convergence threshold
    conv_tol = 0.1
    df = DataFrame()
    for gdf in groupby(df_raw, :method)
        cutoff = findfirst(<(conv_tol), gdf.max_force)
        n = isnothing(cutoff) ? nrow(gdf) : cutoff
        append!(df, gdf[1:n, :])
    end

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
        axis=(xlabel="Oracle calls", ylabel=L"$|F|_\mathrm{CI}$ (eV/\AA)", yscale=log10),
        figure=(size=(504, 350),),
    )

    # Convergence threshold (CI force tolerance = 0.1 eV/A)
    hlines!(current_axis(), [0.1]; color=:gray, linewidth=0.8, linestyle=:dash)

    save_figure(fg, "hcn_convergence")
end

main()
