# LEPS-RFF plotter: RFF approximation quality vs exact GP on LEPS
#
# Reads RFF comparison data from HDF5, plots 2-panel (energy + gradient MAE
# vs D_rff) on log scale.
#
# Output: leps_rff_quality.pdf

include(joinpath(@__DIR__, "common_plot.jl"))

function main()
    hp = get_h5_path(; fallback_stem="leps_rff")
    if !isfile(hp)
        error("HDF5 not found: $hp -- run the generator first")
    end

    tbl = h5_read_table(hp, "table")
    meta = h5_read_metadata(hp)

    D_rff = tbl["D_rff"]
    e_mae_true = tbl["energy_mae_vs_true"]
    g_mae_true = tbl["gradient_mae_vs_true"]
    e_mae_gp = tbl["energy_mae_vs_gp"]
    g_mae_gp = tbl["gradient_mae_vs_gp"]
    gp_e_mae = meta["gp_e_mae"]
    gp_g_mae = meta["gp_g_mae"]

    set_theme!(PUBLICATION_THEME)

    fig = Figure(; size=(504, 400))

    # Energy panel
    ax1 = Axis(
        fig[1, 1];
        ylabel=L"\text{Energy MAE (eV)}",
        yscale=log10,
        xticklabelsvisible=false,
        title=L"\text{RFF approximation quality on LEPS}",
    )
    lines!(ax1, D_rff, e_mae_true;
        color=RUHI.teal, linewidth=1.5, label="RFF vs true PES")
    scatter!(ax1, D_rff, e_mae_true; color=RUHI.teal, markersize=6)
    lines!(ax1, D_rff, e_mae_gp;
        color=RUHI.sky, linewidth=1.5, label="RFF vs exact GP")
    scatter!(ax1, D_rff, e_mae_gp; color=RUHI.sky, markersize=6)
    hlines!(ax1, [gp_e_mae];
        color=RUHI.magenta, linewidth=0.8, linestyle=:dash, label="Exact GP vs true PES")
    axislegend(ax1; position=:rt, framevisible=false, labelsize=9)

    # Gradient panel
    ax2 = Axis(
        fig[2, 1];
        xlabel=L"D_\mathrm{rff}",
        ylabel=L"\text{Gradient MAE (eV/\AA)}",
        yscale=log10,
    )
    lines!(ax2, D_rff, g_mae_true; color=RUHI.teal, linewidth=1.5)
    scatter!(ax2, D_rff, g_mae_true; color=RUHI.teal, markersize=6)
    lines!(ax2, D_rff, g_mae_gp; color=RUHI.sky, linewidth=1.5)
    scatter!(ax2, D_rff, g_mae_gp; color=RUHI.sky, markersize=6)
    hlines!(ax2, [gp_g_mae]; color=RUHI.magenta, linewidth=0.8, linestyle=:dash)

    rowgap!(fig.layout, 5)

    save_figure(fig, "leps_rff_quality")
end

main()
