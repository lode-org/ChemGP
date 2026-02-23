# RFF-COMBINED: RFF approximation quality for LEPS and PET-MAD on same axes
#
# Reads cached CSV data from the individual RFF quality scripts and plots
# a single metric (RFF vs exact GP) for both surfaces in a combined 2-panel
# figure (energy MAE top, gradient MAE bottom).
#
# Expects:
#   output/leps_rff_quality.csv      (from leps_rff_quality.jl)
#   output/petmad_cache/rff_quality.csv  (from petmad_rff_quality.jl)
#
# Output: rff_quality_combined.pdf
#
# Intended for sec:rff of the GPR tutorial review.

using DataFrames
using CSV
using LaTeXStrings
include(joinpath(@__DIR__, "common.jl"))

function main()
    leps_csv = joinpath(OUTPUT_DIR, "leps_rff_quality.csv")
    petmad_csv = joinpath(OUTPUT_DIR, "petmad_cache", "rff_quality.csv")

    if !isfile(leps_csv)
        error("Missing $leps_csv -- run leps_rff_quality.jl first")
    end
    if !isfile(petmad_csv)
        error("Missing $petmad_csv -- run petmad_rff_quality.jl first")
    end

    df_leps = CSV.read(leps_csv, DataFrame)
    df_petmad = CSV.read(petmad_csv, DataFrame)

    # --- Plot: 2-panel combined (energy + gradient MAE vs D_rff) ---
    set_theme!(PUBLICATION_THEME)

    fig = Figure(; size=(504, 400))

    # Energy panel
    ax1 = Axis(
        fig[1, 1];
        ylabel=L"Energy MAE (eV)",
        yscale=log10,
        xticklabelsvisible=false,
        title=L"RFF vs exact GP approximation error",
    )
    lines!(
        ax1,
        df_leps.D_rff,
        df_leps.energy_mae_vs_gp;
        color=RUHI.teal,
        linewidth=1.5,
        label="LEPS",
    )
    scatter!(ax1, df_leps.D_rff, df_leps.energy_mae_vs_gp; color=RUHI.teal, markersize=6)
    lines!(
        ax1,
        df_petmad.D_rff,
        df_petmad.energy_mae_vs_gp;
        color=RUHI.coral,
        linewidth=1.5,
        label="PET-MAD",
    )
    scatter!(
        ax1, df_petmad.D_rff, df_petmad.energy_mae_vs_gp; color=RUHI.coral, markersize=6
    )
    axislegend(ax1; position=:rt, framevisible=false, labelsize=10)

    # Gradient panel
    ax2 = Axis(
        fig[2, 1]; xlabel=L"$D_\mathrm{rff}$", ylabel=L"Gradient MAE (eV/\AA)", yscale=log10
    )
    lines!(ax2, df_leps.D_rff, df_leps.gradient_mae_vs_gp; color=RUHI.teal, linewidth=1.5)
    scatter!(ax2, df_leps.D_rff, df_leps.gradient_mae_vs_gp; color=RUHI.teal, markersize=6)
    lines!(
        ax2, df_petmad.D_rff, df_petmad.gradient_mae_vs_gp; color=RUHI.coral, linewidth=1.5
    )
    scatter!(
        ax2, df_petmad.D_rff, df_petmad.gradient_mae_vs_gp; color=RUHI.coral, markersize=6
    )

    rowgap!(fig.layout, 5)

    save_figure(fig, "rff_quality_combined")
end

main()
