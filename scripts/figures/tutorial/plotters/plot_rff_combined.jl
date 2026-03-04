# RFF-COMBINED plotter: RFF approximation quality for LEPS and C2H4NO
#
# Reads HDF5 data from both LEPS and C2H4NO (PET-MAD) RFF quality generators.
# Plots a single metric (RFF vs exact GP) for both surfaces in a combined
# 2-panel figure (energy MAE top, gradient MAE bottom).
#
# Expects:
#   output/leps_rff.h5       (from leps_rff_quality.rs)
#   output/petmad_rff.h5     (from petmad_rff_quality.rs)
#
# Output: rff_quality_combined.pdf

include(joinpath(@__DIR__, "common_plot.jl"))

function main()
    leps_h5 = joinpath(OUTPUT_DIR, "leps_rff.h5")
    petmad_h5 = joinpath(OUTPUT_DIR, "petmad_rff.h5")

    if !isfile(leps_h5)
        error("Missing $leps_h5 -- run gen_leps_rff.jl first")
    end
    if !isfile(petmad_h5)
        error("Missing $petmad_h5 -- run gen_petmad_rff.jl first")
    end

    tbl_leps = h5_read_table(leps_h5, "table")
    tbl_petmad = h5_read_table(petmad_h5, "table")

    # Read molecule label from HDF5 attrs (fallback to C2H4NO)
    mol_label = HDF5.h5open(petmad_h5) do f
        haskey(HDF5.attributes(f), "system") ?
            HDF5.read_attribute(f, "system") : "C2H4NO"
    end
    leps_label = "H3 (LEPS)"
    mol_label_full = "$mol_label (PET-MAD)"

    set_theme!(PUBLICATION_THEME)

    fig = Figure(; size=(504, 400))

    # Energy panel
    ax1 = Axis(
        fig[1, 1];
        ylabel=L"\text{Energy MAE (eV)}",
        yscale=log10,
        xticklabelsvisible=false,
        title=L"\text{RFF vs exact GP approximation error}",
    )
    lines!(
        ax1,
        tbl_leps["D_rff"],
        tbl_leps["energy_mae_vs_gp"];
        color=RUHI.teal,
        linewidth=1.5,
        label=leps_label,
    )
    scatter!(ax1, tbl_leps["D_rff"], tbl_leps["energy_mae_vs_gp"];
        color=RUHI.teal, markersize=6)
    lines!(
        ax1,
        tbl_petmad["D_rff"],
        tbl_petmad["energy_mae_vs_gp"];
        color=RUHI.coral,
        linewidth=1.5,
        label=mol_label_full,
    )
    scatter!(ax1, tbl_petmad["D_rff"], tbl_petmad["energy_mae_vs_gp"];
        color=RUHI.coral, markersize=6)
    axislegend(ax1; position=:rt, framevisible=false, labelsize=10)

    # Gradient panel
    ax2 = Axis(
        fig[2, 1];
        xlabel=L"D_\mathrm{rff}",
        ylabel=L"\text{Gradient MAE (eV/\AA)}",
        yscale=log10,
    )
    lines!(ax2, tbl_leps["D_rff"], tbl_leps["gradient_mae_vs_gp"];
        color=RUHI.teal, linewidth=1.5)
    scatter!(ax2, tbl_leps["D_rff"], tbl_leps["gradient_mae_vs_gp"];
        color=RUHI.teal, markersize=6)
    lines!(ax2, tbl_petmad["D_rff"], tbl_petmad["gradient_mae_vs_gp"];
        color=RUHI.coral, linewidth=1.5)
    scatter!(ax2, tbl_petmad["D_rff"], tbl_petmad["gradient_mae_vs_gp"];
        color=RUHI.coral, markersize=6)

    rowgap!(fig.layout, 5)

    save_figure(fig, "rff_quality_combined")
end

main()
