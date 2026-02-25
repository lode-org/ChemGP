# LEPS-MIN plotter: GP-minimization vs classical L-BFGS convergence
#
# Reads convergence data from HDF5, plots max per-atom force vs oracle calls.
#
# Output: leps_minimize_convergence.pdf

include(joinpath(@__DIR__, "common_plot.jl"))

function main()
    hp = get_h5_path(; fallback_stem="leps_minimize")
    if !isfile(hp)
        error("HDF5 not found: $hp -- run the generator first")
    end

    tbl = h5_read_table(hp, "table")
    oc = tbl["oracle_calls"]
    fatom = tbl["max_fatom"]
    method = tbl["method"]

    # Split by method
    gp_mask = method .== "GP-minimization"
    cl_mask = method .== "Classical L-BFGS"

    set_theme!(PUBLICATION_THEME)

    fig = Figure(; size=(504, 350))
    ax = Axis(
        fig[1, 1];
        xlabel="Oracle calls",
        ylabel=L"max $|F_\mathrm{atom}|$ (eV/\AA)",
        yscale=log10,
    )

    lines!(ax, oc[gp_mask], fatom[gp_mask];
        color=RUHI.teal, linewidth=1.5, label="GP-minimization")
    lines!(ax, oc[cl_mask], fatom[cl_mask];
        color=RUHI.sky, linewidth=1.5, label="Classical L-BFGS")
    hlines!(ax, [0.01]; color=:gray, linewidth=0.8, linestyle=:dash)

    axislegend(ax; position=:rt, framevisible=false, labelsize=10)

    save_figure(fig, "leps_minimize_convergence")
end

main()
