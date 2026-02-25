# MB-HYPERPARAMS plotter: 3x3 hyperparameter sensitivity grid
#
# Reads slice data, true surface, GP predictions, and training points
# from HDF5. 3x3 grid showing GP mean +/- 2sigma along 1D slice.
#
# Output: mb_hyperparams.pdf

include(joinpath(@__DIR__, "common_plot.jl"))
using LaTeXStrings

function main()
    hp = get_h5_path(; fallback_stem="mb_hyperparams")
    if !isfile(hp)
        error("HDF5 not found: $hp -- run the generator first")
    end

    slice = h5_read_table(hp, "slice")
    x_slice = slice["x"]

    true_surf = h5_read_table(hp, "true_surface")
    E_true = true_surf["E_true"]

    lengthscales = [0.05, 0.3, 2.0]
    signal_vars = [0.1, 1.0, 100.0]
    ls_labels = [L"$\ell = 0.05$", L"$\ell = 0.3$", L"$\ell = 2.0$"]
    sv_labels = [L"$\sigma_f = 0.1$", L"$\sigma_f = 1.0$", L"$\sigma_f = 100.0$"]

    set_theme!(PUBLICATION_THEME)

    fig = Figure(; size=(750, 650))

    for (j, _ls) in enumerate(lengthscales)
        for (i, _sv) in enumerate(signal_vars)
            ax = Axis(
                fig[i, j];
                title=i == 1 ? ls_labels[j] : "",
                ylabel=j == 1 ? sv_labels[i] : "",
                xlabel=i == 3 ? L"$x$" : "",
                xticklabelsvisible=i == 3,
                yticklabelsvisible=j == 1,
            )

            name = "gp_ls$(j)_sv$(i)"
            gp_data = h5_read_table(hp, name)
            E_pred = gp_data["E_pred"]
            E_std = gp_data["E_std"]

            # Confidence band
            band!(
                ax,
                x_slice,
                E_pred .- 2 .* E_std,
                E_pred .+ 2 .* E_std;
                color=(RUHI.sky, 0.3),
            )

            # True surface
            lines!(ax, x_slice, E_true; color=:black, linewidth=1.0, linestyle=:dash)

            # GP mean
            lines!(ax, x_slice, E_pred; color=RUHI.teal, linewidth=1.5)

            ylims!(ax, -250, 100)
        end
    end

    # Legend
    Legend(
        fig[4, 1:3],
        [
            LineElement(; color=:black, linestyle=:dash, linewidth=1.0),
            LineElement(; color=RUHI.teal, linewidth=1.5),
            PolyElement(; color=(RUHI.sky, 0.3)),
        ],
        ["True surface", "GP mean", L"$\pm 2\sigma$"];
        orientation=:horizontal,
        tellwidth=false,
        tellheight=true,
    )

    save_figure(fig, "mb_hyperparams")
end

main()
