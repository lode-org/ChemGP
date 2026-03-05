# LEPS-NLL plotter: MAP-NLL landscape in hyperparameter space
#
# Reads 2D grids (nll, grad_norm) and MAP optimum from HDF5.
# Contourf of NLL with gradient norm contour overlay.
#
# Output: leps_nll_landscape.pdf

include(joinpath(@__DIR__, "common_plot.jl"))
using LaTeXStrings
using Statistics

function main()
    hp = get_h5_path(; fallback_stem="leps_nll")
    if !isfile(hp)
        error("HDF5 not found: $hp -- run the generator first")
    end

    nll_data, xr, yr = h5_read_grid(hp, "nll")
    grad_data, _, _ = h5_read_grid(hp, "grad_norm")
    opt = h5_read_points(hp, "optimum")

    set_theme!(PUBLICATION_THEME)

    fig = Figure(; size=(504, 400))

    ax = Axis(
        fig[1, 1];
        xlabel=L"$\ln\,\sigma^2$",
        ylabel=L"$\ln\,\theta$",
    )

    # Clip NLL for visualization (remove extreme outliers)
    nll_finite = filter(isfinite, vec(nll_data))
    if isempty(nll_finite)
        error("No finite NLL values")
    end
    lo = quantile(nll_finite, 0.02)
    hi = quantile(nll_finite, 0.98)
    nll_clipped = clamp.(nll_data, lo, hi)

    # NLL contourf
    hm = contourf!(
        ax,
        xr, yr, nll_clipped;
        colormap=Reverse(ENERGY_COLORMAP),
        levels=20,
    )
    Colorbar(fig[1, 2], hm; label=L"$\mathcal{L}_{\mathrm{MAP}}$")

    # Gradient norm contours (overlay)
    grad_finite = filter(isfinite, vec(grad_data))
    if !isempty(grad_finite)
        g_lo = quantile(grad_finite, 0.05)
        g_hi = quantile(grad_finite, 0.80)
        grad_clipped = clamp.(grad_data, g_lo, g_hi)
        contour!(
            ax,
            xr, yr, grad_clipped;
            color=:black,
            linewidth=0.5,
            levels=8,
            linestyle=:dash,
        )
    end

    # MAP optimum marker
    scatter!(
        ax,
        opt["log_sigma2"], opt["log_theta"];
        color=RUHI.coral,
        markersize=10,
        marker=:star5,
        strokecolor=:white,
        strokewidth=1,
    )

    save_figure(fig, "leps_nll_landscape")
end

main()
