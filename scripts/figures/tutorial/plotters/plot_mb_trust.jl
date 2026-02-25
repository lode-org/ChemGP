# MB-TRUST plotter: trust region illustration
#
# Reads slice data and training points from HDF5. Plots confidence band,
# true surface, GP mean, trust boundary lines, training points projected
# to slice, hypothetical bad step + oracle fallback.
#
# Output: mb_trust_region.pdf

include(joinpath(@__DIR__, "common_plot.jl"))
using LaTeXStrings

function main()
    hp = get_h5_path(; fallback_stem="mb_trust")
    if !isfile(hp)
        error("HDF5 not found: $hp -- run the generator first")
    end

    tbl = h5_read_table(hp, "slice")
    x_slice = tbl["x"]
    E_true = tbl["E_true"]
    E_pred = tbl["E_pred"]
    E_std = tbl["E_std"]
    in_trust = tbl["in_trust"] .> 0.5  # convert back to Bool

    training = h5_read_points(hp, "training")
    meta = h5_read_metadata(hp)
    y_slice = meta["y_slice"]

    # Find trust boundary crossings
    boundary_idx = findall(diff(in_trust) .!= 0)

    set_theme!(PUBLICATION_THEME)

    fig = Figure(; size=(504, 350))
    ax = Axis(fig[1, 1]; xlabel=L"$x$", ylabel=L"$E$ (a.u.)")

    # Confidence band
    band!(
        ax,
        x_slice,
        E_pred .- 2 .* E_std,
        E_pred .+ 2 .* E_std;
        color=(RUHI.sky, 0.25),
    )

    # True surface
    lines!(
        ax,
        x_slice,
        E_true;
        color=:black,
        linewidth=1.0,
        linestyle=:dash,
        label="True surface",
    )

    # GP mean
    lines!(ax, x_slice, E_pred; color=RUHI.teal, linewidth=1.5, label="GP mean")

    # Trust boundary vertical lines
    for idx in boundary_idx
        vlines!(ax, [x_slice[idx]]; color=RUHI.magenta, linewidth=1.0, linestyle=:dot)
    end

    # Training points projected to slice (show their x-coordinates)
    train_x = training["x"]
    train_y = training["y"]
    for i in eachindex(train_x)
        if abs(train_y[i] - y_slice) < 0.3  # nearby points
            # Evaluate true energy at training x along slice for projection
            idx_closest = argmin(abs.(x_slice .- train_x[i]))
            scatter!(ax, [train_x[i]], [E_true[idx_closest]];
                marker=:circle, markersize=6, color=:black)
        end
    end

    # Hypothetical bad step outside trust region
    x_bad = 1.0
    idx_bad = argmin(abs.(x_slice .- x_bad))
    e_bad_pred = E_pred[idx_bad]
    scatter!(
        ax,
        [x_bad],
        [e_bad_pred];
        marker=:xcross,
        markersize=14,
        color=RUHI.coral,
        strokewidth=2,
    )

    # Fallback oracle evaluation
    e_bad_true = E_true[idx_bad]
    scatter!(ax, [x_bad], [e_bad_true]; marker=:star5, markersize=12, color=RUHI.teal)

    # Annotations
    text!(ax, x_bad + 0.05, e_bad_pred + 10;
        text="GP step", fontsize=9, color=RUHI.coral)
    text!(ax, x_bad + 0.05, e_bad_true + 10;
        text="Oracle fallback", fontsize=9, color=RUHI.teal)

    # Trust region label
    if !isempty(boundary_idx)
        bx = x_slice[boundary_idx[end]]
        text!(ax, bx + 0.03, -50;
            text="trust\nboundary", fontsize=8, color=RUHI.magenta)
    end

    ylims!(ax, -250, 100)

    Legend(
        fig[2, 1],
        [
            LineElement(; color=:black, linestyle=:dash),
            LineElement(; color=RUHI.teal, linewidth=1.5),
            PolyElement(; color=(RUHI.sky, 0.25)),
        ],
        ["True surface", "GP mean", L"$\pm 2\sigma$"];
        orientation=:horizontal,
        tellwidth=false,
        tellheight=true,
    )

    save_figure(fig, "mb_trust_region")
end

main()
