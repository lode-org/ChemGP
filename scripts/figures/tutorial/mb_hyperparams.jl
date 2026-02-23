# MB-5: Hyperparameter sensitivity (3x3 grid)
#
# Shows the effect of lengthscale and signal variance on GP regression quality.
# Uses ~15 training points near the MEP. Each cell shows GP mean +/- 2sigma
# along a 1D slice at y=0.5. Center cell has the "good" hyperparameters.

using ChemGP
using KernelFunctions
using Random
using LinearAlgebra
include(joinpath(@__DIR__, "common.jl"))

Random.seed!(42)

# --- Sample ~15 training points along the MEP region ---
D = 2
td = TrainingData(D)

rng = MersenneTwister(42)
# Points from the three stationary-point clusters
for center in [MULLER_BROWN_MINIMA..., MULLER_BROWN_SADDLES...]
    for _ in 1:3
        pt = center .+ 0.15 * randn(rng, 2)
        pt[1] = clamp(pt[1], -1.5, 1.2)
        pt[2] = clamp(pt[2], -0.5, 2.0)
        E, G = muller_brown_energy_gradient(pt)
        add_point!(td, pt, E, G)
    end
end

println("Training points: $(npoints(td))")

# Normalize once (shared across all panels)
y_full, y_mean, y_std = ChemGP.normalize(td)

# --- 1D slice ---
y_slice = 0.5
x_slice = range(-1.5, 1.2; length=200)

# True surface along slice
E_true = [muller_brown_energy_gradient([x, y_slice])[1] for x in x_slice]

# --- Hyperparameter grid ---
lengthscales = [0.05, 0.3, 2.0]
signal_vars = [0.1, 1.0, 100.0]
ls_labels = [L"$\ell = 0.05$", L"$\ell = 0.3$", L"$\ell = 2.0$"]
sv_labels = [L"$\sigma_f = 0.1$", L"$\sigma_f = 1.0$", L"$\sigma_f = 100.0$"]

# --- Plot ---
set_theme!(PUBLICATION_THEME)

fig = Figure(; size=(750, 650))

for (j, ls) in enumerate(lengthscales)
    for (i, sv) in enumerate(signal_vars)
        ax = Axis(
            fig[i, j];
            title=i == 1 ? ls_labels[j] : "",
            ylabel=j == 1 ? sv_labels[i] : "",
            xlabel=i == 3 ? L"$x$" : "",
            xticklabelsvisible=i == 3,
            yticklabelsvisible=j == 1,
        )

        # Build kernel with fixed hyperparameters
        kernel = sv * with_lengthscale(SqExponentialKernel(), ls)
        model = GPModel(kernel, td.X, y_full)
        # Set noise manually (do not train -- fixed hyperparams)
        model.noise_var = 1e-4
        model.grad_noise_var = 1e-4

        # Predict along slice
        E_pred = Float64[]
        E_std = Float64[]
        for x in x_slice
            X_test = reshape([x, y_slice], 2, 1)
            mu, var = predict_with_variance(model, X_test)
            push!(E_pred, mu[1] * y_std + y_mean)
            push!(E_std, sqrt(max(var[1], 0.0)) * y_std)
        end

        # Confidence band
        band!(
            ax,
            collect(x_slice),
            E_pred .- 2 .* E_std,
            E_pred .+ 2 .* E_std;
            color=(RUHI.sky, 0.3),
        )

        # True surface
        lines!(ax, collect(x_slice), E_true; color=:black, linewidth=1.0, linestyle=:dash)

        # GP mean
        lines!(ax, collect(x_slice), E_pred; color=RUHI.teal, linewidth=1.5)

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
