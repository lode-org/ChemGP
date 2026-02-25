# MB-HYPERPARAMS generator: hyperparameter sensitivity data
#
# Seeds ~15 training points near the MEP. For each (ls, sv) combination
# in a 3x3 grid, builds GP with fixed hyperparams, predicts along 1D
# slice at y=0.5.
#
# Output: {stem}.h5 with /slice, /true_surface, /gp_ls{j}_sv{i},
#         /points/training

include(joinpath(@__DIR__, "common_data.jl"))
using ChemGP
using KernelFunctions
using Random
using LinearAlgebra

function main()
    Random.seed!(42)

    # --- Sample ~15 training points along the MEP region ---
    D = 2
    td = TrainingData(D)

    rng = MersenneTwister(42)
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
    x_slice = collect(range(-1.5, 1.2; length=200))

    # True surface along slice
    E_true = [muller_brown_energy_gradient([x, y_slice])[1] for x in x_slice]

    h5_write_table(h5_path(), "slice", Dict("x" => x_slice))
    h5_write_table(h5_path(), "true_surface", Dict("E_true" => E_true))

    # --- Hyperparameter grid ---
    lengthscales = [0.05, 0.3, 2.0]
    signal_vars = [0.1, 1.0, 100.0]

    for (j, ls) in enumerate(lengthscales)
        for (i, sv) in enumerate(signal_vars)
            # Build kernel with fixed hyperparameters
            kernel = sv * with_lengthscale(SqExponentialKernel(), ls)
            model = GPModel(kernel, td.X, y_full)
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

            name = "gp_ls$(j)_sv$(i)"
            h5_write_table(h5_path(), name, Dict(
                "E_pred" => E_pred,
                "E_std" => E_std,
            ))

            println("ls=$ls, sv=$sv -> $(name)")
        end
    end

    # Training points
    train_x = [td.X[1, i] for i in 1:npoints(td)]
    train_y = [td.X[2, i] for i in 1:npoints(td)]
    h5_write_points(h5_path(), "training"; x=train_x, y=train_y)

    println("Wrote HDF5: $(h5_path())")
end

main()
