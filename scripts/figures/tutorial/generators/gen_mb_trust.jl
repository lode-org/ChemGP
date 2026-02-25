# MB-TRUST generator: trust region illustration data
#
# 10 training points clustered in [-0.5, 0.5] x [0.3, 0.8]. Trains GP.
# Evaluates along 1D slice at y=0.5 (300 points from -1.5 to 1.2).
# Computes distances to data and trust mask.
#
# Output: {stem}.h5 with /slice (x, E_true, E_pred, E_std, dist_to_data,
#         in_trust), /points/training, metadata: trust_radius, y_slice

include(joinpath(@__DIR__, "common_data.jl"))
using ChemGP
using KernelFunctions
using Random
using LinearAlgebra

function main()
    Random.seed!(42)

    # --- Training points clustered in a small region ---
    D = 2
    td = TrainingData(D)

    rng = MersenneTwister(42)
    for _ in 1:10
        x = -0.5 + 1.0 * rand(rng)    # [-0.5, 0.5]
        y = 0.3 + 0.5 * rand(rng)     # [0.3, 0.8]
        E, G = muller_brown_energy_gradient([x, y])
        add_point!(td, [x, y], E, G)
    end

    println("Training points: $(npoints(td))")

    # --- Build GP model ---
    y_full, y_mean, y_std = ChemGP.normalize(td)
    kernel = 1.0 * with_lengthscale(SqExponentialKernel(), 0.3)
    model = GPModel(kernel, td.X, y_full)
    train_model!(model; iterations=300)

    # --- 1D slice at y=0.5 ---
    y_slice = 0.5
    trust_r = 0.3
    x_slice = collect(range(-1.5, 1.2; length=300))

    E_true = Float64[]
    E_pred = Float64[]
    E_std_vals = Float64[]
    dists = Float64[]

    for x in x_slice
        e_true, _ = muller_brown_energy_gradient([x, y_slice])
        push!(E_true, e_true)

        X_test = reshape([x, y_slice], 2, 1)
        mu, var = predict_with_variance(model, X_test)
        push!(E_pred, mu[1] * y_std + y_mean)
        push!(E_std_vals, sqrt(max(var[1], 0.0)) * y_std)

        d = min_distance_to_data([x, y_slice], td.X)
        push!(dists, d)
    end

    in_trust = Float64.(dists .<= trust_r)  # store as Float64 for HDF5

    h5_write_table(h5_path(), "slice", Dict(
        "x" => x_slice,
        "E_true" => E_true,
        "E_pred" => E_pred,
        "E_std" => E_std_vals,
        "dist_to_data" => dists,
        "in_trust" => in_trust,
    ))

    # Training points
    train_x = [td.X[1, i] for i in 1:npoints(td)]
    train_y = [td.X[2, i] for i in 1:npoints(td)]
    h5_write_points(h5_path(), "training"; x=train_x, y=train_y)

    h5_write_metadata(h5_path(); trust_radius=trust_r, y_slice=y_slice)

    println("Wrote HDF5: $(h5_path())")
end

main()
