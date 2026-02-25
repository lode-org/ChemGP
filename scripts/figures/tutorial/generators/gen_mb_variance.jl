# MB-VARIANCE generator: GP variance overlaid on Muller-Brown PES
#
# Seeds ~21 training points near minima A, C, and saddle S1.
# Trains GP. Evaluates PES and variance on 150x150 grid.
#
# Output: {stem}.h5 with /grids/energy, /grids/variance,
#         /points/training, /points/minima, /points/saddles,
#         metadata: max_var_x, max_var_y, max_var_val, var_clip

include(joinpath(@__DIR__, "common_data.jl"))
using ChemGP
using KernelFunctions
using Random
using LinearAlgebra
using Statistics

function main()
    Random.seed!(42)

    # --- Sample ~21 training points near the A-to-C region ---
    D = 2
    td = TrainingData(D)

    centers = [
        MULLER_BROWN_MINIMA[1],  # A
        MULLER_BROWN_SADDLES[1], # S1
        MULLER_BROWN_MINIMA[3],  # C
    ]

    rng = MersenneTwister(42)
    for center in centers
        for _ in 1:7
            pt = center .+ 0.2 * randn(rng, 2)
            pt[1] = clamp(pt[1], -1.5, 1.2)
            pt[2] = clamp(pt[2], -0.5, 2.0)
            E, G = muller_brown_energy_gradient(pt)
            add_point!(td, pt, E, G)
        end
    end

    println("Training points: $(npoints(td))")

    # --- Build GP model ---
    y_full, y_mean, y_std = ChemGP.normalize(td)
    kernel = 1.0 * with_lengthscale(SqExponentialKernel(), 0.3)
    model = GPModel(kernel, td.X, y_full)
    train_model!(model; iterations=300)

    # --- Evaluate PES and variance on grid ---
    x_range = range(-1.5, 1.2; length=150)
    y_range = range(-0.5, 2.0; length=150)

    E_pes = eval_grid(muller_brown_energy_gradient, x_range, y_range)
    _, E_var = gp_predict_grid(model, x_range, y_range, y_mean, y_std)

    h5_write_grid(h5_path(), "energy", E_pes; x_range=x_range, y_range=y_range)
    h5_write_grid(h5_path(), "variance", E_var; x_range=x_range, y_range=y_range)

    # Training points
    train_x = [td.X[1, i] for i in 1:npoints(td)]
    train_y = [td.X[2, i] for i in 1:npoints(td)]
    h5_write_points(h5_path(), "training"; x=train_x, y=train_y)

    # Minima and saddles
    h5_write_points(h5_path(), "minima";
        x=[m[1] for m in MULLER_BROWN_MINIMA],
        y=[m[2] for m in MULLER_BROWN_MINIMA])
    h5_write_points(h5_path(), "saddles";
        x=[s[1] for s in MULLER_BROWN_SADDLES],
        y=[s[2] for s in MULLER_BROWN_SADDLES])

    # Find max variance in interior (skip boundary to avoid edge artifacts)
    xs = collect(x_range)
    ys = collect(y_range)
    margin = 10
    interior_var = E_var[margin:(end - margin), margin:(end - margin)]
    int_idx = argmax(interior_var)
    max_var_x = xs[margin - 1 + int_idx[1]]
    max_var_y = ys[margin - 1 + int_idx[2]]
    max_var_val = interior_var[int_idx]

    var_clip = quantile(vec(E_var), 0.95)

    h5_write_metadata(h5_path();
        max_var_x=max_var_x,
        max_var_y=max_var_y,
        max_var_val=max_var_val,
        var_clip=var_clip,
    )

    println("Max interior variance at ($max_var_x, $max_var_y) = $max_var_val")
    println("95th percentile variance: $var_clip")
    println("Wrote HDF5: $(h5_path())")
end

main()
