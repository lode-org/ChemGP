# MB-GP generator: GP surrogate quality progression
#
# For N = [5, 15, 30, 50]: generates Latin hypercube points, trains GP
# (SE kernel, lengthscale 0.3, 300 iterations), evaluates GP mean on grid.
# Writes true surface grid, GP mean grids, and training point sets.
#
# Output: {stem}.h5 with /grids/true_energy, /grids/gp_mean_N{n},
#         /points/train_N{n}

include(joinpath(@__DIR__, "common_data.jl"))
using ChemGP
using KernelFunctions
using Random
using LinearAlgebra

"""Latin hypercube sampling in 2D."""
function latin_hypercube_2d(n, x_lo, x_hi, y_lo, y_hi; rng=Random.GLOBAL_RNG)
    xs = Float64[]
    ys = Float64[]
    dx = (x_hi - x_lo) / n
    dy = (y_hi - y_lo) / n
    for i in 1:n
        push!(xs, x_lo + (i - 1 + rand(rng)) * dx)
        push!(ys, y_lo + (i - 1 + rand(rng)) * dy)
    end
    shuffle!(rng, ys)
    return xs, ys
end

"""Build and train GP from 2D training points on MB surface."""
function build_gp(xs, ys)
    D = 2
    td = TrainingData(D)
    for i in eachindex(xs)
        E, G = muller_brown_energy_gradient([xs[i], ys[i]])
        add_point!(td, [xs[i], ys[i]], E, G)
    end
    y_full, y_mean, y_std = ChemGP.normalize(td)
    kernel = 1.0 * with_lengthscale(SqExponentialKernel(), 0.3)
    model = GPModel(kernel, td.X, y_full)
    train_model!(model; iterations=300)
    return model, y_mean, y_std
end

function main()
    Random.seed!(42)

    # --- Grid ---
    x_range = range(-1.5, 1.2; length=100)
    y_range = range(-0.5, 2.0; length=100)

    # True surface
    E_true = eval_grid(muller_brown_energy_gradient, x_range, y_range)
    h5_write_grid(h5_path(), "true_energy", E_true; x_range=x_range, y_range=y_range)

    # --- For each N, sample, train, predict ---
    n_points_list = [5, 15, 30, 50]

    for n_pts in n_points_list
        xs, ys = latin_hypercube_2d(n_pts, -1.5, 1.2, -0.5, 2.0;
            rng=MersenneTwister(42))

        model, y_mean, y_std = build_gp(xs, ys)
        E_gp, _ = gp_predict_grid(model, x_range, y_range, y_mean, y_std)

        h5_write_grid(h5_path(), "gp_mean_N$(n_pts)", E_gp;
            x_range=x_range, y_range=y_range)
        h5_write_points(h5_path(), "train_N$(n_pts)"; x=xs, y=ys)

        println("N=$n_pts: GP trained and predicted")
    end

    println("Wrote HDF5: $(h5_path())")
end

main()
