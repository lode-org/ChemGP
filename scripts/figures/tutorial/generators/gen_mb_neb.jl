# MB-NEB generator: NEB path on Muller-Brown surface
#
# Evaluates MB PES on 200x200 grid, runs NEB (11 images, spring 10.0,
# climbing image, max_iter 500). Writes grid, path, minima, and metadata.
#
# Output: {stem}.h5 with /grids/energy, /paths/neb, /points/minima

include(joinpath(@__DIR__, "common_data.jl"))
using ChemGP

function main()
    # --- Grid evaluation (background contour) ---
    x_range = range(-1.5, 1.2; length=200)
    y_range = range(-0.5, 2.0; length=200)
    E = eval_grid(muller_brown_energy_gradient, x_range, y_range)

    h5_write_grid(h5_path(), "energy", E; x_range=x_range, y_range=y_range)

    # --- NEB optimization ---
    x_start = Float64.(MULLER_BROWN_MINIMA[1])  # A: deepest
    x_end = Float64.(MULLER_BROWN_MINIMA[2])    # B: second

    config = NEBConfig(;
        images=11,
        spring_constant=10.0,
        climbing_image=true,
        max_iter=500,
        conv_tol=0.1,
        step_size=1e-4,
    )

    result = neb_optimize(muller_brown_energy_gradient, x_start, x_end; config=config)

    println("NEB converged: $(result.converged)")
    println("Oracle calls: $(result.oracle_calls)")

    # Extract path coordinates
    path_x = [img[1] for img in result.path.images]
    path_y = [img[2] for img in result.path.images]

    h5_write_path(h5_path(), "neb"; x=path_x, y=path_y)

    # Minima
    min_x = [m[1] for m in MULLER_BROWN_MINIMA]
    min_y = [m[2] for m in MULLER_BROWN_MINIMA]
    h5_write_points(h5_path(), "minima"; x=min_x, y=min_y)

    h5_write_metadata(h5_path();
        converged=result.converged,
        oracle_calls=result.oracle_calls,
    )

    println("Wrote HDF5: $(h5_path())")
end

main()
