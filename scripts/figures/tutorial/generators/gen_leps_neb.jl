# LEPS-NEB generator: NEB path on LEPS surface
#
# Evaluates 2D LEPS contour (r_AB, r_BC grid). Runs NEB in 9D.
# Projects images to 2D. Writes grid, path, and endpoints.
#
# Output: {stem}.h5 with /grids/energy, /paths/neb, /points/endpoints

include(joinpath(@__DIR__, "common_data.jl"))
using ChemGP
using LinearAlgebra

function main()
    # --- Grid evaluation (2D background) ---
    r_AB_range = range(0.5, 4.0; length=200)
    r_BC_range = range(0.5, 4.0; length=200)
    E = eval_grid(leps_energy_gradient_2d, r_AB_range, r_BC_range)

    h5_write_grid(h5_path(), "energy", E;
        x_range=r_AB_range, y_range=r_BC_range)

    # --- NEB optimization in 9D ---
    config = NEBConfig(;
        images=9,
        spring_constant=5.0,
        climbing_image=true,
        max_iter=500,
        conv_tol=0.05,
        step_size=0.005,
    )

    result = neb_optimize(
        leps_energy_gradient,
        Float64.(LEPS_REACTANT),
        Float64.(LEPS_PRODUCT);
        config=config,
    )

    println("NEB converged: $(result.converged)")
    println("Oracle calls: $(result.oracle_calls)")

    # Project 9D images to 2D (r_AB, r_BC)
    path_rAB = Float64[]
    path_rBC = Float64[]
    for img in result.path.images
        rA = img[1:3]
        rB = img[4:6]
        rC = img[7:9]
        push!(path_rAB, norm(rB - rA))
        push!(path_rBC, norm(rC - rB))
    end

    h5_write_path(h5_path(), "neb"; rAB=path_rAB, rBC=path_rBC)

    # Endpoints
    h5_write_points(h5_path(), "endpoints";
        rAB=[path_rAB[1], path_rAB[end]],
        rBC=[path_rBC[1], path_rBC[end]])

    h5_write_metadata(h5_path();
        converged=result.converged,
        oracle_calls=result.oracle_calls,
    )

    println("Wrote HDF5: $(h5_path())")
end

main()
