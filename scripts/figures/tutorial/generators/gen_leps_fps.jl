# LEPS-FPS generator: farthest point sampling subset selection
#
# Runs GP-NEB AIE to collect training data. Computes inverse distance
# features. Runs FPS to select 20 from ~50 candidates. Projects to 2D
# via PCA.
#
# Output: {stem}.h5 with /points/selected, /points/pruned (pc1, pc2)

include(joinpath(@__DIR__, "common_data.jl"))
using ChemGP
using KernelFunctions
using LinearAlgebra
using Statistics
using Random

function main()
    Random.seed!(42)

    # --- Run GP-NEB AIE to collect training data ---
    println("Running GP-NEB AIE on LEPS to collect training points...")
    kernel = MolInvDistSE([1, 1, 1], Float64[])

    config = NEBConfig(;
        images=7,
        spring_constant=5.0,
        climbing_image=true,
        conv_tol=0.5,
        gp_train_iter=50,
        max_outer_iter=10,
        trust_radius=0.1,
        verbose=true,
    )

    result = gp_neb_aie(
        leps_energy_gradient,
        Float64.(LEPS_REACTANT),
        Float64.(LEPS_PRODUCT),
        kernel;
        config=config,
    )

    # Collect all images from the final path as candidate training points
    candidates = hcat(result.path.images...)

    # Add perturbed copies to get ~50 candidates
    rng = MersenneTwister(42)
    n_needed = max(0, 50 - size(candidates, 2))
    perturbed_cols = [
        result.path.images[rand(rng, 1:length(result.path.images))] .+ 0.1 * randn(rng, 9)
        for _ in 1:n_needed
    ]
    if !isempty(perturbed_cols)
        candidates = hcat(candidates, perturbed_cols...)
    end

    n_candidates = size(candidates, 2)
    println("Candidate points: $n_candidates")

    # --- Compute inverse distance features ---
    frozen = Float64[]
    feature_matrix = zeros(3, n_candidates)
    for i in 1:n_candidates
        feature_matrix[:, i] = compute_inverse_distances(candidates[:, i], frozen)
    end

    # --- FPS: select 20 points ---
    n_select = 20
    X_selected = reshape(feature_matrix[:, 1], 3, 1)
    remaining_mat = Matrix(feature_matrix[:, 2:end])

    euclidean_dist(x, y) = LinearAlgebra.norm(x - y)
    selected_indices = farthest_point_sampling(
        remaining_mat, X_selected, n_select - 1; distance_fn=euclidean_dist)
    all_selected = vcat([1], selected_indices .+ 1)

    println("Selected $(length(all_selected)) points via FPS")

    # --- PCA for 2D projection ---
    F = feature_matrix
    F_centered = F .- mean(F; dims=2)
    C = F_centered * F_centered' / n_candidates
    eigvals, eigvecs = eigen(C; sortby=x -> -x)
    proj = eigvecs[:, 1:2]' * F_centered  # 2 x N

    # Split into selected and pruned
    is_selected = falses(n_candidates)
    for idx in all_selected
        is_selected[idx] = true
    end

    proj_sel = proj[:, is_selected]
    proj_pru = proj[:, .!is_selected]

    h5_write_points(h5_path(), "selected"; pc1=proj_sel[1, :], pc2=proj_sel[2, :])
    h5_write_points(h5_path(), "pruned"; pc1=proj_pru[1, :], pc2=proj_pru[2, :])

    println("Wrote HDF5: $(h5_path())")
end

main()
