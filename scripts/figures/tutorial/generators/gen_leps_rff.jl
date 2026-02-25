# LEPS-RFF generator: RFF approximation quality vs exact GP on LEPS
#
# Trains an exact GP on LEPS data accumulated during a short minimization,
# then builds RFF models at varying D_rff and compares prediction accuracy
# (energy MAE and gradient MAE) against the exact GP on held-out test points.
#
# Output: {stem}.h5 with /table group and metadata (gp_e_mae, gp_g_mae)

include(joinpath(@__DIR__, "common_data.jl"))
using ChemGP
using LinearAlgebra
using Random
using Statistics

function main()
    Random.seed!(123)

    # --- Generate training data from a short GP-minimization on LEPS ---
    x_init = Float64.(LEPS_REACTANT) .+ 0.2 .* (rand(9) .- 0.5)
    oracle = leps_energy_gradient
    D = length(x_init)

    println("Collecting training data from GP-minimization on LEPS...")
    kernel_init = MolInvDistSE([1, 1, 1], Float64[])
    config = MinimizationConfig(;
        trust_radius=0.15,
        conv_tol=1e-2,
        max_iter=40,
        gp_train_iter=200,
        n_initial_perturb=6,
        perturb_scale=0.1,
        verbose=false,
    )
    result = gp_minimize(oracle, copy(x_init), kernel_init; config=config)

    # Build training set from trajectory
    td = TrainingData(D)
    for x in result.trajectory
        E, G = oracle(x)
        add_point!(td, x, E, G)
    end
    N_train = npoints(td)
    println("Training points: $N_train")

    # --- Generate test data: perturb final geometry ---
    N_test = 20
    X_test = zeros(D, N_test)
    E_true = zeros(N_test)
    G_true = zeros(D, N_test)
    for i in 1:N_test
        x_t = result.x_final .+ 0.08 .* (rand(D) .- 0.5)
        E, G = oracle(x_t)
        X_test[:, i] = x_t
        E_true[i] = E
        G_true[:, i] = G
    end

    # --- Train exact GP ---
    println("Training exact GP...")
    y_gp, y_mean, y_std = ChemGP.normalize(td)

    mol_kernel = MolInvDistSE([1, 1, 1], Float64[])
    model = GPModel(mol_kernel, td.X, y_gp; noise_var=1e-2, grad_noise_var=1e-1, jitter=1e-3)
    train_model!(model; iterations=300)

    # Exact GP predictions on test set
    exact_preds = predict(model, X_test)
    dim_block = D + 1
    exact_E = [exact_preds[(i - 1) * dim_block + 1] * y_std + y_mean for i in 1:N_test]
    exact_G = zeros(D, N_test)
    for i in 1:N_test
        offset = (i - 1) * dim_block
        exact_G[:, i] = exact_preds[(offset + 2):(offset + dim_block)] .* y_std
    end

    # --- RFF at varying D_rff ---
    D_rff_values = [25, 50, 100, 150, 200, 300]

    all_drff = Int[]
    all_e_mae_true = Float64[]
    all_g_mae_true = Float64[]
    all_e_mae_gp = Float64[]
    all_g_mae_gp = Float64[]

    for D_rff in D_rff_values
        println("Building RFF with D_rff = $D_rff...")

        rff = build_rff(mol_kernel, td.X, y_gp, D_rff; noise_var=1e-2, grad_noise_var=1e-1)

        rff_preds = predict(rff, X_test)
        rff_E = [rff_preds[(i - 1) * dim_block + 1] * y_std + y_mean for i in 1:N_test]
        rff_G = zeros(D, N_test)
        for i in 1:N_test
            offset = (i - 1) * dim_block
            rff_G[:, i] = rff_preds[(offset + 2):(offset + dim_block)] .* y_std
        end

        e_mae_true = mean(abs.(rff_E .- E_true))
        g_mae_true = mean(abs.(rff_G .- G_true))
        e_mae_gp = mean(abs.(rff_E .- exact_E))
        g_mae_gp = mean(abs.(rff_G .- exact_G))

        push!(all_drff, D_rff)
        push!(all_e_mae_true, e_mae_true)
        push!(all_g_mae_true, g_mae_true)
        push!(all_e_mae_gp, e_mae_gp)
        push!(all_g_mae_gp, g_mae_gp)

        println("  Energy MAE vs true: $(round(e_mae_true; sigdigits=3))")
        println("  Energy MAE vs GP:   $(round(e_mae_gp; sigdigits=3))")
    end

    # Exact GP errors for reference
    gp_e_mae = mean(abs.(exact_E .- E_true))
    gp_g_mae = mean(abs.(exact_G .- G_true))
    println("\nExact GP energy MAE vs true: $(round(gp_e_mae; sigdigits=3))")
    println("Exact GP gradient MAE vs true: $(round(gp_g_mae; sigdigits=3))")

    # --- Write to HDF5 ---
    h5_write_table(h5_path(), "table", Dict(
        "D_rff" => all_drff,
        "energy_mae_vs_true" => all_e_mae_true,
        "gradient_mae_vs_true" => all_g_mae_true,
        "energy_mae_vs_gp" => all_e_mae_gp,
        "gradient_mae_vs_gp" => all_g_mae_gp,
    ))

    h5_write_metadata(h5_path(); gp_e_mae=gp_e_mae, gp_g_mae=gp_g_mae)
    println("Wrote HDF5: $(h5_path())")
end

main()
