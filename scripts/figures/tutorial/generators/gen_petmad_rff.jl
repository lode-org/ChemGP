# PETMAD-RFF generator: RFF approximation quality vs exact GP on PET-MAD
#
# Trains an exact GP on PET-MAD data accumulated during a short minimization
# of the system100 reactant, then builds RFF models at varying D_rff and
# compares prediction accuracy (energy MAE, gradient MAE) against the exact
# GP on held-out test points.
#
# Uses cache pattern: checks for existing HDF5 before running.
# Requires: PET-MAD server running at localhost:12345
#
# Output: {stem}.h5 with /table group and metadata (gp_e_mae, gp_g_mae)

include(joinpath(@__DIR__, "common_data.jl"))
using ChemGP
using LinearAlgebra
using Random
using Statistics

# --- System100 reactant: 9-atom fragment (2C, 1O, 2N, 4H) ---
const SYSTEM100_REACT = Float64[
    -1.58572291100237,
    -0.84160847213746,
    -0.00000339907657,
    -0.53056971192710,
    -1.65722303210517,
    0.00000434652695,
    1.82767320854265,
    0.45290828278285,
    -0.00002187280664,
    0.97442679271533,
    1.26997020651757,
    0.00006031628749,
    0.15721755319950,
    2.05013813569860,
    -0.00004056288043,
    -2.04209833505612,
    -0.48866007686699,
    0.93039929342888,
    -2.04208985274357,
    -0.48866569200588,
    -0.93041253064561,
    -0.07175706986641,
    -2.00739679244130,
    0.93006512898811,
    -0.07174967386193,
    -2.00740255964219,
    -0.93005072002220,
]
const SYSTEM100_ATMNRS = Int32[6, 6, 8, 7, 7, 1, 1, 1, 1]
const SYSTEM100_BOX = Float64[20, 0, 0, 0, 20, 0, 0, 0, 20]

function run_rff_comparison()
    host = get(ENV, "RGPOT_HOST", "localhost")
    port = parse(Int, get(ENV, "RGPOT_PORT", "12345"))

    println("Connecting to PET-MAD server at $host:$port")
    pot = RpcPotential(host, port, SYSTEM100_ATMNRS, SYSTEM100_BOX)
    oracle = make_rpc_oracle(pot)

    Random.seed!(123)
    x_init = copy(SYSTEM100_REACT)
    D = length(x_init)

    # --- Collect training data from a short GP-minimization ---
    println("Collecting training data from GP-minimization...")
    kernel_init = MolInvDistSE(SYSTEM100_ATMNRS, Float64[])
    config = MinimizationConfig(;
        trust_radius=0.08,
        conv_tol=0.05,
        max_iter=30,
        gp_train_iter=100,
        n_initial_perturb=6,
        perturb_scale=0.05,
        max_move=0.05,
        dedup_tol=1e-4,
        rff_features=200,
        max_training_points=40,
        verbose=false,
    )
    result = gp_minimize(oracle, copy(x_init), kernel_init; config=config)

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
        x_t = result.x_final .+ 0.06 .* (rand(D) .- 0.5)
        E, G = oracle(x_t)
        X_test[:, i] = x_t
        E_true[i] = E
        G_true[:, i] = G
    end

    # --- Train exact GP ---
    println("Training exact GP...")
    y_gp, y_mean, y_std = ChemGP.normalize(td)

    mol_kernel = MolInvDistSE(SYSTEM100_ATMNRS, Float64[])
    model = GPModel(
        mol_kernel, td.X, y_gp; noise_var=1e-2, grad_noise_var=1e-1, jitter=1e-3
    )
    train_model!(model; iterations=300)

    # Exact GP predictions
    exact_preds = predict(model, X_test)
    dim_block = D + 1
    exact_E = [exact_preds[(i - 1) * dim_block + 1] * y_std + y_mean for i in 1:N_test]
    exact_G = zeros(D, N_test)
    for i in 1:N_test
        offset = (i - 1) * dim_block
        exact_G[:, i] = exact_preds[(offset + 2):(offset + dim_block)] .* y_std
    end

    # --- RFF at varying D_rff ---
    D_rff_values = [25, 50, 100, 200, 300, 500]

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

    gp_e_mae = mean(abs.(exact_E .- E_true))
    gp_g_mae = mean(abs.(exact_G .- G_true))
    println("\nExact GP energy MAE vs true: $(round(gp_e_mae; sigdigits=3))")
    println("Exact GP gradient MAE vs true: $(round(gp_g_mae; sigdigits=3))")

    close(pot)
    return all_drff, all_e_mae_true, all_g_mae_true, all_e_mae_gp, all_g_mae_gp, gp_e_mae, gp_g_mae
end

function main()
    hp = h5_path()

    if isfile(hp)
        println("Using cached data from $hp")
        return
    end

    all_drff, all_e_mae_true, all_g_mae_true, all_e_mae_gp, all_g_mae_gp, gp_e_mae, gp_g_mae = run_rff_comparison()

    h5_write_table(hp, "table", Dict(
        "D_rff" => all_drff,
        "energy_mae_vs_true" => all_e_mae_true,
        "gradient_mae_vs_true" => all_g_mae_true,
        "energy_mae_vs_gp" => all_e_mae_gp,
        "gradient_mae_vs_gp" => all_g_mae_gp,
    ))

    h5_write_metadata(hp; gp_e_mae=gp_e_mae, gp_g_mae=gp_g_mae)
    println("Wrote HDF5: $hp")
end

main()
