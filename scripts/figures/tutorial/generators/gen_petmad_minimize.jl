# PETMAD-MIN generator: GP-minimization vs classical L-BFGS on PET-MAD
#
# Compares oracle-call efficiency of GP-guided minimization against
# naive oracle-every-step L-BFGS on a 9-atom organic fragment (2C, 1O, 2N, 4H)
# evaluated with PET-MAD universal ML potential via RPC.
#
# Uses cache pattern: checks for existing HDF5 before running.
# Requires: PET-MAD server running at localhost:12345
#
# Output: {stem}.h5 with /table group (oracle_calls, max_fatom, method)

include(joinpath(@__DIR__, "common_data.jl"))
using ChemGP
using LinearAlgebra
using Random

# --- System100 reactant: 9-atom fragment (2C, 1O, 2N, 4H) ---
const SYSTEM100_REACT = Float64[
    -1.58572291100237,
    -0.84160847213746,
    -0.00000339907657,  # C
    -0.53056971192710,
    -1.65722303210517,
    0.00000434652695,  # C
    1.82767320854265,
    0.45290828278285,
    -0.00002187280664,  # O
    0.97442679271533,
    1.26997020651757,
    0.00006031628749,  # N
    0.15721755319950,
    2.05013813569860,
    -0.00004056288043,  # N
    -2.04209833505612,
    -0.48866007686699,
    0.93039929342888,  # H
    -2.04208985274357,
    -0.48866569200588,
    -0.93041253064561,  # H
    -0.07175706986641,
    -2.00739679244130,
    0.93006512898811,  # H
    -0.07174967386193,
    -2.00740255964219,
    -0.93005072002220,  # H
]
const SYSTEM100_ATMNRS = Int32[6, 6, 8, 7, 7, 1, 1, 1, 1]
const SYSTEM100_BOX = Float64[20, 0, 0, 0, 20, 0, 0, 0, 20]

"""Max per-atom force magnitude (3D norm per atom, then max)."""
function max_fatom(G)
    n_atoms = div(length(G), 3)
    return maximum(norm(@view G[(3 * (i - 1) + 1):(3 * i)]) for i in 1:n_atoms)
end

function run_convergence()
    pot = RpcPotential("localhost", 12345, SYSTEM100_ATMNRS, SYSTEM100_BOX)
    oracle = make_rpc_oracle(pot)

    x_init = copy(SYSTEM100_REACT)

    # Use TCP socket for machine_output if port is available
    machine_out = FIG_PORT === nothing ? "" : "localhost:$(FIG_PORT)"

    # --- GP-accelerated minimization ---
    println("Running GP-minimization on system100...")

    kernel = MolInvDistSE(SYSTEM100_ATMNRS, Float64[])

    gp_config = MinimizationConfig(;
        trust_radius=0.10,
        conv_tol=0.05,
        max_iter=80,
        gp_opt_tol=1e-2,
        gp_train_iter=200,
        n_initial_perturb=3,
        perturb_scale=0.06,
        penalty_coeff=1e3,
        max_move=0.04,
        explosion_recovery=:perturb_best,
        energy_regression_tol=0.5,
        rff_features=300,
        max_training_points=60,
        verbose=false,
        fps_history=40,
        fps_latest_points=10,
        fps_metric=:emd,
        machine_output=machine_out,
    )

    result_gp = gp_minimize(oracle, copy(x_init), kernel; config=gp_config)
    println("GP-min: $(result_gp.stop_reason), oracle calls: $(result_gp.oracle_calls)")

    # --- Parse GP data from JSONL (written by socket) ---
    jsonl_path = joinpath(FIG_OUTPUT, FIG_STEM * ".jsonl")

    gp_ocs = Int[]
    gp_fatom = Float64[]
    if isfile(jsonl_path) && filesize(jsonl_path) > 0
        for line in readlines(jsonl_path)
            startswith(line, "{\"status\"") && continue
            m_oc = match(r"\"oc\":(\d+)", line)
            m_f = match(r"\"F\":([\d.eE+-]+)", line)
            m_oc === nothing && continue
            m_f === nothing && continue
            push!(gp_ocs, parse(Int, m_oc[1]))
            push!(gp_fatom, parse(Float64, m_f[1]))
        end
    end

    # --- Classical minimization (oracle every step, no GP) ---
    println("Running classical L-BFGS on system100...")

    classical_fatom = Float64[]
    classical_oc = Int[]
    x_curr = copy(x_init)
    opt_state = OptimState(10)
    oc_count = 0

    E_curr, G_curr = oracle(x_curr)
    oc_count += 1
    push!(classical_fatom, max_fatom(G_curr))
    push!(classical_oc, oc_count)

    for iter in 1:200
        step = optim_step!(opt_state, x_curr, -G_curr, 0.1)
        x_new = x_curr + step
        E_new, G_new = oracle(x_new)
        oc_count += 1

        if E_new > E_curr
            x_new = x_curr + 0.5 .* step
            E_new, G_new = oracle(x_new)
            oc_count += 1
        end

        x_curr = x_new
        E_curr = E_new
        G_curr = G_new

        push!(classical_fatom, max_fatom(G_curr))
        push!(classical_oc, oc_count)

        max_fatom(G_curr) < 0.05 && break
    end

    println("Classical: $oc_count oracle calls")
    close(pot)

    return gp_ocs, gp_fatom, classical_oc, classical_fatom
end

function main()
    hp = h5_path()

    if isfile(hp)
        println("Using cached data from $hp")
        return
    end

    gp_ocs, gp_fatom, classical_oc, classical_fatom = run_convergence()

    n_gp = length(gp_ocs)
    n_cl = length(classical_fatom)

    all_oc = vcat(gp_ocs, classical_oc)
    all_fatom = vcat(gp_fatom, classical_fatom)
    all_method = vcat(fill("GP-minimization", n_gp), fill("Classical L-BFGS", n_cl))

    h5_write_table(hp, "table", Dict(
        "oracle_calls" => all_oc,
        "max_fatom" => all_fatom,
        "method" => all_method,
    ))

    h5_write_metadata(hp; n_gp=n_gp, n_cl=n_cl)
    println("Wrote HDF5: $hp")
end

main()
