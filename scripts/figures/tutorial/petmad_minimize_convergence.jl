# PETMAD-MIN: GP-minimization vs classical L-BFGS on PET-MAD
#
# Compares oracle-call efficiency of GP-guided minimization against
# naive oracle-every-step L-BFGS on a 9-atom organic fragment (2C, 1O, 2N, 4H)
# evaluated with PET-MAD universal ML potential via RPC.
# Tracks max per-atom force vs oracle evaluations.
#
# GP iteration data streamed to JSONL writer via TCP socket.
#
# Requires: PET-MAD server running at localhost:12345
#
# Output: petmad_minimize_convergence.pdf, petmad_minimize_convergence.csv
#
# Intended for sec:gprmin of the GPR tutorial review.

using ChemGP
using DataFrames
using CSV
using LinearAlgebra
using LaTeXStrings
include(joinpath(@__DIR__, "common.jl"))

const PETMAD_CACHE_DIR = joinpath(OUTPUT_DIR, "petmad_cache")
mkpath(PETMAD_CACHE_DIR)

"""Max per-atom force magnitude (3D norm per atom, then max)."""
function max_fatom(G)
    n_atoms = div(length(G), 3)
    return maximum(norm(@view G[(3 * (i - 1) + 1):(3 * i)]) for i in 1:n_atoms)
end

# --- System100 reactant: 9-atom fragment (2C, 1O, 2N, 4H) ---
# Source: system100-react.xyz from eOn ewNEB benchmark set
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

function run_convergence()
    pot = RpcPotential("localhost", 12345, SYSTEM100_ATMNRS, SYSTEM100_BOX)
    oracle = make_rpc_oracle(pot)

    x_init = copy(SYSTEM100_REACT)

    # --- JSONL writer ---
    jsonl_path = joinpath(PETMAD_CACHE_DIR, "minimize.jsonl")
    writer, port = start_jsonl_writer(jsonl_path; port=9877)

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
        machine_output="localhost:$port",
    )

    result_gp = gp_minimize(oracle, copy(x_init), kernel; config=gp_config)
    println("GP-min: $(result_gp.stop_reason), oracle calls: $(result_gp.oracle_calls)")

    stop_jsonl_writer(writer)

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

    # --- Build dataframes from JSONL ---
    gp_data = parse_minimize_jsonl(jsonl_path)
    n_gp = length(gp_data.oracle_calls)
    n_cl = length(classical_fatom)

    df_gp = DataFrame(;
        oracle_calls=gp_data.oracle_calls,
        max_fatom=gp_data.max_fatom,
        method=fill("GP-minimization", n_gp),
    )
    df_cl = DataFrame(;
        oracle_calls=classical_oc,
        max_fatom=classical_fatom,
        method=fill("Classical L-BFGS", n_cl),
    )
    return vcat(df_gp, df_cl)
end

function main()
    csv_path = joinpath(PETMAD_CACHE_DIR, "minimize_convergence.csv")

    df = if isfile(csv_path)
        println("Using cached data from $csv_path")
        CSV.read(csv_path, DataFrame)
    else
        df_all = run_convergence()
        CSV.write(csv_path, df_all)
        println("Cached convergence data to $csv_path")
        df_all
    end

    # --- Plot ---
    set_theme!(PUBLICATION_THEME)

    fig = Figure(; size=(504, 350))
    ax = Axis(
        fig[1, 1];
        xlabel="Oracle calls",
        ylabel=L"max $|F_\mathrm{atom}|$ (eV/\AA)",
        yscale=log10,
    )

    df_gp = filter(:method => ==("GP-minimization"), df)
    df_cl = filter(:method => ==("Classical L-BFGS"), df)

    lines!(ax, df_gp.oracle_calls, df_gp.max_fatom;
        color=RUHI.teal, linewidth=1.5, label="GP-minimization")
    lines!(ax, df_cl.oracle_calls, df_cl.max_fatom;
        color=RUHI.sky, linewidth=1.5, label="Classical L-BFGS")
    hlines!(ax, [0.05]; color=:gray, linewidth=0.8, linestyle=:dash)

    axislegend(ax; position=:rt, framevisible=false, labelsize=10)

    save_figure(fig, "petmad_minimize_convergence")
    CSV.write(joinpath(OUTPUT_DIR, "petmad_minimize_convergence.csv"), df)
end

main()
