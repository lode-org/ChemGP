# LEPS-MIN: GP-minimization vs classical L-BFGS on LEPS
#
# Compares oracle-call efficiency of GP-guided minimization against
# naive oracle-every-step L-BFGS on the 9D LEPS surface (3-atom collinear).
# Tracks max per-atom force vs oracle evaluations.
#
# GP iteration data streamed to JSONL writer via TCP socket.
#
# Output: leps_minimize_convergence.pdf, leps_minimize_convergence.csv
#
# Intended for sec:gprmin of the GPR tutorial review.

using ChemGP
using DataFrames
using CSV
using LinearAlgebra
using Random
using LaTeXStrings
include(joinpath(@__DIR__, "common.jl"))

"""Max per-atom force magnitude (3D norm per atom, then max)."""
function max_fatom(G)
    n_atoms = div(length(G), 3)
    return maximum(norm(@view G[(3 * (i - 1) + 1):(3 * i)]) for i in 1:n_atoms)
end

function main()
    Random.seed!(42)

    # --- Starting geometry: perturbed LEPS reactant ---
    x_init = Float64.(LEPS_REACTANT) .+ 0.4 .* (rand(9) .- 0.5)
    oracle = leps_energy_gradient

    # --- JSONL writer ---
    jsonl_path = joinpath(OUTPUT_DIR, "leps_minimize.jsonl")
    writer, port = start_jsonl_writer(jsonl_path)

    # --- GP-accelerated minimization ---
    kernel = MolInvDistSE([1, 1, 1], Float64[])

    gp_config = MinimizationConfig(;
        trust_radius=0.15,
        conv_tol=0.01,
        max_iter=80,
        gp_opt_tol=1e-2,
        gp_train_iter=200,
        n_initial_perturb=4,
        perturb_scale=0.08,
        penalty_coeff=1e3,
        max_move=0.1,
        explosion_recovery=:perturb_best,
        verbose=false,
        machine_output="localhost:$port",
    )

    result_gp = gp_minimize(oracle, copy(x_init), kernel; config=gp_config)
    println("GP-min: $(result_gp.stop_reason), oracle calls: $(result_gp.oracle_calls)")

    stop_jsonl_writer(writer)

    # --- Classical minimization (oracle every step, no GP) ---
    println("Running classical L-BFGS...")

    classical_fatom = Float64[]
    classical_oc = Int[]
    x_curr = copy(x_init)
    oc_count = 0

    E_curr, G_curr = oracle(x_curr)
    oc_count += 1
    push!(classical_fatom, max_fatom(G_curr))
    push!(classical_oc, oc_count)

    opt_state = OptimState(10)

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

        max_fatom(G_curr) < 0.01 && break
    end

    println("Classical: $oc_count oracle calls")

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
    df = vcat(df_gp, df_cl)

    # --- Plot ---
    set_theme!(PUBLICATION_THEME)

    fig = Figure(; size=(504, 350))
    ax = Axis(
        fig[1, 1];
        xlabel="Oracle calls",
        ylabel=L"max $|F_\mathrm{atom}|$ (eV/\AA)",
        yscale=log10,
    )

    lines!(ax, df_gp.oracle_calls, df_gp.max_fatom;
        color=RUHI.teal, linewidth=1.5, label="GP-minimization")
    lines!(ax, df_cl.oracle_calls, df_cl.max_fatom;
        color=RUHI.sky, linewidth=1.5, label="Classical L-BFGS")
    hlines!(ax, [0.01]; color=:gray, linewidth=0.8, linestyle=:dash)

    axislegend(ax; position=:rt, framevisible=false, labelsize=10)

    save_figure(fig, "leps_minimize_convergence")
    CSV.write(joinpath(OUTPUT_DIR, "leps_minimize_convergence.csv"), df)
end

main()
