# LEPS-MIN generator: GP-minimization vs classical L-BFGS on LEPS
#
# Runs GP-guided minimization and classical L-BFGS on the 9D LEPS surface,
# writes convergence data to HDF5, and streams iteration metrics via TCP socket.
#
# Output: {stem}.h5 with /table group (oracle_calls, max_fatom, method)

include(joinpath(@__DIR__, "common_data.jl"))
using ChemGP
using LinearAlgebra
using Random

"""Max per-atom force magnitude (3D norm per atom, then max)."""
function max_fatom(G)
    n_atoms = div(length(G), 3)
    return maximum(norm(@view G[(3 * (i - 1) + 1):(3 * i)]) for i in 1:n_atoms)
end

function main()
    Random.seed!(42)

    x_init = Float64.(LEPS_REACTANT) .+ 0.4 .* (rand(9) .- 0.5)
    oracle = leps_energy_gradient

    # --- GP-accelerated minimization ---
    kernel = MolInvDistSE([1, 1, 1], Float64[])

    # Use TCP socket for machine_output if port is available
    machine_out = FIG_PORT === nothing ? "" : "localhost:$(FIG_PORT)"

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
        machine_output=machine_out,
    )

    result_gp = gp_minimize(oracle, copy(x_init), kernel; config=gp_config)
    println("GP-min: $(result_gp.stop_reason), oracle calls: $(result_gp.oracle_calls)")

    # --- Parse GP data from JSONL (written by socket) ---
    jsonl_path = joinpath(FIG_OUTPUT, FIG_STEM * ".jsonl")

    # Fallback: if no JSONL (socket disabled), extract from trajectory
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

    # --- Write combined table to HDF5 ---
    n_gp = length(gp_ocs)
    n_cl = length(classical_fatom)

    all_oc = vcat(gp_ocs, classical_oc)
    all_fatom = vcat(gp_fatom, classical_fatom)
    all_method = vcat(fill("GP-minimization", n_gp), fill("Classical L-BFGS", n_cl))

    h5_write_table(h5_path(), "table", Dict(
        "oracle_calls" => all_oc,
        "max_fatom" => all_fatom,
        "method" => all_method,
    ))

    h5_write_metadata(h5_path(); n_gp=n_gp, n_cl=n_cl)
    println("Wrote HDF5: $(h5_path())")
end

main()
