# PETMAD-MIN: GP-minimization vs classical L-BFGS on PET-MAD
#
# Compares oracle-call efficiency of GP-guided minimization against
# naive oracle-every-step L-BFGS on a 9-atom organic fragment (2C, 1O, 2N, 4H)
# evaluated with PET-MAD universal ML potential via RPC.
# Tracks max per-atom force vs oracle evaluations.
#
# Requires: PET-MAD server running via eonclient or rgpot potserv.
#   export RGPOT_HOST=localhost RGPOT_PORT=12345
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
    host = get(ENV, "RGPOT_HOST", "localhost")
    port = parse(Int, get(ENV, "RGPOT_PORT", "12345"))

    println("Connecting to PET-MAD server at $host:$port")
    pot = RpcPotential(host, port, SYSTEM100_ATMNRS, SYSTEM100_BOX)
    oracle = make_rpc_oracle(pot)

    x_init = copy(SYSTEM100_REACT)
    D = length(x_init)

    # --- GP-accelerated minimization ---
    println("Running GP-minimization on system100...")

    kernel = MolInvDistSE(1.0, [1.0], Float64[])

    gp_config = MinimizationConfig(;
        trust_radius=0.15,
        conv_tol=0.05,
        max_iter=80,
        gp_opt_tol=1e-2,
        gp_train_iter=200,
        n_initial_perturb=4,
        perturb_scale=0.08,
        penalty_coeff=1e3,
        max_move=0.1,
        explosion_recovery=:perturb_best,
        rff_features=200,
        max_training_points=50,
        verbose=true,
    )

    result_gp = gp_minimize(oracle, copy(x_init), kernel; config=gp_config)
    println("GP-min converged: $(result_gp.converged)")
    println("GP-min oracle calls: $(result_gp.oracle_calls)")

    # --- Classical minimization (oracle every step, no GP) ---
    println("\nRunning classical L-BFGS on system100 (oracle every step)...")

    classical_fatom = Float64[]
    x_curr = copy(x_init)
    max_classical_iter = 200
    opt_state = OptimState(10)

    E_curr, G_curr = oracle(x_curr)
    push!(classical_fatom, max_fatom(G_curr))

    for iter in 1:max_classical_iter
        step = optim_step!(opt_state, x_curr, -G_curr, 0.1)
        x_new = x_curr + step
        E_new, G_new = oracle(x_new)

        # Accept if energy decreased, otherwise halve step
        if E_new > E_curr
            x_new = x_curr + 0.5 .* step
            E_new, G_new = oracle(x_new)
        end

        x_curr = x_new
        E_curr = E_new
        G_curr = G_new

        push!(classical_fatom, max_fatom(G_curr))

        if max_fatom(G_curr) < 0.05
            println("Classical converged at iter $iter")
            break
        end
    end

    println("Classical oracle calls: $(length(classical_fatom))")

    close(pot)
    return result_gp, classical_fatom
end

function main()
    csv_path = joinpath(PETMAD_CACHE_DIR, "minimize_convergence.csv")

    df = if isfile(csv_path)
        println("Using cached data from $csv_path")
        CSV.read(csv_path, DataFrame)
    else
        result_gp, classical_fatom = run_convergence()

        # Re-evaluate trajectory for per-atom force (need oracle again)
        # Use cached energies/trajectory from result_gp
        # The GP trajectory stores evaluated points; re-derive forces from
        # the energies and gradients stored during the run.
        # Since oracle calls are expensive, we cache the result.
        host = get(ENV, "RGPOT_HOST", "localhost")
        port = parse(Int, get(ENV, "RGPOT_PORT", "12345"))
        pot = RpcPotential(host, port, SYSTEM100_ATMNRS, SYSTEM100_BOX)
        oracle = make_rpc_oracle(pot)

        gp_fatom = [max_fatom(oracle(x)[2]) for x in result_gp.trajectory]
        close(pot)

        n_gp = length(gp_fatom)
        n_cl = length(classical_fatom)

        df_gp = DataFrame(;
            oracle_calls=1:n_gp,
            max_fatom=gp_fatom,
            method=fill("GP-minimization", n_gp),
        )
        df_cl = DataFrame(;
            oracle_calls=1:n_cl,
            max_fatom=classical_fatom,
            method=fill("Classical L-BFGS", n_cl),
        )
        df_all = vcat(df_gp, df_cl)

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

    lines!(
        ax,
        df_gp.oracle_calls,
        df_gp.max_fatom;
        color=RUHI.teal,
        linewidth=1.5,
        label="GP-minimization",
    )
    lines!(
        ax,
        df_cl.oracle_calls,
        df_cl.max_fatom;
        color=RUHI.sky,
        linewidth=1.5,
        label="Classical L-BFGS",
    )
    hlines!(ax, [0.05]; color=:gray, linewidth=0.8, linestyle=:dash)

    axislegend(ax; position=:rt, framevisible=false, labelsize=10)

    save_figure(fig, "petmad_minimize_convergence")
    CSV.write(joinpath(OUTPUT_DIR, "petmad_minimize_convergence.csv"), df)
end

main()
