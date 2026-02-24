# LEPS-MIN: GP-minimization vs classical L-BFGS on LEPS
#
# Compares oracle-call efficiency of GP-guided minimization against
# naive oracle-every-step L-BFGS on the 9D LEPS surface (3-atom collinear).
# Tracks max per-atom force vs oracle evaluations.
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
    D = length(x_init)
    oracle = leps_energy_gradient

    # --- GP-accelerated minimization ---
    println("Running GP-minimization on LEPS...")

    kernel = MolInvDistSE(1.0, [1.0, 1.0, 1.0], Float64[])

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
        verbose=true,
    )

    result_gp = gp_minimize(oracle, copy(x_init), kernel; config=gp_config)
    println("GP-min converged: $(result_gp.converged)")
    println("GP-min oracle calls: $(result_gp.oracle_calls)")

    # --- Classical minimization (oracle every step, no GP) ---
    println("\nRunning classical L-BFGS on LEPS (oracle every step)...")

    classical_fatom = Float64[]
    x_curr = copy(x_init)
    max_classical_iter = 200

    E_curr, G_curr = oracle(x_curr)
    push!(classical_fatom, max_fatom(G_curr))

    opt_state = OptimState(10)

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

        if max_fatom(G_curr) < 0.01
            println("Classical converged at iter $iter")
            break
        end
    end

    println("Classical oracle calls: $(length(classical_fatom))")

    # --- Build dataframes ---
    gp_fatom = [max_fatom(oracle(x)[2]) for x in result_gp.trajectory]

    n_gp = length(gp_fatom)
    n_cl = length(classical_fatom)

    df_gp = DataFrame(;
        oracle_calls=1:n_gp, max_fatom=gp_fatom, method=fill("GP-minimization", n_gp)
    )
    df_cl = DataFrame(;
        oracle_calls=1:n_cl,
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
    hlines!(ax, [0.01]; color=:gray, linewidth=0.8, linestyle=:dash)

    axislegend(ax; position=:rt, framevisible=false, labelsize=10)

    save_figure(fig, "leps_minimize_convergence")
    CSV.write(joinpath(OUTPUT_DIR, "leps_minimize_convergence.csv"), df)
end

main()
