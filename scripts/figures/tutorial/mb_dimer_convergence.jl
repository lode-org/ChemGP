# MB-6: GP-dimer vs classical dimer convergence on Muller-Brown
#
# Starts near Saddle 2 and tracks force norm vs oracle calls for the
# GP-dimer method. The "classical" baseline uses max_inner_iter=0.
# The dimer history tracks F_true (force norm) and oracle_calls.

using ChemGP
using KernelFunctions
using DataFrames
using CSV
using LinearAlgebra
using LaTeXStrings
include(joinpath(@__DIR__, "common.jl"))

# Starting point: offset from saddle S2
x_init = [0.3, 0.4]
orient_init = LinearAlgebra.normalize([1.0, 0.5])

# --- GP-dimer run ---
println("Running GP-dimer...")
kernel = 1.0 * with_lengthscale(SqExponentialKernel(), 0.3)

gp_config = DimerConfig(;
    T_force_true = 1e-3,
    T_force_gp = 1e-2,
    trust_radius = 0.3,
    max_outer_iter = 30,
    max_inner_iter = 100,
    alpha_trans = 0.01,
    gp_train_iter = 300,
    n_initial_perturb = 4,
    perturb_scale = 0.15,
    verbose = true,
)

result_gp = gp_dimer(
    muller_brown_energy_gradient, x_init, orient_init, kernel;
    config = gp_config,
)

println("GP-dimer converged: $(result_gp.converged)")
println("GP-dimer oracle calls: $(result_gp.oracle_calls)")
println("GP-dimer final position: $(result_gp.state.R)")

# --- Classical dimer (oracle-every-step) ---
println("Running classical dimer (max_inner_iter=0)...")
classical_config = DimerConfig(;
    T_force_true = 1e-3,
    T_force_gp = 1e-2,
    trust_radius = 0.3,
    max_outer_iter = 60,
    max_inner_iter = 0,
    alpha_trans = 0.01,
    gp_train_iter = 300,
    n_initial_perturb = 4,
    perturb_scale = 0.15,
    verbose = false,
)

result_cl = gp_dimer(
    muller_brown_energy_gradient, x_init, orient_init, kernel;
    config = classical_config,
)

println("Classical dimer converged: $(result_cl.converged)")
println("Classical dimer oracle calls: $(result_cl.oracle_calls)")

# --- Extract convergence data from history ---
# Dimer history has: E_true, F_true, curv_true, oracle_calls
function extract_convergence(result, label)
    oracle_calls = result.history["oracle_calls"]
    force_norms = result.history["F_true"]
    n = min(length(oracle_calls), length(force_norms))
    DataFrame(;
        oracle_calls = oracle_calls[1:n],
        force_norm = force_norms[1:n],
        method = fill(label, n),
    )
end

df_gp = extract_convergence(result_gp, "GP-dimer")
df_cl = extract_convergence(result_cl, "Classical dimer")
df = vcat(df_gp, df_cl)

# --- Plot ---
set_theme!(PUBLICATION_THEME)

plt = data(df) *
      mapping(:oracle_calls, :force_norm; color = :method) *
      visual(Lines; linewidth = 1.5)

fg = draw(plt, scales(Color = (; palette = RUHI_CYCLE));
    axis = (xlabel = "Oracle calls",
            ylabel = L"$|F|_{\mathrm{true}}$",
            yscale = log10),
    figure = (size = (504, 350),))

# Convergence threshold
hlines!(current_axis(), [1e-3];
    color = :gray, linewidth = 0.8, linestyle = :dash)

save_figure(fg, "mb_dimer_convergence")

# Export data
CSV.write(joinpath(OUTPUT_DIR, "mb_dimer_convergence.csv"), df)
