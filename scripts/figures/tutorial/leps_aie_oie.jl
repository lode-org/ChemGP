# LEPS-3: AIE vs OIE oracle efficiency comparison on LEPS
#
# Runs GP-NEB AIE and GP-NEB OIE on the 9D LEPS surface and compares
# convergence (max force vs oracle calls).

using ChemGP
using KernelFunctions
using DataFrames
using CSV
include(joinpath(@__DIR__, "common.jl"))

# --- Kernel: MolInvDistSE for 3-atom system ---
# 3 atoms -> 3 inverse distances (AB, BC, AC), no frozen atoms
kernel = MolInvDistSE(1.0, [1.0, 1.0, 1.0], Float64[], Int[])

config = NEBConfig(;
    n_images=7,
    spring_constant=5.0,
    climbing_image=true,
    conv_tol=0.1,
    gp_train_iter=50,
    max_outer_iter=20,
    trust_radius=0.1,
    verbose=true,
)

# --- GP-NEB AIE ---
println("Running GP-NEB AIE on LEPS...")
result_aie = gp_neb_aie(
    leps_energy_gradient,
    Float64.(LEPS_REACTANT),
    Float64.(LEPS_PRODUCT),
    kernel;
    config=config,
)
println("AIE converged: $(result_aie.converged), oracle calls: $(result_aie.oracle_calls)")

# --- GP-NEB OIE ---
println("Running GP-NEB OIE on LEPS...")
oie_config = NEBConfig(;
    n_images=7,
    spring_constant=5.0,
    climbing_image=true,
    conv_tol=0.1,
    gp_train_iter=50,
    max_outer_iter=30,
    trust_radius=0.1,
    verbose=true,
)

result_oie = gp_neb_oie(
    leps_energy_gradient,
    Float64.(LEPS_REACTANT),
    Float64.(LEPS_PRODUCT),
    kernel;
    config=oie_config,
)
println("OIE converged: $(result_oie.converged), oracle calls: $(result_oie.oracle_calls)")

# --- Extract convergence history ---
function extract_history(result, label)
    calls = result.history["oracle_calls"]
    forces = result.history["max_force"]
    n = min(length(calls), length(forces))
    DataFrame(; oracle_calls=calls[1:n], max_force=forces[1:n], method=fill(label, n))
end

df_aie = extract_history(result_aie, "AIE")
df_oie = extract_history(result_oie, "OIE")
df = vcat(df_aie, df_oie)

# --- Plot ---
set_theme!(PUBLICATION_THEME)

plt =
    data(df) *
    mapping(:oracle_calls, :max_force; color=:method) *
    visual(Lines; linewidth=1.5)

fg = draw(
    plt,
    scales(; Color=(; palette=RUHI_CYCLE));
    axis=(xlabel="Oracle calls", ylabel=L"$|F|_\mathrm{max}$ (eV/\AA)", yscale=log10),
    figure=(size=(504, 350),),
)

# Convergence threshold
hlines!(current_axis(), [0.1]; color=:gray, linewidth=0.8, linestyle=:dash)

save_figure(fg, "leps_aie_oie")

# Export data
CSV.write(joinpath(OUTPUT_DIR, "leps_aie_oie.csv"), df)
