# LEPS-3: Standard NEB vs GP-NEB AIE/OIE on LEPS
#
# Compares oracle efficiency of standard NEB, GP-NEB AIE, and GP-NEB OIE
# on the 9D LEPS surface (3-atom collinear H+H2 exchange).

using ChemGP
using KernelFunctions
using DataFrames
using CSV
include(joinpath(@__DIR__, "common.jl"))

# --- Kernel: MolInvDistSE for 3-atom system ---
# 3 H atoms -> 1 unique pair type (H-H), 3 inv-distance features sharing 1 lengthscale
kernel = MolInvDistSE([1, 1, 1], Float64[])

# --- Standard NEB (oracle every step) ---
println("Running standard NEB on LEPS...")
std_config = NEBConfig(;
    images=7,
    spring_constant=5.0,
    climbing_image=true,
    conv_tol=0.1,
    max_iter=200,
    step_size=0.01,
    verbose=true,
)

result_std = neb_optimize(
    leps_energy_gradient,
    Float64.(LEPS_REACTANT),
    Float64.(LEPS_PRODUCT);
    config=std_config,
)
println("Standard NEB converged: $(result_std.converged), oracle calls: $(result_std.oracle_calls)")

# --- GP-NEB AIE ---
println("Running GP-NEB AIE on LEPS...")
aie_config = NEBConfig(;
    images=7,
    spring_constant=5.0,
    climbing_image=true,
    conv_tol=0.1,
    gp_train_iter=50,
    max_outer_iter=20,
    trust_radius=0.1,
    atom_types=Int[1, 1, 1],
    max_gp_points=20,
    rff_features=200,
    verbose=true,
)

result_aie = gp_neb_aie(
    leps_energy_gradient,
    Float64.(LEPS_REACTANT),
    Float64.(LEPS_PRODUCT),
    kernel;
    config=aie_config,
)
println("AIE converged: $(result_aie.converged), oracle calls: $(result_aie.oracle_calls)")

# --- GP-NEB OIE ---
println("Running GP-NEB OIE on LEPS...")
oie_config = NEBConfig(;
    images=7,
    spring_constant=5.0,
    climbing_image=true,
    conv_tol=0.1,
    gp_train_iter=50,
    max_outer_iter=60,
    trust_radius=0.1,
    eps_hess=0.01,
    atom_types=Int[1, 1, 1],
    max_gp_points=20,
    rff_features=200,
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

df = extract_history(result_std, "Standard NEB")
df = vcat(df, extract_history(result_aie, "GP-NEB AIE"))
df = vcat(df, extract_history(result_oie, "GP-NEB OIE"))

n_methods = length(unique(df.method))
palette_slice = RUHI_CYCLE[1:min(n_methods, length(RUHI_CYCLE))]

# --- Plot ---
set_theme!(PUBLICATION_THEME)

plt =
    data(df) *
    mapping(:oracle_calls, :max_force; color=:method) *
    visual(Lines; linewidth=1.5)

fg = draw(
    plt,
    scales(; Color=(; palette=palette_slice));
    axis=(xlabel="Oracle calls", ylabel=L"$|F|_\mathrm{max}$ (eV/\AA)", yscale=log10),
    figure=(size=(504, 350),),
)

# Convergence threshold
hlines!(current_axis(), [0.1]; color=:gray, linewidth=0.8, linestyle=:dash)

save_figure(fg, "leps_aie_oie")

# Export data
CSV.write(joinpath(OUTPUT_DIR, "leps_aie_oie.csv"), df)
