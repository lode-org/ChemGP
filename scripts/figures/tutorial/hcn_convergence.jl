# REAL-2: HCN convergence comparison
#
# Tracks max force residual vs oracle calls for standard NEB, GP-NEB AIE,
# and GP-NEB OIE on the HCN -> HNC isomerization. Reuses cached data from
# REAL-1 if available; otherwise requires PET-MAD server.

using ChemGP
using DataFrames
using CSV
include(joinpath(@__DIR__, "common.jl"))

const HCN_CACHE_DIR = joinpath(OUTPUT_DIR, "hcn_cache")
mkpath(HCN_CACHE_DIR)

# HCN/HNC coordinates (same as hcn_neb_profile.jl)
const X_HCN = [
    0.0, -0.0002, 0.4954,
    0.0, 0.0001, -0.6503,
    0.0, -0.0005, 1.5653,
]
const X_HNC = [
    0.0, 0.0, 0.7366,
    0.0, 0.0, -0.4277,
    0.0, 0.0, -1.4258,
]

function run_convergence()
    host = get(ENV, "RGPOT_HOST", "localhost")
    port = parse(Int, get(ENV, "RGPOT_PORT", "12345"))
    atomic_numbers = Int32[6, 7, 1]
    box = 20.0 * [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

    println("Connecting to PET-MAD server at $host:$port")
    pot = RpcPotential(host, port, atomic_numbers, box)
    oracle = make_rpc_oracle(pot)

    neb_cfg = NEBConfig(;
        images = 8,
        spring_constant = 1.0,
        climbing_image = true,
        energy_weighted = true,
        ew_k_min = 0.972,
        ew_k_max = 9.72,
        max_iter = 200,
        conv_tol = 0.05,
        step_size = 0.01,
        verbose = true,
    )

    # Standard NEB
    println("Running standard NEB...")
    result_std = neb_optimize(oracle, X_HCN, X_HNC; config = neb_cfg)

    # GP-NEB AIE
    result_aie = nothing
    try
        println("Running GP-NEB AIE...")
        kernel = MolInvDistSE(1.0, [1.0], Float64[])
        gp_cfg = NEBConfig(;
            images = 8,
            spring_constant = 1.0,
            climbing_image = true,
            energy_weighted = true,
            ew_k_min = 0.972,
            ew_k_max = 9.72,
            conv_tol = 0.05,
            gp_train_iter = 300,
            max_outer_iter = 50,
            trust_radius = 0.1,
            atom_types = Int[6, 7, 1],
            max_gp_points = 40,
            rff_features = 300,
        )
        result_aie = gp_neb_aie(oracle, X_HCN, X_HNC, kernel; config = gp_cfg)
    catch e
        @warn "GP-NEB AIE failed" exception = e
    end

    # GP-NEB OIE
    result_oie = nothing
    try
        println("Running GP-NEB OIE...")
        kernel = MolInvDistSE(1.0, [1.0], Float64[])
        oie_cfg = NEBConfig(;
            images = 8,
            spring_constant = 1.0,
            climbing_image = true,
            energy_weighted = true,
            ew_k_min = 0.972,
            ew_k_max = 9.72,
            conv_tol = 0.05,
            gp_train_iter = 300,
            max_outer_iter = 80,
            trust_radius = 0.1,
            atom_types = Int[6, 7, 1],
            max_gp_points = 10,
            rff_features = 300,
        )
        result_oie = gp_neb_oie(oracle, X_HCN, X_HNC, kernel; config = oie_cfg)
    catch e
        @warn "GP-NEB OIE failed" exception = e
    end

    close(pot)
    return result_std, result_aie, result_oie
end

function extract_convergence(result, label)
    calls = result.history["oracle_calls"]
    forces = result.history["max_force"]
    n = min(length(calls), length(forces))
    DataFrame(;
        oracle_calls = calls[1:n],
        max_force = forces[1:n],
        method = fill(label, n),
    )
end

function cache_or_run()
    csv_path = joinpath(HCN_CACHE_DIR, "convergence_all.csv")

    if isfile(csv_path)
        println("Using cached convergence data from $csv_path")
        return CSV.read(csv_path, DataFrame)
    end

    result_std, result_aie, result_oie = run_convergence()

    df = extract_convergence(result_std, "Standard NEB")

    if result_aie !== nothing
        df = vcat(df, extract_convergence(result_aie, "GP-NEB AIE"))
    end
    if result_oie !== nothing
        df = vcat(df, extract_convergence(result_oie, "GP-NEB OIE"))
    end

    CSV.write(csv_path, df)
    println("Cached convergence data to $csv_path")
    return df
end

# --- Run or load ---
df = cache_or_run()

# --- Plot ---
set_theme!(PUBLICATION_THEME)

plt = data(df) *
      mapping(:oracle_calls, :max_force; color = :method) *
      visual(Lines; linewidth = 1.5)

n_methods = length(unique(df.method))
palette_slice = RUHI_CYCLE[1:min(n_methods, length(RUHI_CYCLE))]

fg = draw(plt, scales(Color = (; palette = palette_slice));
    axis = (xlabel = "Oracle calls",
            ylabel = L"$|F|_\mathrm{max}$ (eV/\AA)",
            yscale = log10),
    figure = (size = (504, 350),))

# Convergence threshold
hlines!(current_axis(), [0.05]; color = :gray, linewidth = 0.8, linestyle = :dash)

save_figure(fg, "hcn_convergence")

# Export for R/BRMS
CSV.write(joinpath(OUTPUT_DIR, "hcn_convergence_data.csv"), df)
println("Convergence data exported to $(joinpath(OUTPUT_DIR, "hcn_convergence_data.csv"))")
