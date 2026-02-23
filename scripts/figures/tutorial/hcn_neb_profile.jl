# REAL-1: HCN isomerization energy profile
#
# Runs standard NEB and (optionally) GP-NEB AIE on the HCN -> HNC reaction
# using a PET-MAD RPC oracle. Produces an energy profile plot (image index vs
# relative energy) and exports data to CSV.
#
# Prerequisites:
#   1. rgpot server running: RGPOT_BUILD_DIR=... RGPOT_PORT=12345
#   2. julia --project=scripts/figures/tutorial scripts/figures/tutorial/hcn_neb_profile.jl
#
# If no server is available, checks for cached results in output/hcn_cache/.

using ChemGP
using DataFrames
using CSV
using LinearAlgebra
include(joinpath(@__DIR__, "common.jl"))

const HCN_CACHE_DIR = joinpath(OUTPUT_DIR, "hcn_cache")
mkpath(HCN_CACHE_DIR)

# HCN and HNC coordinates (Baker test set 01_hcn)
# C, N, H in Angstroms (non-periodic, 20x20x20 box)
const X_HCN = [
    0.0,
    -0.0002,
    0.4954,   # C
    0.0,
    0.0001,
    -0.6503,   # N
    0.0,
    -0.0005,
    1.5653,   # H
]
const X_HNC = [
    0.0,
    0.0,
    0.7366,       # C
    0.0,
    0.0,
    -0.4277,      # N
    0.0,
    0.0,
    -1.4258,      # H
]

function run_with_rpc()
    host = get(ENV, "RGPOT_HOST", "localhost")
    port = parse(Int, get(ENV, "RGPOT_PORT", "12345"))

    # 3 atoms: C(6), N(7), H(1)
    atomic_numbers = Int32[6, 7, 1]
    box = 20.0 * [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

    println("Connecting to PET-MAD server at $host:$port")
    pot = RpcPotential(host, port, atomic_numbers, box)
    oracle = make_rpc_oracle(pot)

    # Standard NEB
    println("Running standard NEB...")
    cfg_std = NEBConfig(;
        images=8,
        spring_constant=1.0,
        climbing_image=true,
        energy_weighted=true,
        ew_k_min=0.972,
        ew_k_max=9.72,
        max_iter=200,
        conv_tol=0.05,
        step_size=0.01,
        verbose=true,
    )
    result_std = neb_optimize(oracle, X_HCN, X_HNC; config=cfg_std)

    # GP-NEB AIE (may fail on ill-conditioned kernel for 3-atom systems)
    result_aie = nothing
    try
        println("Running GP-NEB AIE...")
        kernel = MolInvDistSE(1.0, [1.0], Float64[])
        cfg_gp = NEBConfig(;
            images=8,
            spring_constant=1.0,
            climbing_image=true,
            energy_weighted=true,
            ew_k_min=0.972,
            ew_k_max=9.72,
            conv_tol=0.05,
            gp_train_iter=300,
            max_outer_iter=50,
            trust_radius=0.1,
            atom_types=Int[6, 7, 1],
            max_gp_points=40,
            rff_features=300,
        )
        result_aie = gp_neb_aie(oracle, X_HCN, X_HNC, kernel; config=cfg_gp)
    catch e
        @warn "GP-NEB AIE failed" exception = e
    end

    close(pot)
    return result_std, result_aie
end

function make_profile_df(result, method_label)
    n = length(result.path.energies)
    e_ref = result.path.energies[1]
    DataFrame(;
        image=1:n, energy=(result.path.energies .- e_ref), method=fill(method_label, n)
    )
end

function cache_or_run()
    csv_std = joinpath(HCN_CACHE_DIR, "profile_standard.csv")

    if isfile(csv_std)
        println("Using cached HCN results from $HCN_CACHE_DIR")
        df = CSV.read(csv_std, DataFrame)
        csv_aie = joinpath(HCN_CACHE_DIR, "profile_aie.csv")
        if isfile(csv_aie)
            df = vcat(df, CSV.read(csv_aie, DataFrame))
        end
        return df
    end

    result_std, result_aie = run_with_rpc()

    df_std = make_profile_df(result_std, "Standard NEB")
    CSV.write(csv_std, df_std)
    df = df_std

    if result_aie !== nothing
        df_aie = make_profile_df(result_aie, "GP-NEB AIE")
        CSV.write(joinpath(HCN_CACHE_DIR, "profile_aie.csv"), df_aie)
        df = vcat(df_std, df_aie)
    end

    println("Cached results to $HCN_CACHE_DIR")
    return df
end

# --- Run or load ---
df = cache_or_run()

# --- Plot ---
set_theme!(PUBLICATION_THEME)

n_methods = length(unique(df.method))

if n_methods > 1
    plt =
        data(df) *
        mapping(:image, :energy; color=:method) *
        (visual(Lines; linewidth=1.5) + visual(Scatter; markersize=6))

    fg = draw(
        plt,
        scales(; Color=(; palette=RUHI_CYCLE));
        axis=(xlabel="Image index", ylabel=L"$\Delta E$ (eV)"),
        figure=(size=(504, 350),),
        legend=(position=:top,),
    )
else
    fig = Figure(; size=(504, 350))
    ax = Axis(fig[1, 1]; xlabel="Image index", ylabel=L"$\Delta E$ (eV)")
    lines!(ax, df.image, df.energy; linewidth=1.5, color=RUHI.teal, label=df.method[1])
    scatter!(ax, df.image, df.energy; markersize=6, color=RUHI.teal)
    Legend(fig[0, 1], ax)
    fg = fig
end

save_figure(fg, "hcn_neb_profile")

# Export data
CSV.write(joinpath(OUTPUT_DIR, "hcn_profile_data.csv"), df)
println("Profile data exported to $(joinpath(OUTPUT_DIR, "hcn_profile_data.csv"))")
