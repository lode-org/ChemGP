# HCN-NEB generator: energy profile for HCN -> HNC isomerization
#
# Runs standard NEB and (optionally) GP-NEB AIE on the HCN -> HNC reaction
# using a PET-MAD RPC oracle. Produces an energy profile (image index vs
# relative energy).
#
# Uses cache pattern: checks for existing HDF5 before running.
# Requires: PET-MAD server running at localhost:12345
#
# Output: {stem}.h5 with /table group (image, energy, method)

include(joinpath(@__DIR__, "common_data.jl"))
using ChemGP
using LinearAlgebra

# Load HCN/HNC from data files (Baker test set 01_hcn)
# From generators/ the path to data is one more .. than from tutorial/
const _hcn_data = read_extxyz(joinpath(@__DIR__, "..", "..", "..", "..", "data", "hcn", "hcn.extxyz"))
const _hnc_data = read_extxyz(joinpath(@__DIR__, "..", "..", "..", "..", "data", "hcn", "hnc.extxyz"))
const X_HCN = _hcn_data.positions
const X_HNC = _hnc_data.positions
const HCN_ATNRS = _hcn_data.atomic_numbers
const HCN_BOX = _hcn_data.box

function make_profile(result, method_label)
    n = length(result.path.energies)
    e_ref = result.path.energies[1]
    images = collect(1:n)
    energies = result.path.energies .- e_ref
    methods = fill(method_label, n)
    return images, energies, methods
end

function run_with_rpc()
    host = get(ENV, "RGPOT_HOST", "localhost")
    port = parse(Int, get(ENV, "RGPOT_PORT", "12345"))

    atomic_numbers = Int32.(HCN_ATNRS)
    box = Float64[HCN_BOX...]

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
        kernel = MolInvDistSE(HCN_ATNRS, Float64[])
        cfg_gp = NEBConfig(;
            images=8,
            spring_constant=1.0,
            climbing_image=true,
            energy_weighted=true,
            ew_k_min=0.972,
            ew_k_max=9.72,
            conv_tol=0.05,
            gp_train_iter=200,
            max_outer_iter=50,
            trust_radius=0.15,
            atom_types=Int.(HCN_ATNRS),
            max_gp_points=40,
            rff_features=200,
        )
        result_aie = gp_neb_aie(oracle, X_HCN, X_HNC, kernel; config=cfg_gp)
    catch e
        @warn "GP-NEB AIE failed" exception = e
    end

    close(pot)
    return result_std, result_aie
end

function main()
    hp = h5_path()

    if isfile(hp)
        println("Using cached data from $hp")
        return
    end

    result_std, result_aie = run_with_rpc()

    img_std, en_std, m_std = make_profile(result_std, "Standard NEB")

    all_img = copy(img_std)
    all_en = copy(en_std)
    all_method = copy(m_std)

    if result_aie !== nothing
        img_aie, en_aie, m_aie = make_profile(result_aie, "GP-NEB AIE")
        append!(all_img, img_aie)
        append!(all_en, en_aie)
        append!(all_method, m_aie)
    end

    h5_write_table(hp, "table", Dict(
        "image" => all_img,
        "energy" => all_en,
        "method" => all_method,
    ))

    println("Wrote HDF5: $hp")
end

main()
