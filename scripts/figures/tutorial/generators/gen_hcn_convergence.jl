# HCN-CONV generator: convergence comparison for HCN -> HNC isomerization
#
# Tracks max force residual vs oracle calls for standard NEB, GP-NEB AIE,
# and GP-NEB OIE on the HCN -> HNC isomerization.
#
# Uses cache pattern: checks for existing HDF5 before running.
# Requires: PET-MAD server running at localhost:12345
#
# Output: {stem}.h5 with /table group (oracle_calls, max_force, method)

include(joinpath(@__DIR__, "common_data.jl"))
using ChemGP
using LinearAlgebra

# Load HCN/HNC from data files
# From generators/ the path to data is one more .. than from tutorial/
const _hcn_data = read_extxyz(joinpath(@__DIR__, "..", "..", "..", "..", "data", "hcn", "hcn.extxyz"))
const _hnc_data = read_extxyz(joinpath(@__DIR__, "..", "..", "..", "..", "data", "hcn", "hnc.extxyz"))
const X_HCN = _hcn_data.positions
const X_HNC = _hnc_data.positions
const HCN_ATNRS = _hcn_data.atomic_numbers
const HCN_BOX = _hcn_data.box

function extract_convergence(result, label)
    calls = result.history["oracle_calls"]
    forces = result.history["max_force"]
    n = min(length(calls), length(forces))
    return calls[1:n], forces[1:n], fill(label, n)
end

function run_convergence()
    host = get(ENV, "RGPOT_HOST", "localhost")
    port = parse(Int, get(ENV, "RGPOT_PORT", "12345"))
    atomic_numbers = Int32.(HCN_ATNRS)
    box = Float64[HCN_BOX...]

    println("Connecting to PET-MAD server at $host:$port")
    pot = RpcPotential(host, port, atomic_numbers, box)
    oracle = make_rpc_oracle(pot)

    neb_cfg = NEBConfig(;
        images=8,
        spring_constant=1.0,
        climbing_image=true,
        energy_weighted=true,
        ew_k_min=0.972,
        ew_k_max=9.72,
        max_iter=1000,
        conv_tol=0.05,
        step_size=0.01,
        verbose=true,
    )

    # Standard NEB
    println("Running standard NEB...")
    result_std = neb_optimize(oracle, X_HCN, X_HNC; config=neb_cfg)

    # GP-NEB AIE
    result_aie = nothing
    try
        println("Running GP-NEB AIE...")
        kernel = MolInvDistSE(HCN_ATNRS, Float64[])
        gp_cfg = NEBConfig(;
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
            verbose=true,
        )
        result_aie = gp_neb_aie(oracle, X_HCN, X_HNC, kernel; config=gp_cfg)
    catch e
        @warn "GP-NEB AIE failed" exception = e
    end

    # GP-NEB OIE
    result_oie = nothing
    try
        println("Running GP-NEB OIE...")
        kernel = MolInvDistSE(HCN_ATNRS, Float64[])
        oie_cfg = NEBConfig(;
            images=8,
            spring_constant=1.0,
            climbing_image=true,
            energy_weighted=true,
            ew_k_min=0.972,
            ew_k_max=9.72,
            conv_tol=0.05,
            gp_train_iter=200,
            max_outer_iter=120,
            trust_radius=0.15,
            atom_types=Int.(HCN_ATNRS),
            max_gp_points=40,
            rff_features=200,
            verbose=true,
        )
        result_oie = gp_neb_oie(oracle, X_HCN, X_HNC, kernel; config=oie_cfg)
    catch e
        @warn "GP-NEB OIE failed" exception = e
    end

    close(pot)
    return result_std, result_aie, result_oie
end

function main()
    hp = h5_path()

    if isfile(hp)
        println("Using cached data from $hp")
        return
    end

    result_std, result_aie, result_oie = run_convergence()

    oc_std, mf_std, m_std = extract_convergence(result_std, "Standard NEB")

    all_oc = copy(oc_std)
    all_mf = copy(mf_std)
    all_method = copy(m_std)

    if result_aie !== nothing
        oc_aie, mf_aie, m_aie = extract_convergence(result_aie, "GP-NEB AIE")
        append!(all_oc, oc_aie)
        append!(all_mf, mf_aie)
        append!(all_method, m_aie)
    end

    if result_oie !== nothing
        oc_oie, mf_oie, m_oie = extract_convergence(result_oie, "GP-NEB OIE")
        append!(all_oc, oc_oie)
        append!(all_mf, mf_oie)
        append!(all_method, m_oie)
    end

    h5_write_table(hp, "table", Dict(
        "oracle_calls" => all_oc,
        "max_force" => all_mf,
        "method" => all_method,
    ))

    println("Wrote HDF5: $hp")
end

main()
