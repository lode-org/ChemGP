# LEPS-NEB generator: Standard NEB vs GP-NEB AIE/OIE on LEPS
#
# Compares oracle efficiency of standard NEB, GP-NEB AIE, and GP-NEB OIE
# on the 9D LEPS surface (3-atom collinear H+H2 exchange).
#
# Output: {stem}.h5 with /table group (oracle_calls, max_force, method)

include(joinpath(@__DIR__, "common_data.jl"))
using ChemGP
using KernelFunctions

function extract_history(result, label)
    calls = result.history["oracle_calls"]
    forces = result.history["max_force"]
    n = min(length(calls), length(forces))
    return calls[1:n], forces[1:n], fill(label, n)
end

function main()
    # Kernel: MolInvDistSE for 3-atom system
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
    oc_std, mf_std, m_std = extract_history(result_std, "Standard NEB")
    oc_aie, mf_aie, m_aie = extract_history(result_aie, "GP-NEB AIE")
    oc_oie, mf_oie, m_oie = extract_history(result_oie, "GP-NEB OIE")

    all_oc = vcat(oc_std, oc_aie, oc_oie)
    all_mf = vcat(mf_std, mf_aie, mf_oie)
    all_method = vcat(m_std, m_aie, m_oie)

    h5_write_table(h5_path(), "table", Dict(
        "oracle_calls" => all_oc,
        "max_force" => all_mf,
        "method" => all_method,
    ))

    println("Wrote HDF5: $(h5_path())")
end

main()
