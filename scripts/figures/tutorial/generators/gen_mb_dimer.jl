# MB-DIMER generator: GP-dimer vs classical dimer on Muller-Brown
#
# Starts near Saddle 2 and tracks force norm vs oracle calls for the
# GP-dimer method. The "classical" baseline uses max_inner_iter=0.
#
# Output: {stem}.h5 with /table group (oracle_calls, force_norm, method)

include(joinpath(@__DIR__, "common_data.jl"))
using ChemGP
using KernelFunctions
using LinearAlgebra

function extract_convergence(result, label)
    oracle_calls = result.history["oracle_calls"]
    force_norms = result.history["F_true"]
    n = min(length(oracle_calls), length(force_norms))
    return oracle_calls[1:n], force_norms[1:n], fill(label, n)
end

function main()
    # Starting point: offset from saddle S2
    x_init = [0.3, 0.4]
    orient_init = LinearAlgebra.normalize([1.0, 0.5])

    # --- GP-dimer run ---
    println("Running GP-dimer...")
    kernel = 1.0 * with_lengthscale(SqExponentialKernel(), 0.3)

    gp_config = DimerConfig(;
        T_force_true=1e-3,
        T_force_gp=1e-2,
        trust_radius=0.3,
        max_outer_iter=30,
        max_inner_iter=100,
        alpha_trans=0.01,
        gp_train_iter=300,
        n_initial_perturb=4,
        perturb_scale=0.15,
        verbose=true,
    )

    result_gp = gp_dimer(
        muller_brown_energy_gradient, x_init, orient_init, kernel; config=gp_config
    )

    println("GP-dimer converged: $(result_gp.converged)")
    println("GP-dimer oracle calls: $(result_gp.oracle_calls)")
    println("GP-dimer final position: $(result_gp.state.R)")

    # --- Classical dimer (oracle-every-step) ---
    println("Running classical dimer (max_inner_iter=0)...")
    classical_config = DimerConfig(;
        T_force_true=1e-3,
        T_force_gp=1e-2,
        trust_radius=0.3,
        max_outer_iter=60,
        max_inner_iter=0,
        alpha_trans=0.01,
        gp_train_iter=300,
        n_initial_perturb=4,
        perturb_scale=0.15,
        verbose=false,
    )

    result_cl = gp_dimer(
        muller_brown_energy_gradient, x_init, orient_init, kernel; config=classical_config
    )

    println("Classical dimer converged: $(result_cl.converged)")
    println("Classical dimer oracle calls: $(result_cl.oracle_calls)")

    # --- Extract convergence data from history ---
    oc_gp, fn_gp, m_gp = extract_convergence(result_gp, "GP-dimer")
    oc_cl, fn_cl, m_cl = extract_convergence(result_cl, "Classical dimer")

    all_oc = vcat(oc_gp, oc_cl)
    all_fn = vcat(fn_gp, fn_cl)
    all_method = vcat(m_gp, m_cl)

    h5_write_table(h5_path(), "table", Dict(
        "oracle_calls" => all_oc,
        "force_norm" => all_fn,
        "method" => all_method,
    ))

    println("Wrote HDF5: $(h5_path())")
end

main()
