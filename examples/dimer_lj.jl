# ==============================================================================
# GP-Dimer Saddle Point Search on a Lennard-Jones Cluster
# ==============================================================================
#
# This example demonstrates the GP-dimer method for finding transition states
# (first-order saddle points) on a potential energy surface.
#
# The dimer method works by:
# 1. Rotating a "dimer" (pair of nearby configurations) to find the direction
#    of lowest curvature
# 2. Translating with a modified force that climbs along the lowest curvature
#    direction while descending in all other directions
#
# Combined with GP regression, the rotation and translation steps are performed
# on the cheap GP surface, and the oracle is only called when needed.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ChemGP
using LinearAlgebra

# ---- Step 1: Start from a configuration ----
N_atoms = 3  # Small system for demonstration
x_init = random_cluster(N_atoms; min_dist=1.2)

E_init, G_init = lj_energy_gradient(x_init)
println("Starting energy: $(round(E_init, digits=4))")

# ---- Step 2: Choose initial dimer orientation ----
# Random unit vector in the configuration space
orient = randn(3 * N_atoms)
orient ./= norm(orient)

# ---- Step 3: Set up kernel and configuration ----
kernel = MolInvDistSE(1.0, [0.5], Float64[])

config = DimerConfig(;
    T_force_true=5e-2,    # Loose tolerance for demo
    T_force_gp=1e-1,
    trust_radius=0.2,
    max_outer_iter=15,
    max_inner_iter=30,
    gp_train_iter=150,
    verbose=true,
)

# ---- Step 4: Run GP-dimer search ----
result = gp_dimer(lj_energy_gradient, x_init, orient, kernel; config=config, dimer_sep=0.01)

# ---- Step 5: Report results ----
E_final, G_final = lj_energy_gradient(result.state.R)
println("\n" * "="^50)
println("Results")
println("="^50)
println("Converged:    $(result.converged)")
println("Final energy: $(round(E_final, digits=6))")
println("|grad|:       $(round(norm(G_final), digits=6))")
println("Oracle calls: $(result.oracle_calls)")
