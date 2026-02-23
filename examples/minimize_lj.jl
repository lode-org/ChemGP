# ==============================================================================
# GP-Guided Minimization of a Lennard-Jones Cluster
# ==============================================================================
#
# This example demonstrates the core GP-guided optimization loop:
# 1. Start from a random cluster
# 2. Evaluate the oracle (LJ potential) at a few initial points
# 3. Train a GP on the accumulated energy + gradient data
# 4. Optimize on the GP surface using L-BFGS
# 5. Evaluate the oracle at the GP-predicted minimum
# 6. Repeat until convergence
#
# The GP acts as a cheap surrogate for the expensive oracle. In real
# applications, the oracle would be a DFT or ab initio calculation.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ChemGP
using LinearAlgebra

# ---- Step 1: Create a random starting cluster ----
N_atoms = 4  # Small cluster for quick demonstration
x_init = random_cluster(N_atoms; min_dist=1.2)

E_init, G_init = lj_energy_gradient(x_init)
println("Initial energy: $(round(E_init, digits=4))")
println("Initial |grad|: $(round(norm(G_init), digits=4))")

# ---- Step 2: Set up the molecular kernel ----
# MolInvDistSE: Squared Exponential kernel on inverse interatomic distances
# This is equivalent to SexpatCF in gpr_optim
kernel = MolInvDistSE(1.0, [0.5], Float64[])

# ---- Step 3: Configure and run GP-guided minimization ----
config = MinimizationConfig(;
    trust_radius=0.15, conv_tol=1e-2, max_iter=30, gp_train_iter=200, verbose=true
)

result = gp_minimize(lj_energy_gradient, x_init, kernel; config=config)

# ---- Step 4: Report results ----
println("\n" * "="^50)
println("Results")
println("="^50)
println("Converged:    $(result.converged)")
println("Final energy: $(round(result.E_final, digits=6))")
println("|grad|:       $(round(norm(result.G_final), digits=6))")
println("Oracle calls: $(result.oracle_calls)")
