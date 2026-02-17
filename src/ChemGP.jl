module ChemGP

using LinearAlgebra
using Statistics
using Optim
using Printf
using ForwardDiff
using KernelFunctions
using ParameterHandling

# ==============================================================================
# Kernel infrastructure
# ==============================================================================
include("kernels/abstract.jl")

# Feature computation (used by molecular kernels)
include("invdist.jl")

# Molecular kernels (SE and Matern 5/2 on inverse distances)
include("kernels/MolInvDistSE.jl")
include("kernels/MolInvDistMat5_2.jl")

# Additional kernels for composition
include("kernels/ConstantKernel.jl")
include("kernels/SumKernel.jl")

# ==============================================================================
# GP core
# ==============================================================================

# Derivative block computation via ForwardDiff
include("derivatives.jl")

# Types: GPModel, TrainingData
include("types.jl")

# GP functions: build_full_covariance, train_model!, predict, predict_with_variance
include("functions.jl")

# ==============================================================================
# Distance metrics and sampling
# ==============================================================================
include("distances.jl")
include("sampling.jl")

# ==============================================================================
# Oracles (pedagogical test potentials)
# ==============================================================================
include("oracles/lennard_jones.jl")

# RPC oracle: connect to remote potential servers via rgpot
include("oracles/rpc.jl")

# ==============================================================================
# Optimization methods
# ==============================================================================
include("optimizers/trust_region.jl")
include("optimizers/minimize.jl")
include("optimizers/dimer.jl")

# ==============================================================================
# Exports
# ==============================================================================

# Types
export GPModel, TrainingData, add_point!, npoints, normalize

# GP core
export train_model!, predict, predict_with_variance, build_full_covariance

# Kernels
export AbstractMoleculeKernel
export MolInvDistSE, MolInvDistMatern52
export OffsetKernel, MolSumKernel
export kernel_blocks, compute_inverse_distances

# Distance metrics
export interatomic_distances, max_1d_log_distance, rmsd_distance

# Sampling
export farthest_point_sampling

# Oracles
export lj_energy_gradient, random_cluster

# RPC oracle (rgpot integration)
export RpcPotential, RpcPotentialCore, make_rpc_oracle

# Minimization
export gp_minimize, MinimizationConfig, MinimizationResult

# Dimer saddle point search
export gp_dimer, DimerState, DimerConfig, DimerResult
export dimer_images, curvature, rotational_force, translational_force

# Trust region utilities
export min_distance_to_data, check_interatomic_ratio

end
