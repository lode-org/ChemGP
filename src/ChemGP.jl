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

# Molecular kernels (SE, Matern 5/2, Matern 3/2 on inverse distances)
include("kernels/MolInvDistSE.jl")
include("kernels/MolInvDistMat5_2.jl")
include("kernels/MolInvDistMat3_2.jl")

# Cartesian SE kernel (for GP-NEB, full-rank gradient blocks)
include("kernels/CartesianSE.jl")

# Additional kernels for composition
include("kernels/ConstantKernel.jl")
include("kernels/SumKernel.jl")
include("kernels/ProductKernel.jl")

# ==============================================================================
# GP core
# ==============================================================================

# Derivative block computation via ForwardDiff
include("derivatives.jl")

# Types: GPModel, TrainingData
include("types.jl")

# GP functions: build_full_covariance, train_model!, predict, predict_with_variance
include("functions.jl")

# Random Fourier Features for scalable GP-NEB with MolInvDistSE
include("rff.jl")

# ==============================================================================
# Distance metrics and sampling
# ==============================================================================
include("distances.jl")
include("distances_emd.jl")
include("distances_trust.jl")
include("sampling.jl")

# ==============================================================================
# Oracles (pedagogical test potentials)
# ==============================================================================
include("oracles/lennard_jones.jl")
include("oracles/muller_brown.jl")
include("oracles/leps.jl")

# RPC oracle: connect to remote potential servers via rgpot
include("oracles/rpc.jl")

# ==============================================================================
# Optimization methods
# ==============================================================================
include("optimizers/lbfgs.jl")
include("optimizers/trust_region.jl")
include("optimizers/minimize.jl")
include("optimizers/dimer.jl")
include("optimizers/optim_step.jl")
include("optimizers/neb_types.jl")
include("optimizers/neb_path.jl")
include("optimizers/idpp.jl")
include("optimizers/neb.jl")
include("optimizers/neb_oie_naive.jl")
include("optimizers/neb_oie.jl")
include("optimizers/otgpd.jl")

# ==============================================================================
# I/O utilities
# ==============================================================================
include("io/extxyz.jl")
include("io/hdf5.jl")

# ==============================================================================
# Exports
# ==============================================================================

# Types
export GPModel, TrainingData, add_point!, npoints, normalize

# GP core
export train_model!, predict, predict_with_variance, build_full_covariance
export RFFModel, build_rff

# Kernels
export AbstractMoleculeKernel
export MolInvDistSE, MolInvDistMatern52, MolInvDistMatern32
export CartesianSE, init_cartesian_se, init_mol_invdist_se
export OffsetKernel, MolSumKernel, MolProductKernel
export kernel_blocks, compute_inverse_distances

# Distance metrics
export interatomic_distances, max_1d_log_distance, rmsd_distance
export emd_distance
export trust_distance_fn, trust_min_distance, adaptive_trust_threshold

# Sampling
export farthest_point_sampling, prune_training_data!

# Oracles
export lj_energy_gradient, random_cluster
export muller_brown_energy_gradient
export MULLER_BROWN_MINIMA, MULLER_BROWN_SADDLES
export leps_energy_gradient, leps_energy_gradient_2d
export LEPS_REACTANT, LEPS_PRODUCT

# RPC oracle (rgpot integration)
export RpcPotential, RpcPotentialCore, make_rpc_oracle, make_oracle_pool
export find_rgpot_lib, find_potserv, with_potserv
export rgpot_available, potserv_available

# Minimization
export gp_minimize, MinimizationConfig, MinimizationResult

# Dimer saddle point search
export gp_dimer, DimerState, DimerConfig, DimerResult
export dimer_images, curvature, rotational_force, translational_force

# L-BFGS optimizer
export LBFGSHistory, push_pair!, compute_direction
export OptimState, optim_step!

# NEB path optimization
export NEBPath, NEBConfig, NEBResult
export linear_interpolation, idpp_interpolation, sidpp_interpolation
export path_tangent, spring_force, neb_force
export energy_weighted_k, get_hessian_points
export neb_optimize, gp_neb_aie, gp_neb_oie, gp_neb_oie_naive

# OTGPD (Optimal Transport GP Dimer)
export otgpd, OTGPDConfig, OTGPDResult

# Trust region utilities
export min_distance_to_data, check_interatomic_ratio, remove_rigid_body_modes!

# I/O
export write_neb_trajectory, write_neb_dat, write_convergence_csv
export write_neb_hdf5, make_neb_writer, make_neb_hdf5_writer, ELEMENT_SYMBOLS

end
