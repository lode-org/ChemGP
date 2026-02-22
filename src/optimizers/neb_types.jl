# ==============================================================================
# NEB Types and Configuration
# ==============================================================================

"""
    NEBPath

Mutable state of a nudged elastic band path.

# Fields
- `images`: Vector of configuration vectors (including fixed endpoints)
- `energies`: Energy at each image
- `gradients`: Gradient at each image
- `spring_constant`: Spring constant for NEB forces
"""
mutable struct NEBPath
    images::Vector{Vector{Float64}}
    energies::Vector{Float64}
    gradients::Vector{Vector{Float64}}
    spring_constant::Float64
end

"""
    NEBConfig

Configuration parameters for NEB path optimization.

# Standard NEB fields
- `n_images`: Number of images including endpoints (default 7)
- `spring_constant`: Spring constant for elastic band (default 5.0)
- `climbing_image`: Whether to enable climbing image NEB (default true)
- `ci_activation_tol`: Absolute force threshold for CI activation (default 0.5)
- `ci_trigger_rel`: Relative CI trigger -- CI activates when force < ci_trigger_rel * baseline (default 0.8)
- `ci_converged_only`: Check convergence at CI image only, not all images (default true)
- `max_iter`: Maximum iterations for standard NEB / inner GP loop (default 1000)
- `conv_tol`: Convergence tolerance on max force norm (default 0.05)
- `step_size`: Step size for steepest descent relaxation (default 0.01)
- `optimizer`: Relaxation method, `:lbfgs` or `:sd` (default `:lbfgs`)
- `max_move`: Maximum per-atom displacement per step (default 0.1)
- `lbfgs_memory`: Number of (s,y) pairs retained by L-BFGS (default 20)
- `initializer`: Path interpolation method, `:sidpp`, `:idpp`, or `:linear` (default `:sidpp`)

# Energy-weighted springs (Henkelman & Jonsson 2000)
- `energy_weighted`: Use energy-dependent spring constants (default false)
- `ew_k_min`: Minimum spring constant for low-energy springs (default 1.0)
- `ew_k_max`: Maximum spring constant for high-energy springs (default 10.0)

# GP-specific fields
- `gp_train_iter`: Nelder-Mead iterations for GP hyperparameter optimization (default 300)
- `max_outer_iter`: Maximum outer iterations for GP-NEB (default 50)
- `trust_radius`: Maximum displacement from training data (default 0.1)

# Virtual Hessian points (Koistinen et al. 2017, J. Chem. Phys. 147, 152720)
- `num_hess_iter`: Outer iterations to include Hessian perturbation points; 0 disables (default 0)
- `eps_hess`: Displacement magnitude for Hessian finite-difference points in Angstrom (default 0.01)
- `verbose`: Print progress (default true)
"""
Base.@kwdef struct NEBConfig
    n_images::Int           = 7
    spring_constant::Float64 = 5.0
    climbing_image::Bool    = true
    ci_activation_tol::Float64 = 0.5
    ci_trigger_rel::Float64 = 0.8
    ci_converged_only::Bool = true
    max_iter::Int           = 1000
    conv_tol::Float64       = 0.05
    step_size::Float64      = 0.01
    # Optimizer: :lbfgs or :sd (steepest descent)
    optimizer::Symbol       = :lbfgs
    max_move::Float64       = 0.1
    lbfgs_memory::Int       = 20
    # Path initialization: :sidpp, :idpp, or :linear
    initializer::Symbol     = :sidpp
    # Energy-weighted springs (Henkelman & Jonsson 2000)
    energy_weighted::Bool   = false
    ew_k_min::Float64       = 1.0
    ew_k_max::Float64       = 10.0
    # GP-specific
    gp_train_iter::Int      = 300
    max_outer_iter::Int     = 50
    trust_radius::Float64   = 0.1
    # Virtual Hessian points (Koistinen et al. 2017):
    # Generate finite-difference perturbation points around endpoints to
    # bootstrap GP training with curvature information.
    num_hess_iter::Int      = 0     # outer iterations to include hessian pts (0=off)
    eps_hess::Float64       = 0.01  # displacement for hessian points (Angstrom)
    verbose::Bool           = true
end

"""
    NEBResult

Result of a NEB path optimization.

# Fields
- `path`: Final [`NEBPath`](@ref) with converged images
- `converged`: Whether convergence criterion was met
- `oracle_calls`: Total number of oracle evaluations
- `max_energy_image`: Index of the highest-energy image (transition state estimate)
- `history`: Dict with convergence history
"""
struct NEBResult
    path::NEBPath
    converged::Bool
    oracle_calls::Int
    max_energy_image::Int
    history::Dict{String,Vector}
end
