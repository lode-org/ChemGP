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

# Fields
- `n_images`: Number of images including endpoints (default 7)
- `spring_constant`: Spring constant for elastic band (default 1.0)
- `climbing_image`: Whether to enable climbing image NEB (default true)
- `ci_activation_tol`: Force norm below which climbing image activates (default 0.5)
- `max_iter`: Maximum iterations for standard NEB (default 500)
- `conv_tol`: Convergence tolerance on max force norm (default 5e-3)
- `step_size`: Step size for steepest descent relaxation (default 0.01)

# GP-specific fields
- `gp_train_iter`: Nelder-Mead iterations for GP training (default 300)
- `max_outer_iter`: Maximum outer iterations for GP-NEB (default 50)
- `trust_radius`: Maximum displacement from training data (default 0.1)
- `verbose`: Print progress (default true)
"""
Base.@kwdef struct NEBConfig
    n_images::Int           = 7
    spring_constant::Float64 = 1.0
    climbing_image::Bool    = true
    ci_activation_tol::Float64 = 0.5
    max_iter::Int           = 500
    conv_tol::Float64       = 5e-3
    step_size::Float64      = 0.01
    # GP-specific
    gp_train_iter::Int      = 300
    max_outer_iter::Int     = 50
    trust_radius::Float64   = 0.1
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
