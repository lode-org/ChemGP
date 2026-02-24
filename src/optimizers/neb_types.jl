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
- `images`: Number of movable intermediate images, excluding fixed endpoints (default 5). Matches eOn convention; total images = images + 2.
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
- `max_gp_points`: Cap GP training set size via FPS subset selection; 0 = use all data (default 0)
- `trust_radius`: Maximum displacement from training data (default 0.1)
- `trust_metric`: Distance metric for trust region (:emd, :max_1d_log, :euclidean; default :emd)
- `atom_types`: Integer element labels per atom for EMD (default Int[] = all same type)
- `use_adaptive_threshold`: Enable sigmoidal trust decay with training set size (default false)
- `adaptive_t_min`: Asymptotic minimum threshold (default 0.15)
- `adaptive_delta_t`: Initial excess range (default 0.35)
- `adaptive_n_half`: Half-life in effective data points (default 50)
- `adaptive_A`: Steepness of sigmoidal transition (default 1.3)
- `adaptive_floor`: Absolute minimum threshold (default 0.2)

# OIE early stopping (Koistinen et al. 2019, J. Chem. Phys. 150, 094106)
- `ci_force_tol`: Convergence threshold on CI force, separate from band `conv_tol` (default -1 = use conv_tol)
- `inner_ci_threshold`: GP force level at which CI turns on during inner relaxation; 0 = no CI (default 0.5)
- `gp_tol_divisor`: Adaptive GP convergence tolerance = smallest_observed_force / divisor; 0 = fixed min(conv_tol, ci_force_tol)/10 (default 10)
- `max_step_frac`: Reject inner step if any image moves further than this fraction of the initial path length from nearest training point (default 0.1)
- `bond_stretch_limit`: Reject inner step if any interatomic distance ratio to nearest training point exceeds |log(limit)|, e.g. 2/3 means bonds cannot shrink below 2/3 or stretch above 3/2 (default 2/3)

# Virtual Hessian points (Koistinen et al. 2017, J. Chem. Phys. 147, 152720)
- `num_hess_iter`: Outer iterations to include Hessian perturbation points; 0 disables (default 0)
- `eps_hess`: Displacement magnitude for Hessian finite-difference points in Angstrom (default 0.01)
- `verbose`: Print progress (default true)
"""
Base.@kwdef struct NEBConfig
    images::Int = 5
    spring_constant::Float64 = 5.0
    climbing_image::Bool = true
    ci_activation_tol::Float64 = 0.5
    ci_trigger_rel::Float64 = 0.8
    ci_converged_only::Bool = true
    max_iter::Int = 1000
    conv_tol::Float64 = 0.05
    step_size::Float64 = 0.01
    # Optimizer: :lbfgs or :sd (steepest descent)
    optimizer::Symbol = :lbfgs
    max_move::Float64 = 0.1
    lbfgs_memory::Int = 20
    # Path initialization: :sidpp, :idpp, or :linear
    initializer::Symbol = :sidpp
    # Energy-weighted springs (Henkelman & Jonsson 2000)
    energy_weighted::Bool = false
    ew_k_min::Float64 = 1.0
    ew_k_max::Float64 = 10.0
    # GP-specific
    gp_train_iter::Int = 300
    max_outer_iter::Int = 50
    max_gp_points::Int = 0     # max training points for GP (0 = no limit; per-bead subset when exceeded)
    rff_features::Int = 0     # RFF feature dimension (0 = exact GP; >0 = RFF approximation for MolInvDistSE)
    trust_radius::Float64 = 0.1
    # Trust region metric and adaptive threshold (shared with OTGPD; see distances_trust.jl)
    trust_metric::Symbol = :emd
    atom_types::Vector{Int} = Int[]
    use_adaptive_threshold::Bool = false
    adaptive_t_min::Float64 = 0.15
    adaptive_delta_t::Float64 = 0.35
    adaptive_n_half::Int = 50
    adaptive_A::Float64 = 1.3
    adaptive_floor::Float64 = 0.2
    # OIE early stopping (Koistinen et al. 2019):
    ci_force_tol::Float64 = -1.0    # CI convergence threshold (-1 = use conv_tol)
    inner_ci_threshold::Float64 = 0.5   # GP force level to activate CI in inner loop (0 = no CI)
    gp_tol_divisor::Int = 10      # GP inner tol = smallest_accurate_force / divisor (0 = fixed)
    max_step_frac::Float64 = 0.1     # max displacement from training data / initial path length
    bond_stretch_limit::Float64 = 2.0/3.0 # bond ratio limit for early stopping
    # Virtual Hessian points (Koistinen et al. 2017):
    # Generate finite-difference perturbation points around endpoints to
    # bootstrap GP training with curvature information.
    num_hess_iter::Int = 0     # outer iterations to include hessian pts (0=off)
    eps_hess::Float64 = 0.01  # displacement for hessian points (Angstrom)
    verbose::Bool = true
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
    stop_reason::StopReason
    oracle_calls::Int
    max_energy_image::Int
    history::Dict{String,Vector}
end
