# ==============================================================================
# Core Types for ChemGP
# ==============================================================================

# ==============================================================================
# TrainingData: Manages the growing dataset of oracle evaluations
# ==============================================================================
#
# In GP-guided optimization, new oracle evaluations are added iteratively.
# This struct provides a clean container for that pattern, making it explicit
# when data flows into the GP model.

"""
    TrainingData

Container for the growing dataset of oracle evaluations in GP-guided optimization.

# Fields
- `X::Matrix{Float64}`: Configuration matrix (D x N), each column is a flat coordinate vector
- `energies::Vector{Float64}`: Raw oracle energies (not normalized), length N
- `gradients::Vector{Float64}`: Raw oracle gradients concatenated, length D*N

New oracle evaluations are added iteratively via [`add_point!`](@ref). The data can be
normalized for GP training via [`normalize`](@ref), which returns zero-mean, unit-variance
targets with gradients scaled by the same factor.

See also: [`GPModel`](@ref), [`add_point!`](@ref), [`normalize`](@ref)
"""
mutable struct TrainingData
    X::Matrix{Float64}         # D x N, each column is a flat coordinate vector
    energies::Vector{Float64}  # Raw oracle energies (not normalized)
    gradients::Vector{Float64} # Raw oracle gradients (concatenated, length D*N)
end

"""
    TrainingData(D::Int)

Create an empty `TrainingData` container for configurations of dimension `D`.
"""
function TrainingData(D::Int)
    return TrainingData(Matrix{Float64}(undef, D, 0), Float64[], Float64[])
end

"""
    add_point!(td::TrainingData, x, E, G)

Add a single oracle evaluation (configuration `x`, energy `E`, gradient `G`)
to the training set.
"""
function add_point!(td::TrainingData, x::AbstractVector{Float64}, E::Real, G::AbstractVector{Float64})
    td.X = hcat(td.X, x)
    push!(td.energies, E)
    append!(td.gradients, G)
    return td
end

"""
    npoints(td::TrainingData)

Number of training points currently stored.
"""
npoints(td::TrainingData) = size(td.X, 2)

"""
    normalize(td::TrainingData)

Prepare normalized targets for GP training. Returns `(y_full, y_mean, y_std)` where
`y_full = [normalized_energies; normalized_gradients]`.

The normalization is:
- Energies: `(E - mean) / std`
- Gradients: `G / std` (shifted by energy mean does not affect gradient)
"""
function normalize(td::TrainingData)
    y_mean = mean(td.energies)
    y_std = max(std(td.energies), 1e-10)
    y_norm = (td.energies .- y_mean) ./ y_std
    g_norm = td.gradients ./ y_std
    return vcat(y_norm, g_norm), y_mean, y_std
end

# ==============================================================================
# GP Model Container
# ==============================================================================
#
# The GP model holds the kernel, training data, and noise parameters.
# It is parameterized on the kernel type so that `train_model!` can
# reconstruct the kernel from optimized hyperparameters.

"""
    GPModel{Tk<:Kernel}

Gaussian process model for molecular energy surfaces with derivative observations.

The model holds a kernel, training data, and noise parameters. It is parameterized
on the kernel type `Tk` so that [`train_model!`](@ref) can reconstruct the kernel
from optimized hyperparameters.

# Fields
- `kernel::Tk`: Covariance kernel (e.g., [`MolInvDistSE`](@ref), [`MolSumKernel`](@ref))
- `X::AbstractMatrix{Float64}`: Training inputs (D x N)
- `y::Vector{Float64}`: Training targets `[energies; gradients]`, length N*(1+D)
- `noise_var::Float64`: Energy observation noise variance (σ_n²)
- `grad_noise_var::Float64`: Gradient observation noise variance (σ_g²)
- `jitter::Float64`: Diagonal jitter for numerical stability

See also: [`train_model!`](@ref), [`predict`](@ref), [`predict_with_variance`](@ref)
"""
mutable struct GPModel{Tk<:Kernel}
    kernel::Tk
    X::AbstractMatrix{Float64}       # Inputs (D x N)
    y::Vector{Float64}               # Targets [Energies; Gradients]
    noise_var::Float64               # Energy noise variance (sigma_n^2)
    grad_noise_var::Float64          # Gradient noise variance (sigma_g^2)
    jitter::Float64                  # Numerical stability jitter
end

function GPModel(kernel, X, y; noise_var = 1e-6, grad_noise_var = 1e-6, jitter = 1e-6)
    return GPModel(kernel, Matrix(X), y, noise_var, grad_noise_var, jitter)
end
