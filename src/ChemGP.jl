module ChemGP

using LinearAlgebra
using Statistics
using Optim
using Printf
using ForwardDiff

# Include core logic
include("functions.jl")

# Export public API
export AbstractGPKernel
export SquaredExpKernel, SquaredExpADKernel
export GPModel
export train_model!, predict

end
