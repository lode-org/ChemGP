module ChemGP

using LinearAlgebra
using Statistics
using Optim
using Printf

# Include core logic
include("functions.jl")

# Export public API
export SquaredExpKernel, GPModel
export train_model!, predict

end
