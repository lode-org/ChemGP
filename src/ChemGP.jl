module ChemGP

using LinearAlgebra
using Statistics
using Optim
using Printf
using ForwardDiff
using KernelFunctions
using ParameterHandling

# Include core logic
include("functions.jl")
include("MolecularKernel.jl")

# Export public API
export GPModel
export train_model!, predict

# Export the custom kernel
export MolecularKernel

end
