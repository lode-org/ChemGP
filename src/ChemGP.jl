module ChemGP

using LinearAlgebra
using Statistics
using Optim
using Printf
using ForwardDiff
using KernelFunctions
using ParameterHandling

# 1. Include the Kernel Module
include("MolecularKernel.jl")
using .MolecularKernels

# 2. Include the Training/Prediction Logic
include("functions.jl")

# 3. Exports
export GPModel
export train_model!, predict

# Re-export the custom kernel
export MolecularKernel, kernel_blocks

end
