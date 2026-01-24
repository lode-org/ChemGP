module ChemGP

using LinearAlgebra
using Statistics
using Optim
using Printf
using ForwardDiff
using KernelFunctions
using ParameterHandling

include("kernels/abstract.jl")
include("kernels/MolInvDistSE.jl")
include("kernels/MolInvDistMat5_2.jl")
include("functions.jl")
include("derivatives.jl")
include("invdist.jl")

# 3. Exports
export GPModel
export train_model!, predict

# Re-export the custom kernel
export MolInvDistSE, kernel_blocks

end
