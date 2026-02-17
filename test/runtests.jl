using ChemGP
using Test
using Statistics
using LinearAlgebra
using KernelFunctions

@testset "ChemGP" begin
    include("test_lbfgs.jl")
    include("test_kernels.jl")
    include("test_variance.jl")
    include("test_distances.jl")
    include("test_sampling.jl")
    include("test_minimize.jl")
    include("test_dimer.jl")
    include("test_muller_brown.jl")
    include("test_neb.jl")

    if isfile(joinpath(@__DIR__, "cpp_consistency.jl"))
        include("cpp_consistency.jl")
    end
end
