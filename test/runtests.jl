using ChemGP
using Test
using Statistics
using LinearAlgebra

# Define the Morse Potential (Energy and Gradient)
function morse(r)
    De, a, re = 10.0, 1.0, 1.5
    val = 1 - exp(-a*(r - re))
    E = De * val^2
    F = -2 * De * val * (exp(-a*(r - re)) * a)
    return E, [F]
end

@testset "ChemGP.jl Tests" begin
    # 1. Generate & Normalize Data
    X_train = reshape(collect(1.0:0.5:3.0), 1, :)
    y_vals = Float64[]
    y_grads = Float64[]
    
    for i in 1:size(X_train, 2)
        E, F = morse(X_train[1, i])
        push!(y_vals, E)
        append!(y_grads, F)
    end
    
    y_mean = mean(y_vals)
    y_std = std(y_vals)
    y_full = [(y_vals .- y_mean) ./ y_std; y_grads ./ y_std]
    
    # 2. Test Analytical Kernel
    @testset "Analytical Kernel" begin
        kern = SquaredExpKernel(1.0, [0.5]) 
        model = GPModel(kern, X_train, y_full, log(1e-4), log(1e-4))
        train_model!(model; iterations=200)
        
        X_test = reshape([1.5], 1, 1) # Equilibrium point
        preds = predict(model, X_test)
        E_pred = (preds[1] * y_std) + y_mean
        @test abs(E_pred) < 0.2
    end
    
    # 3. Test AD Kernel
    @testset "AD Kernel (ForwardDiff)" begin
        kern = SquaredExpADKernel(1.0, [0.5]) 
        model = GPModel(kern, X_train, y_full, log(1e-4), log(1e-4))
        train_model!(model; iterations=200)
        
        X_test = reshape([1.5], 1, 1)
        preds = predict(model, X_test)
        E_pred = (preds[1] * y_std) + y_mean
        @test abs(E_pred) < 0.2
    end
end
