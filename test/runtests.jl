using ChemGP
using Test
using Statistics
using LinearAlgebra
using KernelFunctions

function morse(r)
    De, a, re = 10.0, 1.0, 1.5
    val = 1 - exp(-a*(r - re))
    E = De * val^2
    F = -2 * De * val * (exp(-a*(r - re)) * a)
    return E, [F]
end

@testset "ChemGP with MolecularKernel" begin
    # 1. Data Generation
    X_train = reshape(collect(1.0:0.5:3.0), 1, :)
    y_vals = Float64[]
    y_grads = Float64[]
    for i = 1:size(X_train, 2)
        E, F = morse(X_train[1, i])
        push!(y_vals, E)
        append!(y_grads, F)
    end

    y_mean = mean(y_vals)
    y_std = std(y_vals)
    y_full = [(y_vals .- y_mean) ./ y_std; y_grads ./ y_std]

    # 2. Setup Custom Kernel
    # Use 1.0 for inv_lengthscale (equivalent to lengthscale=1.0)
    k_init = MolecularKernel(1.0, [1.0])

    model = GPModel(k_init, X_train, y_full; noise_var = 1e-4, grad_noise_var = 1e-4)

    # 3. Train
    train_model!(model; iterations = 200)

    # 4. Predict
    X_test = reshape([1.5], 1, 1)
    preds = predict(model, X_test)
    E_pred = (preds[1] * y_std) + y_mean

    println("Predicted E at 1.5: $E_pred")
    @test abs(E_pred) < 0.2

    # 5. Verify Interface Compliance
    @test k_init isa KernelFunctions.Kernel
end
