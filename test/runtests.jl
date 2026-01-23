using ChemGP
using Test
using Statistics
using LinearAlgebra

# Define the Morse Potential (Energy and Gradient)
function morse(r)
    De, a, re = 10.0, 1.0, 1.5
    val = 1 - exp(-a*(r - re))
    E = De * val^2
    F = -2 * De * val * (exp(-a*(r - re)) * a) # F = -dE/dr
    return E, [F]
end

@testset "ChemGP.jl Tests" begin
    # 1. Generate Data
    X_train = reshape(collect(1.0:0.5:3.0), 1, :)
    y_vals = Float64[]
    y_grads = Float64[]

    for i = 1:size(X_train, 2)
        E, F = morse(X_train[1, i])
        push!(y_vals, E)
        append!(y_grads, F)
    end

    # 2. Normalize Data
    # TODO(rg): see if we need this
    # GPs are sensitive to scale. Without this, NLL can be Inf.
    y_mean = mean(y_vals)
    y_std = std(y_vals)

    y_vals_norm = (y_vals .- y_mean) ./ y_std
    y_grads_norm = y_grads ./ y_std
    y_full = [y_vals_norm; y_grads_norm]

    # 3. Initialize Model
    # Initial guess: signal variance ~ 1.0 (since normalized), lengthscale ~ 0.5
    kern = SquaredExpKernel(1.0, [0.5])
    model = GPModel(kern, X_train, y_full, log(1e-4), log(1e-4))

    # 4. Train
    train_model!(model; iterations = 500)

    # 5. Predict
    X_test = reshape(collect(1.0:0.1:3.0), 1, :)
    preds = predict(model, X_test)

    # 6. Un-normalize Predictions
    D = size(X_train, 1)
    stride = D + 1

    E_pred_norm = preds[1:stride:end]
    E_pred = (E_pred_norm .* y_std) .+ y_mean

    F_pred_norm = preds[2:stride:end]
    F_pred = F_pred_norm .* y_std

    # 7. Validation
    # Check if prediction at 1.5 (equilibrium) is close to 0
    idx_eq = findfirst(x -> isapprox(x, 1.5), vec(X_test))
    @test abs(E_pred[idx_eq]) < 0.2

    println("Test Complete.")
    println("Pred Energy at equilibrium (1.5): ", E_pred[idx_eq])
end
