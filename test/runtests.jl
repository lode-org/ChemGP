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

@testset "ChemGP Suite" begin
    kernels_to_test = [MolInvDistSE, MolInvDistMatern52]

    @testset "Kernel: $KType" for KType in kernels_to_test
        println("\nTesting Kernel: $KType")
        
        # 1. Generate 3D data (Bond stretching)
        r_vals = collect(1.0:0.5:3.0)
        N = length(r_vals)
        X_train = zeros(6, N) 

        y_vals = Float64[]
        y_grads = Float64[] 

        for i = 1:N
            r = r_vals[i]
            X_train[1, i] = 0.0 
            X_train[4, i] = r   

            E, F_scalar = morse(r)
            push!(y_vals, E)
            
            # Map scalar Force to 6D gradient vector (F = -Grad)
            # Atom 1 moves left (-1), Atom 2 moves right (+1)
            # Gradient = -Force
            g_vec = [ F_scalar[1], 0.0, 0.0, -F_scalar[1], 0.0, 0.0 ]
            append!(y_grads, g_vec)
        end

        # Normalize
        y_mean = mean(y_vals)
        y_std = std(y_vals)
        y_full = [(y_vals .- y_mean) ./ y_std; y_grads ./ y_std]

        # 2. Instantiate the specific Kernel Type
        # Both share the same isotropic constructor signature
        k_init = KType(1.0, [1.0], Float64[]) 
        
        # 3. Create and Train Model
        model = GPModel(k_init, X_train, y_full; noise_var = 1e-4, grad_noise_var = 1e-4)
        train_model!(model; iterations = 150) # Slightly more iters for safety

        # ---------------------------------------------------------
        # Test 1: Reconstruction (Training Point r=1.5)
        # ---------------------------------------------------------
        # Note: GP will not be exactly 0.0 due to noise_var (smoothing).
        X_test_1 = zeros(6, 1); X_test_1[4, 1] = 1.5
        preds_1 = predict(model, X_test_1)
        E_pred_1 = (preds_1[1] * y_std) + y_mean
        E_true_1, _ = morse(1.5)
        
        println("  Pred E at 1.5: $E_pred_1 (True: $E_true_1)")
        @test isapprox(E_pred_1, E_true_1, atol=0.005)

        # ---------------------------------------------------------
        # Test 2: Generalization (Unseen Point r=1.6)
        # ---------------------------------------------------------
        test_r = 1.6
        X_test_2 = zeros(6, 1); X_test_2[4, 1] = test_r
        preds_2 = predict(model, X_test_2)
        E_pred_2 = (preds_2[1] * y_std) + y_mean
        E_true_2, _ = morse(test_r)

        println("  Pred E at $test_r: $E_pred_2 (True: $E_true_2)")
        @test isapprox(E_pred_2, E_true_2, atol=0.05)
    end

    if isfile("cpp_consistency.jl")
        include("cpp_consistency.jl")
    end
end
