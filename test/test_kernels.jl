function morse(r)
    De, a, re = 10.0, 1.0, 1.5
    val = 1 - exp(-a * (r - re))
    E = De * val^2
    F = -2 * De * val * (exp(-a * (r - re)) * a)
    return E, [F]
end

@testset "Molecular Kernels" begin
    kernels_to_test = [MolInvDistSE, MolInvDistMatern52, MolInvDistMatern32]

    @testset "Kernel: $KType" for KType in kernels_to_test
        println("\nTesting Kernel: $KType")

        # 1. Generate 3D data (Bond stretching)
        r_vals = collect(1.0:0.5:3.0)
        N = length(r_vals)
        X_train = zeros(6, N)

        y_vals = Float64[]
        y_grads = Float64[]

        for i in 1:N
            r = r_vals[i]
            X_train[1, i] = 0.0
            X_train[4, i] = r

            E, F_scalar = morse(r)
            push!(y_vals, E)

            # Map scalar Force to 6D gradient vector (F = -Grad)
            g_vec = [F_scalar[1], 0.0, 0.0, -F_scalar[1], 0.0, 0.0]
            append!(y_grads, g_vec)
        end

        # Normalize
        y_mean = mean(y_vals)
        y_std = std(y_vals)
        y_full = [(y_vals .- y_mean) ./ y_std; y_grads ./ y_std]

        # 2. Instantiate the specific Kernel Type
        k_init = KType(1.0, [1.0], Float64[])

        # 3. Create and Train Model
        model = GPModel(k_init, X_train, y_full; noise_var=1e-4, grad_noise_var=1e-4)
        train_model!(model; iterations=150)

        # Test 1: Reconstruction (Training Point r=1.5)
        X_test_1 = zeros(6, 1)
        X_test_1[4, 1] = 1.5
        preds_1 = predict(model, X_test_1)
        E_pred_1 = (preds_1[1] * y_std) + y_mean
        E_true_1, _ = morse(1.5)

        println("  Pred E at 1.5: $E_pred_1 (True: $E_true_1)")
        @test isapprox(E_pred_1, E_true_1, atol=0.005)

        # Test 2: Generalization (Unseen Point r=1.6)
        test_r = 1.6
        X_test_2 = zeros(6, 1)
        X_test_2[4, 1] = test_r
        preds_2 = predict(model, X_test_2)
        E_pred_2 = (preds_2[1] * y_std) + y_mean
        E_true_2, _ = morse(test_r)

        println("  Pred E at $test_r: $E_pred_2 (True: $E_true_2)")
        @test isapprox(E_pred_2, E_true_2, atol=0.05)
    end
end

@testset "OffsetKernel" begin
    k = OffsetKernel(2.5)
    x1 = [1.0, 2.0, 3.0]
    x2 = [4.0, 5.0, 6.0]

    # Kernel value is always the constant
    @test k(x1, x2) == 2.5
    @test k(x1, x1) == 2.5

    # kernel_blocks should give zero derivatives
    k_ee, k_ef, k_fe, k_ff = kernel_blocks(k, x1, x2)
    @test k_ee == 2.5
    @test all(k_ef .== 0.0)
    @test all(k_fe .== 0.0)
    @test all(k_ff .== 0.0)
end

@testset "MolSumKernel" begin
    k_se = MolInvDistSE(1.0, [1.0], Float64[])
    k_const = OffsetKernel(0.5)
    k_sum = MolSumKernel(k_se, k_const)

    # Test that sum kernel works
    x1 = [0.0, 0.0, 0.0, 1.5, 0.0, 0.0]
    x2 = [0.0, 0.0, 0.0, 1.6, 0.0, 0.0]

    @test k_sum(x1, x2) == k_se(x1, x2) + k_const(x1, x2)
    @test k_sum(x1, x1) == k_se(x1, x1) + 0.5

    # Test that kernel_blocks works through ForwardDiff
    ee_sum, ef_sum, fe_sum, ff_sum = kernel_blocks(k_sum, x1, x2)
    ee_se, ef_se, fe_se, ff_se = kernel_blocks(k_se, x1, x2)

    @test isapprox(ee_sum, ee_se + 0.5, atol=1e-12)
    @test isapprox(ef_sum, ef_se, atol=1e-12)
    @test isapprox(fe_sum, fe_se, atol=1e-12)
    @test isapprox(ff_sum, ff_se, atol=1e-12)
end
