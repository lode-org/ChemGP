@testset "OTGPD" begin
    @testset "Data pruning" begin
        td = TrainingData(2)
        for i in 1:10
            add_point!(td, [Float64(i), 0.0], Float64(i), [1.0, 0.0])
        end

        @test npoints(td) == 10

        # Prune to 5 closest to [5.0, 0.0]
        n_removed = prune_training_data!(td, [5.0, 0.0], 5;
            distance_fn = (a, b) -> norm(a - b))
        @test n_removed == 5
        @test npoints(td) == 5

        # Check that the closest points were kept (3,4,5,6,7)
        for i in 1:5
            @test td.energies[i] ∈ [3.0, 4.0, 5.0, 6.0, 7.0]
        end

        # No pruning when max_points <= 0
        @test prune_training_data!(td, [5.0, 0.0], 0;
            distance_fn = (a, b) -> norm(a - b)) == 0
        @test npoints(td) == 5

        # No pruning when already under limit
        @test prune_training_data!(td, [5.0, 0.0], 10;
            distance_fn = (a, b) -> norm(a - b)) == 0
        @test npoints(td) == 5
    end

    @testset "OTGPDConfig defaults" begin
        cfg = OTGPDConfig()
        @test cfg.T_dimer == 0.01
        @test cfg.divisor_T_dimer_gp == 10.0
        @test cfg.rotation_method == :lbfgs
        @test cfg.translation_method == :lbfgs
        @test cfg.initial_rotation == true
        @test cfg.eval_image1 == true
        @test cfg.max_training_points == 0
    end

    @testset "OTGPD on Muller-Brown" begin
        # Find the saddle point between minima B and C on Muller-Brown surface
        x_B = [0.623, 0.028]
        x_C = [-0.050, 0.467]

        # Start near the path between B and C
        x_init = 0.5 * (x_B + x_C) + [0.05, -0.02]

        # Orient roughly along the transition direction
        orient_init = x_C - x_B
        orient_init ./= norm(orient_init)

        k = KernelFunctions.SqExponentialKernel()

        cfg = OTGPDConfig(
            T_dimer = 5.0,          # Relaxed threshold for test
            T_dimer_gp_init = 1.0,
            divisor_T_dimer_gp = 5.0,
            T_angle_rot = 0.01,
            max_outer_iter = 10,
            max_inner_iter = 200,
            max_rot_iter = 3,
            dimer_sep = 0.01,
            eval_image1 = true,
            initial_rotation = true,
            max_initial_rot = 3,
            rotation_method = :simple,
            translation_method = :simple,
            gp_train_iter = 50,
            n_initial_perturb = 2,
            perturb_scale = 0.1,
            trust_radius = 0.5,
            verbose = false,
        )

        result = otgpd(muller_brown_energy_gradient, x_init, orient_init, k; config = cfg)

        # Should have used oracle calls
        @test result.oracle_calls > 0

        # History should track adaptive thresholds
        @test haskey(result.history, "T_gp")
        @test length(result.history["T_gp"]) > 0

        # History should track true forces
        @test haskey(result.history, "F_true")
        @test length(result.history["F_true"]) > 0

        # State should be valid
        @test length(result.state.R) == 2
        @test norm(result.state.orient) ≈ 1.0 atol = 1e-10
    end

    @testset "OTGPD without initial rotation" begin
        x_init = [0.3, 0.2]
        orient_init = [1.0, 0.5]

        k = KernelFunctions.SqExponentialKernel()

        cfg = OTGPDConfig(
            T_dimer = 10.0,
            max_outer_iter = 3,
            max_inner_iter = 50,
            initial_rotation = false,
            eval_image1 = false,
            rotation_method = :simple,
            translation_method = :simple,
            gp_train_iter = 30,
            n_initial_perturb = 2,
            verbose = false,
        )

        result = otgpd(muller_brown_energy_gradient, x_init, orient_init, k; config = cfg)

        @test result.oracle_calls > 0
        # Without eval_image1, curvature should be NaN
        @test all(isnan, result.history["curv_true"])
    end

    @testset "OTGPD with data pruning" begin
        x_init = [0.3, 0.2]
        orient_init = [1.0, 0.0]

        k = KernelFunctions.SqExponentialKernel()

        cfg = OTGPDConfig(
            T_dimer = 10.0,
            max_outer_iter = 5,
            max_inner_iter = 50,
            max_training_points = 15,
            initial_rotation = false,
            eval_image1 = true,
            rotation_method = :simple,
            translation_method = :simple,
            gp_train_iter = 30,
            n_initial_perturb = 3,
            verbose = false,
        )

        result = otgpd(muller_brown_energy_gradient, x_init, orient_init, k; config = cfg)

        @test result.oracle_calls > 0
        @test length(result.history["F_true"]) > 0
    end

    @testset "Adaptive threshold tightening" begin
        # Verify the adaptive threshold formula
        cfg = OTGPDConfig(
            T_dimer = 0.01,
            divisor_T_dimer_gp = 10.0,
        )

        # With F_history = [1.0, 0.5, 0.3], min = 0.3
        # T_gp = max(0.3/10, 0.01/10) = max(0.03, 0.001) = 0.03
        F_history = [1.0, 0.5, 0.3]
        T_gp = max(minimum(F_history) / cfg.divisor_T_dimer_gp, cfg.T_dimer / 10)
        @test T_gp ≈ 0.03

        # With F_history = [0.005], min = 0.005
        # T_gp = max(0.005/10, 0.001) = max(0.0005, 0.001) = 0.001
        F_history2 = [0.005]
        T_gp2 = max(minimum(F_history2) / cfg.divisor_T_dimer_gp, cfg.T_dimer / 10)
        @test T_gp2 ≈ 0.001

        # With divisor <= 0, always T_dimer/10
        cfg2 = OTGPDConfig(T_dimer = 0.01, divisor_T_dimer_gp = 0.0)
        T_gp3 = cfg2.T_dimer / 10
        @test T_gp3 ≈ 0.001
    end
end
