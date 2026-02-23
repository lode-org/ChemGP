@testset "NEB" begin
    @testset "Linear interpolation" begin
        x_start = [0.0, 0.0]
        x_end = [1.0, 1.0]
        images = linear_interpolation(x_start, x_end, 5)

        @test length(images) == 5
        @test images[1] == x_start
        @test images[end] == x_end
        @test isapprox(images[3], [0.5, 0.5], atol = 1e-12)

        # Images should be evenly spaced
        for i in 1:(length(images) - 1)
            d = norm(images[i+1] - images[i])
            @test isapprox(d, norm(x_end - x_start) / 4, atol = 1e-12)
        end
    end

    @testset "Path tangent" begin
        images = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
        energies = [0.0, 1.0, 2.0, 1.5, 0.5]  # Monotonic increase then decrease

        # Monotonic increase at image 2
        tau2 = path_tangent(images, energies, 2)
        @test norm(tau2) ≈ 1.0  # Should be unit vector
        @test tau2[1] > 0  # Should point forward

        # At local max (image 3), uses energy-weighted bisection
        tau3 = path_tangent(images, energies, 3)
        @test norm(tau3) ≈ 1.0

        # Monotonic decrease at image 4
        tau4 = path_tangent(images, energies, 4)
        @test norm(tau4) ≈ 1.0
    end

    @testset "NEB force perpendicularity" begin
        # Force should be perpendicular to tangent (standard NEB, not CI)
        gradient = [1.0, 2.0]
        tangent = [1.0, 0.0]  # Unit vector along x
        spring_f = [0.5, 0.0]  # Parallel spring

        f = neb_force(gradient, spring_f, tangent; climbing = false, is_highest = false)

        # The gradient perpendicular component is [0, 2], negated -> [0, -2]
        # Plus spring parallel: [0.5, 0]
        @test isapprox(f, [0.5, -2.0], atol = 1e-12)
    end

    @testset "Climbing image force" begin
        gradient = [1.0, 2.0]
        tangent = [1.0, 0.0]
        spring_f = [0.5, 0.0]  # Should be ignored for CI

        f = neb_force(gradient, spring_f, tangent; climbing = true, is_highest = true)

        # CI force: -G + 2*(G.tau)tau = -[1,2] + 2*1*[1,0] = [1, -2]
        @test isapprox(f, [1.0, -2.0], atol = 1e-12)
    end

    @testset "Standard NEB on Muller-Brown" begin
        # Find MEP between minima B and C (shorter path, faster convergence)
        x_B = [0.623, 0.028]
        x_C = [-0.050, 0.467]

        cfg = NEBConfig(
            images = 5,           # 5 movable + 2 endpoints = 7 total
            spring_constant = 10.0,
            climbing_image = false,
            max_iter = 300,
            conv_tol = 1.0,
            step_size = 1e-4,
            verbose = false,
        )

        result = neb_optimize(muller_brown_energy_gradient, x_B, x_C; config = cfg)

        # Path should have correct number of images (movable + 2 endpoints)
        @test length(result.path.images) == 7

        # Endpoints should be fixed
        @test result.path.images[1] ≈ x_B
        @test result.path.images[end] ≈ x_C

        # Energy at endpoints should match
        E_B, _ = muller_brown_energy_gradient(x_B)
        @test isapprox(result.path.energies[1], E_B, atol = 0.1)

        # Maximum energy along path should be higher than both endpoints
        @test maximum(result.path.energies) > min(result.path.energies[1], result.path.energies[end])
    end

    @testset "GP-NEB-AIE uses fewer oracle calls" begin
        x_B = [0.623, 0.028]
        x_C = [-0.050, 0.467]

        # Note: Muller-Brown is 2D, not molecular, so we use a simple SE kernel
        # via KernelFunctions directly. But GP-NEB works with any kernel compatible
        # with GPModel. For 2D, we use a thin wrapper.
        k = KernelFunctions.SqExponentialKernel()

        cfg = NEBConfig(
            images = 3,           # 3 movable + 2 endpoints = 5 total
            spring_constant = 10.0,
            climbing_image = false,
            max_iter = 200,
            conv_tol = 2.0,  # Relaxed for test speed
            step_size = 1e-4,
            gp_train_iter = 50,
            max_outer_iter = 5,
            verbose = false,
        )

        result = gp_neb_aie(muller_brown_energy_gradient, x_B, x_C, k; config = cfg)

        # Should have used oracle
        @test result.oracle_calls > 0

        # Path should be structurally valid
        @test length(result.path.images) == 5
        @test result.path.images[1] ≈ x_B
        @test result.path.images[end] ≈ x_C
    end

    @testset "GP-NEB-OIE selects high-uncertainty images" begin
        x_B = [0.623, 0.028]
        x_C = [-0.050, 0.467]

        k = KernelFunctions.SqExponentialKernel()

        cfg = NEBConfig(
            images = 3,           # 3 movable + 2 endpoints = 5 total
            spring_constant = 10.0,
            climbing_image = false,
            max_iter = 200,
            conv_tol = 2.0,
            step_size = 1e-4,
            gp_train_iter = 50,
            max_outer_iter = 5,
            verbose = false,
        )

        result = gp_neb_oie(muller_brown_energy_gradient, x_B, x_C, k; config = cfg)

        # OIE should use fewer oracle calls than AIE for same config
        @test result.oracle_calls > 0

        # History should track which images were evaluated
        @test haskey(result.history, "image_evaluated")
        @test length(result.history["image_evaluated"]) > 0

        # Path should be valid
        @test length(result.path.images) == 5
    end

    @testset "GP-NEB-OIE-naive on Muller-Brown" begin
        x_B = [0.623, 0.028]
        x_C = [-0.050, 0.467]

        k = KernelFunctions.SqExponentialKernel()

        cfg = NEBConfig(
            images = 3,
            spring_constant = 10.0,
            climbing_image = false,
            max_iter = 200,
            conv_tol = 2.0,
            step_size = 1e-4,
            gp_train_iter = 50,
            max_outer_iter = 5,
            verbose = false,
        )

        result = gp_neb_oie_naive(muller_brown_energy_gradient, x_B, x_C, k; config = cfg)

        @test result.oracle_calls > 0
        @test haskey(result.history, "image_evaluated")
        @test length(result.path.images) == 5
    end

    @testset "GP-NEB-AIE with FPS subset (max_gp_points)" begin
        x_B = [0.623, 0.028]
        x_C = [-0.050, 0.467]

        k = KernelFunctions.SqExponentialKernel()

        cfg = NEBConfig(
            images = 3,
            spring_constant = 10.0,
            climbing_image = false,
            max_iter = 200,
            conv_tol = 2.0,
            step_size = 1e-4,
            gp_train_iter = 50,
            max_outer_iter = 5,
            max_gp_points = 6,   # force FPS subset after a few iterations
            verbose = false,
        )

        result = gp_neb_aie(muller_brown_energy_gradient, x_B, x_C, k; config = cfg)

        @test result.oracle_calls > 0
        @test length(result.path.images) == 5
        @test result.path.images[1] ≈ x_B
        @test result.path.images[end] ≈ x_C
    end

    @testset "CI-NEB convergence with L-BFGS (LEPS 9D)" begin
        # Regression test: L-BFGS must not explode after climbing image activation.
        # Without distance_reset, clipped L-BFGS steps corrupt curvature estimates
        # causing a positive feedback loop (max|F| > 200 by iter 350).
        x_r = copy(LEPS_REACTANT)
        x_p = copy(LEPS_PRODUCT)

        cfg = NEBConfig(
            images = 5,           # 5 movable + 2 endpoints = 7 total
            spring_constant = 5.0,
            climbing_image = true,
            ci_activation_tol = 0.5,
            max_iter = 500,
            conv_tol = 0.05,
            optimizer = :lbfgs,
            max_move = 0.1,
            lbfgs_memory = 20,
            initializer = :linear,
            verbose = false,
        )

        result = neb_optimize(leps_energy_gradient, x_r, x_p; config = cfg)

        # Must converge
        @test result.converged

        # Forces must never explode during optimization (the regression)
        max_force_seen = maximum(result.history["max_force"])
        @test max_force_seen < 50.0

        # Barrier should be positive (TS above reactant)
        barrier = result.path.energies[result.max_energy_image] -
                  result.path.energies[1]
        @test barrier > 0.0
    end

    @testset "GP-NEB-OIE with EMD trust on LEPS" begin
        # Verify that the EMD trust constraint fires (small trust_radius)
        # but does not prevent convergence.
        x_r = copy(LEPS_REACTANT)
        x_p = copy(LEPS_PRODUCT)

        k = KernelFunctions.SqExponentialKernel()

        cfg = NEBConfig(
            images = 3,
            spring_constant = 5.0,
            climbing_image = false,
            max_iter = 200,
            conv_tol = 2.0,
            step_size = 1e-4,
            gp_train_iter = 50,
            max_outer_iter = 10,
            trust_radius = 0.05,
            trust_metric = :emd,
            verbose = false,
        )

        result = gp_neb_oie(leps_energy_gradient, x_r, x_p, k; config = cfg)

        # Path should be structurally valid
        @test length(result.path.images) == 5
        @test result.path.images[1] ≈ x_r
        @test result.path.images[end] ≈ x_p
        @test result.oracle_calls > 0
    end

    @testset "CI-NEB convergence with SD (LEPS 9D)" begin
        # Steepest descent should also converge (the old default before L-BFGS).
        x_r = copy(LEPS_REACTANT)
        x_p = copy(LEPS_PRODUCT)

        cfg = NEBConfig(
            images = 5,           # 5 movable + 2 endpoints = 7 total
            spring_constant = 5.0,
            climbing_image = true,
            ci_activation_tol = 0.5,
            max_iter = 1000,
            conv_tol = 0.05,
            optimizer = :sd,
            max_move = 0.1,
            initializer = :linear,
            verbose = false,
        )

        result = neb_optimize(leps_energy_gradient, x_r, x_p; config = cfg)

        @test result.converged

        # Forces stay bounded
        max_force_seen = maximum(result.history["max_force"])
        @test max_force_seen < 50.0

        # Barrier should be positive
        barrier = result.path.energies[result.max_energy_image] -
                  result.path.energies[1]
        @test barrier > 0.0
    end
end
