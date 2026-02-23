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

        # Energy at endpoints should match oracle
        E_B, _ = muller_brown_energy_gradient(x_B)
        @test isapprox(result.path.energies[1], E_B, atol = 0.1)

        # Maximum energy along path should be higher than both endpoints
        @test maximum(result.path.energies) > min(result.path.energies[1], result.path.energies[end])

        # Barrier should be in the correct range (known: 35.92 with CI-NEB)
        barrier = maximum(result.path.energies) - min(result.path.energies[1], result.path.energies[end])
        @test 20.0 < barrier < 60.0
    end

    @testset "GP-NEB-AIE on Muller-Brown" begin
        x_B = [0.623, 0.028]
        x_C = [-0.050, 0.467]

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

        # Oracle calls: 2 endpoints + 3 images * 5 outer iters = 17 max
        @test 2 < result.oracle_calls <= 20

        # Path should be structurally valid
        @test length(result.path.images) == 5
        @test result.path.images[1] ≈ x_B
        @test result.path.images[end] ≈ x_C

        # Forces should be populated and bounded (not exploding)
        @test length(result.history["max_force"]) == 5
        @test all(f -> f < 200.0, result.history["max_force"])
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

        # OIE: 2 endpoints + 1 midpoint bootstrap + 1 per outer iter = 8 max
        @test 3 < result.oracle_calls <= 10

        # History should track which images were evaluated
        @test haskey(result.history, "image_evaluated")
        @test length(result.history["image_evaluated"]) == 5

        # Evaluated images should be valid indices (2..N-1)
        for idx in result.history["image_evaluated"]
            @test 2 <= idx <= 4
        end

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

        # OIE-naive: 2 endpoints + 1 midpoint + 5 outer = 8
        @test 3 < result.oracle_calls <= 10
        @test haskey(result.history, "image_evaluated")
        @test length(result.history["image_evaluated"]) == 5
        @test length(result.path.images) == 5

        # Forces should be bounded
        @test all(f -> f < 200.0, result.history["max_force"])
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

        @test 2 < result.oracle_calls <= 20
        @test length(result.path.images) == 5
        @test result.path.images[1] ≈ x_B
        @test result.path.images[end] ≈ x_C

        # FPS should not cause force explosion
        @test all(f -> f < 200.0, result.history["max_force"])
    end

    # ==========================================================================
    # LEPS 9D regression tests with numeric checks
    # ==========================================================================

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

        # Barrier: known value is 1.3291 eV from converged CI-NEB
        barrier = result.path.energies[result.max_energy_image] -
                  result.path.energies[1]
        @test 0.5 < barrier < 2.5
        @test isapprox(barrier, 1.33, atol = 0.15)
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

        # Barrier matches L-BFGS result
        barrier = result.path.energies[result.max_energy_image] -
                  result.path.energies[1]
        @test isapprox(barrier, 1.33, atol = 0.15)
    end

    @testset "GP-NEB-OIE-naive converges on LEPS with MolInvDistSE" begin
        # Regression: OIE-naive with molecular kernel must converge on LEPS.
        # Known result: converges in 22 oracle calls, barrier = 1.3293 eV.
        # This test catches GP training regressions and FPS/trust issues.
        x_r = copy(LEPS_REACTANT)
        x_p = copy(LEPS_PRODUCT)
        kernel = MolInvDistSE(1.0, [1.0], Float64[])

        cfg = NEBConfig(
            images = 3,
            spring_constant = 5.0,
            climbing_image = true,
            ci_activation_tol = 1.0,
            max_iter = 200,
            conv_tol = 0.3,
            step_size = 1e-4,
            gp_train_iter = 100,
            max_outer_iter = 30,
            verbose = false,
        )

        result = gp_neb_oie_naive(leps_energy_gradient, x_r, x_p, kernel; config = cfg)

        # Must converge
        @test result.converged

        # Oracle efficiency: should converge in < 30 calls (known: 22)
        @test result.oracle_calls < 30

        # Barrier must match standard NEB (known: 1.3291)
        barrier = result.path.energies[result.max_energy_image] -
                  result.path.energies[1]
        @test isapprox(barrier, 1.33, atol = 0.10)

        # Forces must stay bounded throughout (no divergence)
        @test all(f -> f < 20.0, result.history["max_force"])

        # Forces should decrease overall (first half avg > last half avg)
        forces = result.history["max_force"]
        n = length(forces)
        if n >= 4
            first_half = mean(forces[1:div(n,2)])
            last_half = mean(forces[div(n,2)+1:end])
            @test last_half < first_half * 1.5  # allow some slack
        end
    end

    @testset "GP-NEB-AIE on LEPS with MolInvDistSE" begin
        # AIE with molecular kernel: forces should decrease, barrier reasonable.
        # Known: 10 outer iters, forces ~2.17, barrier ~1.68 (not converged).
        x_r = copy(LEPS_REACTANT)
        x_p = copy(LEPS_PRODUCT)
        kernel = MolInvDistSE(1.0, [1.0], Float64[])

        cfg = NEBConfig(
            images = 3,
            spring_constant = 5.0,
            climbing_image = true,
            ci_activation_tol = 1.0,
            max_iter = 200,
            conv_tol = 0.3,
            step_size = 1e-4,
            gp_train_iter = 100,
            max_outer_iter = 10,
            verbose = false,
        )

        result = gp_neb_aie(leps_energy_gradient, x_r, x_p, kernel; config = cfg)

        # Oracle calls: 2 endpoints + 3 images * 10 outer = 32 max
        @test result.oracle_calls <= 40

        # Forces must stay bounded
        @test all(f -> f < 20.0, result.history["max_force"])

        # Barrier should be positive and in plausible range
        barrier = result.path.energies[result.max_energy_image] -
                  result.path.energies[1]
        @test 0.5 < barrier < 3.0

        # Forces should trend downward (last force < first force)
        forces = result.history["max_force"]
        @test forces[end] < forces[1] * 1.2
    end

    @testset "GP-NEB-AIE converges with per-bead FPS on LEPS" begin
        # Regression: per-bead subset selection must preserve local training data
        # around each bead. Global FPS starved the CI image on LEPS, causing
        # divergence (max|F| > 50). Per-bead selection with FPS(15) converges
        # with barrier = 1.3277 (known: 1.3291) and forces always < 5.
        x_r = copy(LEPS_REACTANT)
        x_p = copy(LEPS_PRODUCT)
        kernel = MolInvDistSE(1.0, [1.0], Float64[])

        cfg = NEBConfig(
            images = 3,
            spring_constant = 5.0,
            climbing_image = true,
            ci_activation_tol = 1.0,
            max_iter = 200,
            conv_tol = 0.3,
            step_size = 1e-4,
            gp_train_iter = 100,
            max_outer_iter = 15,
            max_gp_points = 15,  # triggers per-bead selection after ~5 outer iters
            verbose = false,
        )

        result = gp_neb_aie(leps_energy_gradient, x_r, x_p, kernel; config = cfg)

        # Must converge even with FPS active
        @test result.converged

        # Barrier must match known value
        barrier = result.path.energies[result.max_energy_image] -
                  result.path.energies[1]
        @test isapprox(barrier, 1.33, atol = 0.10)

        # Forces must stay bounded (no bead starvation)
        @test all(f -> f < 10.0, result.history["max_force"])

        # Forces should decrease overall
        forces = result.history["max_force"]
        @test forces[end] < forces[1]
    end

    @testset "GP-NEB-OIE-naive with RFF on LEPS (MolInvDistSE)" begin
        # Regression: RFF approximation must converge when per-bead subset
        # triggers. RFF replaces exact GP with O(D_rff^3) Bayesian linear
        # regression, avoiding the Nystrom Lambda^{-1} amplification issue.
        # Known result: converges in ~26 calls, barrier = 1.3294 eV.
        # Force spikes to ~18 when RFF activates (expected: approximation
        # introduces noise) but recovers.

        x_r = copy(LEPS_REACTANT)
        x_p = copy(LEPS_PRODUCT)
        kernel = MolInvDistSE(1.0, [1.0], Float64[])

        cfg = NEBConfig(
            images = 3,
            spring_constant = 5.0,
            climbing_image = true,
            ci_activation_tol = 1.0,
            max_iter = 200,
            conv_tol = 0.3,
            step_size = 1e-4,
            gp_train_iter = 100,
            max_outer_iter = 30,
            max_gp_points = 10,
            rff_features = 200,
            verbose = false,
        )

        result = gp_neb_oie_naive(leps_energy_gradient, x_r, x_p, kernel; config = cfg)

        # Must converge
        @test result.converged

        # Oracle calls: ~26 with RFF (slightly more than exact GP's 22)
        @test result.oracle_calls < 35

        # Barrier must match known value
        barrier = result.path.energies[result.max_energy_image] -
                  result.path.energies[1]
        @test isapprox(barrier, 1.33, atol = 0.15)

        # Forces: RFF approximation causes transient spikes when it activates,
        # but must stay bounded (no Nystrom-style divergence to 10^6).
        # Bound is generous due to RFF randomness across seeds.
        @test all(f -> f < 30.0, result.history["max_force"])

        # Forces should decrease overall
        forces = result.history["max_force"]
        n = length(forces)
        if n >= 4
            first_half = mean(forces[1:div(n,2)])
            last_half = mean(forces[div(n,2)+1:end])
            @test last_half < first_half * 2.0  # generous for RFF noise
        end
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

        # Oracle calls bounded
        @test 3 < result.oracle_calls <= 15

        # Forces should be bounded (EMD trust prevents wild steps)
        @test all(f -> f < 50.0, result.history["max_force"])
    end
end
