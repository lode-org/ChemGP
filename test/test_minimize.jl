@testset "GP Minimization" begin
    @testset "LJ oracle" begin
        # Two atoms at equilibrium (~2^(1/6) * sigma for LJ minimum)
        r_eq = 2.0^(1 / 6)
        x = [0.0, 0.0, 0.0, r_eq, 0.0, 0.0]
        E, G = lj_energy_gradient(x)

        # At equilibrium, energy should be -epsilon = -1.0
        @test isapprox(E, -1.0, atol=1e-10)
        # Gradient should be ~ 0 at equilibrium
        @test norm(G) < 1e-10
    end

    @testset "random_cluster" begin
        x = random_cluster(4; min_dist=1.0)
        @test length(x) == 12  # 4 atoms * 3 coords
        # Check minimum distance constraint
        dists = interatomic_distances(x)
        @test minimum(dists) >= 1.0
    end

    @testset "GP minimize smoke test" begin
        # Small 3-atom cluster, very loose tolerance, few iterations
        x_init = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.5, 0.0]
        kernel = MolInvDistSE(1.0, [0.5], Float64[])

        config = MinimizationConfig(
            trust_radius=0.3,
            conv_tol=0.5,     # Very loose for fast test
            max_iter=5,       # Few iterations
            gp_train_iter=50, # Minimal training
            verbose=false,
        )

        result = gp_minimize(lj_energy_gradient, x_init, kernel; config=config)

        @test result.oracle_calls >= 5  # At least initial points
        @test length(result.trajectory) >= 1
        @test isfinite(result.E_final)
    end
end
