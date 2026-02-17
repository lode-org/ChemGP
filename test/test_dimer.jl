@testset "Dimer Method" begin
    @testset "Dimer utilities" begin
        # Test dimer_images
        state = DimerState([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.1)
        R0, R1, R2 = dimer_images(state)
        @test R0 == [1.0, 0.0, 0.0]
        @test isapprox(R1, [1.1, 0.0, 0.0], atol = 1e-12)
        @test isapprox(R2, [0.9, 0.0, 0.0], atol = 1e-12)

        # Test curvature: parallel gradients -> zero curvature
        G0 = [1.0, 0.0, 0.0]
        G1 = [1.0, 0.0, 0.0]
        @test curvature(G0, G1, [1.0, 0.0, 0.0], 0.1) == 0.0

        # Test curvature: different gradients -> nonzero curvature
        G1_diff = [2.0, 0.0, 0.0]
        C = curvature(G0, G1_diff, [1.0, 0.0, 0.0], 0.1)
        @test C != 0.0
        @test isapprox(C, 10.0, atol = 1e-12)  # (2-1)*1 / 0.1 = 10

        # Test translational_force: inverts component along dimer
        G_test = [1.0, 1.0, 0.0]
        orient = [1.0, 0.0, 0.0]
        F = translational_force(G_test, orient)
        # F_parallel = 1.0 * [1,0,0] = [1,0,0]
        # F_perp = [0,1,0]
        # F_trans = F_perp - F_parallel = [-1, 1, 0]
        @test isapprox(F, [-1.0, 1.0, 0.0], atol = 1e-12)
    end

    @testset "Trust region utilities" begin
        X_train = [0.0 1.0; 0.0 0.0; 0.0 0.0; 1.0 1.0; 0.0 0.0; 0.0 0.0]
        x_near = [0.5, 0.0, 0.0, 1.0, 0.0, 0.0]

        d = min_distance_to_data(x_near, X_train)
        @test d > 0.0
        @test d <= norm(x_near - X_train[:, 1])  # Should be min of the two

        # check_interatomic_ratio: point close to training should pass
        @test check_interatomic_ratio(X_train[:, 1], X_train, 0.5)
    end
end
