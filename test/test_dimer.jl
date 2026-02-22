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

        # Test translational_force: effective force for saddle search
        # F_eff = -G + 2*(G.n)n: descends perp to dimer, ascends along it
        G_test = [1.0, 1.0, 0.0]
        orient = [1.0, 0.0, 0.0]
        F = translational_force(G_test, orient)
        # G_parallel = (G.n)n = [1,0,0]
        # F_eff = -G + 2*G_parallel = [-1,-1,0] + [2,0,0] = [1,-1,0]
        @test isapprox(F, [1.0, -1.0, 0.0], atol = 1e-12)
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

    @testset "Rigid body mode removal" begin
        n_atoms = 3

        # Equilateral triangle in xy-plane
        x = [0.0, 0.0, 0.0,
             1.0, 0.0, 0.0,
             0.5, sqrt(3)/2, 0.0]

        # Pure translation should be fully removed
        step_trans = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        step = copy(step_trans)
        removed_mag = remove_rigid_body_modes!(step, x, n_atoms)
        @test norm(step) < 1e-8
        @test removed_mag > 0.0

        # Pure internal motion (stretch atom 2 away from atoms 1,3) should be preserved
        # Move atom 2 in +x, atoms 1 and 3 in -x/2
        step_internal = [-0.5, 0.0, 0.0,
                          1.0, 0.0, 0.0,
                         -0.5, 0.0, 0.0]
        step = copy(step_internal)
        removed_mag = remove_rigid_body_modes!(step, x, n_atoms)
        # Internal motion should be mostly preserved (small rotation component possible)
        @test norm(step) > 0.5 * norm(step_internal)

        # Pure rotation about z-axis: should be fully removed
        com = [0.5, sqrt(3)/6, 0.0]
        step_rot = zeros(9)
        for i in 1:n_atoms
            pos = x[(3*(i-1)+1):(3*i)] .- com
            step_rot[3*(i-1)+1] = -pos[2]
            step_rot[3*(i-1)+2] =  pos[1]
        end
        step = copy(step_rot)
        removed_mag = remove_rigid_body_modes!(step, x, n_atoms)
        @test norm(step) < 1e-8

        # Linear molecule: only 5 modes removed (2 rotation, not 3)
        x_linear = [0.0, 0.0, 0.0,
                     1.0, 0.0, 0.0,
                     2.0, 0.0, 0.0]
        step_stretch = [0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        1.0, 0.0, 0.0]
        step = copy(step_stretch)
        remove_rigid_body_modes!(step, x_linear, 3)
        # Stretching along the line should be preserved
        @test norm(step) > 0.5 * norm(step_stretch)
    end

    @testset "DimerConfig rotation/translation methods" begin
        # Default config should use L-BFGS for both
        cfg = DimerConfig()
        @test cfg.rotation_method == :lbfgs
        @test cfg.translation_method == :lbfgs
        @test cfg.lbfgs_memory == 5

        # Simple mode should work too
        cfg_simple = DimerConfig(rotation_method = :simple, translation_method = :simple)
        @test cfg_simple.rotation_method == :simple
        @test cfg_simple.translation_method == :simple

        # CG rotation
        cfg_cg = DimerConfig(rotation_method = :cg)
        @test cfg_cg.rotation_method == :cg
    end
end
