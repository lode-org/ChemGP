@testset "Trust region distance utilities" begin
    @testset "trust_distance_fn returns correct function for each metric" begin
        # :euclidean
        fn_euc = trust_distance_fn(:euclidean)
        @test fn_euc([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]) ≈ 1.0

        # :max_1d_log
        fn_log = trust_distance_fn(:max_1d_log)
        x1 = Float64[0, 0, 0, 1, 0, 0, 0, 1, 0]  # 3-atom config
        x2 = Float64[0, 0, 0, 2, 0, 0, 0, 1, 0]   # atom 2 moved
        @test fn_log(x1, x2) > 0.0

        # :emd (3D coords, all same type)
        fn_emd = trust_distance_fn(:emd)
        @test fn_emd(x1, x1) ≈ 0.0 atol = 1e-12
        @test fn_emd(x1, x2) > 0.0

        # :emd with non-3D falls back to euclidean
        fn_emd2 = trust_distance_fn(:emd)
        @test fn_emd2([1.0, 2.0], [3.0, 4.0]) ≈ norm([1.0, 2.0] - [3.0, 4.0])
    end

    @testset "trust_min_distance finds closest training point" begin
        D = 3
        X_train = Float64[0.0 1.0 5.0;
                           0.0 0.0 0.0;
                           0.0 0.0 0.0]
        x_query = Float64[0.9, 0.0, 0.0]

        d = trust_min_distance(x_query, X_train, :euclidean)
        @test d ≈ 0.1 atol = 1e-12  # closest to column 2 at [1,0,0]

        # Empty training set returns Inf
        X_empty = Matrix{Float64}(undef, D, 0)
        @test trust_min_distance(x_query, X_empty, :euclidean) == Inf
    end

    @testset "trust_min_distance with EMD" begin
        # 2-atom system, atom_types = [1, 1] (same type -> permutation invariant)
        x1 = Float64[0, 0, 0, 1, 0, 0]
        x2 = Float64[1, 0, 0, 0, 0, 0]  # swapped atoms
        X_train = reshape(x1, :, 1)

        # EMD should recognize the permutation => distance ~ 0
        d = trust_min_distance(x2, X_train, :emd; atom_types = Int[1, 1])
        @test d < 1e-10
    end

    @testset "adaptive_trust_threshold fixed mode" begin
        # When disabled, returns fixed trust_radius
        t = adaptive_trust_threshold(0.5, 100, 3; use_adaptive = false)
        @test t == 0.5
    end

    @testset "adaptive_trust_threshold adaptive mode" begin
        # Enabled: threshold should decay with n_data
        t_small = adaptive_trust_threshold(0.5, 10, 3; use_adaptive = true)
        t_large = adaptive_trust_threshold(0.5, 200, 3; use_adaptive = true)

        @test t_small > t_large  # decays with more data

        # Floor is respected
        t_huge = adaptive_trust_threshold(0.5, 100000, 3;
            use_adaptive = true, floor = 0.2)
        @test t_huge >= 0.2
    end
end
