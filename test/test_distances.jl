@testset "Distance Metrics" begin
    @testset "interatomic_distances" begin
        # Two atoms: distance should be the Euclidean distance
        x = [0.0, 0.0, 0.0, 3.0, 4.0, 0.0]
        dists = interatomic_distances(x)
        @test length(dists) == 1  # 2 atoms = 1 pair
        @test isapprox(dists[1], 5.0, atol = 1e-12)

        # Three atoms: 3 pairs
        x3 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        dists3 = interatomic_distances(x3)
        @test length(dists3) == 3  # 3 atoms = 3 pairs
        @test isapprox(dists3[1], 1.0, atol = 1e-12)  # atoms 1-2
        @test isapprox(dists3[2], 1.0, atol = 1e-12)  # atoms 1-3
        @test isapprox(dists3[3], sqrt(2), atol = 1e-12)  # atoms 2-3
    end

    @testset "max_1d_log_distance" begin
        x1 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        x2 = copy(x1)

        # Identical configurations: distance should be ~0
        @test max_1d_log_distance(x1, x2) < 1e-10

        # Scaled configuration: one distance doubled
        x3 = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        d = max_1d_log_distance(x1, x3)
        @test d > 0.0
        # The atom 1-2 distance changed from 1.0 to 2.0, so log(2) ~ 0.693
        @test isapprox(d, log(2), atol = 0.1)  # Approximate due to other pairs changing too
    end

    @testset "rmsd_distance" begin
        x1 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        x2 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

        @test rmsd_distance(x1, x2) == 0.0

        x3 = [0.1, 0.0, 0.0, 1.1, 0.0, 0.0]
        @test rmsd_distance(x1, x3) > 0.0
    end
end
