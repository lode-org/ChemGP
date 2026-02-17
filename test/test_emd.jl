@testset "Intensive EMD Distance" begin
    @testset "Identical configurations" begin
        x = [1.0, 0.0, 0.0,  2.0, 0.0, 0.0,  3.0, 0.0, 0.0]
        @test emd_distance(x, x) ≈ 0.0 atol = 1e-12
    end

    @testset "Simple displacement" begin
        x1 = [0.0, 0.0, 0.0,  1.0, 0.0, 0.0]
        x2 = [0.1, 0.0, 0.0,  1.1, 0.0, 0.0]

        d = emd_distance(x1, x2)
        @test d ≈ 0.1 atol = 1e-10
    end

    @testset "Permutation invariance (same type)" begin
        # Two atoms, swapped
        x1 = [0.0, 0.0, 0.0,  3.0, 0.0, 0.0]
        x2 = [3.0, 0.0, 0.0,  0.0, 0.0, 0.0]  # Swapped

        # Without type info, should find optimal permutation (distance = 0)
        d = emd_distance(x1, x2)
        @test d ≈ 0.0 atol = 1e-10
    end

    @testset "Type-aware matching" begin
        # Two types: atom 1 is type 1, atom 2 is type 2
        x1 = [0.0, 0.0, 0.0,  3.0, 0.0, 0.0]
        x2 = [3.0, 0.0, 0.0,  0.0, 0.0, 0.0]

        # With different types, each atom must match itself (can't swap)
        d = emd_distance(x1, x2; atom_types = [1, 2])
        @test d ≈ 3.0 atol = 1e-10

        # With same type, optimal matching gives 0
        d2 = emd_distance(x1, x2; atom_types = [1, 1])
        @test d2 ≈ 0.0 atol = 1e-10
    end

    @testset "Intensive (size-independent)" begin
        # 3 atoms of same type, each displaced by 0.1 in x
        x1 = [0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  2.0, 0.0, 0.0]
        x2 = [0.1, 0.0, 0.0,  1.1, 0.0, 0.0,  2.1, 0.0, 0.0]

        d = emd_distance(x1, x2)
        # Mean displacement = 0.1 (intensive, not 0.3)
        @test d ≈ 0.1 atol = 1e-10
    end

    @testset "Max over types" begin
        # Type 1 displaced by 0.1, type 2 displaced by 0.5
        x1 = [0.0, 0.0, 0.0,  1.0, 0.0, 0.0]
        x2 = [0.1, 0.0, 0.0,  1.5, 0.0, 0.0]

        d = emd_distance(x1, x2; atom_types = [1, 2])
        @test d ≈ 0.5 atol = 1e-10
    end

    @testset "Symmetry" begin
        x1 = [0.0, 0.0, 0.0,  1.0, 2.0, 3.0]
        x2 = [0.5, 0.5, 0.5,  1.5, 2.5, 3.5]

        @test emd_distance(x1, x2) ≈ emd_distance(x2, x1) atol = 1e-12
    end

    @testset "Triangle inequality" begin
        x1 = [0.0, 0.0, 0.0,  1.0, 0.0, 0.0]
        x2 = [0.3, 0.0, 0.0,  1.3, 0.0, 0.0]
        x3 = [0.7, 0.0, 0.0,  1.7, 0.0, 0.0]

        d12 = emd_distance(x1, x2)
        d23 = emd_distance(x2, x3)
        d13 = emd_distance(x1, x3)
        @test d13 ≤ d12 + d23 + 1e-10
    end
end
