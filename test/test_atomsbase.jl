using AtomsBase
using Unitful: @u_str

@testset "AtomsBase conversion" begin
    @testset "chemgp_coords (isolated system)" begin
        sys = isolated_system([
            :C => [0.0, 0.0, 0.5]u"Å",
            :N => [0.0, 0.0, -0.65]u"Å",
            :H => [0.0, 0.0, 1.57]u"Å",
        ])
        c = chemgp_coords(sys)

        @test length(c.positions) == 9
        @test c.positions[3] ≈ 0.5
        @test c.positions[6] ≈ -0.65
        @test c.positions[9] ≈ 1.57
        @test c.atomic_numbers == Int32[6, 7, 1]
        @test c.box == Float64[20, 0, 0, 0, 20, 0, 0, 0, 20]
    end

    @testset "chemgp_coords (periodic system)" begin
        box = ([10.0, 0.0, 0.0]u"Å", [0.0, 12.0, 0.0]u"Å", [0.0, 0.0, 14.0]u"Å")
        sys = periodic_system([:H => [1.0, 2.0, 3.0]u"Å", :He => [4.0, 5.0, 6.0]u"Å"], box)
        c = chemgp_coords(sys)

        @test c.positions == Float64[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        @test c.atomic_numbers == Int32[1, 2]
        @test c.box ≈ Float64[10, 0, 0, 0, 12, 0, 0, 0, 14]
    end

    @testset "atomsbase_system round-trip (isolated)" begin
        sys = isolated_system([
            :C => [0.0, 0.0, 0.5]u"Å",
            :N => [0.0, 0.0, -0.65]u"Å",
            :H => [0.0, 0.0, 1.57]u"Å",
        ])
        c = chemgp_coords(sys)

        sys2 = atomsbase_system(c.positions, c.atomic_numbers, c.box)
        @test length(sys2) == 3
        @test all(.!periodicity(sys2))

        c2 = chemgp_coords(sys2)
        @test c2.positions ≈ c.positions
        @test c2.atomic_numbers == c.atomic_numbers
    end

    @testset "atomsbase_system round-trip (periodic)" begin
        pos = Float64[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        atnrs = Int32[8, 1]
        box = Float64[10, 0, 0, 0, 10, 0, 0, 0, 10]

        sys = atomsbase_system(pos, atnrs, box; pbc=true)
        @test length(sys) == 2
        @test all(periodicity(sys))

        c = chemgp_coords(sys)
        @test c.positions ≈ pos
        @test c.atomic_numbers == atnrs
        @test c.box ≈ box
    end
end
