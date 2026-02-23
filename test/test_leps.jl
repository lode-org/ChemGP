@testset "LEPS Potential" begin
    @testset "3D interface" begin
        # Collinear A-B-C with B at equilibrium distance from A
        x = [0.0, 0.0, 0.0, 0.742, 0.0, 0.0, 3.742, 0.0, 0.0]
        E, G = leps_energy_gradient(x)

        @test isfinite(E)
        @test length(G) == 9
        @test all(isfinite, G)

        # Energy should be negative (bound state)
        @test E < 0.0
    end

    @testset "2D reduced interface" begin
        # Same geometry in reduced coordinates
        E_2d, G_2d = leps_energy_gradient_2d([0.742, 3.0])

        @test isfinite(E_2d)
        @test length(G_2d) == 2
        @test E_2d < 0.0
    end

    @testset "2D and 3D consistency" begin
        # For a collinear arrangement, 2D and 3D should give same energy
        r_AB = 1.0
        r_BC = 1.5

        x_3d = [0.0, 0.0, 0.0, r_AB, 0.0, 0.0, r_AB + r_BC, 0.0, 0.0]
        E_3d, _ = leps_energy_gradient(x_3d)
        E_2d, _ = leps_energy_gradient_2d([r_AB, r_BC])

        @test isapprox(E_3d, E_2d, atol=1e-10)
    end

    @testset "Finite difference gradient (3D)" begin
        x = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.5, 0.0, 0.0]
        E0, G = leps_energy_gradient(x)

        h = 1e-5
        for i in 1:9
            xp = copy(x);
            xp[i] += h
            xm = copy(x);
            xm[i] -= h
            Ep, _ = leps_energy_gradient(xp)
            Em, _ = leps_energy_gradient(xm)
            fd = (Ep - Em) / (2h)
            @test isapprox(G[i], fd, atol=1e-4) || (abs(G[i]) < 1e-8 && abs(fd) < 1e-4)
        end
    end

    @testset "Finite difference gradient (2D)" begin
        r = [1.2, 1.8]
        E0, G = leps_energy_gradient_2d(r)

        h = 1e-5
        for i in 1:2
            rp = copy(r);
            rp[i] += h
            rm = copy(r);
            rm[i] -= h
            Ep, _ = leps_energy_gradient_2d(rp)
            Em, _ = leps_energy_gradient_2d(rm)
            fd = (Ep - Em) / (2h)
            @test isapprox(G[i], fd, atol=1e-4)
        end
    end

    @testset "Reactant and product energies" begin
        E_react, _ = leps_energy_gradient(LEPS_REACTANT)
        E_prod, _ = leps_energy_gradient(LEPS_PRODUCT)

        # Both should be bound states (negative energy)
        @test E_react < 0.0
        @test E_prod < 0.0

        # The asymmetric Sato parameters make them unequal
        @test E_react != E_prod
    end

    @testset "Barrier exists between reactant and product" begin
        # Sample along the exchange coordinate
        n_pts = 20
        energies = Float64[]
        for t in range(0, 1, length=n_pts)
            r_AB = 0.742 + t * 2.258    # 0.742 → 3.0
            r_BC = 3.0 - t * 2.258      # 3.0 → 0.742
            E, _ = leps_energy_gradient_2d([r_AB, r_BC])
            push!(energies, E)
        end

        # There should be a barrier (max > endpoints)
        E_max = maximum(energies)
        @test E_max > energies[1]
        @test E_max > energies[end]
    end
end
