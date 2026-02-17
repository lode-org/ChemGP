@testset "Muller-Brown" begin
    @testset "Known minima have small gradients" begin
        for (i, x_min) in enumerate(ChemGP.MULLER_BROWN_MINIMA)
            E, G = muller_brown_energy_gradient(x_min)
            gnorm = norm(G)
            # These are approximate minima, so gradient should be small but not zero
            @test gnorm < 5.0
            println("  Minimum $i: E = $(round(E, digits=1)), |G| = $(round(gnorm, digits=3))")
        end
    end

    @testset "Known saddle points have small gradients" begin
        for (i, x_sp) in enumerate(ChemGP.MULLER_BROWN_SADDLES)
            E, G = muller_brown_energy_gradient(x_sp)
            gnorm = norm(G)
            @test gnorm < 5.0
            println("  Saddle $i: E = $(round(E, digits=1)), |G| = $(round(gnorm, digits=3))")
        end
    end

    @testset "Energy ordering" begin
        E_A, _ = muller_brown_energy_gradient(ChemGP.MULLER_BROWN_MINIMA[1])
        E_B, _ = muller_brown_energy_gradient(ChemGP.MULLER_BROWN_MINIMA[2])
        E_C, _ = muller_brown_energy_gradient(ChemGP.MULLER_BROWN_MINIMA[3])

        # A is deepest, C is shallowest
        @test E_A < E_B < E_C
    end

    @testset "Gradient finite differences" begin
        # Verify analytical gradient against numerical finite differences
        x0 = [0.3, 0.7]
        E0, G0 = muller_brown_energy_gradient(x0)

        h = 1e-6
        G_fd = zeros(2)
        for i in 1:2
            xp = copy(x0)
            xm = copy(x0)
            xp[i] += h
            xm[i] -= h
            Ep, _ = muller_brown_energy_gradient(xp)
            Em, _ = muller_brown_energy_gradient(xm)
            G_fd[i] = (Ep - Em) / (2h)
        end

        @test isapprox(G0, G_fd, rtol = 1e-5)
    end

    @testset "Steepest descent to minimum A" begin
        # Start near minimum A and verify we converge there
        x = [-0.5, 1.5]

        for _ in 1:500
            E, G = muller_brown_energy_gradient(x)
            if norm(G) < 1e-5
                break
            end
            x = x - 1e-4 * G
        end

        E_final, G_final = muller_brown_energy_gradient(x)
        @test norm(G_final) < 0.1
        @test E_final < -140.0  # Should be near -146.7
    end
end
