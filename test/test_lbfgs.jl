@testset "L-BFGS" begin
    @testset "Circular buffer" begin
        hist = ChemGP.LBFGSHistory(3)

        # Push 5 pairs that satisfy curvature condition y'*s > 0
        for i in 1:5
            s = [Float64(i), Float64(i)]
            y = [Float64(i), Float64(i)]  # y'*s = 2i^2 > 0
            ChemGP.push_pair!(hist, s, y)
        end

        @test length(hist.s) == 3
        @test length(hist.y) == 3
        @test hist.count == 5

        # Oldest surviving pair should be i=3
        @test hist.s[1] == [3.0, 3.0]
        @test hist.y[1] == [3.0, 3.0]

        # Newest should be i=5
        @test hist.s[3] == [5.0, 5.0]
        @test hist.y[3] == [5.0, 5.0]
    end

    @testset "Reset" begin
        hist = ChemGP.LBFGSHistory(5)
        ChemGP.push_pair!(hist, [1.0, 1.0], [1.0, 1.0])
        ChemGP.push_pair!(hist, [2.0, 2.0], [2.0, 2.0])
        ChemGP.reset!(hist)

        @test length(hist.s) == 0
        @test length(hist.y) == 0
        @test hist.count == 0
        @test hist.m == 5  # Memory depth preserved
    end

    @testset "Steepest descent fallback" begin
        hist = ChemGP.LBFGSHistory(5)
        g = [3.0, -1.0, 2.0]
        d = ChemGP.compute_direction(hist, g)

        # With empty history, direction should be -gradient
        @test d ≈ -g
    end

    @testset "Curvature condition rejection" begin
        hist = ChemGP.LBFGSHistory(5)
        # y'*s <= 0 should be silently skipped
        ChemGP.push_pair!(hist, [1.0, 0.0], [-1.0, 0.0])  # y's = -1 < 0
        @test length(hist.s) == 0

        ChemGP.push_pair!(hist, [1.0, 0.0], [0.0, 1.0])  # y's = 0, skip
        @test length(hist.s) == 0
    end

    @testset "Converge on quadratic" begin
        # Minimize f(x) = 0.5 * x' * A * x where A = [4 1; 1 2]
        A = [4.0 1.0; 1.0 2.0]
        grad(x) = A * x

        hist = ChemGP.LBFGSHistory(5)
        x = [10.0, -5.0]

        for iter in 1:50
            g = grad(x)

            if norm(g) < 1e-10
                break
            end

            d = ChemGP.compute_direction(hist, g)

            # Exact line search for quadratic: α = -(g'd)/(d'Ad)
            α = -dot(g, d) / dot(d, A * d)
            α = max(α, 1e-12)  # Safety

            x_new = x + α * d
            g_new = grad(x_new)

            s = x_new - x
            y = g_new - g
            ChemGP.push_pair!(hist, s, y)

            x = x_new
        end

        @test norm(x) < 1e-8
    end

    @testset "Converge on Rosenbrock (2D)" begin
        # f(x,y) = (1-x)^2 + 100(y-x^2)^2, minimum at (1,1)
        function rosenbrock_grad(x)
            gx = -2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]^2)
            gy = 200 * (x[2] - x[1]^2)
            return [gx, gy]
        end

        hist = ChemGP.LBFGSHistory(10)
        x = [-1.0, 1.0]

        for iter in 1:2000
            g = rosenbrock_grad(x)
            if norm(g) < 1e-6
                break
            end

            d = ChemGP.compute_direction(hist, g)

            # Backtracking line search
            α = 1.0
            f(z) = (1 - z[1])^2 + 100 * (z[2] - z[1]^2)^2
            fx = f(x)
            for _ in 1:30
                x_trial = x + α * d
                if f(x_trial) < fx + 1e-4 * α * dot(g, d)
                    break
                end
                α *= 0.5
            end

            x_new = x + α * d
            g_new = rosenbrock_grad(x_new)

            s = x_new - x
            y = g_new - g
            ChemGP.push_pair!(hist, s, y)

            x = x_new
        end

        @test isapprox(x, [1.0, 1.0], atol = 1e-4)
    end
end
