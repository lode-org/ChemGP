# ==============================================================================
# RPC Integration Tests
# ==============================================================================
#
# These tests verify ChemGP's RPC oracle against rgpot's potserv.
# They require:
#   - rgpot built with RPC support (libpot_client_bridge.so + potserv)
#   - RGPOT_BUILD_DIR set to the meson builddir, OR rgpot built in ./rgpot/builddir
#
# Run via pixi:
#   pixi run integration-test
#
# Or manually:
#   RGPOT_BUILD_DIR=/path/to/rgpot/builddir julia --project=. test/test_rpc.jl

using ChemGP
using Test
using LinearAlgebra

# ==============================================================================
# Skip checks
# ==============================================================================

function rpc_tests_available()
    if !rgpot_available()
        @warn "Skipping RPC tests: rgpot shared library not found. " *
            "Set RGPOT_LIB_PATH or RGPOT_BUILD_DIR."
        return false
    end
    if !potserv_available()
        @warn "Skipping RPC tests: potserv executable not found. " *
            "Build rgpot with -Dwith_rpc=true."
        return false
    end
    return true
end

if !rpc_tests_available()
    @info "RPC integration tests skipped (rgpot not available)"
    exit(0)
end

# ==============================================================================
# Test configuration
# ==============================================================================

# Use a high port to avoid conflicts
const TEST_PORT = 18923
const LJ_ATMNRS = Int32[0, 0, 0]  # LJ uses type 0 (rgpot convention)
const LJ_BOX = Float64[100, 0, 0, 0, 100, 0, 0, 0, 100]  # Large non-periodic box

@testset "RPC Integration" begin
    @testset "Library discovery" begin
        lib = find_rgpot_lib()
        @test lib !== nothing
        @test isfile(lib)
        @test endswith(lib, Sys.isapple() ? ".dylib" : ".so")

        exe = find_potserv()
        @test exe !== nothing
        @test isfile(exe)
    end

    @testset "LJ potential via RPC" begin
        with_potserv(TEST_PORT, "LJ") do
            # Connect
            pot = RpcPotential("localhost", TEST_PORT, find_rgpot_lib(), LJ_ATMNRS, LJ_BOX)

            # Three atoms in a line
            x = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0]

            E_rpc, F_rpc = ChemGP.calculate(pot, x)

            @test isfinite(E_rpc)
            @test length(F_rpc) == 9
            @test all(isfinite, F_rpc)

            # Oracle adapter
            oracle = make_rpc_oracle(pot)
            E, G = oracle(x)

            @test E == E_rpc
            # G should be -F (gradient = -forces)
            @test isapprox(G, -F_rpc; atol=1e-12)

            close(pot)
        end
    end

    @testset "Auto-discovery constructor" begin
        with_potserv(TEST_PORT + 1, "LJ") do
            # This constructor should find the library automatically
            pot = RpcPotential("localhost", TEST_PORT + 1, LJ_ATMNRS, LJ_BOX)

            x = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0]
            E, F = ChemGP.calculate(pot, x)

            @test isfinite(E)
            @test all(isfinite, F)

            close(pot)
        end
    end

    @testset "GP minimize with RPC oracle" begin
        with_potserv(TEST_PORT + 2, "LJ") do
            pot = RpcPotential("localhost", TEST_PORT + 2, LJ_ATMNRS, LJ_BOX)
            oracle = make_rpc_oracle(pot)

            # 3-atom LJ cluster with reasonable initial geometry
            x_init = [0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 1.25, 2.0, 0.0]
            kernel = MolInvDistSE(1.0, [0.5], Float64[])

            config = MinimizationConfig(
                trust_radius=0.3,
                conv_tol=0.5,     # Loose for fast test
                max_iter=3,       # Very few iterations
                gp_train_iter=50,
                verbose=false,
            )

            result = gp_minimize(oracle, x_init, kernel; config=config)

            @test result.oracle_calls >= 3
            @test isfinite(result.E_final)
            @test length(result.trajectory) >= 1

            close(pot)
        end
    end

    @testset "Error handling" begin
        # Cap'n Proto may not fail on connect, only on calculate.
        # So we test that calculate on a dead server fails.
        # We connect to a port where nothing is running, then try to calculate.
        lib = find_rgpot_lib()
        # pot_client_init may or may not throw depending on implementation.
        # If it connects without error, the calculate should fail.
        try
            pot = RpcPotential("localhost", 19999, lib, LJ_ATMNRS, LJ_BOX)
            x = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0]
            # This should fail since no server is running
            @test_throws Exception ChemGP.calculate(pot, x)
        catch e
            # If the constructor itself throws, that's also fine
            @test e isa Exception
        end
    end
end
