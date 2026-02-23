# ==============================================================================
# GP-NEB OIE Integration Test (PET-MAD via RPC)
# ==============================================================================
#
# Verifies GP-NEB OIE convergence on HCN -> HNC isomerization using PET-MAD.
# Requires:
#   - rgpot built with RPC support
#   - eOn built with serve + metatomic support
#   - PET-MAD model (set PETMAD_MODEL_PATH)
#
# Run via pixi:
#   pixi run neb-integration-test
#
# Or manually:
#   PETMAD_MODEL_PATH=/path/to/pet-mad.pt EON_SERVE_BIN=/path/to/eonclient \
#     julia --project=. test/test_neb_rpc.jl

using ChemGP
using Test
using Printf
using LinearAlgebra

# ==============================================================================
# Skip checks
# ==============================================================================

function petmad_available()
    model = get(ENV, "PETMAD_MODEL_PATH", nothing)
    if model === nothing || !isfile(model)
        @warn "Skipping PET-MAD NEB tests: PETMAD_MODEL_PATH not set or file missing."
        return false
    end
    if !rgpot_available()
        @warn "Skipping PET-MAD NEB tests: rgpot shared library not found."
        return false
    end
    bin = _find_eon_serve()
    if bin === nothing
        @warn "Skipping PET-MAD NEB tests: eonclient not found. Set EON_SERVE_BIN."
        return false
    end
    return true
end

function _find_eon_serve()
    # Explicit env var
    env_bin = get(ENV, "EON_SERVE_BIN", nothing)
    if env_bin !== nothing && isfile(env_bin)
        return env_bin
    end
    # Check PATH
    try
        path = strip(read(`which eonclient`, String))
        if isfile(path)
            return path
        end
    catch
    end
    # Check common pixi locations
    prefix = get(ENV, "CONDA_PREFIX", nothing)
    if prefix !== nothing
        p = joinpath(prefix, "bin", "eonclient")
        if isfile(p)
            return p
        end
    end
    return nothing
end

function with_eon_serve(f::Function, port::Integer; startup_time::Real = 5.0)
    bin = _find_eon_serve()
    bin === nothing && error("eonclient not found")
    model = ENV["PETMAD_MODEL_PATH"]

    # Write temporary config
    config_path = tempname() * ".ini"
    open(config_path, "w") do io
        println(io, "[Metatomic]")
        println(io, "model_path = $model")
        println(io, "device = cpu")
        println(io, "length_unit = angstrom")
    end

    proc = run(
        pipeline(
            `$bin --serve "metatomic:$port" --config $config_path`;
            stdout = devnull, stderr = devnull,
        );
        wait = false,
    )
    sleep(startup_time)

    try
        f()
    finally
        kill(proc)
        wait(proc)
        rm(config_path; force = true)
    end
end

if !petmad_available()
    @info "PET-MAD NEB integration tests skipped"
    exit(0)
end

# ==============================================================================
# Test configuration
# ==============================================================================

const TEST_PORT = 18950
const ATOMIC_NUMBERS = Int32[6, 7, 1]
const BOX = Float64[20, 0, 0, 0, 20, 0, 0, 0, 20]

# HCN reactant (Baker 01_hcn, frame 1)
const X_HCN = Float64[
   -0.0000000000, -0.0001901002,  0.4953725273,   # C
    0.0000000000,  0.0001075881, -0.6502937324,   # N
   -0.0000000000, -0.0004700964,  1.5653497002,   # H
]

# HNC product (Baker 01_hcn, frame 2)
const X_HNC = Float64[
    0.0000000000,  0.0000000000,  0.7365959260,   # C
    0.0000000000,  0.0000000000, -0.4276753515,   # N
    0.0000000000,  0.0000000000, -1.4258476271,   # H
]

# eOn CI-NEB reference: barrier = 2.918 eV
const REFERENCE_BARRIER = 2.918

# ==============================================================================
# Tests
# ==============================================================================

@testset "PET-MAD NEB Integration" begin
    with_eon_serve(TEST_PORT) do
        pot = RpcPotential("localhost", TEST_PORT, ATOMIC_NUMBERS, BOX)
        oracle = make_rpc_oracle(pot)

        @testset "Standard CI-NEB convergence" begin
            cfg = NEBConfig(
                images = 8,
                spring_constant = 1.0,
                climbing_image = true,
                energy_weighted = true,
                ew_k_min = 0.972,
                ew_k_max = 9.72,
                max_iter = 1000,
                conv_tol = 0.05,
                atom_types = Int[6, 7, 1],
                verbose = false,
            )

            result = neb_optimize(oracle, X_HCN, X_HNC; config = cfg)

            @test result.converged

            barrier = result.path.energies[result.max_energy_image] -
                      result.path.energies[1]
            @test barrier > 0.0
            @test isapprox(barrier, REFERENCE_BARRIER, atol = 0.1)

            @printf("Standard NEB: %d calls, barrier = %.4f eV\n",
                    result.oracle_calls, barrier)
        end

        @testset "GP-NEB OIE uses fewer oracle calls" begin
            kernel = MolInvDistSE(1.0, [1.0], Float64[])

            cfg = NEBConfig(
                images = 8,
                spring_constant = 1.0,
                climbing_image = true,
                energy_weighted = true,
                ew_k_min = 0.972,
                ew_k_max = 9.72,
                conv_tol = 0.05,
                trust_radius = 0.1,
                atom_types = Int[6, 7, 1],
                gp_train_iter = 300,
                max_outer_iter = 80,
                verbose = false,
            )

            result = gp_neb_oie(oracle, X_HCN, X_HNC, kernel; config = cfg)

            @test result.converged

            barrier = result.path.energies[result.max_energy_image] -
                      result.path.energies[1]
            @test barrier > 0.0
            @test isapprox(barrier, REFERENCE_BARRIER, atol = 0.1)

            # OIE must use significantly fewer oracle calls than standard
            # Standard uses ~362 calls; OIE should use < 50
            @test result.oracle_calls < 50

            @printf("GP-NEB OIE: %d calls, barrier = %.4f eV\n",
                    result.oracle_calls, barrier)
        end

        close(pot)
    end
end
