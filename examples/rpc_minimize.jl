# ==============================================================================
# Example: GP-guided minimization using a remote potential via rgpot RPC
# ==============================================================================
#
# This connects ChemGP to a potential served over Cap'n Proto RPC using the
# rgpot library (https://github.com/OmniPotentRPC/rgpot).
#
# Prerequisites:
#   1. Build rgpot with RPC support (produces libpot_client_bridge.so or librgpot.so)
#   2. Start a potential server:
#        ./potserv 12345 CuH2    # or any supported potential
#   3. Run this script:
#        julia --project=. examples/rpc_minimize.jl
#
# The server can serve any potential supported by rgpot: EAM, metatensor ML
# potentials, MACE, etc. ChemGP sees it as a black-box oracle returning
# (energy, forces) and wraps it for GP-guided optimization.

using ChemGP

# ---------------------------------------------------------------------------
# Configuration: adjust these for your setup
# ---------------------------------------------------------------------------
const SERVER_HOST = get(ENV, "RGPOT_HOST", "localhost")
const SERVER_PORT = parse(Int, get(ENV, "RGPOT_PORT", "12345"))
const LIB_PATH = get(
    ENV,
    "RGPOT_LIB",
    joinpath(homedir(), "Git/Github/OmniPotentRPC/rgpot/bbdir/CppCore/librgpot.so"),
)

# System definition: 4-atom copper cluster
const ATOMIC_NUMBERS = Int32[29, 29, 29, 29]
const BOX = Float64[20, 0, 0, 0, 20, 0, 0, 0, 20]  # large box (cluster in vacuum)

# Initial positions (flat: [x1,y1,z1, x2,y2,z2, ...])
const X_INIT = Float64[
    0.0,
    0.0,
    0.0,     # atom 1
    2.5,
    0.0,
    0.0,     # atom 2
    1.25,
    2.2,
    0.0,    # atom 3
    1.25,
    0.7,
    2.0,    # atom 4
]

# ---------------------------------------------------------------------------
# Connect and optimize
# ---------------------------------------------------------------------------
function main()
    println("Connecting to rgpot server at $SERVER_HOST:$SERVER_PORT")
    println("Using library: $LIB_PATH")

    # Connect to the remote potential
    pot = RpcPotential(SERVER_HOST, SERVER_PORT, LIB_PATH, ATOMIC_NUMBERS, BOX)

    # Wrap as a ChemGP oracle: x -> (energy, gradient)
    oracle = make_rpc_oracle(pot)

    # Test a single evaluation
    E, G = oracle(X_INIT)
    println("Initial energy: $E")
    println("Max |gradient|: $(maximum(abs.(G)))")

    # GP-guided minimization
    kernel = MolInvDistSE(1.0, [1.0], Float64[])
    config = MinimizationConfig(; max_iter=50, conv_tol=1e-3, trust_radius=0.5)

    result = gp_minimize(oracle, X_INIT, kernel; config=config)

    println("\n--- Result ---")
    println("Converged: $(result.converged)")
    println("Oracle calls: $(result.oracle_calls)")
    println("Final energy: $(result.E_final)")
    println("Max |gradient|: $(maximum(abs.(result.G_final)))")

    # Clean up the RPC connection
    close(pot)
end

main()
