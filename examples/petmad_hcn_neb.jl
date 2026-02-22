# ==============================================================================
# Example: GP-guided NEB on HCN -> HNC isomerization via PET-MAD RPC
# ==============================================================================
#
# Runs both standard NEB and GP-NEB (AIE) on the hydrogen cyanide to hydrogen
# isocyanide reaction using PET-MAD (metatomic potential) served over RPC.
#
# Per-step output mirrors eOn: neb_NNN.dat + neb_path_NNN.xyz per iteration,
# plus a single HDF5 file with the full optimization history.
# 
# Reference (eOn CI-NEB with PET-MAD): barrier=2.918 eV, C-N=1.195, C-H=1.292, N-H=1.447, angle=71 deg
#
# Prerequisites:
#   1. Build rgpot with RPC support
#   2. Start a PET-MAD potential server:
#        ./potserv 12345 pet-mad
#   3. Run this script:
#        julia --project=. examples/petmad_hcn_neb.jl
#
# Coordinates from Baker test set (01_hcn):
#   nebmmf/eonRuns/resources/icFSM/baker/01_hcn/initial.xyz

using ChemGP
using Printf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const SERVER_HOST = get(ENV, "RGPOT_HOST", "localhost")
const SERVER_PORT = parse(Int, get(ENV, "RGPOT_PORT", "12345"))

# System: C, N, H (order matches initial.xyz)
const ATOMIC_NUMBERS = Int32[6, 7, 1]
const BOX = Float64[20, 0, 0, 0, 20, 0, 0, 0, 20]  # cluster in vacuum

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

const OUTDIR = get(ENV, "CHEMGP_OUTDIR", "results_hcn_neb")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    mkpath(OUTDIR)

    println("Connecting to PET-MAD server at $SERVER_HOST:$SERVER_PORT")
    pot = RpcPotential(SERVER_HOST, SERVER_PORT, ATOMIC_NUMBERS, BOX)
    oracle = make_rpc_oracle(pot)

    # Verify connectivity
    E_r, _ = oracle(X_HCN)
    E_p, _ = oracle(X_HNC)
    @printf("HCN energy: %.6f\n", E_r)
    @printf("HNC energy: %.6f\n", E_p)

    neb_cfg = NEBConfig(
        n_images = 10,
        spring_constant = 1.0,
        climbing_image = true,
        energy_weighted = true,
        ew_k_min = 0.972,
        ew_k_max = 9.72,
        max_iter = 1000,
        conv_tol = 0.05,
        step_size = 0.01,
        verbose = true,
    )

    # Parallel oracle pool: one connection per movable image (or thread count)
    n_workers = min(Threads.nthreads(), neb_cfg.n_images - 2)
    oracles = if n_workers > 1
        println("Creating $n_workers parallel oracle connections")
        make_oracle_pool(SERVER_HOST, SERVER_PORT, ATOMIC_NUMBERS, BOX, n_workers)
    else
        oracle
    end

    # --- Standard NEB (baseline) ---
    println("\n=== Standard NEB ===")
    std_dir = joinpath(OUTDIR, "standard")

    # Per-step .dat/.xyz writer + HDF5 history
    std_writer = make_neb_writer(std_dir, ATOMIC_NUMBERS, BOX)
    std_h5 = make_neb_hdf5_writer(
        joinpath(std_dir, "neb_history.h5");
        atomic_numbers = ATOMIC_NUMBERS, cell = BOX,
    )
    std_callback = (path, iter) -> begin
        std_writer(path, iter)
        std_h5(path, iter)
    end

    result_std = neb_optimize(oracles, X_HCN, X_HNC;
        config = neb_cfg, on_step = std_callback)

    # Final outputs
    write_neb_trajectory(result_std, joinpath(std_dir, "neb_final.xyz"), ATOMIC_NUMBERS, BOX)
    write_neb_hdf5(result_std, joinpath(std_dir, "neb_result.h5");
        atomic_numbers = ATOMIC_NUMBERS, cell = BOX)
    write_convergence_csv(result_std, joinpath(std_dir, "convergence.csv"))

    # --- GP-NEB AIE ---
    println("\n=== GP-NEB (AIE) ===")
    gp_dir = joinpath(OUTDIR, "gp_aie")
    kernel = MolInvDistSE(1.0, [1.0], Float64[])

    gp_writer = make_neb_writer(gp_dir, ATOMIC_NUMBERS, BOX)
    gp_h5 = make_neb_hdf5_writer(
        joinpath(gp_dir, "neb_history.h5");
        atomic_numbers = ATOMIC_NUMBERS, cell = BOX,
    )
    gp_callback = (path, iter) -> begin
        gp_writer(path, iter)
        gp_h5(path, iter)
    end

    gp_cfg = NEBConfig(
        n_images = 10,
        spring_constant = 1.0,
        climbing_image = true,
        energy_weighted = true,
        ew_k_min = 0.972,
        ew_k_max = 9.72,
        conv_tol = 0.05,
        gp_train_iter = 300,
        max_outer_iter = 50,
        trust_radius = 0.1,
        verbose = true,
    )

    result_gp = gp_neb_aie(oracles, X_HCN, X_HNC, kernel;
        config = gp_cfg, on_step = gp_callback)

    write_neb_trajectory(result_gp, joinpath(gp_dir, "neb_final.xyz"), ATOMIC_NUMBERS, BOX)
    write_neb_hdf5(result_gp, joinpath(gp_dir, "neb_result.h5");
        atomic_numbers = ATOMIC_NUMBERS, cell = BOX)
    write_convergence_csv(result_gp, joinpath(gp_dir, "convergence.csv"))

    # --- GP-NEB OIE ---
    # OIE evaluates one image per iteration (max uncertainty), so parallelism
    # does not apply -- pass single oracle.
    println("\n=== GP-NEB (OIE) ===")
    oie_dir = joinpath(OUTDIR, "gp_oie")

    oie_writer = make_neb_writer(oie_dir, ATOMIC_NUMBERS, BOX)
    oie_h5 = make_neb_hdf5_writer(
        joinpath(oie_dir, "neb_history.h5");
        atomic_numbers = ATOMIC_NUMBERS, cell = BOX,
    )
    oie_callback = (path, iter) -> begin
        oie_writer(path, iter)
        oie_h5(path, iter)
    end

    oie_cfg = NEBConfig(
        n_images = 10,
        spring_constant = 1.0,
        climbing_image = true,
        energy_weighted = true,
        ew_k_min = 0.972,
        ew_k_max = 9.72,
        conv_tol = 0.05,
        gp_train_iter = 300,
        max_outer_iter = 80,
        trust_radius = 0.1,
        verbose = true,
    )

    result_oie = gp_neb_oie(oracle, X_HCN, X_HNC, kernel;
        config = oie_cfg, on_step = oie_callback)

    write_neb_trajectory(result_oie, joinpath(oie_dir, "neb_final.xyz"), ATOMIC_NUMBERS, BOX)
    write_neb_hdf5(result_oie, joinpath(oie_dir, "neb_result.h5");
        atomic_numbers = ATOMIC_NUMBERS, cell = BOX)
    write_convergence_csv(result_oie, joinpath(oie_dir, "convergence.csv"))

    # --- Generate profile plots ---
    for (label, dir) in [("standard", std_dir), ("gp_aie", gp_dir), ("gp_oie", oie_dir)]
        traj = joinpath(dir, "neb_final.xyz")
        png = joinpath(dir, "profile.png")
        cmd = `uv run rgpycrumbs eon plt-neb --source traj --input-traj $traj -o $png --title "HCN->HNC ($label)"`
        println("Plotting: $cmd")
        try
            run(cmd)
        catch e
            @warn "Plot command failed" exception = e
        end
    end

    # --- Comparison table ---
    println("\n" * "="^70)
    @printf("%-12s %12s %10s %15s\n", "Method", "Oracle Calls", "Converged", "Barrier (E_TS)")
    println("-"^70)

    for (label, res) in [("Standard", result_std), ("GP-NEB AIE", result_gp),
                          ("GP-NEB OIE", result_oie)]
        ts_idx = res.max_energy_image
        barrier = res.path.energies[ts_idx] - res.path.energies[1]
        @printf("%-12s %12d %10s %15.6f\n",
            label, res.oracle_calls, res.converged, barrier)
    end
    println("="^70)

    close(pot)
    println("\nResults written to $OUTDIR/")
end

main()
