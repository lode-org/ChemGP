# Architecture

## Workspace Layout

```
ChemGP/
  Cargo.toml              # workspace root
  crates/
    chemgp-core/          # all GP + optimizer logic
      src/
        lib.rs            # module exports, StopReason enum
        invdist.rs        # inverse distance features + Jacobian
        kernel.rs         # Kernel enum, MolInvDistSE, CartesianSE
        types.rs          # TrainingData, GPModel, init_kernel
        covariance.rs     # build_full_covariance, robust_cholesky
        predict.rs        # GP posterior mean + variance, PredModel, build_pred_model
        nll.rs            # MAP NLL + analytical gradient
        scg.rs            # Moller (1993) SCG optimizer
        train.rs          # train_model dispatcher
        distances.rs      # max_1d_log, euclidean distance
        emd.rs            # brute-force EMD
        sampling.rs       # FPS, select_optim_subset, prune
        lbfgs.rs          # L-BFGS two-loop recursion
        optim_step.rs     # FIRE/LBFGS unified step
        trust.rs          # adaptive trust thresholds, EMD-based trust
        rff.rs            # Random Fourier Features (O(D_rff^3) GP approx)
        minimize.rs       # gp_minimize
        dimer.rs          # gp_dimer, standard_dimer
        neb_path.rs       # NEBConfig, NEBPath, NEB force computation
        idpp.rs           # IDPP/sIDPP path initialization
        neb.rs            # neb_optimize, gp_neb_aie
        neb_oie.rs        # gp_neb_oie (LCB-guided image selection)
        otgpd.rs          # otgpd (HOD, adaptive threshold)
        potentials.rs     # LJ, Muller-Brown, LEPS
        io.rs             # readcon-core + chemfiles (feature-gated)
        oracle.rs         # rgpot-core wrapper (feature-gated)
      examples/
        mb_minimize.rs    # GP minimize on Muller-Brown (CartesianSE)
        leps_minimize.rs  # GP minimize on LEPS (MolInvDistSE)
        leps_neb.rs       # NEB vs AIE vs OIE
        leps_dimer.rs     # GP-Dimer vs OTGPD
    chemgp-py/            # pyo3 bindings
  docs/                   # Sphinx + Shibuya documentation
  scripts/                # figure generation and plotting
  data/                   # test structures (HCN, system100)
```

## Four Key Components

Every GP-accelerated optimizer in ChemGP shares four mechanisms:

1. **FPS subset selection** for hyperparameter training
2. **Trust region clipping** for step proposals (EMD for molecules,
   Euclidean for Cartesian surfaces)
3. **RFF approximation** for large training sets
4. **LCB exploration** adapted per method

## Kernel Dispatch

The `Kernel` enum wraps `MolInvDistSE` (molecules) and `CartesianSE`
(arbitrary surfaces). All code accepts `&Kernel` and dispatches through
unified methods. See [Kernel Design](kernel_design) for details.

## PredModel Dispatch

The `PredModel` enum in `predict.rs` dispatches between exact GP and
RFF for inner-loop predictions:

```rust
pub enum PredModel {
    Gp(GPModel),
    Rff(RffModel),
}
```

All optimizers (minimize, dimer, otgpd, neb, neb_oie) use
`build_pred_model()` to construct the appropriate variant based on
`cfg.rff_features`. Hyperparameters are trained on the FPS subset
(exact GP via SCG), then the prediction model is built on all training
data for fast inner-loop evaluation.

## Outer Loop Pattern

All GP-accelerated optimizers follow the same pattern:

1. **FPS subset selection**: select the K most informative training
   points near the current position
2. **Train GP**: SCG-optimize hyperparameters on the subset
3. **Build PredModel**: exact GP or RFF on full training data
4. **Inner optimization**: optimize on the surrogate surface
   (L-BFGS for minimize/dimer, FIRE/LBFGS for NEB)
5. **Trust clip**: clip the proposed step to the trust region
6. **Oracle call**: evaluate the true potential at the new point
7. **Convergence check**: compare force to tolerance
