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
        kernel.rs         # MolInvDistSE, kernel_blocks, hypergrad blocks
        types.rs          # TrainingData, GPModel, init_mol_invdist_se
        covariance.rs     # build_full_covariance, robust_cholesky
        predict.rs        # GP posterior mean + variance
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
        neb.rs            # neb_optimize, gp_neb_aie, PredModel
        neb_oie.rs        # gp_neb_oie (LCB-guided image selection)
        otgpd.rs          # otgpd (HOD, adaptive threshold)
        potentials.rs     # LJ, Muller-Brown, LEPS
        io.rs             # readcon-core + chemfiles (feature-gated)
        oracle.rs         # rgpot-core wrapper (feature-gated)
      examples/
        leps_minimize.rs  # GP minimize vs direct GD
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
2. **EMD trust region** clipping for step proposals
3. **RFF approximation** for large training sets
4. **LCB exploration** adapted per method

## PredModel Dispatch

The `PredModel` enum dispatches between exact GP and RFF for inner-loop
predictions:

```rust
pub enum PredModel {
    Gp(GPModel),
    Rff(RffModel),
}
```

When `rff_features > 0`, hyperparameters are trained on the FPS subset
(exact GP), then an RFF model is built on all training data for fast
O(D_rff) prediction during inner relaxation.
