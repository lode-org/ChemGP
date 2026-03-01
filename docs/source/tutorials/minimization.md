# Minimization

## GP-Guided Minimization

The GP minimizer (`gp_minimize`) replaces most oracle calls with GP
predictions, querying the true potential only when the GP uncertainty
is high or when the predicted step leaves the trust region.

### Key components

1. **FPS subset selection** limits the GP training set to the most
   informative points, keeping Cholesky cost manageable.

2. **Trust regions** clip proposed steps to regions where the GP
   has sufficient training data coverage. EMD distance for molecules,
   Euclidean distance for Cartesian surfaces.

3. **LCB exploration** adds a variance penalty to the predicted energy,
   encouraging sampling of uncertain regions.

4. **Energy regression gate** rejects steps where the GP predicts
   implausibly low energies (extrapolation artifacts).

5. **PredModel dispatch** uses RFF approximation for scalable prediction
   when `rff_features > 0`, otherwise exact GP.

## Muller-Brown Example

The Muller-Brown surface is a standard 2D test potential with three
minima and two saddle points. Using the `CartesianSE` kernel:

```shell
cargo run --release --example mb_minimize
```

GP minimization converges to the nearest local minimum in 7 oracle
calls. Direct gradient descent with the same convergence tolerance
needs 34 calls, a 4.9x improvement.

Key configuration for non-molecular surfaces: set `dedup_tol`
explicitly (the default `conv_tol * 0.1` can be too large relative to
the coordinate space), and use `TrustMetric::Euclidean` instead of
the default EMD metric.

## LEPS Example

The LEPS surface models a collinear H + H2 reaction. Using the
`MolInvDistSE` kernel:

```shell
cargo run --release --example leps_minimize
```

GP minimization converges in 9 oracle calls to the reactant minimum
(E = -4.515). Direct gradient descent needs 200 calls and still has
not converged to the same precision.

## Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `conv_tol` | 5e-3 | Gradient norm convergence threshold |
| `trust_radius` | 0.1 | Max distance from training data |
| `max_oracle_calls` | 0 (unlimited) | Oracle call budget |
| `n_initial_perturb` | 4 | Bootstrap perturbation points |
| `perturb_scale` | 0.1 | Perturbation radius |
| `penalty_coeff` | 1e3 | Trust region penalty in inner L-BFGS |
| `rff_features` | 0 (exact GP) | RFF feature count |
| `fps_history` | 0 (use all) | FPS subset size |
| `trust_metric` | Emd | Distance metric for trust region |
| `dedup_tol` | 0 (auto) | Minimum distance between training points |
