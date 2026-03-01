# Minimization

```{note}
Full tutorial with interactive examples via Python bindings to follow.
```

## GP-Guided Minimization

The GP minimizer (`gp_minimize`) replaces most oracle calls with GP
predictions, querying the true potential only when the GP uncertainty
is high or when the predicted step leaves the trust region.

### Key components

1. **FPS subset selection** limits the GP training set to the most
   informative points, keeping Cholesky cost manageable.

2. **EMD trust regions** clip proposed steps to regions where the GP
   has sufficient training data coverage.

3. **LCB exploration** adds a variance penalty to the predicted energy,
   encouraging sampling of uncertain regions.

4. **Energy regression gate** rejects steps where the GP predicts
   implausibly low energies (extrapolation artifacts).

## LEPS Example

```shell
cargo run --release --example leps_minimize
```

On the LEPS surface (collinear H + H2), GP minimization converges in
~15 oracle calls. Direct gradient descent with the same step size
needs 200+ calls and still has not converged.
