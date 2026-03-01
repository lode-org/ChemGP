# Nudged Elastic Band

```{note}
Full tutorial with interactive examples via Python bindings to follow.
```

## Minimum Energy Path

The Nudged Elastic Band (NEB) method finds minimum energy paths between
two known states (reactant and product). ChemGP provides three variants:

Standard NEB
: All images evaluated at every iteration. The spring forces keep images
  distributed along the path while the perpendicular force component
  drives relaxation.

GP-NEB AIE (All Image Evaluation)
: Evaluates all images with the oracle each outer iteration, then
  relaxes on the GP surrogate. Per-bead FPS subset selection ensures
  each image has local GP coverage.

GP-NEB OIE (One Image Evaluation)
: Evaluates only one image per outer iteration, selected by LCB scoring
  (force magnitude + uncertainty). Most oracle-efficient variant.

## Scalability

For large systems, the GP covariance matrix grows as O((N*(1+D))^2)
where N is the number of training points and D the dimensionality. Two
mechanisms keep this tractable:

1. **Per-bead FPS subset**: selects the K nearest training points for
   each NEB image, then takes the global union. Limits the training
   set size.

2. **Random Fourier Features (RFF)**: approximates the GP with a linear
   model that has O(D_rff) prediction cost. Trained on the FPS subset
   hyperparameters but evaluated on all data.

## LEPS Example

```shell
cargo run --release --example leps_neb
```

On the LEPS surface, OIE converges in ~49 oracle calls, AIE in ~62,
and standard NEB in ~127.
