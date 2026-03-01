# Quickstart

## From Rust

Add `chemgp-core` to your `Cargo.toml`:

```toml
[dependencies]
chemgp-core = "0.1"
```

Run the Muller-Brown minimization example (2D, CartesianSE kernel):

```shell
cargo run --release --example mb_minimize
```

Or the LEPS minimization example (molecular, MolInvDistSE kernel):

```shell
cargo run --release --example leps_minimize
```

Both produce `.jsonl` files with convergence data for GP-guided vs
direct gradient descent minimization.

## From Python

Install the Python bindings:

```shell
pip install chemgp
```

Python API documentation will be added as the `chemgp-py` bindings are
expanded. The bindings use [pyo3](https://pyo3.rs/) and expose the same
core algorithms available in Rust.

## Examples

Four self-contained examples demonstrate the core methods:

`mb_minimize`
: GP-guided minimization on the Muller-Brown 2D surface using the
  `CartesianSE` kernel. GP converges in 7 oracle calls; direct GD
  needs 34.

`leps_minimize`
: GP-guided minimization on the LEPS surface (collinear H + H2) using
  the `MolInvDistSE` kernel. GP converges in 9 oracle calls; direct
  GD needs 200+.

`leps_dimer`
: Standard dimer vs GP-Dimer vs OTGPD for saddle point search.
  GP methods converge in ~13 calls; standard dimer needs ~45.

`leps_neb`
: Standard NEB vs GP-NEB AIE vs GP-NEB OIE for minimum energy path.
  OIE converges in ~49 calls; standard NEB needs ~127.

Run all examples:

```shell
cargo run --release --example mb_minimize
cargo run --release --example leps_minimize
cargo run --release --example leps_dimer
cargo run --release --example leps_neb
```

Each writes a `.jsonl` file with per-step convergence data suitable for
plotting.
