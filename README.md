[![CI](https://github.com/HaoZeke/ChemGP/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HaoZeke/ChemGP/actions/workflows/CI.yml?query=branch%3Amain)
[![Documentation](https://github.com/HaoZeke/ChemGP/actions/workflows/Documentation.yml/badge.svg)](https://chemgp.rgoswami.me)

Gaussian Process accelerated optimization for computational chemistry.
ChemGP provides GP surrogate models for energy surface exploration,
reducing expensive oracle (electronic structure) evaluations by 3-22x.

Two kernel types are provided: `MolInvDistSE` for molecular systems
(operates on inverse interatomic distances, providing rotational and
translational invariance) and `CartesianSE` for arbitrary smooth
surfaces (operates directly on coordinates).

# Methods

- **Minimization**: GP-guided local optimization with FPS subset
  selection, EMD trust regions, and LCB exploration
- **Dimer**: GP-accelerated saddle point search with L-BFGS translation
- **NEB**: All-image (AIE) and one-image (OIE) evaluation with per-bead
  FPS and RFF approximation
- **OTGPD**: Adaptive threshold GP dimer with HOD training data management

All methods share four mechanisms: FPS subset selection, EMD/Euclidean
trust regions, RFF approximation for scalable prediction, and
method-adapted LCB exploration. The `PredModel` enum dispatches between
exact GP and RFF uniformly across all optimizers.

# Benchmark Results

| Surface | Method | Oracle calls | Speedup |
|---|---|---|---|
| Muller-Brown | GP minimize | 7 | 4.9x vs direct GD |
| LEPS | GP minimize | 9 | 22x vs direct GD |
| LEPS | GP-Dimer | 13 | 3.5x vs standard |
| LEPS | OTGPD | 13 | 3.5x vs standard |
| LEPS | GP-NEB AIE | 62 | 2x vs standard |
| LEPS | GP-NEB OIE | 49 | 2.6x vs standard |

# Building

```shell
cargo build --release
cargo test -p chemgp-core
```

# Examples

```shell
# 2D analytical surface (CartesianSE kernel)
cargo run --release --example mb_minimize

# Collinear H + H2 reaction (MolInvDistSE kernel)
cargo run --release --example leps_minimize
cargo run --release --example leps_dimer
cargo run --release --example leps_neb
```

Each writes a `.jsonl` file with per-step convergence data.

# Python Bindings

```shell
pip install maturin
cd crates/chemgp-py
maturin develop --release
```

# Documentation

```shell
pixi run -e docs docbld
```

Or visit [chemgp.rgoswami.me](https://chemgp.rgoswami.me).

# License

MIT License. See [LICENSE](LICENSE) for details.
