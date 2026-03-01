# Installation

## Requirements

- Rust 1.75+ (for building from source)
- Python 3.10+ (for Python bindings, optional)

## Building from source

```shell
git clone https://github.com/HaoZeke/ChemGP.git
cd ChemGP
cargo build --release
```

## Running tests

```shell
cargo test -p chemgp-core
```

## Optional features

The `chemgp-core` crate has two optional feature flags:

`io`
: Enables file I/O through `chemfiles` (for .extxyz) and `readcon-core`
  (for .con files).

`rgpot`
: Enables the RPC oracle interface through `rgpot-core` for connecting
  to external potential energy calculators.

Enable features in your `Cargo.toml`:

```toml
[dependencies]
chemgp-core = { version = "0.1", features = ["io", "rgpot"] }
```

## Python bindings

The Python package is built with [maturin](https://www.maturin.rs/):

```shell
pip install maturin
cd crates/chemgp-py
maturin develop --release
```

Or install directly:

```shell
pip install chemgp
```
