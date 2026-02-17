[![CI](https://github.com/HaoZeke/ChemGP.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HaoZeke/ChemGP.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Documentation](https://github.com/HaoZeke/ChemGP.jl/actions/workflows/Documentation.yml/badge.svg)](https://lode-org.github.io/ChemGP.jl/dev/)
[![Pre-commit](https://github.com/HaoZeke/ChemGP.jl/actions/workflows/pre_commit.yml/badge.svg)](https://github.com/HaoZeke/ChemGP.jl/actions/workflows/pre_commit.yml)

A Julia package for Gaussian Process (GP) guided molecular geometry
optimization. ChemGP provides GP surrogate models with molecular kernels for
energy surface exploration, including minimization, saddle-point search (dimer
method), nudged elastic band (NEB) path optimization, and optimal transport GP
dimer (OTGPD).


# Features

-   **Molecular kernels**: Inverse-distance based SE, Matern 5/2, and Matern 3/2
    kernels with kernel composition (sum, product, constant offset)
-   **GP surrogate**: Full covariance with energy+gradient observations, automatic
    hyperparameter optimization via marginal likelihood
-   **Minimization**: Trust-region GP-guided geometry optimization
-   **Dimer method**: GP-accelerated saddle point search
-   **NEB**: Standard NEB, GP-NEB with all-image evaluation (AIE) and
    outer-image-only evaluation (OIE)
-   **OTGPD**: Optimal Transport GP Dimer for transition state search
-   **Oracles**: Built-in Lennard-Jones, Muller-Brown, and LEPS potentials; RPC
    oracle for remote potential evaluation via [rgpot](https://github.com/HaoZeke/rgpot)


# Installation

ChemGP.jl is not yet registered. Install directly from GitHub:

    using Pkg
    Pkg.add(url="https://github.com/HaoZeke/ChemGP.jl")


# Quick Start

    using ChemGP
    
    # Create a GP model with a molecular kernel
    kernel = MolInvDistSE(1.0, 1.0)
    model = GPModel(kernel; noise=1e-6)
    
    # Set up training data for a 3-atom system
    td = TrainingData(3)
    
    # Evaluate oracle and add point
    coords = random_cluster(3, 2.0)
    E, G = lj_energy_gradient(coords)
    add_point!(td, coords, E, G)
    
    # Train and predict
    train_model!(model, td)
    E_pred, G_pred = predict(model, td, coords)

See the [Quick Start tutorial](https://lode-org.github.io/ChemGP.jl/dev/tutorials/quickstart/) for a complete walkthrough.


# Development Setup

ChemGP uses [pixi](https://pixi.sh) for environment management:

    pixi install
    pixi r instantiate


## Available Tasks

| Task | Command | Description |
|------|---------|-------------|
| Instantiate | `pixi r instantiate` | Install Julia project dependencies |
| Test | `pixi r test` | Run the Julia test suite |
| Format | `pixi r fmt` | Format all Julia files |
| Lint | `pixi r lint` | Check formatting and run all pre-commit hooks |
| Pre-commit | `pixi r pre-commit` | Run all pre-commit hooks |
| Generate README | `pixi r -e docs gen-readme` | Export `readme_src.org` to `README.md` |
| Install docs deps | `pixi r -e docs docinstall` | Install documentation dependencies |
| Build docs | `pixi r -e docs docbld` | Build documentation locally |
| Clean docs | `pixi r -e docs docdel` | Remove documentation build artifacts |
| Diagrams | `pixi r -e docs diagrams` | Regenerate architecture diagrams from `.dot` sources |


## Running Tests

    pixi r test


## Building Documentation

    pixi r -e docs docinstall
    pixi r -e docs docbld

The built documentation will be in `docs/build/`.


## Pre-commit Hooks

Install pre-commit hooks for automatic formatting and linting:

    uvx pre-commit install

Run all hooks manually:

    pixi r pre-commit


# Contributing

1.  Clone and set up the environment:
    
        git clone https://github.com/HaoZeke/ChemGP.jl.git
        cd ChemGP.jl
        pixi install
        pixi r instantiate

2.  Install pre-commit hooks:
    
        uvx pre-commit install

3.  Make your changes, then verify:
    
        pixi r test
        pixi r lint

4.  Code style is enforced by [JuliaFormatter](https://domluna.github.io/JuliaFormatter.jl/stable/) (Blue style,
    4-space indent). Run `pixi r fmt` to auto-format before committing.

5.  Architecture diagrams live in `docs/src/assets/diagrams/` as Graphviz
    `.dot` files. After editing, regenerate with `pixi r -e docs diagrams`.


# License

MIT License. See [LICENSE](LICENSE) for details.

