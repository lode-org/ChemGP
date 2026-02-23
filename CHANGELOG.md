# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- towncrier release notes start -->

## [1.0.0-DEV](https://github.com/lode-org/ChemGP.jl/tree/1.0.0-DEV) - 2026-02-23

### Added

- Analytical kernel block derivatives for MolInvDistSE, replacing ForwardDiff in inner loops.
- AtomsBase.jl package extension for reading/writing molecular structures via extxyz (``atoms_to_flat_coords``, ``flat_coords_to_system``).
- Climbing-image NEB with energy-weighted spring constants and dynamic CI activation matching eOn behavior.
- EMD-based trust region for molecular systems with type-aware optimal transport distance (``emd_distance``).
- Farthest Point Sampling (FPS) subset selection for GP training data (``max_gp_points`` config parameter).
- GP-NEB optimizer with three oracle strategies: standard NEB (``neb_optimize``), All-Images Evaluation (``gp_neb_aie``), and One-Image Evaluation (``gp_neb_oie``, ``gp_neb_oie_naive``).
- NEB output writers for extxyz trajectories, eOn-compatible ``.dat`` files, and HDF5 history (``make_neb_writer``, ``make_neb_hdf5_writer``).
- OTGPD extensions: HOD rotation, variance barrier early stopping, adaptive trust radius, and FPS subset selection for hyperparameter optimization.
- PET-MAD HCN to HNC worked example with standard and GP-NEB (``examples/petmad_hcn_neb.jl``).
- Parallel oracle evaluation for NEB via thread-safe oracle pools (``make_oracle_pool``).
- RPC oracle with auto-discovery of rgpot shared libraries and eOn serve mode (``RpcPotential``, ``RpcPotentialCore``).
- Random Fourier Features (RFF) approximation for scalable GP-NEB with MolInvDistSE kernels (``rff_features`` config parameter).
- SIDPP (squared inverse distance pairwise potential) initial path interpolation for NEB.
- Tutorial figure generation scripts for GPR review paper (14+ figures covering PES, GP progression, hyperparameters, variance, trust regions, FPS, dimer, NEB on Muller-Brown, LEPS, and HCN surfaces).

### Changed

- GP-NEB inner optimization loops use L-BFGS instead of steepest descent.
- NEBConfig field ``n_images`` renamed to ``images`` matching eOn convention.

### Fixed

- Dimer translational force sign and L-BFGS step direction now match the original algorithm.
- GP-NEB trust radius enforcement in inner optimization loops.
- HDF5 writer row-major convention for Python/h5py interoperability; history files now include self-contained ``/path`` group.
- L-BFGS negative curvature reset and distance reset matching eOn behavior.
- NEB climbing image activation timing on SIDPP-initialized paths; CI toggle uses max force rather than energy.

### Developer

- CI workflows: pre-commit deduplication, macOS matrix, doc preview comments, codecov, bumped actions to v6.
- Documentation overhaul: OTGPD, NEB, trust regions, RPC integration tutorials; architecture diagrams; references.
- JuliaFormatter applied across entire codebase; pre-commit CI installs JuliaFormatter globally.
