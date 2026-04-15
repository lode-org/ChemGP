# Data for: Bayesian Optimization with Gaussian Processes to Accelerate Stationary Point Searches

This Materials Cloud Archive record contains the complete reproduction data
for all numerical figures in the tutorial review:

> Goswami, R. *Bayesian Optimization with Gaussian Processes to Accelerate
> Stationary Point Searches.* arXiv:2603.10992 (2026).
> Invited article, ACS Physical Chemistry Au.
> <https://arxiv.org/abs/2603.10992>
> doi:10.48550/arXiv.2603.10992

The data was produced by the `chemgp-core` Rust crate with fixed RNG seeds
(`StdRng::seed_from_u64`), so every record is bit-for-bit reproducible from
the manuscript source code.

## Archive

- `chemgp_tutorial_data.tar.xz` (single XZ-compressed tarball, 114 entries)

Extract with:

```bash
tar -xJf chemgp_tutorial_data.tar.xz
```

## Contents

After extraction the archive restores the directory tree used by the
manuscript pipeline:

```
config/                              # eOn serve INI for PET-MAD
docs/readme_materialscloudarchive.md # this file
docs/source/_static/figures/         # 300 DPI PNGs for the tutorial docs
models/                              # PET-MAD-XS v1.5.0 .ckpt and .pt
scripts/figures/README.md            # pipeline documentation
scripts/figures/tutorial/output/     # JSONL, HDF5, PDF, CON, DAT
scripts/proofs/                      # standalone symbolic proofs (PEP 723)
scripts/sympy/                       # SymPy verification scripts (kernel,
                                     #   NLL, RFF, dimer, constant kernel)
*.jsonl                              # root-level traces (Rust example outputs)
*.con, *.dat                         # root-level NEB path data (System100)
```

### File formats

| Extension | Format              | Reader              |
|-----------|---------------------|---------------------|
| `.jsonl`  | JSON Lines (traces) | any JSON parser     |
| `.h5`     | HDF5 (structured)   | h5py, HDFView       |
| `.pdf`    | Vector figure       | any PDF viewer      |
| `.png`    | 300 DPI raster      | any image viewer    |
| `.con`    | eOn atomic config   | ASE, readcon, eOn   |
| `.dat`    | TSV NEB path        | any text reader     |
| `.pt`     | PyTorch model       | PyTorch, metatensor |
| `.py`     | Python source       | any text reader; runnable with `uv run` (PEP 723) |

JSONL records carry `method`, `step`, `energy` (eV), `force` (eV/A), and
`oracle_calls` fields. HDF5 groups are organised as `/table`, `/grids/{name}`,
`/paths/{name}`, `/points/{name}` with plain NumPy arrays.

## Systems and potentials

| Surface                 | System                   | Atoms | DOF | Potential           |
|-------------------------|--------------------------|-------|-----|---------------------|
| Muller-Brown            | 2D analytical            | N/A   | 2   | built-in            |
| LEPS                    | H + H2 collinear         | 3     | 9   | built-in            |
| PET-MAD minimize        | molecular                | 9     | 27  | PET-MAD-XS v1.5.0   |
| PET-MAD dimer           | C3H5 allyl radical       | 8     | 24  | PET-MAD-XS v1.5.0   |
| PET-MAD NEB (System100) | C2H4 + N2O cycloaddition | 9     | 27  | PET-MAD-XS v1.5.0   |

The PET-MAD-XS v1.5.0 model is shipped inside `models/` for convenience.

- Source: [lab-cosmo/pet-mad](https://huggingface.co/lab-cosmo/pet-mad)
- Exported via `mtt export` (metatrain)
- SHA256: `5ea4404e44f281087fa5f294040546b10bf3b9c8110b166fdf4755fc39aa4a59`

## Rehydrating the manuscript repository

This archive is the exact working-tree snapshot referenced by the ChemGP
manuscript repository. Drop it at the repo root and unpack:

```bash
cd /path/to/ChemGP
tar -xJf chemgp_tutorial_data.tar.xz
```

Once restored, all figures can be regenerated from the JSONL traces without
rerunning the Rust examples or starting an eOn/PET-MAD RPC server. See
`scripts/figures/README.md` (inside the archive) for the full pipeline.

## Figure inventory

| Figure | PDF                            | Surface    |
|--------|--------------------------------|------------|
| 3      | mb_gp_progression.pdf          | MB 2D      |
| 4      | mb_variance.pdf                | MB 2D      |
| 7      | mb_hyperparams.pdf             | MB 2D      |
| 9      | rpc_dimer_convergence.pdf      | C3H5 (RPC) |
| 11     | mb_trust_region.pdf            | MB 2D      |
| 12     | mb_neb.pdf                     | MB 2D      |
| 13     | leps_neb.pdf                   | LEPS 9D    |
| 14     | leps_aie_oie.pdf               | LEPS 9D    |
| 15     | leps_minimize_convergence.pdf  | LEPS 9D    |
| 18     | leps_fps.pdf                   | LEPS 9D    |
| 19     | leps_rff_quality.pdf           | LEPS 9D    |
| 20     | petmad_minimize_convergence.pdf| PET-MAD    |
| 21     | system100_convergence.pdf      | PET-MAD    |
| 22     | system100_neb_profile.pdf      | PET-MAD    |
| 23     | system100_neb_landscape.pdf    | PET-MAD    |
| S3     | leps_nll_landscape.pdf         | LEPS 9D    |

TikZ figures from the manuscript (1, 2, 5, 6, 8, 10, 16, 17, S1, S2, TOC)
live in the manuscript source repository and are not part of this archive.

## Symbolic proofs and verification scripts

The archive ships two directories of standalone symbolic and numerical
verification scripts that accompany the derivations in the manuscript.

### `scripts/proofs/`

Single-file proofs intended to be self-contained and runnable without
project setup. Each script carries a [PEP 723](https://peps.python.org/pep-0723/)
inline metadata header, so

```bash
uv run scripts/proofs/<file>.py
```

materialises a temporary environment with the right dependencies and
executes the script.

| Script                       | What it proves                                                                  |
|------------------------------|---------------------------------------------------------------------------------|
| `invdist_planar_jacobian.py` | The inverse-distance feature map is well-behaved at planar geometries: in-plane Jacobian has full row-rank, the out-of-plane block vanishes by reflection symmetry (the GP-predicted out-of-plane force is identically zero, which is the physical answer), and rank is recovered for any infinitesimal off-plane perturbation.  Settles the corresponding reviewer concern by symbolic computation. |

### `scripts/sympy/`

Manuscript-aligned SymPy scripts that derive or validate intermediate
results in the GP framework. These are the worked notebooks behind the
derivations in the main text and the SI appendix. Run any of them with
`python scripts/sympy/<file>.py` after installing `sympy`.

| Script                          | Topic                                                                |
|---------------------------------|----------------------------------------------------------------------|
| `t1_cartesian_se_blocks.py`     | Energy/force covariance blocks for the Cartesian SE kernel.          |
| `t3_invdist_jacobian.py`        | Closed-form inverse-distance feature Jacobian and chain-rule check.  |
| `t4_nll_gradient.py`            | Negative log marginal likelihood gradient w.r.t. hyperparameters.    |
| `t5_rff_expectation.py`         | Random-Fourier-feature expectation that recovers the SE kernel.      |
| `t7_lcb_scoring.py`             | LCB convergence rule for the inner loop.                             |
| `t8_constant_kernel.py`         | Constant-kernel block structure (vanishing in derivative blocks).    |
| `t9_dimer_gp_conditioning.py`   | GP-dimer conditioning and the rotation-translation linear algebra.   |

## Software versions

- `chemgp-core` 0.1.0 (Rust)
- eOn >= 2.12.0 (serve mode with metatomic)
- `rgpycrumbs`, `chemparseplot` (Python, editable install)
- `metatrain` for `mtt export`
- `sympy` >= 1.11 (proofs and verification scripts)
- `uv` >= 0.5 (recommended; resolves PEP-723 headers in `scripts/proofs/`)

## License

- Data and figures: CC-BY-4.0
- Code (`chemgp-core`): MIT
- PET-MAD model: see [lab-cosmo/pet-mad](https://huggingface.co/lab-cosmo/pet-mad)

## Citation

Please cite both the manuscript and this archive:

```bibtex
@article{goswami2026bayesian,
  author  = {Goswami, Rohit},
  title   = {Bayesian Optimization with Gaussian Processes to Accelerate
             Stationary Point Searches},
  journal = {ACS Physical Chemistry Au},
  year    = {2026},
  eprint  = {2603.10992},
  archivePrefix = {arXiv},
  doi     = {10.48550/arXiv.2603.10992},
}

@misc{goswami2026chemgp_data,
  author    = {Goswami, Rohit},
  title     = {Data for: Bayesian Optimization with Gaussian Processes to
               Accelerate Stationary Point Searches},
  year      = {2026},
  publisher = {Materials Cloud Archive},
  doi       = {10.24435/materialscloud:XXXX},
}
```

## Contact

Rohit Goswami, Science Institute, University of Iceland.
