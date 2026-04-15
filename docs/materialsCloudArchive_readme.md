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
docs/source/_static/figures/         # 300 DPI PNGs for the tutorial docs
models/                              # PET-MAD-XS v1.5.0 .ckpt and .pt
scripts/figures/README.md            # pipeline documentation
scripts/figures/tutorial/output/     # JSONL, HDF5, PDF, CON, DAT
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

## Software versions

- `chemgp-core` 0.1.0 (Rust)
- eOn >= 2.12.0 (serve mode with metatomic)
- `rgpycrumbs`, `chemparseplot` (Python, editable install)
- `metatrain` for `mtt export`

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
