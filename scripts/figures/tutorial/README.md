# Tutorial figures for GPR review paper

Publication-quality figures for the GP-accelerated saddle point search
tutorial/review paper.

## Architecture

A Snakemake pipeline orchestrates four stages, from Rust example binaries
through to PNGs for Sphinx documentation.

```
Rust examples  -->  JSONL  -->  HDF5  -->  PDF  -->  PNG
  (Stage 1)        (Stage 2)     (Stage 3)    (Stage 4)
```

- **Stage 1** (`run_standalone`, `run_rpc`): Rust examples from
  `crates/chemgp-core/examples/` produce JSONL files at the repo root.
  Standalone examples need no server; RPC examples require eOn serve on
  port 12345.
- **Stage 2** (`h5_standalone`, `h5_rpc`): Python converter
  (`scripts/jsonl_to_h5.py`) reads JSONL and writes HDF5 to `output/`.
- **Stage 3** (`pdfs_standalone`, `pdfs_rpc`, `hcn_landscape`):
  `rgpycrumbs.chemgp.plt_gp batch` reads HDF5 and produces PDFs in
  `output/`, configured via `figures.toml`.
- **Stage 4** (`pdf_to_png`): `pdftoppm` converts PDFs to 300 DPI PNGs
  in `docs/source/_static/figures/`.

### Directory structure

```
ChemGP/
  crates/chemgp-core/examples/    # Rust examples (12 standalone + 4 RPC)
  scripts/
    figures/
      Snakefile                   # 4-stage pipeline orchestration
      tutorial/
        figures.toml              # Plot config: HDF5 -> PDF mappings
        output/                   # HDF5, PDF intermediates
        README.md                 # This file
    jsonl_to_h5.py                # JSONL -> HDF5 converter
  docs/source/_static/figures/    # Final PNGs for Sphinx
  pixi.toml                       # Task definitions (dev, rpc envs)
```

### Snakemake rules

| Rule | Stage | Description |
|------|-------|-------------|
| `run_standalone` | 1 | Run standalone Rust examples (no server) |
| `run_rpc` | 1 | Run RPC examples (checks server first) |
| `scattered_sideeffect` | 1 | Side-effect from `mb_gp_quality` |
| `h5_standalone` | 2 | Convert standalone JSONL to HDF5 |
| `h5_rpc` | 2 | Convert all JSONL (standalone + RPC) to HDF5 |
| `pdfs_standalone` | 3 | Generate standalone PDFs via rgpycrumbs |
| `pdfs_rpc` | 3 | Generate RPC PDFs via rgpycrumbs |
| `hcn_landscape` | 3b | NEB landscape from .con/.dat files |
| `pdf_to_png` | 4 | Convert PDFs to PNGs for Sphinx |

### Diagnostics

Stderr from Rust examples is captured to `.snakemake/logs/{jsonl}.log`.
If a rule fails, check the log for the panic message or runtime error.

RPC rules verify the server is reachable before running. If port 12345 is
down, the rule fails immediately with a clear message.

### HDF5 file structure

The converter writes data to `output/{stem}.h5` using these groups:

| Group | Purpose | Converter function |
|-------|---------|-------------------|
| `/table` | Tabular data (convergence curves, RFF metrics) | `h5_write_table()` |
| `/grids/{name}` | 2D matrices (PES energy, GP variance) | `h5_write_grid()` |
| `/paths/{name}` | Ordered point sequences (NEB paths) | `h5_write_path()` |
| `/points/{name}` | Unordered point sets (minima, training pts) | `h5_write_points()` |
| root attrs | Scalar metadata (convergence status, thresholds) | via `h5py` attrs |

Grid datasets store `x_range`, `y_range`, `x_length`, `y_length` as HDF5
attributes so plotters reconstruct the exact axis ranges.

## Test surfaces

- **Muller-Brown (2D)**: Standard analytic surface with 3 minima and 2
  saddles. Uses `CartesianSE` kernel. Suited for illustrating GP concepts
  (variance, hyperparameters, trust regions).

- **LEPS (9D, 3 atoms)**: Analytic triatomic potential (H + H2 collinear
  exchange). Uses `MolInvDistSE` with 3 inverse distances. Demonstrates
  NEB, minimization, RFF approximation, and FPS selection.

- **PET-MAD (27D, 9 atoms)**: Real ML universal potential via RPC. Uses
  `MolInvDistSE` with RFF approximation. Requires a running eOn serve
  instance. Test system: 9-atom organic fragment (2C, 1O, 2N, 4H) from
  the eOn ewNEB benchmark set (`data/system100/`).

## Setup

### pixi environments

- **dev**: Snakemake, Python plotting stack (rgpycrumbs, chemparseplot,
  matplotlib, h5py), Rust toolchain
- **rpc**: eOn (eonclient), Rust toolchain

Both `rgpycrumbs` and `chemparseplot` are installed as editable packages
from `external/`.

## Running

### Full pipeline (all figures)

Requires the PET-MAD RPC server running in a separate terminal:

```bash
# Terminal 1: start the potential server
pixi run -e rpc serve-petmad

# Terminal 2: build all figures
pixi run -e dev figures-all
```

### Standalone figures only (no server needed)

```bash
pixi run -e dev figures
```

### RPC figures only (server required)

```bash
pixi run -e rpc serve-petmad   # terminal 1
pixi run -e dev figures-rpc    # terminal 2
```

### Individual examples

Run Rust examples directly:

```bash
# Standalone (no server)
cargo run --release --example leps_neb

# RPC (server required)
cargo run --release --features io,rgpot --example system100_neb -- --method oie
```

### pixi task reference

| Task | Environment | Description |
|------|-------------|-------------|
| `figures` | dev | Build standalone figures (no server) |
| `figures-rpc` | dev | Build RPC figures (server required) |
| `figures-all` | dev | Build all figures (server required) |
| `examples` | dev | Run all 12 standalone Rust examples |
| `jsonl2h5` | dev | Convert JSONL to HDF5 (depends on examples) |
| `serve-petmad` | rpc | Start PET-MAD RPC server on port 12345 |
| `serve-lj` | rpc | Start Lennard-Jones RPC server on port 12345 |
| `rpc-examples` | rpc | Run all 4 RPC Rust examples |
| `rpc-smoke` | rpc | Quick RPC connectivity test |

## Example registry

### Standalone examples (12, no server)

| Example | Output JSONL |
|---------|-------------|
| `leps_minimize` | `leps_minimize_comparison.jsonl` |
| `leps_dimer` | `leps_dimer_comparison.jsonl` |
| `leps_neb` | `leps_neb_comparison.jsonl` |
| `leps_rff_quality` | `leps_rff_quality.jsonl` |
| `leps_fps` | `leps_fps.jsonl` |
| `leps_nll_landscape` | `leps_nll_landscape.jsonl` |
| `mb_minimize` | `mb_minimize_comparison.jsonl` |
| `mb_dimer` | `mb_dimer_comparison.jsonl` |
| `mb_neb` | `mb_neb.jsonl` |
| `mb_gp_quality` | `mb_gp_quality.jsonl` (+ `mb_gp_scattered.jsonl`) |
| `mb_hyperparams` | `mb_hyperparams.jsonl` |
| `mb_trust` | `mb_trust.jsonl` |

### RPC examples (4, server required)

| Example | Features | Output JSONL |
|---------|----------|-------------|
| `petmad_minimize` | `rgpot` | `petmad_minimize_comparison.jsonl` |
| `rpc_dimer` | `io,rgpot` | `rpc_dimer.jsonl` |
| `petmad_rff_quality` | `io,rgpot` | `petmad_rff_quality.jsonl` |
| `system100_neb` | `io,rgpot` | `system100_neb_comparison.jsonl` |

## Figure inventory

### Standalone PDFs (12)

| PDF | Source HDF5 | Plot type |
|-----|-------------|-----------|
| `leps_minimize_convergence.pdf` | `leps_minimize.h5` | convergence |
| `leps_neb.pdf` | `leps_neb.h5` | surface |
| `leps_aie_oie.pdf` | `leps_aie_oie.h5` | convergence |
| `leps_rff_quality.pdf` | `leps_rff.h5` | rff |
| `leps_fps.pdf` | `leps_fps.h5` | fps |
| `leps_nll_landscape.pdf` | `leps_nll.h5` | nll |
| `mb_neb.pdf` | `mb_neb.h5` | surface |
| `mb_gp_progression.pdf` | `mb_gp.h5` | quality |
| `mb_gp_progression_scattered.pdf` | `mb_gp_scattered.h5` | quality |
| `mb_hyperparams.pdf` | `mb_hyperparams.h5` | sensitivity |
| `mb_trust_region.pdf` | `mb_trust.h5` | trust |
| `mb_variance.pdf` | `mb_variance.h5` | variance |

### RPC PDFs (5)

| PDF | Source HDF5 | Plot type |
|-----|-------------|-----------|
| `petmad_minimize_convergence.pdf` | `petmad_minimize.h5` | convergence |
| `petmad_rff_quality.pdf` | `petmad_rff.h5` | rff |
| `rpc_dimer_convergence.pdf` | `rpc_dimer.h5` | convergence |
| `system100_convergence.pdf` | `system100_convergence.h5` | convergence |
| `system100_neb_profile.pdf` | `system100_neb.h5` | profile |

### Landscape PDFs (1)

| PDF | Source | Plot type |
|-----|--------|-----------|
| `system100_neb_landscape.pdf` | `system100_neb_oie_enh.dat` + `system100_neb_path_oie_enh.con` | landscape |

## How to add a new figure

1. Create a Rust example in `crates/chemgp-core/examples/<name>.rs`
   that writes JSONL output to the repo root.
2. Add a converter section in `scripts/jsonl_to_h5.py` to transform the
   JSONL into HDF5 with the appropriate group structure.
3. Add a plot entry in `scripts/figures/tutorial/figures.toml` mapping
   the HDF5 to a PDF with the desired plot type.
4. Register the example in the Snakefile:
   - Standalone: add to the `STANDALONE` dict
   - RPC: add to the `RPC` dict (with features and extra args)
5. Add the H5 and PDF filenames to the appropriate lists (`H5_STANDALONE`
   or `H5_RPC`, `PDF_STANDALONE` or `PDF_RPC`).
6. Run `pixi run -e dev figures` (or `figures-all`) to verify.

## Styling

All figures use the Ruhi color palette (coral, sunshine, teal, sky,
magenta) with Jost font. Python-side plots use `--theme ruhi` flag in
`rgpycrumbs`. Figure dimensions target ACS 2-column format (~3.25in
single column).

### Font requirements

The Jost font must be available to matplotlib. Install via:

```bash
# Arch Linux
pacman -S otf-jost

# Or download from Google Fonts
```
