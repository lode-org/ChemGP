# Tutorial figures for GPR review paper

Publication-quality figures for the GP-accelerated saddle point search tutorial/review paper.

## Test surfaces

The scripts use three classes of test surface at increasing molecular complexity:

- **Muller-Brown (2D)**: Standard analytic surface with 3 minima and 2 saddles.
  Uses plain `SqExponentialKernel` (Cartesian coordinates). Suited for
  illustrating GP concepts (variance, hyperparameters, trust regions, PES
  topology). Not suited for NEB/dimer convergence because `MolInvDistSE`
  (inverse interatomic distance features) is the relevant molecular kernel
  and requires actual atomic coordinates.

- **LEPS (9D, 3 atoms)**: Analytic triatomic potential (H + H2 collinear
  exchange). Uses `MolInvDistSE` with 3 inverse distances. Demonstrates
  NEB, minimization, RFF approximation, and FPS selection without needing
  an external server.

- **PET-MAD (27D, 9 atoms)**: Real ML universal potential via RPC. Uses
  `MolInvDistSE` with RFF approximation (`rff_features=200`) because
  exact GP is prohibitively expensive at this dimension. Requires a running
  `potserv` or `eonclient` server. Test system: 9-atom organic fragment
  (2C, 1O, 2N, 4H) from the eOn ewNEB benchmark set (`data/system100/`).

## Setup

### Julia dependencies

```bash
pixi run -e figures fig-install
```

Or manually:

```bash
julia --project=scripts/figures/tutorial -e \
  'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
```

### Python dependencies (landscape plots only)

```bash
pixi run -e figures fig-pyinstall
```

Requires `rgpycrumbs` and `chemparseplot` installed from local checkouts.

## Running

### Muller-Brown figures (no server needed)

```bash
pixi run -e figures fig-mb-pes     # MB-1: PES contour
pixi run -e figures fig-mb-gp      # MB-3: GP progression
pixi run -e figures fig-mb-var     # MB-4: Variance overlay with crosshatch
pixi run -e figures fig-mb-hyp     # MB-5: Hyperparameter sensitivity
pixi run -e figures fig-mb-trust   # MB-7: Trust region geometry
```

### LEPS figures (no server needed)

```bash
pixi run -e figures fig-leps-pes   # LEPS-1: 2D contour
pixi run -e figures fig-leps-neb   # LEPS-2: NEB path
pixi run -e figures fig-leps-aie   # LEPS-3: AIE vs OIE convergence
pixi run -e figures fig-leps-fps   # LEPS-4: FPS selection
pixi run -e figures fig-leps-min   # LEPS-5: GP vs classical minimization
pixi run -e figures fig-leps-rff   # LEPS-6: RFF approximation quality
```

### Batch (no server needed)

```bash
pixi run -e figures fig-tier1      # All MB + LEPS figures
```

### PET-MAD and HCN figures (require RPC server)

Start the potential server first:

```bash
./potserv 12345 pet-mad
```

Then:

```bash
pixi run -e figures fig-petmad-min   # PETMAD-1: GP minimization convergence
pixi run -e figures fig-petmad-rff   # PETMAD-2: RFF quality
pixi run -e figures fig-rff-combined # RFF-1: Combined LEPS + PET-MAD RFF
pixi run -e figures fig-hcn          # REAL-1: HCN energy profile
pixi run -e figures fig-hcn-land     # REAL-1B: HCN 2D landscape (Python)
pixi run -e figures fig-hcn-conv     # REAL-2: HCN convergence comparison
```

HCN and PET-MAD scripts cache results so figures can be regenerated without
rerunning the optimization.

## Output

PDFs are written to `scripts/figures/tutorial/output/`. Override with
`CHEMGP_FIG_OUTPUT` environment variable.

CSV convergence data is exported alongside PDFs for optional R/BRMS analysis.

## Figure inventory

| File | ID | Paper section |
|------|----|---------------|
| `mb_pes.pdf` | MB-1 | Introduction |
| `mb_gp_progression.pdf` | MB-3 | GPR |
| `mb_variance.pdf` | MB-4 | GPR |
| `mb_hyperparams.pdf` | MB-5 | GPR |
| `mb_trust_region.pdf` | MB-7 | GP-Dimer |
| `leps_contour.pdf` | LEPS-1 | GP-NEB |
| `leps_neb.pdf` | LEPS-2 | GP-NEB |
| `leps_aie_oie.pdf` | LEPS-3 | GP-NEB |
| `leps_fps.pdf` | LEPS-4 | OT-GP |
| `leps_minimize_convergence.pdf` | LEPS-5 | GP-MIN |
| `leps_rff_quality.pdf` | LEPS-6 | RFF |
| `rff_quality_combined.pdf` | RFF-1 | RFF |
| `petmad_minimize_convergence.pdf` | PETMAD-1 | GP-MIN |
| `petmad_rff_quality.pdf` | PETMAD-2 | Examples |
| `hcn_neb_profile.pdf` | REAL-1 | Examples |
| `hcn_landscape.pdf` | REAL-1B | Examples |
| `hcn_convergence.pdf` | REAL-2 | Examples |

## Styling

All figures use the Ruhi color palette (coral, sunshine, teal, sky, magenta)
with Jost font. Python-side plots use `--theme ruhi` flag in `rgpycrumbs`.
Figure dimensions target ACS 2-column format (~3.25in single column).

## Font requirements

The Jost font must be installed system-wide or available to CairoMakie.
Falls back to sans-serif if missing. Install via:

```bash
# Arch Linux
pacman -S otf-jost

# Or download from Google Fonts
```
