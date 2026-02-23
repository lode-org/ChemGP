# Tutorial figures for GPR review paper

Publication-quality figures for the GP-accelerated saddle point search tutorial/review paper.

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

### Python dependencies (REAL-* landscape plots only)

```bash
pixi run -e figures fig-pyinstall
```

Requires `rgpycrumbs` and `chemparseplot` installed from local checkouts.

## Running

### Individual figures

```bash
pixi run -e figures fig-mb-pes     # MB-1: PES contour
pixi run -e figures fig-mb-neb     # MB-2: NEB path
pixi run -e figures fig-mb-gp      # MB-3: GP progression
pixi run -e figures fig-mb-var     # MB-4: Variance heatmap
pixi run -e figures fig-mb-hyp     # MB-5: Hyperparameter sensitivity
pixi run -e figures fig-leps-pes   # LEPS-1: 2D contour
pixi run -e figures fig-leps-neb   # LEPS-2: NEB path
pixi run -e figures fig-mb-dimer   # MB-6: GP-dimer convergence
pixi run -e figures fig-mb-trust   # MB-7: Trust region
pixi run -e figures fig-leps-aie   # LEPS-3: AIE vs OIE
pixi run -e figures fig-leps-fps   # LEPS-4: FPS selection
```

### Batch

```bash
pixi run -e figures fig-tier1      # All P1 (MB + LEPS, no RPC needed)
pixi run -e figures fig-tier2      # All P2 (no RPC needed)
```

### REAL-* figures (require PET-MAD RPC server)

Start the potential server first:

```bash
./potserv 12345 pet-mad
```

Then:

```bash
pixi run -e figures fig-hcn        # REAL-1: HCN energy profile
pixi run -e figures fig-hcn-land   # REAL-1B: HCN 2D landscape (Python)
pixi run -e figures fig-hcn-conv   # REAL-2: HCN convergence comparison
```

HCN scripts cache results to `output/hcn_cache/` so the landscape plot can
be regenerated without rerunning the NEB.

## Output

PDFs are written to `scripts/figures/tutorial/output/`. Override with
`CHEMGP_FIG_OUTPUT` environment variable.

CSV convergence data is exported alongside PDFs for optional R/BRMS analysis.

## Figure inventory

| File | ID | Paper section |
|------|----|---------------|
| `mb_pes.pdf` | MB-1 | Introduction |
| `mb_neb.pdf` | MB-2 | GP-NEB |
| `mb_gp_progression.pdf` | MB-3 | GPR |
| `mb_variance.pdf` | MB-4 | GPR |
| `mb_hyperparams.pdf` | MB-5 | GPR |
| `mb_dimer_convergence.pdf` | MB-6 | GP-Dimer |
| `mb_trust_region.pdf` | MB-7 | GP-Dimer |
| `leps_contour.pdf` | LEPS-1 | GP-NEB |
| `leps_neb.pdf` | LEPS-2 | GP-NEB |
| `leps_aie_oie.pdf` | LEPS-3 | GP-NEB |
| `leps_fps.pdf` | LEPS-4 | OT-GP |
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
