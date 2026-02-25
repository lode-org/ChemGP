# Tutorial figures for GPR review paper

Publication-quality figures for the GP-accelerated saddle point search
tutorial/review paper.

## Architecture

Figures are split into **generators** (data production) and **plotters** (PDF
rendering), connected via HDF5 files. A Python orchestrator manages the
lifecycle and provides real-time metrics via a TCP JSONL socket.

```
Julia gen script
  |-- TCP socket (metrics) --> figure_runner.py --> stdout + JSONL log
  |-- HDF5 file (all data) --> Julia plot script --> PDF
```

- **Generators** (`generators/gen_*.jl`) run ChemGP optimizations, write
  results to HDF5. Send lightweight iteration metrics through TCP socket.
- **Plotters** (`plotters/plot_*.jl`) read HDF5 files, produce PDFs. No
  ChemGP dependency, no socket communication.
- **Orchestrator** (`figure_runner.py`) creates TCP listener, launches gen
  script, renders human-readable stdout + writes JSONL log, then launches
  plot script.
- **Kept-as-is** scripts (`mb_pes.jl`, `leps_contour.jl`, `hcn_landscape.sh`)
  are trivially fast analytic PES plots that don't need the split.

### Directory structure

```
scripts/figures/tutorial/
  generators/
    common_data.jl           # HDF5 write helpers, socket emit, eval_grid
    gen_leps_minimize.jl     # ... (15 gen scripts)
  plotters/
    common_plot.jl           # Theme, palette, HDF5 read helpers, save_figure
    plot_leps_minimize.jl    # ... (16 plot scripts incl. plot_rff_combined)
  figure_runner.py           # Python orchestrator (socket + subprocess mgmt)
  common.jl                  # Backward-compat for kept-as-is scripts
  mb_pes.jl                  # Kept as-is (pure analytic PES plot)
  leps_contour.jl            # Kept as-is (pure analytic PES plot)
  hcn_landscape.sh           # Kept as-is (shell+Python wrapper)
  Project.toml
  output/
    *.h5                     # HDF5 data files from generators
    *.jsonl                  # JSONL metric logs from orchestrator
    *.pdf                    # Figures from plotters
```

### HDF5 file structure

Generators write data to `output/{stem}.h5` using these groups:

| Group | Purpose | Writer helper |
|-------|---------|---------------|
| `/table` | Tabular data (convergence curves, RFF metrics) | `h5_write_table(path, name, Dict(...))` |
| `/grids/{name}` | 2D matrices (PES energy, GP variance) | `h5_write_grid(path, name, M; x_range, y_range)` |
| `/paths/{name}` | Ordered point sequences (NEB paths) | `h5_write_path(path, name; x=..., y=...)` |
| `/points/{name}` | Unordered point sets (minima, training pts) | `h5_write_points(path, name; x=..., y=...)` |
| root attrs | Scalar metadata (convergence status, thresholds) | `h5_write_metadata(path; key=val, ...)` |

Grid datasets store `x_range`, `y_range`, `x_length`, `y_length` as HDF5
attributes so plotters can reconstruct the exact range objects.

### Socket metrics protocol

The TCP socket carries only lightweight iteration metrics as JSONL (one JSON
object per line). Same format as ChemGP's `machine_output` protocol:

- **Minimize iterations**: `{i, E, F, oc, tp, t, sv, ls, td, gate}`
- **NEB/dimer iterations**: `{i, max_force, oracle_calls, method, ...}`
- **Summary lines**: `{status, oc, E, F, iters}`

### Environment variables

Set by `figure_runner.py` for child processes:

| Variable | Description |
|----------|-------------|
| `CHEMGP_FIG_PORT` | TCP port for metrics socket (0 = disabled) |
| `CHEMGP_FIG_OUTPUT` | Output directory path |
| `CHEMGP_FIG_STEM` | Output file stem (e.g., `leps_minimize`) |
| `CHEMGP_FIG_H5` | Explicit HDF5 path (overrides stem-based default) |

## Test surfaces

- **Muller-Brown (2D)**: Standard analytic surface with 3 minima and 2 saddles.
  Uses plain `SqExponentialKernel` (Cartesian coordinates). Suited for
  illustrating GP concepts (variance, hyperparameters, trust regions).

- **LEPS (9D, 3 atoms)**: Analytic triatomic potential (H + H2 collinear
  exchange). Uses `MolInvDistSE` with 3 inverse distances. Demonstrates
  NEB, minimization, RFF approximation, and FPS selection.

- **PET-MAD (27D, 9 atoms)**: Real ML universal potential via RPC. Uses
  `MolInvDistSE` with RFF approximation (`rff_features=200`). Requires a
  running `potserv` server. Test system: 9-atom organic fragment (2C, 1O,
  2N, 4H) from the eOn ewNEB benchmark set (`data/system100/`).

## Setup

### Julia dependencies

```bash
pixi run -e figures fig-install
```

### Python dependencies (landscape plots only)

```bash
pixi run -e figures fig-pyinstall
```

Requires `rgpycrumbs` and `chemparseplot` installed from local checkouts.

## Running

### Batch tasks

```bash
pixi run -e figures fig-tier1   # MB + LEPS basics (no RPC)
pixi run -e figures fig-tier2   # MB + LEPS advanced (no RPC)
pixi run -e figures fig-tier3   # PET-MAD + HCN (RPC required)
```

### Muller-Brown figures (no server needed)

```bash
pixi run -e figures fig-mb-pes     # MB-1: PES contour
pixi run -e figures fig-mb-neb     # MB-2: NEB path
pixi run -e figures fig-mb-gp      # MB-3: GP progression
pixi run -e figures fig-mb-var     # MB-4: Variance overlay
pixi run -e figures fig-mb-hyp     # MB-5: Hyperparameter sensitivity
pixi run -e figures fig-mb-dimer   # MB-6: GP-dimer convergence
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

### PET-MAD and HCN figures (require RPC server)

Start the potential server first:

```bash
./potserv 12345 pet-mad
```

Then:

```bash
pixi run -e figures fig-petmad-min   # REAL-3: PET-MAD GP minimization
pixi run -e figures fig-petmad-rff   # REAL-4: PET-MAD RFF quality
pixi run -e figures fig-rff-combined # RFF-5: Combined LEPS + PET-MAD RFF
pixi run -e figures fig-hcn          # REAL-1: HCN energy profile
pixi run -e figures fig-hcn-land     # REAL-1B: HCN 2D landscape (Python)
pixi run -e figures fig-hcn-conv     # REAL-2: HCN convergence comparison
```

### Generation-only and plot-only

Append `-gen-` or `-plot-` to the task name for partial runs:

```bash
pixi run -e figures fig-gen-leps-min    # Data generation only
pixi run -e figures fig-plot-leps-min   # Plot only (reuse existing HDF5)
```

### Direct orchestrator usage

```bash
python3 scripts/figures/tutorial/figure_runner.py \
  --gen generators/gen_leps_minimize.jl \
  --plot plotters/plot_leps_minimize.jl \
  [--port 0]              # 0 = ephemeral (default)
  [--output-dir output/]
  [--skip-gen]            # reuse existing HDF5
  [--skip-plot]           # generate data only
```

## Output

PDFs and HDF5 data are written to `scripts/figures/tutorial/output/`.
Override with `CHEMGP_FIG_OUTPUT` environment variable.

## Figure inventory

| File | ID | Generator | Plotter | Section |
|------|----|-----------|---------|---------|
| `mb_pes.pdf` | MB-1 | `mb_pes.jl` (kept-as-is) | -- | Introduction |
| `mb_neb.pdf` | MB-2 | `gen_mb_neb.jl` | `plot_mb_neb.jl` | GP-NEB |
| `mb_gp_progression.pdf` | MB-3 | `gen_mb_gp.jl` | `plot_mb_gp.jl` | GPR |
| `mb_variance.pdf` | MB-4 | `gen_mb_variance.jl` | `plot_mb_variance.jl` | GPR |
| `mb_hyperparams.pdf` | MB-5 | `gen_mb_hyperparams.jl` | `plot_mb_hyperparams.jl` | GPR |
| `mb_dimer_convergence.pdf` | MB-6 | `gen_mb_dimer.jl` | `plot_mb_dimer.jl` | GP-Dimer |
| `mb_trust_region.pdf` | MB-7 | `gen_mb_trust.jl` | `plot_mb_trust.jl` | GP-Dimer |
| `leps_contour.pdf` | LEPS-1 | `leps_contour.jl` (kept-as-is) | -- | GP-NEB |
| `leps_neb.pdf` | LEPS-2 | `gen_leps_neb.jl` | `plot_leps_neb.jl` | GP-NEB |
| `leps_aie_oie.pdf` | LEPS-3 | `gen_leps_aie_oie.jl` | `plot_leps_aie_oie.jl` | GP-NEB |
| `leps_fps.pdf` | LEPS-4 | `gen_leps_fps.jl` | `plot_leps_fps.jl` | OT-GP |
| `leps_minimize_convergence.pdf` | LEPS-5 | `gen_leps_minimize.jl` | `plot_leps_minimize.jl` | GP-MIN |
| `leps_rff_quality.pdf` | LEPS-6 | `gen_leps_rff.jl` | `plot_leps_rff.jl` | RFF |
| `rff_quality_combined.pdf` | RFF-5 | -- | `plot_rff_combined.jl` | RFF |
| `petmad_minimize_convergence.pdf` | REAL-3 | `gen_petmad_minimize.jl` | `plot_petmad_minimize.jl` | GP-MIN |
| `petmad_rff_quality.pdf` | REAL-4 | `gen_petmad_rff.jl` | `plot_petmad_rff.jl` | Examples |
| `hcn_neb_profile.pdf` | REAL-1 | `gen_hcn_neb.jl` | `plot_hcn_neb.jl` | Examples |
| `hcn_landscape.pdf` | REAL-1B | `hcn_landscape.sh` (kept-as-is) | -- | Examples |
| `hcn_convergence.pdf` | REAL-2 | `gen_hcn_convergence.jl` | `plot_hcn_convergence.jl` | Examples |

## How to add a new figure

1. Create `generators/gen_<name>.jl`:
   - `include(joinpath(@__DIR__, "common_data.jl"))`
   - Run optimization, write results via `h5_write_*` helpers
   - Optionally use `machine_output="localhost:$(FIG_PORT)"` for metrics
2. Create `plotters/plot_<name>.jl`:
   - `include(joinpath(@__DIR__, "common_plot.jl"))`
   - Read HDF5 via `h5_read_*` helpers
   - Plot with CairoMakie, call `save_figure(fig, "<name>")`
3. Add a pixi task in `pixi.toml` under `[feature.figures.tasks]`:
   ```toml
   fig-<name> = { cmd = "python3 scripts/figures/tutorial/figure_runner.py --gen generators/gen_<name>.jl --plot plotters/plot_<name>.jl", depends-on = ["fig-install"], description = "..." }
   ```
4. Add it to the appropriate `fig-tier*` batch task

## MLflow live tracking

All GP optimizers support an `on_step` callback for live monitoring. The
`ChemGPMLflowExt` package extension provides an MLflow REST client that logs
metrics automatically when `HTTP` and `JSON3` are loaded.

### Quick start

1. Start the MLflow tracking server:

```bash
pixi run mlflow-ui
```

2. Open <http://localhost:5000> in a browser.

3. Use the MLflow callback in your script:

```julia
using ChemGP, HTTP, JSON3

tracker = MLflowTracker(; uri="http://localhost:5000", experiment="ChemGP")

log_params!(tracker, Dict(
    "trust_radius" => "0.15",
    "conv_tol" => "0.01",
    "kernel" => "MolInvDistSE",
))

cb = mlflow_callback(tracker, :minimize)
result = gp_minimize(oracle, x_init, kernel; config=cfg, on_step=cb)
finish_run!(tracker; status=result.converged ? "FINISHED" : "FAILED")
```

### on_step callback protocol

All optimizers accept `on_step::Union{Function,Nothing}=nothing`. Return
`:stop` from the callback to trigger early termination (`USER_CALLBACK`
stop reason).

- **minimize/dimer**: `on_step(info::Dict{String,Any})` with keys
  `"step"`, `"energy"`, `"max_force"`, `"oracle_calls"`, `"training_points"`.
- **NEB variants**: `on_step(path::NEBPath, iter::Int)`.

### StopReason

All result structs include a `stop_reason::StopReason` field:

- `CONVERGED` -- convergence criterion met
- `MAX_ITERATIONS` -- iteration limit reached
- `ORACLE_CAP` -- oracle call budget exhausted (minimize only)
- `FORCE_STAGNATION` -- force metric unchanged for 3 consecutive steps
- `USER_CALLBACK` -- `on_step` returned `:stop`

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
