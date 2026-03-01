# ChemGP

Gaussian Process accelerated optimization for computational chemistry.

## Overview

ChemGP provides GP-surrogate methods that reduce the number of expensive
electronic structure evaluations (oracle calls) needed for geometry
optimization, saddle point search, and minimum energy path finding.

The core library (`chemgp-core`) is written in Rust for performance and
reproducibility. Python bindings (`chemgp`) expose the full API for
scripting and integration with existing workflows.

### Methods

Minimization
: GP-guided local minimization with FPS subset selection, EMD trust
  regions, and LCB exploration. Converges in ~15 oracle calls where
  gradient descent needs ~200 on LEPS.

Dimer (saddle point search)
: GP-accelerated dimer method with L-BFGS translation. Finds transition
  states in ~13 oracle calls vs ~45 for the standard dimer.

NEB (minimum energy path)
: All-image evaluation (AIE) and one-image evaluation (OIE) variants
  with per-bead FPS subset selection and RFF approximation for scalable
  inner relaxation. OIE converges in ~49 oracle calls vs ~127 standard.

OTGPD (optimal transport GP dimer)
: Adaptive threshold GP dimer with HOD training data management.
  Matches GP-Dimer efficiency (~13 calls) with automatic convergence
  threshold adjustment.

## Getting started

```{toctree}
:maxdepth: 1
:caption: Getting Started

quickstart
installation
```

## Tutorials

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/gp_basics
tutorials/minimization
tutorials/dimer_method
tutorials/neb_method
tutorials/otgpd
```

## Reference

```{toctree}
:maxdepth: 2
:caption: Reference

reference/architecture
reference/kernel_design
reference/trust_regions
api/index
```
