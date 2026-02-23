# Nudged Elastic Band (NEB)

This tutorial covers the NEB method for finding minimum energy paths (MEPs)
between two configurations, and how GP acceleration reduces oracle calls.

## What is a Minimum Energy Path?

The MEP is the lowest-energy pathway connecting two local minima on the potential
energy surface. The highest point along the MEP is the transition state (first-order
saddle point). Finding the MEP is essential for computing reaction rates and
understanding chemical mechanisms.

## The NEB Concept

NEB represents the path as a chain of "images" (configurations) connected by
springs. The optimization minimizes the energy of each image while maintaining
even spacing:

```
  x_start ── x_2 ── x_3 ── ... ── x_{N-1} ── x_end
     (fixed)                              (fixed)
```

### Tangent Estimation

At each image, the tangent to the path is estimated using the improved method
of Henkelman & Jónsson (2000), which uses energy-weighted bisection at local
extrema to avoid path oscillations:

- **Monotonic energy increase**: Use forward tangent ``\tau^+ = R_{i+1} - R_i``
- **Monotonic energy decrease**: Use backward tangent ``\tau^- = R_i - R_{i-1}``
- **Local extremum**: Weighted average ``\tau = \Delta E_{\max} \tau^+ + \Delta E_{\min} \tau^-``

See [`path_tangent`](@ref).

### NEB Forces

The NEB force at each image combines two components:

```math
F_i = -\nabla E_i^{\perp} + F_i^{\text{spring},\parallel}
```

- ``\nabla E_i^{\perp}``: True gradient projected perpendicular to the path tangent
- ``F_i^{\text{spring},\parallel}``: Spring force projected parallel to the tangent

See [`neb_force`](@ref), [`spring_force`](@ref).

### Climbing Image

After the path partially converges, the highest-energy image switches to
"climbing image" mode:

```math
F_{\text{CI}} = -\nabla E + 2(\nabla E \cdot \hat{\tau})\hat{\tau}
```

This image moves uphill along the tangent and downhill perpendicular to it,
converging to the exact saddle point.

## Three NEB Variants

### Standard NEB ([`neb_optimize`](@ref))

Oracle-only baseline. Evaluates all images at every step. Useful as a reference
for measuring GP speedup.

```julia
result = neb_optimize(muller_brown_energy_gradient, x_start, x_end;
    config = NEBConfig(images = 5, spring_constant = 10.0))
```

### GP-NEB-AIE ([`gp_neb_aie`](@ref))

All Images Evaluated per outer iteration. The inner relaxation (many steps)
operates on the cheap GP surface. Uses warm-started GP hyperparameters and
regularized noise settings for stability (Goswami, Gunde & Jonsson 2026).

```julia
result = gp_neb_aie(oracle, x_start, x_end, kernel; config = NEBConfig())
```

### GP-NEB-OIE ([`gp_neb_oie`](@ref))

One Image Evaluated per outer iteration, selected by maximum predictive
variance. Most oracle-efficient but requires more GP training.

```julia
result = gp_neb_oie(oracle, x_start, x_end, kernel; config = NEBConfig())
```

## Example: Muller-Brown Surface

```julia
using ChemGP
using KernelFunctions

# Path between minima B and C
x_B = [0.623, 0.028]
x_C = [-0.050, 0.467]

# For 2D (non-molecular) surfaces, use KernelFunctions directly
k = SqExponentialKernel()

config = NEBConfig(
    images = 5,
    spring_constant = 10.0,
    climbing_image = true,
    conv_tol = 0.1,
    max_outer_iter = 20,
    step_size = 1e-4,
)

# Standard NEB (baseline)
result_std = neb_optimize(muller_brown_energy_gradient, x_B, x_C; config)
println("Standard NEB: $(result_std.oracle_calls) oracle calls")

# GP-NEB-AIE
result_aie = gp_neb_aie(muller_brown_energy_gradient, x_B, x_C, k; config)
println("GP-NEB-AIE: $(result_aie.oracle_calls) oracle calls")

# GP-NEB-OIE
result_oie = gp_neb_oie(muller_brown_energy_gradient, x_B, x_C, k; config)
println("GP-NEB-OIE: $(result_oie.oracle_calls) oracle calls")
```

## Parallel Oracle Evaluation

Each NEB iteration evaluates the oracle at N-2 intermediate images. These
evaluations are independent and can run in parallel when the server supports
concurrent connections.

All three NEB functions (`neb_optimize`, `gp_neb_aie`, `gp_neb_oie`) accept
either a single oracle function or a vector of oracle functions (an oracle pool).
When a pool is provided, image evaluations are dispatched across workers using
`Threads.@spawn`:

```julia
using ChemGP

# Single oracle (sequential)
result = neb_optimize(oracle, x_start, x_end; config)

# Oracle pool (parallel evaluation)
n_workers = min(Threads.nthreads(), config.images)
oracles = make_oracle_pool("localhost", 12345, atmnrs, box, n_workers)
result = neb_optimize(oracles, x_start, x_end; config)
```

Launch Julia with multiple threads for parallel evaluation:

```bash
julia -t auto --project=. examples/petmad_hcn_neb.jl
# or specify explicitly:
julia -t 8 --project=. examples/petmad_hcn_neb.jl
```

A single `Function` oracle still works -- the pool is optional. For GP-NEB
OIE, which evaluates one image per iteration, parallelism does not apply.

See also: [RPC Integration](@ref) for setting up parallel servers with
gateway mode.

## GP Training: Warm-Start and Noise Regularization

The GP-NEB variants (AIE and OIE) share a common GP training strategy
that differs from naive implementation in two important ways:

### Warm-Start

Rather than re-initializing kernel hyperparameters from scratch at each outer
iteration, the optimized hyperparameters from the previous iteration seed the
next optimization. This provides two benefits:
- Fewer optimization iterations needed (the starting point is already close)
- More stable convergence, especially for molecular kernels with many
  lengthscale parameters

### Noise Regularization

For molecular kernels (e.g., `MolInvDistSE`), the gradient-gradient block
``K_{GG}`` of the covariance matrix can be rank-deficient for small molecules
where the number of inverse-distance features is less than the number of
gradient components. The default noise settings prevent numerical instability:

| Parameter | Value | Purpose |
|:----------|:------|:--------|
| `noise_var` | 1e-6 | Energy observation noise |
| `grad_noise_var` | 1e-4 | Gradient observation noise (regularizes K\_GG) |
| `jitter` | 1e-6 | Diagonal jitter for Cholesky stability |

These are set with `fix_noise = true`, meaning they are not optimized but
treated as fixed regularization. This is critical for small systems (e.g.,
HCN with 3 atoms, 3 inverse distances, but 9 gradient components).

## Configuration

The [`NEBConfig`](@ref) struct controls all parameters:

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `images` | 5 | Number of movable images (total = images + 2) |
| `spring_constant` | 1.0 | Spring constant for elastic band |
| `climbing_image` | true | Enable climbing image |
| `ci_activation_tol` | 0.5 | Force norm to activate climbing image |
| `max_iter` | 500 | Max iterations (standard NEB or inner loop) |
| `conv_tol` | 5e-3 | Convergence on max force norm |
| `step_size` | 0.01 | Steepest descent step size |
| `gp_train_iter` | 300 | GP hyperparameter optimization iterations |
| `max_outer_iter` | 50 | Max outer iterations (GP-NEB) |
| `max_gp_points` | 0 | Cap GP training set via FPS subset (0 = all data) |
| `rff_features` | 0 | RFF feature dimension (0 = exact GP; >0 = RFF for MolInvDistSE) |
| `trust_radius` | 0.1 | Maximum distance from training data |
| `trust_metric` | `:emd` | Distance metric (`:emd`, `:max_1d_log`, `:euclidean`) |
| `atom_types` | `Int[]` | Element labels per atom for EMD (empty = all same) |
| `use_adaptive_threshold` | false | Sigmoidal trust decay with training set size |

## Random Fourier Features (RFF)

For large training sets, exact GP scales as ``O((N(D+1))^3)``. When
`rff_features > 0` and the kernel is `MolInvDistSE`, the GP-NEB pipeline
replaces the exact GP with a Random Fourier Features approximation:

1. Hyperparameters are optimized on the FPS subset (exact GP, ``O(M^3)``)
2. An RFF model is built using ALL training data (``O(N \cdot D \cdot D_\text{rff} + D_\text{rff}^3)``)
3. Predictions use the RFF model (``O(D_\text{rff} \cdot D)`` per test point)

RFF is a Bayesian linear regression in a ``D_\text{rff}``-dimensional random
feature space that approximates the kernel. The feature map is derived from
Bochner's theorem: the SE kernel's spectral density is Gaussian, so random
frequencies are sampled from ``\mathcal{N}(0, 2\theta^2 I)`` where ``\theta``
is the trained inverse lengthscale.

```julia
# Enable RFF with 200 features on an OIE run
cfg = NEBConfig(
    images = 3,
    max_gp_points = 10,   # FPS subset for hyperparameter training
    rff_features = 200,   # RFF approximation for prediction
    conv_tol = 0.3,
    verbose = true,
)

kernel = MolInvDistSE(1.0, [1.0], Float64[])
result = gp_neb_oie_naive(oracle, x_start, x_end, kernel; config = cfg)
```

RFF activates only when the full training set exceeds the FPS subset
(i.e., `npoints(td) > npoints(td_subset)`). With fewer data points than
the subset cap, the exact GP is used directly.

On the LEPS benchmark, RFF with 200 features converges in 25-26 oracle
calls with a barrier of 1.3293 eV (exact GP: 22 calls, 1.3291 eV). The
approximation introduces transient force spikes when it activates, but
these recover within a few iterations.

## Trust Region Configuration

GP-NEB uses the same EMD trust region infrastructure as OTGPD (see
[Trust Regions](@ref) for details). After inner relaxation on the GP surface
completes, each image is checked against all training data using the configured
metric. Images that exceeded the trust threshold are scaled back toward their
nearest training point before oracle evaluation. The EMD check is applied at
the outer loop boundary, not inside the inner loop, to avoid disrupting L-BFGS
curvature estimates.

```julia
# HCN -> HNC with type-aware EMD trust
cfg = NEBConfig(
    images = 8,
    trust_radius = 0.1,
    trust_metric = :emd,
    atom_types = Int[6, 7, 1],  # C, N, H
)
```

For systems with identical atoms (e.g., metal clusters), the EMD metric is
permutation-invariant and prevents the trust check from being confused by
atom relabeling. For small unique-atom systems, `:euclidean` or `:max_1d_log`
are equally effective and slightly faster.

The adaptive threshold option tightens the trust region as more training data
accumulates:

```julia
cfg = NEBConfig(
    trust_radius = 0.1,
    trust_metric = :emd,
    atom_types = Int[6, 7, 1],
    use_adaptive_threshold = true,
    adaptive_t_min = 0.15,
    adaptive_delta_t = 0.35,
    adaptive_n_half = 50,
)
```

## Loading Structures from Files

For molecular systems, hardcoding coordinates is impractical. ChemGP integrates
with [AtomsBase.jl](https://github.com/JuliaMolSim/AtomsBase.jl) via a package
extension. Load structures from extended XYZ, POSCAR, or other formats using
[AtomsIO.jl](https://github.com/mfherbst/AtomsIO.jl), then convert to
ChemGP's flat-vector convention:

```julia
using ChemGP
using AtomsBase, AtomsIO

# Load reactant and product from extxyz files
sys_r = load_system("reactant.extxyz")
sys_p = load_system("product.extxyz")

(; positions = X_r, atomic_numbers, box) = chemgp_coords(sys_r)
(; positions = X_p) = chemgp_coords(sys_p)

# Use X_r, X_p as NEB endpoints
result = neb_optimize(oracle, X_r, X_p; config = NEBConfig())
```

After optimization, convert the NEB path back to AtomsBase systems for
visualization or further computation:

```julia
trajectory = atomsbase_neb_trajectory(result, atomic_numbers, box)
```

See `examples/petmad_hcn_neb.jl` for a complete worked example using PET-MAD
over RPC with AtomsIO-loaded HCN/HNC structures.

## Further Reading

- Goswami, Gunde & Jónsson (2026) [arXiv:2601.12630](https://arxiv.org/abs/2601.12630) — enhanced CI-NEB with Hessian eigenmode alignment
- Goswami et al., *J. Chem. Theory Comput.* (2025) [doi:10.1021/acs.jctc.5c00866](https://doi.org/10.1021/acs.jctc.5c00866) — efficient GP-accelerated saddle point searches
- Goswami, *Efficient exploration of chemical kinetics* (2025) [arXiv:2510.21368](https://arxiv.org/abs/2510.21368) — thesis
- Koistinen et al., *J. Chem. Phys.* 147, 152720 (2017) — GP-NEB
- Henkelman, Uberuaga & Jonsson, *J. Chem. Phys.* 113, 9901 (2000) — climbing image NEB

## Next Steps

- [GP-Guided Minimization](@ref): Understand GP surrogate optimization
- [Dimer Method](@ref): Transition state search without knowing endpoints
- [References](@ref references)
