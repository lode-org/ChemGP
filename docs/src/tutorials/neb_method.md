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
    config = NEBConfig(n_images = 7, spring_constant = 10.0))
```

### GP-NEB-AIE ([`gp_neb_aie`](@ref))

All Images Evaluated per outer iteration. The inner relaxation (many steps)
operates on the cheap GP surface.

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
    n_images = 7,
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

## Configuration

The [`NEBConfig`](@ref) struct controls all parameters:

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `n_images` | 7 | Number of images (including endpoints) |
| `spring_constant` | 1.0 | Spring constant for elastic band |
| `climbing_image` | true | Enable climbing image |
| `ci_activation_tol` | 0.5 | Force norm to activate climbing image |
| `max_iter` | 500 | Max iterations (standard NEB or inner loop) |
| `conv_tol` | 5e-3 | Convergence on max force norm |
| `step_size` | 0.01 | Steepest descent step size |
| `gp_train_iter` | 300 | GP hyperparameter optimization iterations |
| `max_outer_iter` | 50 | Max outer iterations (GP-NEB) |

## Next Steps

- [GP-Guided Minimization](@ref): Understand GP surrogate optimization
- [Dimer Method](@ref): Transition state search without knowing endpoints
- [References](@ref references)
