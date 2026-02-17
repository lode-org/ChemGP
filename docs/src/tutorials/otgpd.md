# OTGPD — Optimal Transport GP Dimer

This tutorial covers the OTGPD algorithm, the full production-grade GP-guided
dimer method. It builds on the [GP-Dimer Saddle Point Search](@ref) by adding adaptive
convergence thresholds, initial rotation, and data management.

## From Basic GP-Dimer to OTGPD

The basic [`gp_dimer`](@ref) is a pedagogical implementation that demonstrates
the core idea: train a GP on accumulated oracle evaluations, optimize the dimer
on the GP surface, then call the oracle when the GP converges or the trust
region is exceeded.

OTGPD ([`otgpd`](@ref)) extends this with features needed for production use:

| Feature | `gp_dimer` | `otgpd` |
|:--------|:-----------|:--------|
| Adaptive GP threshold | Fixed `T_force_gp` | Tightens as forces decrease |
| Initial rotation | None | Optional true-PES rotation phase |
| Image 1 evaluation | Always | Configurable (`eval_image1`) |
| Data pruning | None | Optional max training set size |
| Convergence tracking | Basic history | Detailed history with `T_gp` tracking |

## Adaptive GP Convergence Threshold

In the basic dimer, the GP convergence threshold `T_force_gp` is fixed. This
means the GP must always achieve the same accuracy, even when the dimer is far
from the saddle point and the forces are large.

OTGPD adapts the threshold based on the true force history:

```math
T_{\text{gp}} = \max\left(\frac{\min(\|F_{\text{true}}\|_{\text{history}})}{\text{divisor}},\; \frac{T_{\text{dimer}}}{10}\right)
```

Early in the search when forces are large (~1.0), `T_gp` might be 0.1 —
allowing coarse GP optimization. As the true forces decrease (~0.01), `T_gp`
tightens to 0.001, demanding precise GP optimization near convergence.

Set `divisor_T_dimer_gp > 0` to enable adaptive mode (default 10.0).
Set `divisor_T_dimer_gp ≤ 0` for the fixed mode (`T_dimer / 10`).

## Initial Rotation Phase

Before the main GP loop, OTGPD can perform an initial rotation phase directly
on the true potential. This aligns the dimer approximately with the lowest
curvature mode using modified Newton angle optimization (parabolic fit) on the
real energy surface.

This avoids the problem of spending many GP outer iterations on large rotations
when the initial orientation is poor. The oracle cost of a few initial rotations
is typically much less than multiple GP train/predict cycles.

Enable with `initial_rotation = true` (default) and control the number of
rotation steps with `max_initial_rot`.

## Example: Muller-Brown Saddle Point

```julia
using ChemGP
using KernelFunctions

# Start between minima B and C
x_B = [0.623, 0.028]
x_C = [-0.050, 0.467]
x_init = 0.5 * (x_B + x_C)

# Orient along the transition direction
orient_init = x_C - x_B

# For 2D surfaces, use a KernelFunctions kernel directly
k = SqExponentialKernel()

config = OTGPDConfig(
    T_dimer = 0.1,
    divisor_T_dimer_gp = 10.0,
    max_outer_iter = 20,
    max_inner_iter = 500,
    dimer_sep = 0.01,
    initial_rotation = true,
    max_initial_rot = 5,
    rotation_method = :simple,
    translation_method = :simple,
    gp_train_iter = 100,
    n_initial_perturb = 3,
)

result = otgpd(muller_brown_energy_gradient, x_init, orient_init, k; config)

println("Converged: $(result.converged)")
println("Oracle calls: $(result.oracle_calls)")
println("Final position: $(result.state.R)")
println("Final |F|: $(result.history["F_true"][end])")
```

## Configuration

The [`OTGPDConfig`](@ref) struct controls all parameters. Key differences from
[`DimerConfig`](@ref):

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `T_dimer` | 0.01 | True force convergence threshold |
| `divisor_T_dimer_gp` | 10.0 | Adaptive threshold divisor (>0 = adaptive) |
| `initial_rotation` | true | Enable initial rotation on true PES |
| `max_initial_rot` | 20 | Max initial rotation iterations |
| `eval_image1` | true | Evaluate oracle at image 1 each outer iter |
| `max_training_points` | 0 | Max training set size (0 = no pruning) |
| `max_inner_iter` | 10000 | Max GP steps per outer iteration |

## Data Pruning

When the training set grows large, GP inference becomes expensive (O(N^3)).
OTGPD can optionally prune the training set to keep only the `max_training_points`
closest to the current dimer position, using [`prune_training_data!`](@ref).

Set `max_training_points > 0` to enable. A value of 0 (default) disables pruning.

## Convergence History

The [`OTGPDResult`](@ref) includes a `history` dict with detailed convergence
information:

```julia
result.history["E_true"]       # True energies at each outer iteration
result.history["F_true"]       # True translational force norms
result.history["curv_true"]    # True curvatures (NaN if eval_image1=false)
result.history["oracle_calls"] # Cumulative oracle call count
result.history["T_gp"]         # Adaptive GP threshold per outer iteration
```

## Next Steps

- [GP-Dimer Saddle Point Search](@ref): The basic pedagogical dimer implementation
- [Nudged Elastic Band (NEB)](@ref): Finding minimum energy paths instead of saddle points
- [Kernel Design](@ref): Choosing the right kernel for your system
