# GP-Dimer Saddle Point Search

This tutorial explains the dimer method for finding transition states (first-order
saddle points), accelerated by GP predictions.

## What is a Saddle Point?

A first-order saddle point on the potential energy surface (PES) has:
- Zero gradient (``\nabla E = 0``)
- Exactly one negative eigenvalue of the Hessian (one downhill direction)

Saddle points correspond to transition states — the highest-energy point along
the minimum energy path between two minima.

## The Dimer Concept

A "dimer" is a pair of configurations separated by a small distance ``2\delta``
along a direction vector ``\hat{n}``:

```
R₁ = R₀ + δn̂     (image 1)
R₂ = R₀ - δn̂     (image 2)
```

where ``R_0`` is the midpoint. The algorithm has two phases:

### Rotation

Rotate ``\hat{n}`` to align with the lowest curvature mode. The curvature along
the dimer direction is estimated by finite differences:

```math
C = \frac{(G_1 - G_0) \cdot \hat{n}}{\delta}
```

The rotational force drives the dimer toward the eigenvector with the most
negative curvature:

```math
F_{\text{rot}} = \frac{G_1 - G_0}{\delta} - \left[\frac{G_1 - G_0}{\delta} \cdot \hat{n}\right] \hat{n}
```

See [`rotational_force`](@ref) and [`curvature`](@ref).

### Translation

The translation force inverts the component along the dimer direction, so the
algorithm climbs along the lowest mode while descending in all other directions:

```math
F_{\text{trans}} = G_0 - 2(G_0 \cdot \hat{n})\hat{n}
```

See [`translational_force`](@ref).

## GP Acceleration

The key insight of GP-Dimer is that rotation and translation steps don't need
oracle evaluations — they can use GP predictions instead. The oracle is only
called when:
- The GP converges to a saddle candidate
- The trust region is exceeded

This dramatically reduces the number of expensive oracle calls.

## Rotation Strategies

Three rotation methods are available, selected via `config.rotation_method`:

### Simple (`:simple`)
The original pedagogical version. Estimates the rotation angle directly from the
ratio of rotational force magnitude to curvature: ``\Delta\theta = 0.5 \arctan(|F_{\text{rot}}| / |C|)``.
Good for understanding the algorithm but converges slowly.

### Modified Newton with L-BFGS (`:lbfgs`, default)
Uses L-BFGS to choose the rotation search direction, then applies a **modified
Newton angle optimization** (parabolic fit) to find the optimal angle:

1. Compute initial test angle from ``|F_{\text{rot}}|`` and curvature ``C``
2. Evaluate GP at a trial rotation by ``\Delta\theta``
3. Fit a parabola ``F(\theta) = a_1 \cos 2\theta + b_1 \sin 2\theta``
4. Optimal angle: ``\theta^* = 0.5 \arctan(b_1 / a_1)``

The L-BFGS direction is projected perpendicular to the current dimer orientation
before use — this is why a custom L-BFGS implementation is needed rather than
Optim.jl's standard optimizer.

Reference: MATLAB `rot_iter_lbfgs.m`, `rotate_dimer.m`

### Conjugate Gradient (`:cg`)
Polak-Ribière-Polyak with automatic restart when ``\gamma < 0`` or the CG direction
exceeds the raw force. Pedagogically useful contrast to L-BFGS.

Reference: MATLAB `rot_iter_cg.m`

## Translation Strategies

Two translation methods, selected via `config.translation_method`:

### Simple (`:simple`)
Adaptive step based on curvature: ``\alpha = \min(\alpha_0, 0.1 |F| / |C|)``.

### Curvature-dependent L-BFGS (`:lbfgs`, default)
- **Negative curvature** (at a saddle): Full L-BFGS on the modified translational
  force, with step bounded by `max_step`. This is the production algorithm.
- **Positive curvature** (still searching): Fixed step along ``-(G \cdot \hat{n})\hat{n}``
  with conservative step `step_convex`. L-BFGS memory is reset.

Reference: MATLAB `trans_iter_lbfgs.m`

## Usage

```julia
using ChemGP

# Start from a configuration between two minima
x_init = random_cluster(4)
orient = randn(length(x_init))  # Random initial orientation

kernel = MolInvDistSE(1.0, [0.5], Float64[])

# Default: L-BFGS rotation + L-BFGS translation
config = DimerConfig(
    T_force_true = 1e-3,
    trust_radius = 0.1,
    max_outer_iter = 50,
)

result = gp_dimer(lj_energy_gradient, x_init, orient, kernel; config)

# Or use simple methods for pedagogical clarity
config_simple = DimerConfig(
    rotation_method = :simple,
    translation_method = :simple,
    T_force_true = 1e-3,
)

result_simple = gp_dimer(lj_energy_gradient, x_init, orient, kernel;
                         config = config_simple)
```

## Convergence Criteria

The algorithm converges when **both** conditions are met on oracle-evaluated values:
1. Translational force norm ``\|F_{\text{trans}}\| <`` `T_force_true`
2. Curvature is negative (``C < 0``)

This ensures the converged point is genuinely a saddle point.

## Result Structure

The [`DimerResult`](@ref) contains:
- `state`: Final [`DimerState`](@ref) (position and orientation)
- `converged`: Whether convergence criteria were met
- `oracle_calls`: Total number of oracle evaluations
- `history`: Dict with convergence history (`"E_true"`, `"F_true"`, `"curv_true"`)

## Further Reading

- Goswami et al., *J. Chem. Theory Comput.* (2025) [doi:10.1021/acs.jctc.5c00866](https://doi.org/10.1021/acs.jctc.5c00866) — efficient GP-accelerated saddle point searches
- Goswami & Jónsson, *ChemPhysChem* (2025) [doi:10.1002/cphc.202500730](https://doi.org/10.1002/cphc.202500730) — adaptive pruning
- Goswami, *Efficient exploration of chemical kinetics* (2025) [arXiv:2510.21368](https://arxiv.org/abs/2510.21368) — thesis
- Henkelman & Jonsson, *J. Chem. Phys.* 111, 7010 (1999) — original dimer method
- Koistinen et al., *J. Chem. Theory Comput.* 16, 499 (2020) — GP-Dimer

See also: [References](@ref references)
