# Trust Regions

The GP model is only reliable near its training data. Trust region management
ensures that optimization steps don't stray into regions where the GP is
uncalibrated.

## Distance Metrics

ChemGP provides three distance metrics for measuring how "far" a configuration
is from training data. The choice of metric affects both the trust region check
and the FPS subset selection in OTGPD.

### Euclidean Distance

The default in [`min_distance_to_data`](@ref):

```math
d(x, x') = \|x - x'\|_2
```

Simple and fast, but **not** rotationally invariant. Two configurations that
differ only by a rotation will have nonzero Euclidean distance.

### MAX\_1D\_LOG Distance

The primary metric from gpr\_optim, implemented in [`max_1d_log_distance`](@ref):

```math
d(x_1, x_2) = \max_k \left|\log\frac{r^{(1)}_k}{r^{(2)}_k}\right|
```

where ``r^{(1)}_k`` and ``r^{(2)}_k`` are corresponding interatomic distances.

Properties:
- Rotationally and translationally invariant
- Sensitive to large relative changes in any single atom pair
- Zero when configurations are identical (up to rigid body motion)

### Earth Mover's Distance (EMD)

Implemented in [`emd_distance`](@ref), this computes the optimal transport cost
between two atomic configurations treated as discrete measures. Each atom is a
point mass located at its 3D coordinates (Goswami 2025).

```math
d_{\text{EMD}}(x_1, x_2) = \min_{\pi \in \Pi} \sum_{i,j} \pi_{ij} \|r^{(1)}_i - r^{(2)}_j\|
```

where ``\Pi`` is the set of valid transport plans (doubly stochastic matrices
for equal-size systems).

Properties:
- **Permutation-invariant**: atoms of the same type can be freely reordered
- **Type-aware**: when `atom_types` are provided, only atoms of the same
  element can be matched, preventing unphysical H-to-Cu assignments
- Rotationally and translationally invariant (in the assignment sense)
- More expensive than MAX\_1D\_LOG but handles identical-atom clusters correctly

The EMD metric is the default for both OTGPD and GP-NEB trust regions
(`trust_metric = :emd`) and FPS subset selection (`fps_metric = :emd`). The
shared implementation lives in `distances_trust.jl`. For systems without
permutation symmetry (e.g., organic molecules with unique atoms), MAX\_1D\_LOG
or Euclidean may be equally effective and faster.

```julia
using ChemGP

x1 = Float64[0,0,0, 1,0,0, 0,1,0]  # 3-atom config A
x2 = Float64[0,0,0, 0,1,0, 1,0,0]  # same config, atoms 2&3 swapped

# Euclidean sees a large difference
norm(x1 - x2)  # > 0

# EMD recognizes the permutation
emd_distance(x1, x2)  # ~ 0
```

## Soft Trust Region Penalty

In [`gp_minimize`](@ref), the trust region is enforced as a soft quadratic penalty
on the GP objective:

```math
f(x) = \mu_{\text{GP}}(x) + \lambda \max(0, d_{\min}(x) - r)^2
```

where ``d_{\min}(x)`` is the distance to the nearest training point, ``r`` is the
trust radius, and ``\lambda`` is the penalty coefficient.

The gradient of the penalty pushes the optimizer back toward the training data
when it tries to step outside the trust region.

## Interatomic Distance Ratio Check

The [`check_interatomic_ratio`](@ref) function provides a complementary physical
sanity check. It verifies that no interatomic distance has changed by more than
a factor of `ratio_limit` relative to the nearest training point:

```julia
# Reject if any r_ij changes by more than 50%
check_interatomic_ratio(x_new, X_train, 2/3)
```

This prevents the optimizer from distorting the molecule beyond recognition,
which the distance-based trust region alone might not catch.

## Adaptive Trust Threshold

In production saddle point searches, the optimal trust radius depends on how
well-sampled the configuration space is. Early in the search when data is
sparse, a generous radius lets the optimizer make large steps. As the
training set grows and the GP becomes better calibrated, tighter trust
regions improve accuracy near the saddle point (Goswami & Jonsson 2025).

Both OTGPD and GP-NEB implement a sigmoidal schedule that decays with
training set size:

```math
T(n) = T_{\min} + \frac{\Delta T}{1 + A \exp(n / n_{1/2})}
```

where:
- ``n = N_{\text{data}} / N_{\text{atoms}}`` is the effective data density
- ``T_{\min}`` is the asymptotic minimum threshold
- ``\Delta T`` is the initial excess range
- ``A`` controls the steepness of the transition
- ``n_{1/2}`` is the half-life in effective data points

A floor parameter ensures the threshold never drops below a minimum value:

```math
T_{\text{final}}(n) = \max(T(n),\; T_{\text{floor}})
```

```julia
config = OTGPDConfig(
    use_adaptive_threshold = true,
    adaptive_t_min = 0.15,      # asymptote
    adaptive_delta_t = 0.35,    # range: starts at 0.15 + 0.35 = 0.50
    adaptive_n_half = 50,       # half-life
    adaptive_A = 1.3,           # steepness
    adaptive_floor = 0.2,       # absolute minimum
)
```

When `use_adaptive_threshold = false`, the fixed `trust_radius` is used.

## Choosing a Trust Configuration

| Scenario | Recommended metric | Adaptive? |
|:---------|:-------------------|:----------|
| Small organic molecule (unique atoms) | `:max_1d_log` | Optional |
| Metal cluster (identical atoms) | `:emd` | Recommended |
| 2D test surface (Muller-Brown) | `:euclidean` | No |
| Production molecular search (OTGPD) | `:emd` | Yes |
| GP-NEB (molecular systems) | `:emd` | Optional |

## Tuning the Trust Radius

| Symptom | Adjustment |
|:--------|:-----------|
| Too many oracle calls | Increase `trust_radius` |
| Optimizer diverges or energy explodes | Decrease `trust_radius` |
| Optimizer oscillates near boundary | Increase `penalty_coeff` |
| Progress is very slow | Decrease `penalty_coeff` |

A typical starting value is `trust_radius = 0.1` with `penalty_coeff = 1e3`.
For systems with stiffer potentials (e.g., covalent bonds), a smaller trust
radius (0.05) may be needed. For softer systems (e.g., van der Waals clusters),
larger values (0.2--0.3) may be appropriate.

## Further Reading

- Goswami & Jonsson, *ChemPhysChem* (2025) [doi:10.1002/cphc.202500730](https://doi.org/10.1002/cphc.202500730) -- adaptive pruning and trust region management
- Goswami, *Efficient exploration of chemical kinetics* (2025) [arXiv:2510.21368](https://arxiv.org/abs/2510.21368) -- thesis
- [OTGPD tutorial](@ref) for EMD trust and adaptive threshold in context
- [NEB Method](@ref) for trust region configuration in GP-NEB
- [References](@ref references)
