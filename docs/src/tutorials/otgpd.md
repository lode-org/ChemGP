# OTGPD -- Optimal Transport GP Dimer

This tutorial covers the OTGPD algorithm, the full production-grade GP-guided
dimer method. It builds on the [GP-Dimer Saddle Point Search](@ref) by adding adaptive
convergence thresholds, initial rotation, data management, and robust
hyperparameter handling (Goswami et al. 2025; Goswami & Jonsson 2025).

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
| FPS subset selection | None | Farthest point sampling for hyperopt |
| Hyperparameter oscillation | None | HOD monitoring with auto-enlargement |
| Variance barrier | None | Log-barrier prevents signal variance collapse |
| Trust metric | Euclidean only | EMD, MAX\_1D\_LOG, or Euclidean |
| Adaptive trust threshold | Fixed radius | Sigmoidal decay with data size |
| Molecular kernel path | Normalize only | Energy shift, fix\_noise, warm-start |
| Convergence tracking | Basic history | Detailed history with `T_gp` tracking |

## Adaptive GP Convergence Threshold

In the basic dimer, the GP convergence threshold `T_force_gp` is fixed. This
means the GP must always achieve the same accuracy, even when the dimer is far
from the saddle point and the forces are large.

OTGPD adapts the threshold based on the true force history:

```math
T_{\text{gp}} = \max\left(\frac{\min(\|F_{\text{true}}\|_{\text{history}})}{\text{divisor}},\; \frac{T_{\text{dimer}}}{10}\right)
```

Early in the search when forces are large (~1.0), `T_gp` might be 0.1 --
allowing coarse GP optimization. As the true forces decrease (~0.01), `T_gp`
tightens to 0.001, demanding precise GP optimization near convergence.

Set `divisor_T_dimer_gp > 0` to enable adaptive mode (default 10.0).
Set `divisor_T_dimer_gp <= 0` for the fixed mode (`T_dimer / 10`).

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

## FPS Subset Selection for Hyperparameter Optimization

GP hyperparameter optimization (marginal likelihood maximization) scales as
O(N^3) in the training set size. For large training sets the cost dominates
the outer loop. OTGPD addresses this with Farthest Point Sampling (FPS):
a representative subset is selected for hyperparameter optimization, while
the full training set is used for the final GP model rebuild (Goswami &
Jonsson 2025).

The selection strategy:
1. The `fps_latest_points` most recently added points are always included
   (they reflect the current dimer neighborhood)
2. The remaining slots (up to `fps_history` total) are filled by FPS from
   the older points, maximizing spatial coverage

The distance metric for FPS is configurable via `fps_metric`:
- `:emd` -- Earth Mover's Distance (permutation-invariant, default)
- `:max_1d_log` -- maximum log-ratio of interatomic distances
- `:euclidean` -- standard L2 norm

```julia
config = OTGPDConfig(
    fps_history = 8,          # use 8 points for hyperopt
    fps_latest_points = 2,    # always include 2 most recent
    fps_metric = :emd,        # permutation-invariant distances
    atom_types = Int[29, 29], # for type-aware EMD (Cu2)
)
```

When `fps_history = 0`, no subset selection is performed and all training
points are used for hyperparameter optimization.

## Hyperparameter Oscillation Detection (HOD)

In GP-accelerated saddle point searches, hyperparameter optimization on a
small training subset can produce oscillating hyperparameters across outer
iterations -- the signal variance, lengthscales, or noise variance flip
direction from iteration to iteration without converging (Goswami & Jonsson
2025). This instability degrades the GP surface quality and wastes oracle
evaluations.

HOD monitors the log-space hyperparameter trajectory across iterations. At
each outer step, the algorithm computes consecutive differences in the
hyperparameter vector and counts sign-flips within a sliding window:

```math
\text{flip\_ratio} = \frac{\text{sign-flips in } [\Delta\theta_i, \Delta\theta_{i+1}]}{\text{total pairs}}
```

When the flip ratio exceeds `hod_flip_threshold` (default 0.8), the FPS
subset size is enlarged by `hod_history_increment` (default 2), capped at
`hod_max_history` (default 30). A larger training subset provides a more
stable optimization landscape, damping the oscillations.

```julia
config = OTGPDConfig(
    use_hod = true,                 # enable HOD monitoring
    hod_monitoring_window = 5,      # look at last 5 iterations
    hod_flip_threshold = 0.8,       # trigger at 80% sign-flips
    hod_history_increment = 2,      # enlarge subset by 2
    hod_max_history = 30,           # max subset size
)
```

Disable with `use_hod = false`.

## Variance Barrier

When training with `fix_noise = true` (as in the molecular kernel path), the
marginal likelihood can be maximized by collapsing the signal variance to zero
-- the GP becomes a constant function with no predictive power. The variance
barrier adds a log-barrier term to the negative log-likelihood:

```math
\text{NLL}_{\text{barrier}} = \text{NLL} - \beta \log\left(\max\left(\log \hat{\sigma}^2_{\text{max}} - \log \sigma^2, \epsilon\right)\right)
```

where ``\hat{\sigma}^2_{\text{max}}`` is the largest singular value of the
covariance factor and ``\sigma^2`` is the signal variance. The barrier
strength ``\beta`` is adapted to the subset size:

```math
\beta = \min(10^{-4} + 10^{-3} n_{\text{subset}},\; 0.5)
```

This prevents variance collapse while remaining weak enough to not distort
the likelihood landscape significantly. The barrier is computed automatically
in OTGPD when using the molecular kernel path.

## Trust Region Metrics

The trust region check ensures GP optimization steps remain in calibrated
regions. OTGPD supports multiple distance metrics via `trust_metric`:

| Metric | Config value | Properties |
|:-------|:-------------|:-----------|
| Earth Mover's Distance | `:emd` | Permutation-invariant, type-aware |
| MAX\_1D\_LOG | `:max_1d_log` | Rotation/translation-invariant |
| Euclidean | `:euclidean` | Fast, simple L2 norm |

The default is `:emd`, which computes the optimal transport cost between
atomic configurations treated as discrete measures (Goswami 2025). This is
particularly important for systems where atom permutation symmetry matters
(e.g., clusters with identical atoms).

See also: [Trust Regions](@ref) for detailed discussion of each metric.

## Adaptive Trust Threshold

Instead of a fixed trust radius, OTGPD can decay the threshold as the
training set grows. More data means the GP is better calibrated, so the
optimizer can take larger steps early and tighter steps later:

```math
T(n) = T_{\min} + \frac{\Delta T}{1 + A \exp(n / n_{1/2})}
```

where ``n = N_{\text{data}} / N_{\text{atoms}}`` is the effective data
density, ``T_{\min}`` is the asymptotic minimum, ``\Delta T`` is the
additional range, ``A`` controls the steepness, and ``n_{1/2}`` is the
half-life in effective data points.

```julia
config = OTGPDConfig(
    use_adaptive_threshold = true,
    adaptive_t_min = 0.15,      # asymptotic threshold
    adaptive_delta_t = 0.35,    # initial excess above t_min
    adaptive_n_half = 50,       # half-life (effective points)
    adaptive_A = 1.3,           # steepness
    adaptive_floor = 0.2,       # absolute minimum threshold
)
```

When `use_adaptive_threshold = false` (default), the fixed `trust_radius`
is used.

## Molecular Kernel Path

When `kernel` is an [`AbstractMoleculeKernel`](@ref) (e.g., `MolInvDistSE`),
OTGPD uses a specialized training strategy:

1. **Energy shift** (not normalization): subtract the first training energy
   as reference, keeping physical energy scale intact
2. **Fixed noise** (`fix_noise = true`): noise and gradient noise variances
   are set to physically motivated values (`1e-6` and `1e-4`) and not
   optimized, since molecular kernels already encode the correct structure
3. **Warm-start**: the kernel hyperparameters from the previous outer
   iteration seed the next optimization, reducing training iterations needed
4. **Variance barrier**: prevents signal variance collapse (see above)

For non-molecular kernels (e.g., `SqExponentialKernel()`), the standard
normalize-and-optimize path is used, where noise is treated as a free
hyperparameter.

```julia
# Molecular kernel: energy shift + fix_noise + warm-start
kernel = MolInvDistSE(1.0, [1.0], Float64[])
result = otgpd(oracle, x0, orient, kernel; config)

# Generic kernel: normalize + optimize noise
k = SqExponentialKernel()
result = otgpd(oracle, x0, orient, k; config)
```

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

## Example: Molecular System with Full Feature Set

```julia
using ChemGP

# LJ cluster saddle point search with all production features
x_init = random_cluster(4)
orient = randn(length(x_init))

kernel = MolInvDistSE(1.0, [0.5], Float64[])

config = OTGPDConfig(
    T_dimer = 1e-3,
    divisor_T_dimer_gp = 10.0,
    max_outer_iter = 50,
    max_inner_iter = 5000,
    dimer_sep = 0.01,
    initial_rotation = true,
    max_initial_rot = 10,
    rotation_method = :lbfgs,
    translation_method = :lbfgs,
    gp_train_iter = 300,

    # FPS subset selection
    fps_history = 8,
    fps_latest_points = 2,
    fps_metric = :emd,

    # HOD monitoring
    use_hod = true,
    hod_monitoring_window = 5,
    hod_flip_threshold = 0.8,
    hod_history_increment = 2,
    hod_max_history = 30,

    # EMD trust region with adaptive threshold
    trust_metric = :emd,
    use_adaptive_threshold = true,
    adaptive_t_min = 0.15,
    adaptive_delta_t = 0.35,
    adaptive_n_half = 50,
)

result = otgpd(lj_energy_gradient, x_init, orient, kernel; config)
```

## Configuration

The [`OTGPDConfig`](@ref) struct controls all parameters. Key differences from
[`DimerConfig`](@ref):

### Core Parameters

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `T_dimer` | 0.01 | True force convergence threshold |
| `divisor_T_dimer_gp` | 10.0 | Adaptive threshold divisor (>0 = adaptive) |
| `initial_rotation` | true | Enable initial rotation on true PES |
| `max_initial_rot` | 20 | Max initial rotation iterations |
| `eval_image1` | true | Evaluate oracle at image 1 each outer iter |
| `max_training_points` | 0 | Max training set size (0 = no pruning) |
| `max_inner_iter` | 10000 | Max GP steps per outer iteration |

### FPS Subset Selection

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `fps_history` | 5 | Subset size for hyperopt (0 = use all) |
| `fps_latest_points` | 2 | Most recent points always included |
| `fps_metric` | `:emd` | Distance metric (`:emd`, `:max_1d_log`, `:euclidean`) |

### Hyperparameter Oscillation Detection

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `use_hod` | true | Enable HOD monitoring |
| `hod_monitoring_window` | 5 | Iterations to look back |
| `hod_flip_threshold` | 0.8 | Fraction of sign-flips to trigger |
| `hod_history_increment` | 2 | FPS enlargement per trigger |
| `hod_max_history` | 30 | Maximum FPS subset size |

### Trust Region

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `trust_metric` | `:emd` | Distance metric for trust check |
| `atom_types` | `Int[]` | Atomic numbers for type-aware EMD |
| `use_adaptive_threshold` | false | Enable sigmoidal threshold decay |
| `adaptive_t_min` | 0.15 | Asymptotic minimum threshold |
| `adaptive_delta_t` | 0.35 | Additional range above t\_min |
| `adaptive_n_half` | 50 | Half-life in effective data points |
| `adaptive_A` | 1.3 | Steepness of sigmoid |
| `adaptive_floor` | 0.2 | Absolute minimum threshold |

## Data Pruning

When the training set grows large, GP inference becomes expensive (O(N^3)).
OTGPD can optionally prune the training set to keep only the `max_training_points`
closest to the current dimer position, using [`prune_training_data!`](@ref).

Set `max_training_points > 0` to enable. A value of 0 (default) disables pruning.

Note that FPS subset selection (above) offers a complementary approach: rather
than discarding data permanently, it selects a representative subset for
hyperparameter optimization only, while the full dataset is retained for the
GP model rebuild. In practice, FPS subset selection is preferred over pruning
for most applications (Goswami & Jonsson 2025).

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

## Further Reading

- Goswami, Masterov, Kamath, Pena-Torres & Jonsson, *J. Chem. Theory Comput.* (2025) [doi:10.1021/acs.jctc.5c00866](https://doi.org/10.1021/acs.jctc.5c00866) -- efficient GP-accelerated saddle point searches
- Goswami & Jonsson, *ChemPhysChem* (2025) [doi:10.1002/cphc.202500730](https://doi.org/10.1002/cphc.202500730) -- adaptive pruning, HOD, FPS subset selection
- Goswami, *Efficient exploration of chemical kinetics* (2025) [arXiv:2510.21368](https://arxiv.org/abs/2510.21368) -- thesis covering the full OTGPD algorithm
- Koistinen et al., *J. Chem. Theory Comput.* 16, 499 (2020) -- GP-Dimer

## Next Steps

- [GP-Dimer Saddle Point Search](@ref): The basic pedagogical dimer implementation
- [Nudged Elastic Band (NEB)](@ref): Finding minimum energy paths instead of saddle points
- [Trust Regions](@ref): Distance metrics, EMD, and adaptive thresholds
- [Kernel Design](@ref): Choosing the right kernel for your system
