# GP-Guided Minimization

This tutorial walks through the full GP-guided geometry optimization algorithm
implemented in [`gp_minimize`](@ref).

## Algorithm Overview

The key idea is to use a GP surrogate as a cheap stand-in for the expensive oracle
(e.g., DFT). The algorithm alternates between optimizing on the GP surface and
validating with the true oracle:

1. **Sample**: Evaluate the oracle at the initial point and a few perturbations
2. **Train**: Fit the GP to accumulated data by optimizing hyperparameters
3. **Optimize**: Find the minimum on the GP surface using L-BFGS
4. **Validate**: Evaluate the oracle at the GP-predicted minimum
5. **Converge**: Check if the true gradient norm is below threshold
6. If not converged, go to step 2

## Step-by-Step Walkthrough

### Initial Sampling

The first step generates initial training data by evaluating the oracle at the
starting configuration and several random perturbations:

```julia
config = MinimizationConfig(n_initial_perturb = 4, perturb_scale = 0.1)
```

Each evaluation provides both energy and gradient, so 5 oracle calls give
`5 * (1 + D)` observations for the GP.

### GP Training

At each outer iteration, the GP is retrained on all accumulated data. For
molecular kernels (MolInvDistSE, MolInvDistMatern52, MolInvDistMatern32),
hyperparameters are optimized by minimizing the MAP negative log-likelihood
using Scaled Conjugate Gradient (SCG) with analytical gradients. Parameters
are packed in log-space as `w = [log(sigma2); log.(inv_lengthscales)]` with
Gaussian priors centered at data-dependent initial values.

The MAP prior variance is adaptive: pair types with many contributing inverse
distances (e.g., C-H with 8 features) receive looser priors than pair types
with few (e.g., C-C with 1 distance), preventing lengthscale collapse.

```julia
train_model!(model, iterations = 100)
```

For non-molecular kernels, Nelder-Mead is used as a fallback.
See [`train_model!`](@ref) and [`scg_optimize`](@ref) for details.

### GP Surface Optimization

The optimizer minimizes the GP-predicted energy plus a soft trust region penalty:

```math
f(x) = \mu_{\text{GP}}(x) + \lambda \max(0, d(x) - r_{\text{trust}})^2
```

where ``d(x)`` is the distance to the nearest training point and ``r_{\text{trust}}``
is the trust radius. This ensures the optimizer doesn't wander into regions where
the GP has never been validated.

The optimization uses L-BFGS with analytical gradients from the GP posterior.

### Oracle Validation and Convergence

After each GP optimization, the oracle is called at the predicted minimum.
Convergence is checked on the **true** gradient norm:

```julia
config = MinimizationConfig(conv_tol = 5e-3)  # ||∇E|| < 5e-3
```

This is critical: convergence is always measured on oracle values, not GP
predictions.

## Configuration

The [`MinimizationConfig`](@ref) struct controls all algorithm parameters:

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `trust_radius` | 0.1 | Max distance from training data |
| `conv_tol` | 5e-3 | Gradient norm convergence threshold |
| `max_iter` | 500 | Max outer iterations (GP-guided steps) |
| `max_oracle_calls` | 0 | Hard limit on oracle evaluations (0 = no cap) |
| `gp_train_iter` | 300 | SCG iterations for hyperparameters |
| `n_initial_perturb` | 4 | Number of initial perturbation samples |
| `perturb_scale` | 0.1 | Scale of perturbations |
| `penalty_coeff` | 1e3 | Trust region penalty strength |
| `max_move` | 0.1 | Per-atom max displacement (Angstrom) |
| `rff_features` | 0 | 0 = exact GP; >0 = RFF approximation |
| `machine_output` | `""` | JSONL output: `""` disabled, `"host:port"` TCP socket, else file |
| `verbose` | true | Print progress |

## Comparison to Direct L-BFGS

Direct L-BFGS on the oracle minimizes each step using oracle evaluations.
GP-guided optimization uses the GP surrogate for most of the work:

- **Direct L-BFGS**: Needs many oracle calls for line searches
- **GP-guided**: Each outer iteration requires only 1 oracle call (the validation step)

The savings are proportional to how expensive the oracle is. For DFT calculations
that take minutes per evaluation, reducing from hundreds to tens of oracle calls
is a major practical improvement.

## Machine-Readable Output

The `machine_output` field enables JSONL logging alongside human-readable output.
Each iteration emits a JSON line with fields `i` (iteration), `E` (energy),
`F` (max force), `oc` (oracle calls), `tp` (training points), `t` (train time),
`sv` (signal variance), `ls` (lengthscales), `td` (trust distance), and
`gate` (acceptance status). A summary line is emitted at convergence.

For real-time output without Julia's stdout buffering, point `machine_output`
at a TCP socket served by `scripts/jsonl_writer.py`:

```bash
python scripts/jsonl_writer.py --port 9876 --output convergence.jsonl
```

```julia
config = MinimizationConfig(machine_output = "localhost:9876")
```

The writer accepts newline-terminated JSON over TCP, writes to file with
`fsync`, and renders human-readable summaries to stdout immediately. This
decouples I/O flushing from the Julia optimizer process.

## Next Steps

- [GP-Dimer Saddle Point Search](@ref): Extend to saddle point search
- [Trust Regions](@ref): Details on trust region management
