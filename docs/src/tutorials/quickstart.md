# Quick Start

This tutorial walks through a complete GP-guided geometry optimization in
about 5 minutes using the built-in Lennard-Jones potential.

## Setup

```julia
using ChemGP
```

## Step 1: Create a Starting Structure

Generate a random cluster of 4 atoms with no pair closer than 1.5 distance units:

```julia
x_init = random_cluster(4)
```

This returns a flat coordinate vector `[x1,y1,z1, x2,y2,z2, ...]` of length 12.

## Step 2: Choose a Kernel

The kernel defines the GP's prior over the potential energy surface. For a system
with no frozen atoms, use the isotropic Squared Exponential kernel:

```julia
kernel = MolInvDistSE(1.0, [0.5], Float64[])
```

Arguments: signal variance (1.0), inverse lengthscale ([0.5]), frozen coordinates
(empty).

## Step 3: Run Minimization

```julia
result = gp_minimize(lj_energy_gradient, x_init, kernel)
```

The optimizer will:
1. Evaluate the oracle at the initial point and a few perturbations
2. Train the GP on accumulated data
3. Optimize on the GP surface using L-BFGS
4. Evaluate the oracle at the GP-predicted minimum
5. Repeat until the true gradient norm is below the convergence threshold

## Step 4: Inspect the Result

```julia
println("Converged: ", result.converged)
println("Final energy: ", result.E_final)
println("Oracle calls: ", result.oracle_calls)
println("Gradient norm: ", sqrt(sum(result.G_final .^ 2)))
```

The `MinimizationResult` contains:
- `x_final`: The optimized configuration
- `E_final`, `G_final`: Energy and gradient at the final point
- `converged`: Whether the gradient norm fell below `conv_tol`
- `oracle_calls`: Total number of expensive oracle evaluations
- `trajectory`, `energies`: Full history for analysis

## Customizing the Optimization

Use `MinimizationConfig` to control the algorithm:

```julia
config = MinimizationConfig(
    trust_radius = 0.15,    # Allow slightly larger steps
    conv_tol = 1e-3,        # Tighter convergence
    max_iter = 200,         # More iterations
    verbose = true,
)

result = gp_minimize(lj_energy_gradient, x_init, kernel; config)
```

## Next Steps

- [GP Basics: Derivative Observations](@ref): Understand the theory behind derivative observations
- [Molecular Kernels](@ref): Learn about inverse distance features and kernel choices
- [GP-Guided Minimization](@ref): Detailed walkthrough of the optimization algorithm
