# Optimizers

GP-guided minimization and saddle point search algorithms.

## Minimization

```@docs
gp_minimize
MinimizationConfig
MinimizationResult
```

## Dimer Saddle Point Search

```@docs
gp_dimer
DimerState
DimerConfig
DimerResult
```

## Dimer Utilities

```@docs
dimer_images
curvature
rotational_force
translational_force
```

## L-BFGS Optimizer

```@docs
LBFGSHistory
push_pair!
compute_direction
ChemGP.reset!
```

## Trust Region

```@docs
min_distance_to_data
check_interatomic_ratio
```
