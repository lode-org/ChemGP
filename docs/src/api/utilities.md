# Utilities

Distance metrics and sampling utilities for GP-guided optimization.

## Distance Metrics

```@docs
interatomic_distances
max_1d_log_distance
rmsd_distance
```

## Earth Mover's Distance

```@docs
emd_distance
```

## Sampling

```@docs
farthest_point_sampling
```

## [AtomsBase Integration](@id AtomsBase-Integration)

These functions are available when the `AtomsBase` package is loaded (via
package extension). They convert between AtomsBase `AbstractSystem` objects
and ChemGP's flat-vector representation.

```@docs
chemgp_coords
atomsbase_system
atomsbase_neb_trajectory
```
