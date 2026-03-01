# Dimer Method

```{note}
Full tutorial with interactive examples via Python bindings to follow.
```

## Saddle Point Search

The dimer method finds first-order saddle points (transition states)
by following the lowest curvature mode uphill while minimizing in all
other directions. ChemGP provides three variants:

Standard Dimer
: Direct oracle evaluation at every step. Reliable but expensive.

GP-Dimer
: Uses a GP surrogate for inner loop predictions. The oracle is called
  only at outer iterations, reducing total evaluations by ~3.5x.

OTGPD
: Adaptive threshold variant that automatically adjusts the GP trust
  threshold based on observed forces. Matches GP-Dimer efficiency with
  less manual tuning.

## Translation Force

The dimer translational force is the effective force that drives the
midpoint toward the saddle:

- **Negative curvature**: modified force (gradient projected out along
  the dimer orientation, reversed component along orient added back)
- **Positive curvature**: step along the orient toward the negative
  curvature region

## LEPS Example

```shell
cargo run --release --example leps_dimer
```

Starting from 0.05 A displaced from the known LEPS saddle along the
negative eigenmode, the GP-Dimer and OTGPD converge in ~13 oracle
calls. The standard dimer needs ~45 calls.
