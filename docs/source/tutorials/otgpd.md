# OTGPD

```{note}
Full tutorial with interactive examples via Python bindings to follow.
```

## Adaptive Threshold GP Dimer

The Optimal Transport GP Dimer (OTGPD) extends the GP-Dimer with an
adaptive convergence threshold. Instead of requiring a fixed GP force
tolerance, OTGPD adjusts the threshold based on the history of observed
true forces:

```
T_gp = max(min(F_true) / divisor, T_dimer / 10)
```

When the GP-predicted force is below T_gp, the GP is trusted. Otherwise,
the oracle is invoked and the GP is retrained.

## HOD Training Data Management

OTGPD uses History-Ordered Data (HOD) management: training points are
kept in temporal order and older points are pruned via FPS when the
dataset exceeds `fps_history`. This differs from the standard GP-Dimer
which uses spatial FPS relative to the current position.

## Relationship to GP-Dimer

OTGPD reduces to the GP-Dimer when:
- `fps_history` is large enough that FPS never activates
- The adaptive threshold saturates at `T_dimer / 10`

On the LEPS surface, both methods converge in ~13 oracle calls.
