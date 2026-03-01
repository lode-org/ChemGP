# Trust Regions

```{note}
Detailed exposition to follow with Python binding examples.
```

## EMD Trust Distance

The Earth Mover's Distance (EMD) between the inverse-distance features
of a proposed point and the training set provides a measure of how far
the GP is extrapolating. Steps that exceed a trust radius (measured in
EMD) are clipped.

## Adaptive Trust Threshold

The OTGPD method uses an adaptive GP trust threshold:

```
T_gp = max(min(F_true) / divisor, T_dimer / 10)
```

This tightens as the optimizer approaches convergence, trusting the GP
more when forces are already small.

## 1D Max-Log Distance

A faster alternative to full EMD: computes the maximum absolute
difference in log-space features across all dimensions. Used for FPS
subset selection and coarse trust screening.
