# Archive-domain numerical sensitivity

Command:

```text
/usr/bin/time -p pixi run python investigation/archive_numerics_probe.py
```

Exit status 0; wall time 1.11 s. All 2,312 configured bins were retained. The
reproduced sample T, density, and baseline were fixed; the science default
`n_tau=600`, nominal-L model is the reference.

Current reconstructed transmission-ratio residuals (data minus prediction,
divided by propagated sample/open counting uncertainty) are:

| Requested feature | Actual bin | residual |
|---|---:|---:|
| 10.35 eV | 10.3510 eV | +6.89σ |
| 13.92 eV | 13.9218 eV | -9.96σ |
| 23.95 eV | 23.9521 eV | -24.59σ |
| 35.17 eV | 35.1811 eV | -16.92σ |
| 39.15 eV | 39.1413 eV | +59.41σ |

The largest absolute residual is +62.09σ at 39.0980 eV. These are not the
joint-Poisson signed-deviance residuals, and they differ from some cached
archive tables (notably the archive's -47.9σ label near 13.92 eV). Residual
definitions in the prior package are therefore not interchangeable.

| Case | Max prediction change | RMS change | Residual SSR change |
|---|---:|---:|---:|
| archived calibration grid, `n_tau=400` | 0.559σ | 0.0845σ | −0.357% |
| calibrator default, `n_tau=500` | 0.971σ | 0.1036σ | −0.406% |
| higher `n_tau=1200` | 0.598σ | 0.0728σ | −0.225% |
| physically consistent `L=25.020943 m` | 0.078σ | 0.00887σ | +0.085% |

The `n_tau` choice is not numerically converged to the data precision at the
archived fitted parameters, and calibration (400) versus science construction
(default 600) are not identical. However, these fixed-parameter changes are
sub-sigma per bin and do not approach the 14–64σ dominant residuals.

The confirmed `L_scale` implementation defect has a small direct effect at the
cached parameters because its fitted scale is only 1.0008377; applying the
effective flight path changes a prediction by at most 0.078σ. This cannot
directly generate the current 62σ feature at fixed parameters, but an
end-to-end recalibration under the corrected model was not tested.
