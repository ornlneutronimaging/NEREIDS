# Dense-grid discrimination on the archived IC/JENDL-5 sample

The code-level coarse-grid defect was first validated independently in
`investigation/evidence/code-probes.md`. These two tests ask only whether
sample-stage application of a frozen cached response explains the supplied
sample's dominant residuals.

## Fixed published parameters and convergence

Command:

```text
/usr/bin/time -p pixi run python investigation/dense_grid_probe.py
```

Exit status 0; wall time 6.56 s. Every one of the 2,312 configured
8.0006–44.9605 eV bins was retained. The cached sample temperature, density,
and KL-fit baseline were held fixed while IC+JENDL-5 was evaluated on the data
grid and on 10k/20k/40k log grids spanning 4–120 eV.

```text
coarse vs 40k max raw-transmission change = 0.00274474
coarse vs 40k RMS raw-transmission change = 0.00026708
20k vs 40k max raw-transmission change    = 0.00003659
20k vs 40k RMS raw-transmission change    = 0.00000445

fixed-baseline weighted SSR:
  coarse = 60322.20
  10k    = 61149.88
  20k    = 61125.78
  40k    = 61139.40
```

At the dominant fixed-parameter bins, the 40k grid changed standardized
residuals as follows:

| Feature | Coarse | Dense 40k | Dense prediction correction |
|---|---:|---:|---:|
| 10.35 eV | +10.84σ | +10.95σ | −0.10σ |
| 13.92 eV | −14.60σ | −14.54σ | −0.06σ |
| 23.95 eV | −31.32σ | −30.99σ | −0.32σ |
| 35.17 eV | −23.53σ | −22.80σ | −0.73σ |
| 39.15 eV | +62.09σ | +63.13σ | −1.04σ |

## Temperature/density re-fit

Command:

```text
/usr/bin/time -p pixi run python investigation/dense_grid_refit.py
```

Exit status 0; wall time 24.76 s. Temperature and density were optimized for
both grids; the multiplicative quadratic baseline was profiled at every step.
Both optimizations met `ftol` in four optimizer evaluations (13 actual
forward calls).

```text
                         coarse             dense 20k
temperature K            1050.6099          1054.9933
density atoms/barn       0.0006375663       0.0006371283
weighted SSR             59829.69           60731.28
weighted residual RMS    5.0870 sigma       5.1252 sigma
dense SSR change                            +1.5069%
```

The 39.15 eV residual changed from +63.69σ to +64.47σ; the other four
predeclared features moved by less than 0.5σ except 39.15 eV.

## Interpretation

The missing auxiliary grid is a real correctness defect and materially changes
the forward prediction. With cached `a0`, `a1`, `beta`, and `R` frozen, it is
not a large improvement in the sample-stage WLS diagnostic: densification
slightly worsens the global fit after temperature, density, and baseline
adjustment. This is negative evidence only for that narrow frozen-response
application. It does **not** test whether RT calibration on the corrected grid
would find a different IC response that transfers better, and it does not use
the production count likelihood. An end-to-end recalibration A/B remains open.
