# Local IC identifiability probe

Command:

```text
/usr/bin/time -p pixi run python investigation/identifiability.py --fit-min 8 --fit-max 45
```

Exit status 0; wall time 1.25 s. The probe uses all 2,312 bins from
8.0006–44.9605 eV, the supplied RT counts, JENDL-5, the cached position, the
notebook's 48×400 synthesis grid, and the exact profiled
normalization+constant+linear calibration objective. No bins were removed from
the configured fit window.

Coordinates are `(log a0, log a1, log beta, logit R)`. After whitening and
projecting the finite-difference model Jacobian away from the three profiled
nuisance columns:

```text
singular values = [430.2792733, 130.4773920, 67.14394945, 0.03258022]
condition number = 13206.76396
column-normalized singular values = [1.6607802, 1.0258835, 0.3904924, 0.1920620]
column-normalized condition number = 8.64710
column norms:
  log_a0   380.6842723
  log_a1     0.1065700
  log_beta 105.2388521
  logit_R  225.1159064
least-informed right-singular direction:
  log_a0   +0.0002873
  log_a1   -0.9999999
  log_beta +0.0001635
  logit_R  +0.0000358
```

Normalized projected-derivative correlation matrix:

```text
             log_a0    log_a1   log_beta   logit_R
log_a0       1.0000     0.9415    -0.1219   -0.8578
log_a1       0.9415     1.0000     0.0221   -0.8349
log_beta    -0.1219     0.0221     1.0000   -0.1880
logit_R     -0.8578    -0.8349    -0.1880    1.0000
```

The pseudoinverse Fisher correlation includes
`corr(log_a0,log_a1)=-0.8192`,
`corr(log_a0,log_beta)=+0.6189`, and
`corr(log_a0,logit_R)=+0.6003`.

Two independently archived JENDL-5 parameter sets illustrate the ridge:

```text
notebook: a0=.8030711 a1=.00100018 beta=.3485746 R=.1570399
robust:   a0=.7839300 a1=.04889000 beta=.3463200 R=.1509000
```

Their fitted transmissions differ by only 0.1491 RMS standard deviations
(maximum 0.9320σ; maximum absolute transmission 0.001692), while their
profiled reduced calibration chi-squares are 13.3250 and 13.3817. Across
8–45 eV their alpha curves differ by only −0.28% to −1.49%, despite a 48.9×
change in `a1`.

This validates local practical non-identifiability—especially `a1`—for the
archived calibration design. It does not prove global structural
non-identifiability; full profile likelihood remains the required nonlinear
uncertainty diagnostic.

The raw condition number is not coordinate invariant: each `log` derivative
contains the corresponding parameter scale, and normalizing the four columns
changes 13,206.8 to 8.65. The evidentiary conclusions are therefore the active
lower-bound hit, the 3,572× column-norm contrast, the nearly pure least-informed
direction, and predictive equivalence—not a universal condition cutoff.

Full cached-range command:

```text
/usr/bin/time -p pixi run python investigation/identifiability.py --fit-min 4 --fit-max 122
```

Exit status 0; wall time 2.20 s. This uses all 4,280 bins selected on the raw
4–120 eV axis; after t0/L correction they span 4.5435–121.6323 eV. Key output:

```text
raw singular values = [524.1311593, 262.7544516, 92.0706681, 0.0392162]
raw coordinate condition = 13365.1800
column-normalized condition = 7.83992
column norms: log_a0=472.685, log_a1=.111622, log_beta=270.682, logit_R=235.650
least direction log_a1=-0.999999963
cache versus robust prediction difference = 0.121136 RMS sigma, 1.336081 max sigma
```

Thus the `a1` conclusion survives removal of the inherited 8–45 eV exclusion,
including the seven corrected-energy bins above 120 eV.
