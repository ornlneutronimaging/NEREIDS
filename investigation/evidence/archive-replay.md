# Cached whole-region replay against synchronized source

Build command:

```text
pixi run build
```

Exit status 0. Relevant output:

```text
Finished `release` profile [optimized] target(s) in 38.07s
Built wheel ... nereids-0.3.0-cp314-cp314-macosx_11_0_arm64.whl
Installed nereids-0.3.0
```

The rebuilt signature includes both archived keywords, `baseline` and
`scale_by_chi2`.

Replay command:

```text
/usr/bin/time -p pixi run python investigation/reproduce_ic_cached.py
```

Exit status 0. Output:

```text
n_resonances=500
bins_total=5304 bins_4_120=4280
t0_us=1.01609978396026
l_scale=1.00083771018426
L_eff_m=25.0209427546065
converged=True iterations=7
temperature_K=1058.69727434866
temperature_unc_K=7.03907637870556
density_atoms_per_barn=0.000632508761631833
density_nominal_fraction=0.898781881990271
deviance_per_dof=28.6325263111934
baseline=[0.9365930840171124, 0.0016145043811995664, 0.00689211751303141]
real 24.76
user 24.32
sys 0.28
```

This independently reproduces the IC+JENDL-5 notebook's cached whole-region
fit from the supplied 1D counts. It does not recreate the RT IC calibration,
the UDR result, data reduction, maps, or profiles because their external
inputs are absent.
