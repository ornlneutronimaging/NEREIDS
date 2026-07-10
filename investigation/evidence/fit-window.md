# Inherited fit-window sensitivity

The 5,304-bin cache spans 4.525 eV–2.2788 MeV on its stored axis. The notebooks
first select 4–120 eV (4,280 bins), then activate only 8–45 eV in the fit (2,312
bins). No independent instrument log or calibration report in the ZIP
physically justifies either exact boundary. Their independent procedural
justification in this audit is exact reproduction of the pre-existing notebook;
neither window was selected because this audit's fit improved.

## Inner 8–45 eV exclusion

Reference replay:

```text
n active bins = 2312
T = 1058.6973 ± 7.0391 K
density = 0.0006325088 atoms/barn
deviance/dof = 28.6325
```

Command with the identical cached IC table synthesized over 4–120 eV, but no
second fit mask:

```text
/usr/bin/time -p pixi run python investigation/fit_window_probe.py
```

Exit status 0; wall time 23.67 s. Every raw-selected 4–120 eV bin was active.
After t0/L correction they span 4.5435–121.6323 eV; seven corrected bins exceed
the IC table and therefore exercise the current endpoint-clamp behavior.

```text
n input/active bins = 4280 / 4280
converged = true (7 iterations)
T = 1051.6716 ± 7.7895 K
density = 0.0006228912 atoms/barn
deviance/dof = 36.1516
```

Relative to 8–45 eV, temperature changes -7.026 K and density -1.521%. This
comparison keeps response construction fixed; the seven clamped bins are a
disclosed limitation, not a reason to delete them.

## Outer 4–120 eV exclusion

A physically attempted full-domain IC synthesis is preserved by:

```text
/usr/bin/time -p pixi run python investigation/full_domain_synthesis_probe.py
```

Exit status 0; wall time 0.12 s. The script catches and reports the expected
construction error. All 5,304 corrected bins span 4.5435 eV–119.346 MeV; 1,024
stored bins are above 120 eV. Synthesis fails at 401,484 eV because the cached
alpha/beta/R combination requires a 0.0056 us tau step while the cap permits
only the 0.0051 us feature floor. A first direct development attempt using
`/usr/bin/time -p pixi run python investigation/fit_window_probe.py --all-raw`
exited 1 with this same error before the durable catch probe was separated.

```text
n_bins=5304
raw_energy_span_ev=[4.525065320800565, 2278807.542654969]
corrected_energy_span_ev=[4.543510205232908, 119345697.08084093]
bins_raw_above_120_ev=1024
full_domain_synthesis=error
Invalid resolution file format: Ikeda–Carpenter kernel at E = 401484.0960255875 eV:
the 8192-sample tau-grid cap cannot resolve alpha = 508.8491 us^-1,
beta = 0.3486 us^-1, R = 0.157; tau-step 0.0056 us exceeds floor 0.0051 us.
```

For disclosure only, the current implementation can still apply the fixed
4–120 eV table outside its range by silently clamping endpoint kernels:

```text
/usr/bin/time -p pixi run python investigation/fit_window_probe.py --all-raw
```

Exit status 0; wall time 16.95 s. This is **not a physically valid full-domain
IC fit**: 1,031 corrected bins use endpoint-clamped resolution and one sparse
Doppler edge is passed through unbroadened.

```text
n input/active bins = 5304 / 5304
raw span = 4.525 eV–2.2788 MeV
corrected span = 4.5435 eV–119.346 MeV
converged = true (3 iterations)
T = 1038.9482 ± 9.7326 K
density = 0.0006137252 atoms/barn
deviance/dof = 56.8114
```

Relative to 8–45 eV, this diagnostic changes temperature -19.749 K and density
-2.970%. It shows that the outer selection matters; it cannot validate those
numbers because the response and high-energy nuclear/transport model are out of
their tested domains. The 4–120 eV outer scope is therefore independently
justified for exact reproduction and by the current IC model's demonstrated
full-domain infeasibility, not because excluding high-energy bins improves the
fit. A physically valid without-exclusion result requires the P0 adaptive
response and explicit validity domains.

## Interpretation

Neither sensitivity validates 8–45 or 4–120 eV as the production window. The
final analysis reports both exclusions, all available current-model controls,
and the full-domain blocker. Future line inclusion must be pre-registered from
independent nuclear/instrument validity criteria and repeated with the corrected
response model.
