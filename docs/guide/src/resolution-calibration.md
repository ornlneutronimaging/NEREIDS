# Instrument Resolution: Models and Calibration

A measured resonance dip is broadened by several largely separate effects: the
**instrument resolution** (moderator pulse + flight-path + detector timing — a
*beamline* property), **Doppler** broadening (sample temperature), the
**background / normalization** (the measurement), and **self-shielding /
multiple scattering** in optically-thick (black) resonances (the sample). On a
single spectrum these trade off — too narrow a resolution looks like too hot a
sample — so the instrument resolution must be characterized separately and then
held fixed when fitting unknown samples.

This page covers (1) the three resolution models NEREIDS provides and (2) the
**calibrate → pin → fit** procedure that determines the resolution from a known
standard.

## The three resolution models

All three are passed to [`forward_model`](./python-api.md) and the fitters the
same way; only the kernel *source* differs.

### Gaussian (analytical)

Energy-dependent Gaussian from instrument timing parameters:

```python
import numpy as np, nereids
hf = nereids.load_endf(72, 177)              # Hf-177
energies = np.linspace(2.0, 30.0, 2000)

T = nereids.forward_model(
    energies, [(hf, 5e-5)], temperature_k=300.0,
    flight_path_m=25.0, delta_t_us=1.0, delta_l_m=1e-3,
)
```

### Tabulated UDD (Monte-Carlo file)

A measured/simulated asymmetric kernel (e.g. a VENUS FTS file):

```python
udd = nereids.load_resolution("fts_bl10.txt", flight_path_m=25.0)
T = nereids.forward_model(energies, [(hf, 5e-5)], temperature_k=300.0, resolution=udd)
```

### Ikeda–Carpenter (analytical moderator model)

A physically-grounded analytic moderator pulse — `α(E)` (fast rate), `β` (slow
storage rate), `R` (storage fraction), with optional proton-burst and channel
terms. Build parameter laws with [`EnergyLaw`](./python-api.md), then synthesize
the kernel:

```python
ic = nereids.IkedaCarpenter(
    flight_path_m=25.0, e_min_ev=0.5e-3, e_max_ev=1000.0,
    alpha=nereids.EnergyLaw.sqrt_e(0.30, 0.0),   # α(E) = 0.30·√E
    beta=0.10,
    r=nereids.EnergyLaw.exp_mev(25.0),           # R(E) → 0 in the eV regime
)
tab = ic.as_tabulated()                          # a TabulatedResolution
T = nereids.forward_model(energies, [(hf, 5e-5)], temperature_k=300.0, resolution=tab)
```

`ic.kernel_at(energy_ev)` returns the `(tof_offsets_us, weights)` kernel at one
energy for inspection.

## The calibrate → pin → fit procedure

Instrument resolution and flight-path geometry are **beamline constants**;
density, temperature, background and normalization are **per-measurement**. So:

1. **Calibrate** — measure a calibrant of *known* density and temperature; fit
   the resolution parameters with `ρ, T` fixed.
2. **Pin** — keep the calibrated resolution; switch to the sample (same geometry).
3. **Fit** — fit the sample `ρ` / `T` / both with the resolution pinned; re-fit
   background and normalization per measurement.

### `calibrate_resolution`

```python
cal = nereids.calibrate_resolution(
    energies, data, uncertainty,
    family="udd_corr",                  # "gaussian" | "udd_corr" | "ic"
    isotopes=[(hf, 5e-5)],              # KNOWN calibrant composition + density
    temperature_k=300.0,                # KNOWN calibrant temperature
    base_udd=udd,                       # required for family="udd_corr"
    restarts=2,
)
print(cal)                # ResolutionCalibration(family=udd_corr, chi2/dof=..., converged=...)
print(cal.params())       # decoded fitted parameters
calibrated = cal.as_tabulated()         # pin this into the sample fit
```

The families calibrate different knobs:

| family       | fits                                  | meaning                                   |
|--------------|---------------------------------------|-------------------------------------------|
| `gaussian`   | `Δt, ΔL`                              | analytical Gaussian width                 |
| `udd_corr`   | `s(E)=s0·(E/E_ref)^p` on a base UDD   | trust the MC *shape*, calibrate its width |
| `ic`         | `α(E)=a0√E+a1, β`                      | free analytic shape                       |

Use `.as_tabulated()` for `udd_corr` / `ic` (a `TabulatedResolution` to pass as
`resolution=`); use `.gaussian_params()` → `(delta_t_us, delta_l_m)` for the
Gaussian family.

### Pin and fit the sample

```python
fit = nereids.fit_spectrum_typed(
    sample_T, sample_unc, energies, [(hf, 1e-4)],
    temperature_k=300.0, fit_temperature=True,   # recover ρ and T
    resolution=calibrated,                        # PINNED
)
print(fit.densities, fit.temperature_k, fit.reduced_chi_squared)
```

## Choosing a calibrant (important)

The fit absorbs *every* unmodeled broadening into the "resolution", so a poor
calibrant yields a contaminated, non-transferable result:

- **Use non-black resonances** (`T_min ≈ 0.2–0.8`). Geometric thinness is not
  enough — at a strong resonance the cross-section is thousands of barns, so even
  a thin foil is optically thick at the peak and its self-shielding / multiple
  scattering would be soaked into the "resolution".
- **Model the background / normalization** during calibration (`fit_background=True`),
  then re-fit it per sample (it does not transfer).
- **Same geometry** for calibrant and sample (flight path, sample-to-detector).

## Guidance

- **Density / isotopic characterization** is robust to the resolution-model
  choice; **temperature** is sensitive — calibrate carefully before trusting a
  fitted temperature.
- A worked end-to-end example (build the models, calibrate, pin, fit) is in
  `examples/notebooks/workflows/06_resolution_calibration.ipynb`.
