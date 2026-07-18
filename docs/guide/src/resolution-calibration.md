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

For separate open/sample counts from an aggregate detector ROI, use the
supported `calibrate_aggregated_1d(...)` then
`fit_frozen_aggregated_1d(...)` workflow described in the
[Python API reference](python-api.md#aggregate-1d-room-calibration-and-frozen-hot-fit).
It applies the response to the two count arms separately and returns the full
signal/background/model curve. The older `calibrate_resolution(...)` API below
fits a normalized transmission spectrum and remains the supported route when
the separate count arms are unavailable.

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

### Tabulated UDR (Monte-Carlo file)

**UDR** = *User-Defined Resolution* — SAMMY's term for a numerical resolution
function supplied as a `(time/energy, weight)` table (SAMMY manual,
"User-Defined Numerical Resolution Function"). In NEREIDS this is a
measured/simulated asymmetric kernel, e.g. a VENUS FTS file:

```python
udr = nereids.load_resolution("fts_bl10.txt", flight_path_m=25.0)
T = nereids.forward_model(energies, [(hf, 5e-5)], temperature_k=300.0, resolution=udr)
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

> **Performance note.** An `IkedaCarpenter` synthesizes its kernel table **once**
> at construction and caches it (`as_tabulated()` just clones it). The cost to know
> about is that IC is **not plan-cached** when fitting the `t0`/`L` energy-scale at
> run time: unlike a loaded `TabulatedResolution`, its broadening plan is rebuilt
> each energy-scale evaluation, so an IC run-time fit over a large grid is slower
> than the tabulated path. Resolution *calibration* is once-per-experiment, so the
> calibrate → pin → fit workflow is unaffected; for production spatial maps, pin
> the calibrated kernel via `as_tabulated()` (which is plan-cached).

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
    family="udr_corr",                  # "gaussian" | "udr_corr" | "ic"
    isotopes=[(hf, 5e-5)],              # KNOWN calibrant composition + density
    temperature_k=300.0,                # KNOWN calibrant temperature
    base_udr=udr,                       # required for family="udr_corr"
    restarts=2,
)
print(cal)                # ResolutionCalibration(family=udr_corr, chi2/dof=..., converged=..., n_free_params=2, bounds_hit=[])
print(cal.params())       # decoded fitted parameters
calibrated = cal.as_tabulated()         # pin this into the sample fit
```

The families calibrate different knobs:

| family       | fits                                        | meaning                                   |
|--------------|---------------------------------------------|-------------------------------------------|
| `gaussian`   | `Δt, ΔL`                                    | analytical Gaussian width                 |
| `udr_corr`   | `s(E)=s0·(E/E_ref)^p` on a base UDR         | trust the MC *shape*, calibrate its width |
| `ic`         | `α(E)=e^c0·√E+e^c1`, `β`, `R` (⊗ PSR triangle) | full bounded analytic moderator shape     |

The `ic` family fits the **complete bounded Ikeda–Carpenter shape**: the
prompt law `α(E) = e^{c0}·√E + e^{c1}` is *positive at every energy by
construction* (the coefficients are exp-encoded — a calibration can no longer
return an `a1 < 0` that flips α negative below the fit window), and the
storage rate `β` and mixing fraction `R ∈ [0, 1]` are free within physics
bounds. The kernel is convolved with the SNS PSR (accumulator-ring) channel
triangle — FWHM `psr_fwhm_ns` (default 350 ns, the value the VENUS FTS file
header records as already folded into the *tabulated* kernel; `0` disables).
Pass `fit_psr=True` to also fit the triangle FWHM as a 5th parameter.
The PSR fold applies to `ic` only: tabulated/UDR kernels already carry it in
the file and are never re-folded.

Every result echoes `n_free_params` and `bounds_hit` — a list of
`"name:lower"` / `"name:upper"` strings for parameters pinned at a box bound.
A pinned bound flags a degenerate direction: e.g. an eV-regime calibrant with
no storage tail drives `R → 0` (`"r:lower"`), and on that β↔R ridge the
reported `β` carries no information.

Use `.as_tabulated()` for `udr_corr` / `ic` (a `TabulatedResolution` to pass as
`resolution=`); use `.gaussian_params()` → `(delta_t_us, delta_l_m)` for the
Gaussian family. For `ic`, `cal.params()` returns the decoded
`{a0, a1, beta, r, psr_fwhm_us}` (the raw `theta` is ln/box-encoded optimizer
space).

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
- **The resolution *width* is well-determined; the resolution *family* is not,
  from a calibrant alone.** The `ic` kernel anchors its *mode* at zero offset, so a
  right-skewed pulse's centroid lags by ~1/α(E), shifting a broadened dip's
  apparent energy. For the `a1=0` prompt law that lag is **≈1/√E — the same basis as
  a flight-path (`L_scale`) error** (leading-order for `a1≠0`) — so absolute dip
  position cannot distinguish an asymmetric kernel from a small `L` miscalibration. By default `calibrate_resolution`
  **pins** the energy scale (`fit_t0=fit_l_scale=False`): a pure shape/width fit on
  the already energy-calibrated grid. The cross-family χ² then discriminates on
  shape **and** position, which is honest only when `(t0, L)` are independently
  known; otherwise the position part is confounded with `L`.
- **To handle position honestly, fit the *shared* energy scale under a metrology
  prior.** Set `fit_t0=True` / `fit_l_scale=True` (centered at `t0_center_us` /
  `l_scale_center`, with Gaussian priors `t0_prior_us` / `l_scale_prior` from the
  instrument's flight-path / timing metrology). The fit reports `position_t0_us`,
  `position_l_scale`, and `prior_penalty`. **Do not fit position with a flat
  prior** — a free `L_scale` absorbs the asymmetric-kernel lag and corrupts the
  calibrated width. With a weak prior, family discrimination collapses toward the
  position-independent *skew/tail* evidence only (χ²/dof ≈ 1.1–1.3 in synthetic
  Hf-177 studies), so report it as a function of the prior strength rather than
  claiming strong discrimination.
- A worked end-to-end example (build the models, calibrate, pin, fit) is in
  `examples/notebooks/workflows/06_resolution_calibration.ipynb`.
