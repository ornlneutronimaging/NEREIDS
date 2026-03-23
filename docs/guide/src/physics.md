# Physics Reference

NEREIDS implements exact SAMMY physics for neutron resonance imaging. This
chapter is a navigation guide to the rustdoc API documentation, not a
standalone physics textbook.

All implementations reference specific sections of the SAMMY manual and
SAMMY Fortran source files. See the rustdoc for each module for detailed
equations and citations.

## Cross-Section Formalisms

| Formalism | ENDF LRF | Module | SAMMY Reference |
|-----------|----------|--------|-----------------|
| Reich-Moore | LRF=3 | [`reich_moore`](api/nereids_physics/reich_moore/) | Manual Sec 2, `rml/` |
| Single-Level Breit-Wigner | LRF=1,2 | [`slbw`](api/nereids_physics/slbw/) | Manual Sec 2, `mlb/` |
| R-Matrix Limited | LRF=7 | [`rmatrix_limited`](api/nereids_physics/rmatrix_limited/) | Manual Sec 2 |
| Unresolved Resonance Region | LRU=2 | [`urr`](api/nereids_physics/urr/) | Hauser-Feshbach |

The [`penetrability`](api/nereids_physics/penetrability/) and
[`channel`](api/nereids_physics/channel/) modules provide the underlying
nuclear physics: hard-sphere phase shifts, penetrability factors, wave numbers,
and statistical spin weights.

## Broadening Models

### Doppler Broadening

Free Gas Model (FGM) convolution accounting for thermal motion of target nuclei.

- Module: [`doppler`](api/nereids_physics/doppler/)
- SAMMY reference: `dop/` module, manual Sec 3.1
- Key function: `doppler_broaden()` using psi/chi auxiliary functions on an adaptive grid

### Resolution Broadening

Instrument resolution broadening from flight-path uncertainty, timing jitter,
and moderator pulse width.

- Module: [`resolution`](api/nereids_physics/resolution/)
- SAMMY reference: `convolution/` module, manual Sec 3.2
- Supports: Gaussian convolution, Gaussian + exponential tail, tabulated resolution functions

## Transmission Model

Beer-Lambert transmission: T(E) = exp(-sum_i n_i sigma_i(E))

Where n_i is the areal density (atoms/barn) and sigma_i(E) is the broadened
total cross-section for isotope i.

- Module: [`transmission`](api/nereids_physics/transmission/)
- SAMMY reference: `cro/`, `xxx/` modules, manual Sec 2, Sec 5
- Handles multi-isotope samples with shared Doppler temperature
  (one global temperature parameter, optionally fitted jointly with densities)

## Fitting Engines

### Levenberg-Marquardt

Standard nonlinear least-squares minimization for Gaussian-distributed data.

- Module: [`lm`](api/nereids_fitting/lm/)
- SAMMY reference: `fit/` module, manual Sec 4
- Parameters: areal densities with optional bounds, optional temperature fitting

### Poisson KL Divergence

Maximum-likelihood fitting for low-count data where Gaussian statistics break down.

- Module: [`poisson`](api/nereids_fitting/poisson/)
- Reference: TRINIDI approach (`trinidi/reconstruct.py`)
- Uses bounds-based preconditioning for joint density + temperature fits

## ENDF Nuclear Data

Resonance parameters are sourced from the ENDF/B library via the IAEA API:

- Module: [`retrieval`](api/nereids_endf/retrieval/) -- download and cache
- Module: [`parser`](api/nereids_endf/parser/) -- parse ENDF-6 File 2
- Module: [`resonance`](api/nereids_endf/resonance/) -- data structures

Supported libraries: ENDF/B-VIII.0, ENDF/B-VIII.1, JEFF-3.3, JENDL-5.

## Further Reading

- SAMMY User's Guide (ORNL/TM-9179/R8)
- [ENDF-6 Formats Manual](https://www.nndc.bnl.gov/csewg/docs/endf-manual.pdf) (BNL-203218-2018-INRE)
- [ENDF/B-VIII.0](https://doi.org/10.1016/j.nds.2018.02.001) (Nuclear Data Sheets, 2018)
