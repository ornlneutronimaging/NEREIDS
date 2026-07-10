# IC resolution-calibration elimination ledger

## Outcome arm

**Unresolved physical root cause.** The synchronized code contains several
validated correctness defects and the archived calibration is demonstrably
practically non-identifiable, but the supplied artifacts do not identify a unique mechanism
for the real-data residuals. A physical count-response approximation and
calibration-stage dense-grid bias remain live implementation hypotheses; the
existing archive probes do not recalibrate IC parameters under either corrected
model.

“Validated mechanism” below means an independent oracle or A/B reproduced the
code behavior. It does not mean that behavior caused the archived residual
unless the archive-domain test says so.

## Candidate-by-candidate record

| Cause class | Candidate mechanism and discriminating test | Evidence/status | Conclusion |
|---|---|---|---|
| Prior result / reproduction | Replay the supplied 1D counts with the cached IC/JENDL-5 calibration on the synchronized v0.3.0 binding. | **Reproduced.** 1058.697 K, density 0.000632509, deviance/dof 28.6325, 7 iterations. [archive replay](evidence/archive-replay.md) | The IC/JENDL-5 result itself reproduces. Its lower error relative to UDR is a coherent cached comparison because the exact UDR input is absent; neither is a causal diagnosis. |
| UDR reference data | Load the exact VENUS FTS file, inspect reference nodes, checksum/provenance, centering, kernel moments, and replay the UDR notebook. | **Blocked.** The named FTS file and raw/SNS inputs are absent. Only cached UDR metrics (992.1 K, deviance 50.0) and notebook prose are present. [archive inventory](archive-inventory.md) | “Sparse UDR is the root cause” is unverified and was overclaimed. |
| Nuclear reference data | Hold resolution fixed and compare Ta-181 libraries/covariances; validate on independent Ta transmission/capture data in the same energy range. | **Supported but confounded.** Cached JENDL alone improves deviance 50.0→37.4. An independent multi-thickness study favors JENDL-5 only over 0.15–100 keV, which does not overlap 8–45 eV. Library choice changes T across 974–1108 K and IC parameters by up to 91×. | Treat JENDL-5 as the strongest archived candidate, not a low-eV oracle; propagate resonance uncertainty and keep Ta out of instrument-model training when Ta thermometry is the target. |
| IC equation, units, normalization | Compare implementation with the Ikeda–Carpenter equation and analytic area/mean/fold moments. | **Validated internally.** 21 IC unit tests pass; area, mean, alpha≈beta, tail direction, fold variance, and plan parity are covered. [tests](evidence/test-results.md) | No current evidence that the base pulse equation or µs/eV units are wrong in the fitted domain. |
| Convolution orientation | Use a delayed-tail oracle and synthetic dip to check whether positive delay shifts apparent energy in the physical direction. | **Validated after recent fix.** Orientation integration test passes; changelog says older correlation-based calibrations are invalid. | Current orientation is correct. Any archive calibration made before commit `c9ccf91` must be discarded; the supplied v0.3.0 replay is after it. |
| Count-response formation | Compare the current `R[T]` path with `R[Phi*T]/R[Phi]` using a true-energy-conditioned response, then recalibrate/freeze on RT data and predict the hot run. | **Mechanism validated synthetically; archive effect untested.** The current code broadens transmission and profiles same-bin flux. A probability-matrix control agrees exactly for flat flux but differs by 0.2106 for structured flux. [probe](evidence/count-response.md) | This is a P0 physical-model defect/candidate cause. Implement the two-arm response before interpreting residuals or IC parameters. |
| Numerical working grid | Compare coarse direct convolution with dense-convolve-then-sample; then recalibrate all IC parameters under both grids and transfer to hot data. | **Code defect validated; sample-stage effect only measured.** Synthetic error is 0.047622. With cached IC parameters frozen, a 20k-grid WLS T/density re-fit worsens SSR 1.507% and shifts T +4.38 K. It does not refit `a0/a1/beta/R` and is not a production count likelihood. [probe](evidence/code-probes.md), [archive A/B](evidence/dense-grid.md) | Must be fixed. The existing A/B rules against a large sample-application effect at the frozen cache, but calibration-stage bias remains open. |
| Energy/flight-path coupling | Derive and probe non-unit `L_scale` using nominal versus physically effective resolution flight path. | **Code defect validated; cached direct effect small.** Width is erroneously multiplied by `L_scale`; at archive scale 1.0008377, fixing L changes at most 0.078σ and SSR by 0.085%. [probe](evidence/code-probes.md), [archive sensitivity](evidence/archive-numerics.md) | Fix and regression-test. It cannot directly explain 14–64σ features at cached parameters; an end-to-end recalibration effect was not tested. |
| Tau discretization | Compare sampled Gamma(3) moments with the analytic variance and vary `n_tau` at archive parameters. | **Boundary defect validated; modest archive sensitivity.** Eight samples understate variance 26.226%; archive 400/500/600/1200 choices change predictions by up to 0.97σ, not 14–64σ. [probe](evidence/code-probes.md), [archive sensitivity](evidence/archive-numerics.md) | Replace the acceptance floor with an error-controlled grid and use the same synthesis settings in calibration and science. This is not sufficient to explain the large residuals at the cached optimum. |
| Parameter bounds/domain | Inventory every IC/PSR/position bound and synthesize the true box corner. | **Validated domain defect.** Only `R in [0,1]` is directly physical; exact rate/PSR/position boxes are heuristic. The true `a0=5,a1=2,beta=.02,R=1,PSR=.35 us` corner fails the tau-cap criterion, while the named worst-corner test uses `a0=.5`. The PSR `.05 us` lower fit bound is infeasible at the default beta/R start. [audit](code-audit.md), [probe](evidence/code-probes.md) | Replace the rectangular search box with a source/prior-backed and numerically feasible domain; correct the misleading boundary test. |
| R threshold / optimizer continuity | Cross `R_NEGLIGIBLE=1e-9` at fixed alpha/beta and compare grid/moments. | **Validated defect.** An infinitesimal crossing changes 1004 points to 16 and variance error −0.02%→−3.58%; at `R=1e-6`, −19.72%. [probe](evidence/code-probes.md) | Remove the hard threshold from grid sizing or make support/error tolerance continuous before using gradients/profile likelihood. Cached R≈0.157 is far from this threshold. |
| Centering convention | Compare mode-centered IC, raw UDR, and SAMMY centroid-centered kernels against an independently timed response. | **Implementation difference confirmed; data effect blocked.** Local SAMMY source recenters UDR to centroid zero; NEREIDS IC uses mode zero and raw UDR is not recentered. The real FTS file is absent. | Do not change convention blindly. Store explicit origin metadata and run mode/centroid/absolute-delay A/B with prompt timing or measured response. |
| IC energy law | Compare `a0√E+a1`, inverse-wavelength alpha, and energy-dependent R on held-out lines. | **Model discrepancy, not yet discriminated.** A KENS H2O-moderator eV measurement found width approximately E^-0.48; Mantid uses inverse-wavelength alpha and energy-dependent R for a different peak model. Neither validates a VENUS law. Archived `a1` is at its lower bound. | Reduce first to `alpha=a√E` as an identifiable candidate; compare R=0 and independently constrained candidate R(E) laws rather than interpreting constant R≈0.15 as moderator physics. |
| Missing physical components | Independently measure/simulate moderator, PSR, path distribution, detector timing, and bin integration; model sample transport per run; add one term at a time and validate held out. | **Model incompleteness confirmed; responsible component unverified.** Calibrator contains IC moderator plus an inherited 350 ns PSR setting; the FTS provenance is absent. Burst Gaussian is disabled and detector/path/bin responses are absent. | Build a shared instrument operator from measured/prior-constrained terms and a separate material/geometry-dependent transport operator; do not add free IC coefficients to absorb both. |
| Calibration identifiability | Compute the concentrated finite-difference Jacobian/Fisher SVD, state parameter scaling, compare independent archived optima, and remove the inner fit mask. | **Validated practical non-identifiability of `a1`.** In log/logit coordinates the raw condition is 13,206.8 but column normalization makes it 8.65. `a1` is at its lower bound, its column norm is 3,572× below `a0`, and 48.9×-different values differ only 0.149 RMS sigma. All 4,280 raw-selected bins give the same least direction. [identifiability](evidence/identifiability.md) | Remove/freeze `a1` first. Use explicit scaling plus nonlinear profiles and predictive transfer; do not use a condition threshold alone. |
| Optimizer / initialization | Vary starts, grids, and restarts; compare fit predictions and profiles rather than parameter vectors alone. | **Sensitivity supported.** Notebook uses 48×400/two restarts; robust grid uses 64×500/three and finds materially different parameters with almost identical alpha/prediction. Starts are hard-coded and Nelder–Mead evaluation counts are discarded. | UDR is not a defensible source of truth. Use moment/transport priors, space-filling multi-start, profile-based selection, and held-out likelihood. |
| Calibration likelihood/background | Fit calibration and science with the same two-arm response-aware count likelihood and nuisance model; compare with current Gaussian-transmission/additive baseline calibration. | **Confirmed mismatch; effect unmeasured.** Calibration profiles additive constant/linear terms; science uses a same-bin joint-Poisson objective and multiplicative log-energy baseline. Neither forms latent open/sample spectra through energy migration. | Implement a true-energy response-matrix count objective, including open-beam uncertainty/latent flux, before interpreting IC parameters physically. |
| Energy/time calibration | Validate t0 and L with prompt timing/Bragg edges or metrology, not the same Ta spectrum. | **Not independently validated.** Archived self-calibration changes hot T only +0.6 K, but it uses the same model/data; the IC sanity position fit is discarded. | Retain t0/L as metrology-prior coordinates shared across all calibrants; do not let IC lag determine them. |
| Baseline and fit range | Disclose both nested exclusions: 5,304 cache bins to 4–120 eV, then to the 8–45 eV fit; run every available with/without control. | **Measured with blocker.** Activating all 4,280 raw-selected bins changes T -7.026 K/density -1.521%. Activating all 5,304 with current endpoint clamp changes T -19.749 K/-2.970%, but 1,031 bins clamp and full-domain IC synthesis fails at 401 keV. [fit-window A/B](evidence/fit-window.md) | Both scopes are justified here only for exact reproduction, not physically. The outer without-exclusion diagnostic is invalid as physics but exposes the blocker rather than hiding the data. |
| Contaminants | Add independently plausible W/Ta-180m/Fe/Cr/Ni/Cu candidates with assay-informed priors and compare held-out features. | **Unverified.** The archived “14 eV not contaminant” script never performs this scan. | Contamination remains open; a library sign flip cannot exclude contaminant plus changed Ta parameters. |
| Self-shielding / multiple scattering | Measure multiple foil thicknesses and use a per-run transport operator; test shared instrument response after changing optical depth. | **Unverified.** Beer–Lambert attenuation order is correct, but multiple scattering/in-scattering and finite geometry are absent and are sample-specific. | Use non-black lines and multi-thickness data, but never fold sample transport into a transferable instrument convolution. |
| Data reduction / detector/background | Rebuild counts from raw data with union masks, per-panel diagnostics, empty/black runs, and compare spectra/maps. | **Blocked, with warning evidence.** Raw cubes are absent; archived `common.py` cannot find the open-beam `ob/` hierarchy and masks only OB dead pixels. Both cached maps retain a column-256 panel step. | Preserve raw inputs and provenance; quantify per-panel timing/efficiency before attributing spatial structure to resolution. |
| Performance-induced approximation | Measure component times and vary numerical grids under parity checks. | **Measured.** 48×400 synthesis 7.53 ms; no-resolution forward 11.70 ms; reused-IC forward 100.19 ms; composite 109.11 ms. Archive maps are 6.08× slower; calibrations 665–1490 s. [performance](evidence/performance.md) | The analytical pulse does not make application analytical. Optimize tabulated convolution/plan construction first; parallel synthesis is secondary. |
| Out-of-range reuse | Apply calibrated IC beyond its synthesized range. | **Confirmed silent behavior and exposed by disclosure controls.** The 8–45 result is inside; the 4–120 raw-bin control has 7 corrected bins beyond 120 eV and the all-cache control has 1,031, all endpoint-clamped. Full-domain law synthesis fails under the tau cap. [fit-window record](evidence/fit-window.md) | Preserve analytical laws or reject out-of-range application explicitly; do not treat the all-cache clamp diagnostic as a physical fit. |
| Input/API validation | Probe invalid UDR flight paths/all-zero kernels and extreme finite IC rates. | **Confirmed boundary defects, no archive evidence.** Rust/GUI constructors accept zero/negative/NaN/infinite L; extreme direct IC rate can yield NaN. [probe](evidence/code-probes.md) | Add validation, but these are not explanations for the supplied valid-domain fit. |

## Exact command register

Every executed test kept its configured data; no affected resonance was
masked, downweighted, smoothed, or removed.

| ID | Exact command | Exit | Preserved output |
|---|---|---:|---|
| G1 | `git fetch --prune origin`; `git merge --ff-only origin/main`; `git rev-parse main origin/main`; `git rev-list --left-right --count main...origin/main` | 0 | [git-sync.md](evidence/git-sync.md) |
| A0 | `unzip -t /Users/chenzhang/Downloads/Archive.zip` | 0 | [archive-inventory.md](archive-inventory.md) |
| A1 | `pixi run python investigation/archive_audit.py --extract-images /tmp/nereids-notebook-images-8b5afb3f1a4e` | 0 | [archive-inventory.md](archive-inventory.md) |
| A2 | `pdfinfo /tmp/nereids-archive-8b5afb3f1a4e/01_spectral_lineshape_bias/report/report.pdf` | 0 | [archive-inventory.md](archive-inventory.md) |
| A3 | `pdftoppm -png -r 120 /tmp/nereids-archive-8b5afb3f1a4e/01_spectral_lineshape_bias/report/report.pdf /tmp/nereids-prior-report-8b5afb3f1a4e` | 0 | [archive-inventory.md](archive-inventory.md) |
| B1 | `pixi run build` | 0 | [archive-replay.md](evidence/archive-replay.md) |
| R1 | `/usr/bin/time -p pixi run python investigation/reproduce_ic_cached.py` | 0 | [archive-replay.md](evidence/archive-replay.md) |
| I1 | `/usr/bin/time -p pixi run python investigation/identifiability.py --fit-min 8 --fit-max 45` | 0 | [identifiability.md](evidence/identifiability.md) |
| I2 | `/usr/bin/time -p pixi run python investigation/identifiability.py --fit-min 4 --fit-max 122` | 0 | [identifiability.md](evidence/identifiability.md) |
| CR1 | `/usr/bin/time -p pixi run python investigation/count_response_probe.py` | 0 | [count-response.md](evidence/count-response.md) |
| P1 | `/usr/bin/time -p pixi run python investigation/performance_probe.py` | 0 | [performance.md](evidence/performance.md) |
| D1 | `/usr/bin/time -p pixi run python investigation/dense_grid_probe.py` | 0 | [dense-grid.md](evidence/dense-grid.md) |
| D2 | `/usr/bin/time -p pixi run python investigation/dense_grid_refit.py` | 0 | [dense-grid.md](evidence/dense-grid.md) |
| N1 | `/usr/bin/time -p pixi run python investigation/archive_numerics_probe.py` | 0 | [archive-numerics.md](evidence/archive-numerics.md) |
| W1 | `/usr/bin/time -p pixi run python investigation/fit_window_probe.py` | 0 | [fit-window.md](evidence/fit-window.md) |
| W2a | `/usr/bin/time -p pixi run python investigation/fit_window_probe.py --all-raw` | 1 in the initial full-domain-synthesis probe version | [fit-window.md](evidence/fit-window.md) |
| W2b | `/usr/bin/time -p pixi run python investigation/fit_window_probe.py --all-raw` | 0 in the final explicit endpoint-clamp control | [fit-window.md](evidence/fit-window.md) |
| W3 | `/usr/bin/time -p pixi run python investigation/full_domain_synthesis_probe.py` | 0 (caught construction error) | [fit-window.md](evidence/fit-window.md) |
| C1 | `/usr/bin/time -p pixi run python investigation/run_code_probes.py` | 0 | [code-probes.md](evidence/code-probes.md) |
| T1 | `cargo test -p nereids-physics ikeda_carpenter::tests -- --nocapture` | 0 | [test-results.md](evidence/test-results.md) |
| T2 | `cargo test -p nereids-physics resolution::tests -- --nocapture` | 0 | [test-results.md](evidence/test-results.md) |
| T3 | `cargo test -p nereids-physics --test kernel_orientation --test kernel_width_interpolation --test venus_usr_resolution -- --nocapture` | 0 | [test-results.md](evidence/test-results.md) |
| T4 | `cargo test -p nereids-fitting resolution_calib::tests -- --nocapture` | 0 | [test-results.md](evidence/test-results.md) |
| T5 | `cargo test -p nereids-physics --test samtry_validation -- --nocapture` | 0 | [test-results.md](evidence/test-results.md) |
| T6 | `cargo test -p nereids-fitting rejects_infeasible_psr_start_width -- --nocapture` | 0 | [test-results.md](evidence/test-results.md) |
| V1 | `pixi run python investigation/verify_artifacts.py` | 0 | [artifact-verification.md](evidence/artifact-verification.md) |

## Most informative next test

After count formation and the four high-priority numerical/cross-layer fixes
are regression-tested,
run one independent VENUS response campaign in unchanged geometry:

1. acquire raw event/TOF data, prompt timing, empty/black background, and exact
   instrument configuration;
2. measure at least two non-Ta thin calibrants and two optical depths at known
   temperature, chosen to provide non-black isolated resonances across the
   eV range; keep Ta and one calibrant completely held out;
3. fit one shared true-energy instrument response (moderator IC or source prior
   + PSR + path + detector + bin integration), applied separately to latent
   open/sample spectra, with per-run sample transport, flux/background
   nuisance, and t0/L metrology priors;
4. publish parameter profiles/Jacobian singular values and predict held-out Ta
   and held-out calibrant spectra without refitting the instrument response.

This is discriminating because it breaks the present Ta-nuclear-data/instrument
confound and tests transfer, not merely fit quality. A successful model must
improve held-out Poisson likelihood and the predeclared 10.35/13.92/23.95/
35.17/39.15 eV residuals, while producing finite profile intervals and
consistent parameters across thicknesses. Otherwise the candidate mechanism is
rejected rather than absorbed into more IC freedom.
