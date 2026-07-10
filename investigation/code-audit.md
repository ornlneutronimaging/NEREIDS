# IC instrument-resolution code audit

Audit started after synchronizing `main` to `origin/main` at
`19d0ea28b967f6772a36025cb4184a3b2e149b0f`. Primary source audit completed
2026-07-09T19:13:22-04:00, before any modeling recommendation was written.
The task contract itself is the only earlier research-branch commit.

This document records code facts and open risks. A risk is not presented as the
cause of a real-data residual unless a later reproduction or discrimination test
validates that mechanism.

## Workspace and ownership map

- `crates/nereids-endf` parses evaluated nuclear data into `ResonanceData`.
- `crates/nereids-physics` owns Reich-Moore cross sections, Doppler broadening,
  Beer-Lambert transmission, all resolution kernels, and the IC pulse.
- `crates/nereids-fitting` owns fit models, LM/Nelder-Mead optimizers, and the
  resolution calibrator.
- `crates/nereids-pipeline` assembles fitting and spatial workflows and caches
  tabulated-resolution plans for production fits.
- `bindings/python` exposes `IkedaCarpenter`, `TabulatedResolution`, and
  `calibrate_resolution`; the GUI currently exposes Gaussian or loaded
  tabulated resolution, not IC calibration (`apps/gui/src/widgets/design.rs`,
  lines 852-877).

## End-to-end calibration data flow

1. Python's `calibrate_resolution` validates the arrays, known calibrant
   composition, family, IC synthesis sizes, and PSR units, then releases the GIL
   and calls Rust (`bindings/python/src/lib.rs`, lines 1440-1676).
2. `nereids-fitting::resolution_calib::calibrate_resolution` requires the
   calibrant density and temperature to be fixed and validates only that the
   number of data points exceeds the nominal parameter count
   (`crates/nereids-fitting/src/resolution_calib.rs`, lines 799-872).
3. Each outer-objective call constructs a candidate resolution with
   `build_resolution` (lines 577-635), optionally maps the nominal energy grid
   through `(t0, L_scale)`, and calls the full physics `forward_model`
   (lines 1084-1115).
4. `forward_model` recomputes Reich-Moore cross sections, Doppler-broadens them,
   forms total attenuation, applies Beer-Lambert, and only then broadens the
   total transmission (`crates/nereids-physics/src/transmission.rs`, lines
   606-704). This is the correct ordering within a transmission-only model, but
   it is not the complete measured-count model: the code computes `R[T]`, not
   `R[Phi*T]/R[Phi]` from separate open/sample count arms. The current
   joint-Poisson objective profiles an independent same-bin flux after this
   broadening (`crates/nereids-fitting/src/joint_poisson.rs`, lines 12-28), so
   it also omits true-to-measured bin migration in the incident flux.
5. The calibrator profiles out one multiplicative normalization, or that term
   plus unconstrained constant and linear additive terms, by solving normal
   equations; the outer loss is raw weighted SSR plus any metrology prior
   (`resolution_calib.rs`, lines 637-761 and 1109-1115).
6. Bounded Nelder-Mead performs the resolution search. One nominal restart may
   launch up to five larger simplex re-inflations (`resolution_calib.rs`, lines
   1074-1150). The optimizer counts function evaluations, but
   `CalibrationResult` discards `n_evals` and exposes only iterations,
   convergence, reduced data chi-square, bound hits, and the best kernel
   (`nelder_mead.rs`, lines 75-89 and 324-329; `resolution_calib.rs`, lines
   1171-1233).

## Resolution families and parameterization

- Gaussian fits `(delta_t_us, delta_l_m)` with two bounded coordinates
  (`resolution_calib.rs`, lines 294-369).
- UDR correction fits `s(E) = s0 (E/10 eV)^p` and scales each reference block
  about its trapezoidal intensity centroid (`resolution.rs`, lines 915-986).
- IC fits four coordinates by default:
  `alpha(E) = exp(theta0) sqrt(E) + exp(theta1)`, `beta = exp(theta2)`, and a
  scalar `R = theta3`; fitting the PSR triangle adds a fifth coordinate
  (`resolution_calib.rs`, lines 304-318, 372-405, and 598-632).
- The IC calibration family hard-codes `burst_sigma_us = None` and includes
  only the optional symmetric PSR triangle (`resolution_calib.rs`, lines
  609-623). It has no calibration coordinate for detector response, flight-path
  distribution, acquisition-bin width, or a general Gaussian timing term.
- The default PSR value is 350 ns. The Rust and Python entry points guard a
  nanosecond/microsecond mistake because the direct sampled triangle convolution
  becomes quadratic in its sampled width (`resolution_calib.rs`, lines 114-153
  and 883-923; Python lines 1622-1654). The comment attributes 350 ns to the
  missing FTS header; the supplied archive does not independently preserve that
  measurement.

### Complete IC/position bound audit

| Coordinate | Encoding / units | lower / start / upper | Assessment |
|---|---|---:|---|
| `a0` | `exp(theta0)`, us^-1 eV^-1/2 | 0.01 / 0.30 / 5.0 | source-scale-informed, exact box/start heuristic |
| `a1` | `exp(theta1)`, us^-1 | 0.001 / 0.05 / 2.0 | positivity guard; exact values heuristic; excludes `a1=0` |
| `beta` | `exp(theta2)`, us^-1 | 0.02 / 0.10 / 5.0 | source-scale-informed, exact box/start heuristic |
| `R` | direct fraction | 0 / 0.1 / 1 | `[0,1]` physical; constant law/start heuristic |
| fitted PSR FWHM | direct us | 0.05 / 0.35 default / 1.0 | heuristic; 350 ns provenance absent |
| optional `t0` | direct us | -5 / 0 / `min(5,minTOF-1e-6)` | guardrail; intended constraint is metrology |
| optional `L_scale` | direct | 0.98 / 1 / 1.02 | heuristic guardrail; intended constraint is metrology |

Values and transforms are in `resolution_calib.rs`, lines 56-129, 171-193,
341-405, 598-630, and 981-1015. Only the fraction bound on `R` is directly
physical. The true rate-box corner (`a0=5`, `a1=2`, `beta=.02`, `R=1`, PSR
`.35 us`) fails the 8192-point tau-cap criterion at 24.178 eV; the test named
`ic_box_worst_corner_synthesizes_within_tau_cap` instead uses `a0=.5`
(`resolution_calib.rs`, lines 2607-2643). The fitted PSR lower bound `.05 us`
is also infeasible at the default beta/R start; the threshold is about
`.0586 us` (lines 1016-1047 and 2460-2506).

The nominal simplex is not a substantive multistart. Its initial raw-coordinate
perturbations are one-sided in physical space, and each extra restart shifts all
coordinates upward by 10% of the raw box. With the default `restarts=1`, no
independent basin is tried (`resolution_calib.rs`, lines 1075-1083;
`nelder_mead.rs`, lines 172-205).

## IC pulse and synthesis

- `ic_pulse` implements a unit-area Gamma/Erlang-3 prompt term plus that prompt
  convolved with an exponential storage term. The documented mean is
  `3/alpha + R/beta` (`crates/nereids-physics/src/ikeda_carpenter.rs`, lines
  13-41 and 232-266).
- The alpha-near-beta limit uses a Taylor branch and the general branch avoids
  the earlier overflow-prone factorization (`ikeda_carpenter.rs`, lines
  220-266).
- Direct API laws include constant, `a0 sqrt(E)+a1`, inverse wavelength, and
  exponential-in-meV storage fraction, but the calibrator exposes only the
  positive `sqrt(E)` alpha law and a constant `R` (`ikeda_carpenter.rs`, lines
  268-317; `resolution_calib.rs`, lines 609-618).
- Every candidate IC resolution synthesizes a log-spaced table. Calibration
  uses 64 reference energies by 500 prompt samples by default and extends the
  table only over `[0.5 E_min, 2 E_max]` (`resolution_calib.rs`, lines 469-496
  and 624-630).
- The tau grid extends to 18 prompt e-folds and, for `R > 1e-9`, 16 storage
  e-folds. A cap of 8192 pulse-body samples may widen the step down to a stated
  floor equivalent to only eight prompt samples; impossible fold/prompt
  combinations fail loudly (`ikeda_carpenter.rs`, lines 157-218 and 583-635).
- Gaussian and triangle folds use direct discrete convolution
  (`ikeda_carpenter.rs`, lines 652-797). Synthesis over reference energies is
  sequential (`IkedaCarpenter::new`, lines 500-504).
- Each synthesized kernel is trimmed at `1e-7` of its peak, peak-normalized, and
  mode-anchored at time offset zero (`ikeda_carpenter.rs`, lines 668-717).

## Tabulated application path

- IC is analytical only at synthesis. It is immediately wrapped as a
  `TabulatedResolution` and subsequently uses exactly the UDR broadener
  (`resolution.rs`, lines 1152-1176 and 2590-2619).
- The broadener performs the convolution gather at `t - offset`, not the old
  time-mirrored correlation (`resolution.rs`, lines 1990-2037 and 2078-2135).
- Between reference energies, NEREIDS departs intentionally from SAMMY's linear
  element-wise interpolation: it blends width-normalized shapes at a
  geometrically interpolated RMS width (`resolution.rs`, lines 2360-2581).
- The blend allocates fresh offset and weight vectors for every target energy
  (`resolution.rs`, lines 2411-2413 and 2503-2507).
- Reusable `ResolutionPlan` and CSR matrix paths exist and support both
  Tabulated and IC variants (`resolution.rs`, lines 2623-2657), but runtime
  energy-scale plan caching deliberately accepts only the Tabulated enum
  variant (`crates/nereids-fitting/src/transmission_model.rs`, lines
  1985-2041). Converting a calibrated IC to `TabulatedResolution` therefore
  changes performance without changing its stored kernel.
- A fitted flight-path scale is not carried into the resolution object's TOF
  transform. The corrected coordinate uses `L_eff=L_nom*L_scale`, while
  `TabulatedResolution::broaden` reconstructs TOF with its stored nominal
  `flight_path_m` (`transmission_model.rs`, lines 1796-1803 and 2059-2074;
  `resolution.rs`, lines 2070-2092). Algebra and an independent probe show
  that the implementation therefore scales a physical kernel delay by
  `L_scale`: width ratios were 1.0049951 at `L_scale=1.005` and 1.0199861
  at 1.020. This is a confirmed cross-layer defect.
- Table lookup clamps below and above the synthesized reference energies
  (`resolution.rs`, line 2439). A calibrated IC table spans only
  `[0.5 E_min, 2 E_max]`, so reuse on a wider experiment silently freezes
  the endpoint kernel instead of continuing the fitted analytical energy law.

## Centering convention: confirmed implementation difference

- `TabulatedResolution::from_text` validates blocks but does not normalize or
  recenter them (`resolution.rs`, lines 1694-1850).
- IC synthesis subtracts the sampled mode, not the intensity centroid
  (`ikeda_carpenter.rs`, lines 687-717).
- Current NEREIDS code explicitly documents this as an intentional departure:
  SAMMY's live `Gen_Udr_Par` path re-aligns each interpolated UDR component to
  trapezoidal centroid zero, while NEREIDS keeps mode zero (`resolution.rs`,
  lines 2380-2402).
- The independent local SAMMY source confirms the mechanism at
  `/Users/chenzhang/code.ornl.gov/SAMMY/SAMMY/sammy/src/udr/mudr3.f90`, lines
  267-292: it calculates `Ct` and subtracts it from every time point before
  convolution. The SAMMY manual additionally treats centering as an explicit
  choice for RPI/GELINA/nTOF resolution functions (manual Section III.C.3.b).
- This is a verified convention difference, but whether the VENUS file is
  already centroid-centered, mode-centered, or carries an absolute delay is a
  data/provenance question. Therefore this audit does not yet label the
  difference a defect or residual cause.

## Working-grid behavior: confirmed family asymmetry

- Fine-structure and boundary auxiliary grids are built only for
  `ResolutionFunction::Gaussian`; Tabulated and IC use the observed data grid
  directly (`transmission.rs`, lines 37-102 and 360-380).
- Consequently a resonance narrower than or off the observed grid can be
  undersampled before tabulated/IC convolution, whereas the Gaussian path gets
  resonance-aware intermediate points. This is a code-level family asymmetry,
  not just a speed difference.
- An independent probe placed a 0.005 eV-wide dip at 15.05 eV between 0.1 eV
  coarse-grid points. The coarse input's minimum was 1.0 versus 0.2 on a dense
  grid, and direct coarse convolution differed from dense-convolve-then-sample
  by 0.047622 at 15.10 eV. This validates a mechanism capable of producing a
  localized, grid-dependent residual. It has not been run on the missing raw
  VENUS/UDR inputs, so attribution of the archived features remains unverified.

## Tau-grid accuracy and threshold behavior: confirmed defects

- The 8,192-point cap may reduce the prompt core to the accepted
  `MIN_N_TAU=8` floor. Against the independent Gamma(3, alpha=1) variance
  oracle of 3, a direct probe returned 2.213235 at 8 samples (−26.226%),
  2.920245 at 16 (−2.658%), and 2.999037 at 64 (−0.032%).
- This regime is reachable inside calibration bounds: at `beta=.02` with an
  active slow component, the 800 us reach forces about 0.097668 us spacing;
  `alpha≈26` then has only about 0.4 samples per prompt time constant.
- `R_NEGLIGIBLE=1e-9` makes the numerical objective discontinuous. At
  `alpha=26`, `beta=.02`, and the default 0.35 us triangle, changing R from
  exactly `1e-9` to `1.0001e-9` changed the retained grid from 1,004 points
  at 0.001387 us spacing to 16 at 0.097668 us; variance error jumped from
  −0.02% to −3.58%. At `R=1e-6` it was −19.72%.
- Existing worst-corner tests assert successful construction but do not test
  moment accuracy or continuity (`resolution_calib.rs`, near line 2606).

## Validation and diagnostics actually returned

- Simple point-count validation is not a rank/identifiability check. There is
  no calibration Jacobian, Hessian/Fisher matrix, singular-value report,
  profile likelihood, covariance, confidence interval, or posterior.
- `bounds_hit` detects coordinates near hard bounds. It exposes the obvious
  `R -> 0` / unconstrained-beta ridge, but an interior ridge is silent
  (`resolution_calib.rs`, lines 1193-1220).
- Model-family results have different parameter counts (2 vs 4-5), but the API
  reports only reduced chi-square; there is no held-out score, AIC/BIC, or
  likelihood-ratio validity check.
- Known calibrant density, temperature, nuclear-data parameters, and their
  uncertainties are fixed rather than propagated.
- Resolution calibration accepts Gaussian transmission uncertainties only. It
  does not expose the repository's raw-count Poisson likelihood or open-beam
  covariance model.
- A finite-difference concentrated Jacobian on all 2,312 archived RT bins had
  singular values `[430.28, 130.48, 67.14, 0.03258]` and condition number
  13,206.8 in `(log a0, log a1, log beta, logit R)`. The least-informed
  direction was 0.9999999 `log a1`; the projected `a0/a1` derivative
  correlation was 0.9415. Column normalization changes the condition number to
  8.65, so the raw value is coordinate-scale dependent; the bound hit, tiny
  `a1` column, and predictive ridge—not a universal cutoff—confirm practical
  underdetermination. The full 4–120 eV probe reaches the same conclusion.

## Real-instrument components represented and absent

The repository's bundled SAMMY manual (Section III.C) says a realistic
instrument resolution is site/setup specific. Its ORR example separately
models source/moderator, burst, flight-path distribution, detector response,
and TOF channel width; its UDR definition is a convolution of burst, channel,
and one or more numerical components. Visual rendering of manual pages 155 and
175 confirmed the extracted equations and component list.

Current calibrated IC represents the moderator IC pulse plus a symmetric PSR
triangle. The other listed components are absent from the calibration family,
even though the direct IC struct has one unused optional Gaussian burst field.
This establishes model incompleteness relative to the general instrument
component decomposition, but not which absent term matters for VENUS.

## Tests and evidence read

- `cargo test -p nereids-physics ikeda_carpenter::tests -- --nocapture` completed with
  `21 passed; 0 failed`; it covers pulse area/mean, alpha-near-beta stability,
  tail orientation, fold variance, tau-cap errors, synthesis validity, and
  direct/plan equality.
- `cargo test -p nereids-fitting resolution_calib::tests -- --nocapture`
  completed with `33 passed; 0 failed` in 1329.99 s. Six IC cases crossed
  Rust's 60-second long-test notice; the final alpha-recovery test consumed most
  of the elapsed time.
- The synthetic closed loop in
  `crates/nereids-fitting/tests/ic_closed_loop.rs` generates truth with the same
  IC family, derived grid, and forward model used for recovery. It is valuable
  loop-closure coverage, not an external physics oracle (lines 96-123).
- Kernel orientation and width interpolation have independent synthetic
  integration tests, but the real VENUS UDR file is intentionally not shipped;
  tests use a much narrower synthetic kernel
  (`crates/nereids-physics/tests/venus_usr_resolution_microbench.rs`, lines
  1-23).

## Recent IC/UDR history inspected

`git log --oneline --all --` over the IC, resolution, and calibrator sources
shows the implementation was materially revised after its first landing:

- `da32da6` / earlier `59d71e2` and `913eb63`: IC model and calibration
  framework;
- `60b4b56`: shared metrology-priored energy scale;
- `c9ccf91`: correlation-to-convolution correction and exact support;
- `4587360`: width-normalized geometric kernel interpolation;
- `42b1c16`: current bounded four/five-parameter IC calibration and tau/fold
  changes;
- `1ee5cc0`: joint energy-scale/temperature fitting support.

This history explains why passing current loop-closure tests cannot validate
older cached calibrations. The v0.3.0 archive replay uses current code; any
calibration artifact lacking a commit ID cannot be assumed current.

## Open code-level risks to discriminate against real data

1. **Confirmed code defect:** missing resonance-aware working grid for IC/UDR.
2. **Unverified data convention:** mode-versus-centroid/absolute-delay mismatch.
3. **Confirmed numerical defect:** cap-limited IC kernels can use an inaccurate
   `n_tau = 8` floor inside calibration bounds.
4. **Confirmed numerical defect:** discontinuity at the `R > 1e-9`
   storage-tail grid-sizing threshold.
5. Missing detector, path-length, digitizer-bin, and additional timing response.
6. Constant eV-regime `R` compensating for omitted physics rather than
   representing moderator storage.
7. **Confirmed API behavior:** IC freezes outside its finite synthesized
   energy range instead of continuing the analytical law.
8. Reference-library/resonance-parameter mismatch or missing covariance.
9. Fixed calibrant density/temperature and energy scale transferring their
   uncertainty into the resolution.
10. Additive baseline/Gaussian-transmission loss mismatching the data-generating
    process.
11. **Confirmed practical identifiability defect:** hard-coded starts and local
    simplex search choose one point on a broad ridge without reporting it.
12. **Measured performance issue:** full physics recomputation and tabulated
    convolution dominate; sequential synthesis and allocation are secondary.
13. **Confirmed physical-model discrepancy:** transmission is broadened
    directly instead of forming open and sample count spectra through a
    true-energy-conditioned response; archive impact remains unmeasured.
14. **Confirmed bounds/test defect:** the documented IC box contains an
    unsynthesizable corner that its named worst-corner test does not exercise.

Additional confirmed boundary defects are invalid Rust/GUI UDR flight paths
being accepted (zero/negative become identity; NaN/infinite propagate NaNs) and
no equality check between a UDR's stored flight path and calibration config.

These fourteen items seed the candidate ledger; none is silently accepted as the
cause of the observed features.
