# Phase 0 test results

Commands are recorded as they are run. A passing regression anchor establishes
route stability; it is not by itself evidence that the route's physical or
statistical model is correct.

## Evidence registry

Each route row in the audit/matrix cites one or more IDs below. The linked
section contains the exact command and observed result; an ID is never used as
evidence for a sibling route not named here.

| ID | Directly exercised route(s) | Evidence kind | Exact-command section |
|---|---|---|---|
| E01 | branch/base prerequisite | live remote | [Live remote/base check](#live-remotebase-check) |
| E02 | F1, F3 | real aggregate Hf | [Committed real-VENUS single-spectrum regression routes](#committed-real-venus-single-spectrum-regression-routes) |
| E03 | F3 | real cached Ta, IC-as-tabulated | [Archived real-Ta IC/JENDL counts replay](#archived-real-ta-icjendl-counts-replay) |
| E04 | F1, F2, F3, F4 | real Hf semantic controls | [Real-data route-semantics probe](#real-data-route-semantics-probe) |
| E05 | S3, S5 | matched spatial flux-gradient synthetic | [Spatial averaged-flux semantic control](#spatial-averaged-flux-semantic-control) |
| E06 | F3–F6, S3–S6, research Fisher helper | existing targeted suites | [Counts-focused existing suites](#counts-focused-existing-suites) |
| E07 | F3, F4 | 300 matched stochastic fits | [Matched-model stochastic counts ensemble](#matched-model-stochastic-counts-ensemble) |
| E08 | F1, F2, S1 plus shared transmission/resolution features | existing targeted suites | [Transmission/resolution-focused existing suites](#transmissionresolution-focused-existing-suites) |
| E09 | C1, C2, C3 | exact synthetic calibration commands | [Resolution-calibration routes](#resolution-calibration-routes) |
| E10 | F4, F5, F6, S2, S4, S5, S6 | direct public Python smoke/rejection synthetic | [Remaining public route cells](#remaining-public-route-cells) |
| E11 | F1, F3 | matched IC-as-tabulated production-fit synthetic | [Ordinary-fit IC handoff](#ordinary-fit-ic-handoff) |
| E12 | S3 assumption | real VENUS open-beam HDF5 | [Real VENUS spatial open-beam diagnostic](#real-venus-spatial-open-beam-diagnostic) |
| E13 | Python/MCP surfaces and rejection contracts | static/helper plus targeted tests | [Public-surface consistency checks](#public-surface-consistency-checks) |
| E14 | F1, F2, F3, F4, S1, S4, C1, C2 | direct route-identity recovery reruns | [Narrow direct R13 route reruns](#narrow-direct-r13-route-reruns) |

## Narrow direct R13 route reruns

These selectors were rerun after the first cold audit because the earlier
suite-level summaries were not sufficient to identify which route had passed.
Each command below exited `0`; the route labels are based on the typed input and
explicit solver in the named test, not on a sibling test in the same suite.

```text
cargo test -p nereids-pipeline test_typed_transmission_lm_recovers_density -- --nocapture
cargo test -p nereids-pipeline test_typed_transmission_kl_recovers_density -- --nocapture
cargo test -p nereids-pipeline test_joint_poisson_density_recovery_c_5_98 -- --nocapture
cargo test -p nereids-pipeline test_typed_counts_lm_auto_converts -- --nocapture
cargo test -p nereids-pipeline test_spatial_map_typed_counts_lm_no_deviance_map -- --nocapture
```

Exact route-specific result lines:

```text
test pipeline::tests::test_typed_transmission_lm_recovers_density ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 175 filtered out; finished in 0.00s

test pipeline::tests::test_typed_transmission_kl_recovers_density ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 175 filtered out; finished in 0.00s

test pipeline::tests::test_joint_poisson_density_recovery_c_5_98 ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 175 filtered out; finished in 0.01s

test pipeline::tests::test_typed_counts_lm_auto_converts ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 175 filtered out; finished in 0.00s

test spatial::tests::test_spatial_map_typed_counts_lm_no_deviance_map ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 175 filtered out; finished in 0.00s
```

The first four commands are direct matched synthetic route gates for F1–F4.
The fifth is a direct S4 execution/result-semantics gate; its successful
execution does not repair the ignored-`c` mechanism demonstrated by E10.

The public Python S1 matched-recovery selector was rerun directly:

```text
pixi run pytest -q \
  tests/test_nereids.py::TestSpatialMapTransmission::test_basic_spatial_map
```

Exit status: 0.

```text
.                                                                        [100%]
1 passed in 0.10s
```

The two available real Hf route gates were also split into exact selectors so
F1 and F3 no longer rely on the two-test class summary:

```text
pixi run pytest -q \
  tests/test_nereids.py::TestVenusMlbwRegression::test_mlbw_lm_fit_matches_baseline
pixi run pytest -q \
  tests/test_nereids.py::TestVenusMlbwRegression::test_counts_kl_fit_matches_baseline
```

Both exited `0`:

```text
.                                                                        [100%]
1 passed in 0.20s

.                                                                        [100%]
1 passed in 0.25s
```

Finally, the matched C1 and C2 calibration selectors were rerun directly:

```text
cargo test -p nereids-fitting \
  resolution_calib::tests::gaussian_recovers_known_width -- --nocapture
cargo test -p nereids-fitting \
  resolution_calib::tests::udr_corr_recovers_known_width_scale_and_exponent \
  -- --nocapture
```

Both exited `0`:

```text
test resolution_calib::tests::gaussian_recovers_known_width ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 188 filtered out; finished in 6.22s

test resolution_calib::tests::udr_corr_recovers_known_width_scale_and_exponent ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 188 filtered out; finished in 2.97s
```

E14 does not add real spatial truth, a real UDR/calibration fixture, a native-IC
ordinary fit, or a fresh C3 full-recovery/PSR closed loop. Those remain gaps.

## Live remote/base check

Command:

```text
git fetch origin
git rev-parse main origin/main HEAD
git rev-list --left-right --count main...origin/main
git rev-list --left-right --count HEAD...origin/main
git merge-base HEAD origin/main
```

Exit status: 0.

```text
main        19d0ea28b967f6772a36025cb4184a3b2e149b0f
origin/main 19d0ea28b967f6772a36025cb4184a3b2e149b0f
audit HEAD  941785913d2bbd9945262fb2455e8a9e8cd902aa
main divergence: 0 0
audit-branch divergence before the staged Phase 0 artifacts: 3 0
merge base: 19d0ea28b967f6772a36025cb4184a3b2e149b0f
```

Observation: the Phase 0 branch was based on the live remote head before its
contract and audit commits. The pre-existing dirty nested test-data submodule
was not modified.

## Committed real-VENUS single-spectrum regression routes

Command:

```text
pixi run pytest -q tests/test_nereids.py::TestVenusMlbwRegression
```

Exit status: 0.

```text
..                                                                       [100%]
2 passed in 0.38s
```

Routes exercised:

- real aggregated VENUS Hf-177 transmission + Gaussian resolution + LM;
- the same raw sample/open-beam counts + Gaussian resolution + counts KL
  (joint-Poisson), including the measured proton-charge ratio.

What this establishes: both committed real-data anchors still execute and
match their stored density/goodness-of-fit values on this checkout.

What this does not establish: either no-background single-isotope model is a
good description of the data (their reported GOF is deliberately very poor),
nor does it validate transmission KL, counts-to-transmission LM, nuisance
backgrounds, spatial variation, UDR, or IC calibration.

## Archived real-Ta IC/JENDL counts replay

Command:

```text
pixi run python investigation/reproduce_ic_cached.py
```

Exit status: 0. Wall time: 22.3 s.

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
```

Route exercised: real cached Ta sample/open-beam counts + counts KL
(joint-Poisson) + JENDL-5 + archived IC converted to tabulated resolution +
NEREIDS multiplicative baseline + fit-energy mask.

What this establishes: the supplied archive's whole-region IC science fit is
reproducible through the current public Python API.

What this does not establish: the archived IC calibration itself, a UDR route,
spatial maps, or the physical correctness of the current same-bin `R[T]`
joint-Poisson measurement model.

## Real-data route-semantics probe

Command:

```text
pixi run python investigation/phase0_route_semantics.py
```

Exit status: 0. Wall time: 1.4 s.

```text
pc_ratio=5.9820330009603486
lm_transmission_reference_density=8.1045852851800799e-05
lm_transmission_reweighted_density=0.013701576773053498
lm_transmission_reweight_delta=0.013620530920201697
kl_transmission_reference_density=3.095027138790765e-05
kl_transmission_reweighted_density=3.095027138790765e-05
kl_transmission_reweight_delta=0
kl_negative_transmission_observation=accepted converged=True density=3.0953861244912064e-05
joint_poisson_raw_counts_density=2.9596692297867937e-05
lm_raw_counts_with_c_density=0
lm_counts_with_prescaled_ob_density=4.3104663890779766e-05
lm_raw_counts_converged=False
lm_prescaled_ob_converged=True
```

Validated mechanisms:

1. Transmission Poisson-KL returned bit-identical fitted density after the
   supplied uncertainty's *relative* weighting was changed by eight orders of
   magnitude across the spectrum. Weighted LM changed substantially on the
   same control. This confirms the code trace: transmission KL does not use
   `uncertainty` in its optimization; it is not a low-count likelihood derived
   from the normalized transmission's stated errors.
2. The same public transmission-KL route accepted an observed transmission of
   `−0.1` and reported a converged fit. The NLL's observed-count non-negativity
   condition is a debug assertion only and is not enforced by the release
   Python boundary.
3. Raw-count LM with the fixture's documented `c=5.982...` ignored `c`, hit
   density zero, and did not converge. Pre-scaling the open beam by `c` and
   passing `c=1` made the fallback converge. This independently confirms that
   `counts_to_transmission(S,O)` does not honor the raw-count API's proton-charge
   contract.
4. Even after manual open-beam normalization, the fallback density differs
   from transmission LM because its internal uncertainty is
   `sqrt(max(S,1))/O` and omits open-beam variance.

The absolute fitted densities are not interpreted as sample truth: the probe's
one-isotope/no-background model is deliberately mis-specified. The invariance,
ignored-`c` failure, and pre-scaling control are the mechanism tests.

## Spatial averaged-flux semantic control

Command:

```text
pixi run python investigation/phase0_spatial_flux_probe.py
```

Exit status: 0. Wall time: 0.2 s.

```text
true_density=8.0000000000000007e-05
flux_levels=[[5000.0, 10000.0], [15000.0, 20000.0]]
averaged_flux=12500
from_counts_density_map=[[0.0006299038934515659, 0.00016719998935087603], [1.3127175420586113e-05, 0.0]]
from_counts_converged_map=[[True, True], [True, True]]
paired_nuisance_density_map=[[7.999999999999998e-05, 7.999999999999998e-05], [7.999999998754448e-05, 7.999999999999998e-05]]
paired_nuisance_converged_map=[[True, True], [True, True]]
paired_max_relative_error=1.5569414247434965e-10
```

Mechanism: the noise-free sample cube was generated as `S_pixel =
O_pixel*T(true)` under a known 4:1 spatial open-beam gradient. The ordinary
`from_counts` spatial route substituted the mean `O=12500` for every pixel and
therefore converted beam-profile differences into a large false density
gradient, despite reporting every pixel converged. The zero-background
`CountsWithNuisance` control retained the paired per-pixel open beam and
recovered the injected density to `1.56e-10` maximum relative error.

This does not prove that the real VENUS open beam has a comparable gradient;
it establishes the documented approximation's failure mode and the exact
condition that must be tested on real beam-flatness data.

## Counts-focused existing suites

Independently run by the counts audit:

```text
cargo test -p nereids-fitting joint_poisson -- --nocapture
```

`30 passed; 0 failed; 159 filtered out` in 0.19 s.

```text
cargo test -p nereids-pipeline counts -- --nocapture
```

`32 passed; 0 failed; 144 filtered out` in 0.20 s.

```text
cargo test -p nereids-pipeline joint_poisson -- --nocapture
```

`8 passed; 0 failed; 168 filtered out` in 0.01 s.

```text
cargo test -p nereids-pipeline spatial_global_baseline -- --nocapture
cargo test -p nereids-pipeline spatial_per_pixel_baseline -- --nocapture
```

`2 passed; 0 failed; 174 filtered out` and `1 passed; 0 failed; 175 filtered
out`, respectively.

```text
cargo test -p nereids-pipeline evaluate_jacobian_and_fisher -- --nocapture
cargo test -p nereids-fitting counts_ -- --nocapture
pixi run pytest -q tests/test_nereids.py -k counts
```

Results: `3 passed`, `4 passed`, and `18 passed, 194 deselected`, respectively.

These suites verify current dispatch/recovery/rejection contracts. They do not
close the missing count-fit integration coverage for tabulated UDR or IC, nor
validate the spatial averaged-open-beam assumption on measured flat-field data.

## Matched-model stochastic counts ensemble

Command:

```text
pixi run python investigation/phase0_counts_ensemble.py
```

Exit status: 0. Wall time: 0.3 s. Fixed RNG seed `20260709`; 50 generated
spectra at each exposure; every fit and non-convergence counted.

| Route | Open-beam expectation/bin | Converged | Relative density bias | RMSE | Reported 1σ coverage |
|---|---:|---:|---:|---:|---:|
| joint-Poisson | 25 | 50/50 | +4.995% | 2.371e-5 | 36/50 (72%) |
| counts→LM fallback | 25 | 50/50 | +52.802% | 5.064e-5 | 28/50 (56%) |
| joint-Poisson | 250 | 50/50 | −0.295% | 7.442e-6 | 37/50 (74%) |
| counts→LM fallback | 250 | 50/50 | +3.711% | 8.036e-6 | 36/50 (72%) |
| joint-Poisson | 2500 | 50/50 | +0.356% | 2.632e-6 | 38/50 (76%) |
| counts→LM fallback | 2500 | 50/50 | +0.790% | 2.710e-6 | 37/50 (74%) |

Interpretation: on data generated exactly from the fitted forward model with
`c=1`, the native counts likelihood has small bias at moderate/high exposure
and materially lower low-count bias. The ratio/LM approximation approaches it
in the high-count limit but is not acceptable as a general low-count engine.
The coverage sample is only 50 per cell; it is consistent with a useful smoke
test, not a precision coverage certification.

## Transmission/resolution-focused existing suites

Independently run by the transmission audit; all passed:

| Command | Result |
|---|---|
| `cargo test -p nereids-pipeline transmission -- --nocapture` | 23 passed |
| `cargo test -p nereids-pipeline resolution_plan` | 1 passed; scalar-surrogate tolerance miss correctly fell back to exact evaluation |
| `cargo test -p nereids-pipeline gaussian_aux_grid` | 3 passed |
| `cargo test -p nereids-pipeline energy_scale` | 19 passed |
| `cargo test -p nereids-pipeline baseline` | 15 passed |
| `cargo test -p nereids-pipeline background` | 8 passed |
| `cargo test -p nereids-pipeline fit_energy_range` | 10 passed |
| `cargo test -p nereids-pipeline fix_densities` | 3 passed |
| `cargo test -p nereids-physics ikeda` | 21 passed; kernel/plan level, not production-fit integration |
| `cargo test -p nereids-physics --test venus_usr_resolution` | 8 passed; synthetic VENUS-like tabulated kernel |
| kernel-orientation integration filter | 1 passed |
| width-interpolation integration filter | 1 passed |

The current extension was rebuilt successfully with `pixi run build`; a
selected Python transmission set then passed 11 tests: the real Hf LM anchor,
all five `TestSpatialMapTransmission` tests, two spatial-baseline tests, and
three fit-range binding tests.

Coverage gaps found: no end-to-end ordinary fit with native IC resolution; no
transmission-KL fit with a nontrivial resolution; the tabulated spatial smoke
generates unresolved truth; and no committed real-data transmission test covers
UDR, IC, SAMMY background, multiplicative baseline, energy scale, spatial
variation, or transmission KL.

## Resolution-calibration routes

The calibration audit first listed `33` unit tests with:

```text
cargo test -p nereids-fitting resolution_calib::tests -- --list
```

Narrow matched-route results:

Each Rust row below used the exact form `cargo test -p nereids-fitting
resolution_calib::tests::<filter> -- --nocapture`; filters are shown in the
table so each run is reproducible.

| Filter | Result/runtime |
|---|---:|
| `gaussian_recovers_known_width` | 1 passed / 6.12 s |
| `udr_corr_recovers_known_width_scale_and_exponent` | 1 passed / 2.85 s |
| `udr_corr_recovers_independent_raw_kernel` | 1 passed / 3.47 s |
| `fit_t0_recovers_injected_energy_scale_shift` | 1 passed / 5.14 s |
| `position_prior_penalizes_displacement` | 1 passed / 13.84 s |
| `calibrate_with_background_runs` | 1 passed / 0.30 s |
| `inner_chi2_background_path_and_degenerate_model` | 1 passed / 0.00 s |
| `ic_box_worst_corner_synthesizes_within_tau_cap` | 1 passed / 0.22 s |
| `ic_unresolvable_theta_errs_in_build_resolution` | 1 passed / 0.00 s |
| selected Python calibration API set | 5 passed / 0.93 s |
| flat-L-scale Python diagnostic | 1 passed / 5.56 s |

An actual IC optimizer/convergence route was independently rerun:

```text
cargo test -p nereids-fitting \
  resolution_calib::tests::gaussian_and_ic_families_run_and_converge \
  -- --nocapture
```

Exit status: 0.

```text
test resolution_calib::tests::gaussian_and_ic_families_run_and_converge ... ok
test result: ok. 1 passed; 0 failed; 188 filtered out; finished in 115.81s
```

This proves the bounded IC calibration optimizer can finish and self-converge
on its selected synthetic case; the nearly two-minute runtime for one small
case also independently confirms the optimization-cost problem.

Not rerun in Phase 0: fitted-PSR recovery, full parameter-recovery cases, and
the two multi-minute `ic_closed_loop` tests. The prior committed investigation
records a full 33-test calibrator pass in 1329.99 s, but this Phase 0 cold audit
does not treat that record as an independent rerun. No real-data calibration
fixture/test exists; production Hf/Ta fits are not calibration evidence.

The five Python calibration boundary/result checks were independently rerun as:

```text
pixi run pytest -q \
  tests/test_nereids.py::TestCalibrateResolution::test_result_exposes_new_position_fields_not_old_nuisance \
  tests/test_nereids.py::TestCalibrateResolution::test_udr_corr_requires_base \
  tests/test_nereids.py::TestCalibrateResolution::test_invalid_position_prior_rejected \
  tests/test_nereids.py::TestCalibrateResolution::test_invalid_flight_path_rejected_not_panicked \
  tests/test_nereids.py::TestCalibrateResolution::test_fit_psr_requires_ic_family
```

`5 passed in 0.93s`.

## Remaining public route cells

Command:

```text
pixi run python investigation/phase0_remaining_routes.py
```

Exit status: 0. Wall time: 0.2 s.

```text
spatial_transmission_kl_density=8.0000000000071306e-05
spatial_transmission_joint_poisson_alias_density=8.0000000000071306e-05
spatial_transmission_kl_negative_observation=converged=True density=8.0066156333984375e-05
single_zero_background_nuisance=converged=True density=7.9999999994368538e-05
single_nonzero_detector_background=rejected type=ValueError message=Invalid parameter: joint-Poisson solver with non-zero detector_background is not yet supported (B_det wiring is deferred).
single_fit_alpha_1=rejected type=ValueError message=Invalid parameter: joint-Poisson solver does not support fit_alpha_1/fit_alpha_2: the profile lambda-hat absorbs the global flux scale (alpha_1 redundant); alpha_2 / B_det wiring is not yet implemented.
single_counts_lm_fit_alpha_1=accepted converged=True density=7.9999999999999993e-05 alpha_1=None alpha_2=None
single_zero_background_nuisance_lm=rejected type=ValueError message=Invalid parameter: CountsWithNuisance requires a counts-domain solver (LM cannot use nuisance parameters)
spatial_nonzero_detector_background=accepted n_failed=1 converged=False density=nan
spatial_counts_lm_c2=converged=False density=nan
spatial_counts_lm_fit_alpha_1=rejected type=ValueError message=counts background scaling requires from_counts_with_nuisance() input
spatial_counts_nuisance_lm=rejected type=ValueError message=Invalid parameter: spatial_map_typed: InputData3D::CountsWithNuisance requires a counts-domain solver (joint-Poisson via SolverConfig::PoissonKL or SolverConfig::Auto); SolverConfig::LevenbergMarquardt cannot use the user-supplied nuisance parameters (alpha_1, alpha_2).  Choose a counts-domain solver, or drop the nuisance arm by passing `InputData3D::Counts` instead.
```

The matched noise-free truth confirms route dispatch and explicit rejections.
It does **not** validate transmission KL's statistic: both `kl` and the
misleading `joint_poisson` alias select the same fractional-transmission NLL,
and spatial KL again accepted a negative observation. Counts-with-zero-
background provides a working paired-open-beam path, while the advertised
nonzero-background/alpha functionality remains unavailable. The single route
rejects nonzero Bdet, but S5 catches the same pixel error and returns an
accepted `SpatialResult` with `n_failed=1`, nonconvergence, and NaN density;
that is result-container reachability, not a working background route. Spatial
counts LM also reproduces the ignored-`c` failure. On single F4, `fit_alpha_1`
is accepted but absent from the result because LM silently ignores the attached
counts config; the analogous Python spatial call rejects at its binding
boundary. That is a verified cross-surface difference, not one shared F4/S4
behavior.

## Ordinary-fit IC handoff

Command:

```text
pixi run python investigation/phase0_ic_fit_probe.py
```

Exit status: 0. Wall time: 0.5 s.

```text
true_density=8.0000000000000007e-05
transmission_lm_converged=True density=8.0000000000000061e-05
counts_joint_poisson_converged=True density=7.9999999983031578e-05
```

This closes the missing Python ordinary-fit smoke for an IC kernel converted to
`TabulatedResolution`: both F1 and F3 recover matched, noise-free truth. It does
not test native Rust `ResolutionFunction::IkedaCarpenter`, independent IC
physics, spatial IC, noisy recovery, or real calibration transfer.

## Real VENUS spatial open-beam diagnostic

Command:

```text
pixi run python investigation/phase0_real_open_beam.py
```

Exit status: 0. Wall time: 3.0 s. The 147 MB HDF5 dataset was streamed once in
its native 32×32 spatial chunks; no data were written or fit.

Key output for all 65,536 pixels over the 2,310 bins from 8.0037–44.9992 eV:

```text
pixels_zero_total=24
pixel_total_all_cv=0.049401331882944835
pixel_total_nonzero_cv=0.045534458595607188
pixel_total_nonzero_over_median p05=0.92500279423270371
pixel_total_nonzero_over_median p95=1.0680675086621214
pixel_total_nonzero_over_median min=0.55124622778584997
pixel_total_nonzero_over_median max=1.3780038001564769
poisson_cv_at_mean_total=0.010582032187852817
right_over_left=0.97807393864818448
quadrant_top_left_nonzero_mean=8880.3591403101727
quadrant_top_right_nonzero_mean=8614.5580188103086
quadrant_bottom_left_nonzero_mean=9178.0427402613259
quadrant_bottom_right_nonzero_mean=9047.7884263215728
```

The observed nonzero-pixel total CV is 4.55%, versus 1.06% Poisson CV at the
same mean total. The central 90% spans roughly −7.5% to +6.8% around the median,
and quadrant/half means differ systematically. Thus the actual fixture does
not support treating spatial open-beam magnitude as uniform at percent-level
precision. The diagnostic cannot separate beam profile from pixel efficiency;
both are precisely effects that paired open-beam normalization retains and
spatial averaging discards.

Per-bin spatial CV is ~0.65 because individual bins contain few counts per
pixel; it is reported in the command output but is not used to claim a 65%
beam-profile variation.

## Public-surface consistency checks

Command:

```text
pixi run python scripts/check_python_api_drift.py
```

Exit status: 0.

```text
python-api.md drift check passed: 63 public symbols, 40 documented, 23 allowlisted
```

This symbol-level check does not inspect defaults, objective semantics,
unsupported-but-exposed arguments, or result labels; the Phase 0 audit found
all of those mismatches despite this pass.

Command:

```text
pixi run pytest -q tests/test_mcp.py
```

Exit status: 0, but no tests executed:

```text
1 skipped in 0.04s
```

FastMCP is unavailable in this environment. MCP routing findings are therefore
source-verified plus direct helper/probe evidence, not a live MCP server test.
GUI source behavior was inspected, but no completed GUI test or interactive
exercise is claimed.

Durable MCP routing probe:

```text
pixi run python investigation/phase0_mcp_route_probe.py
```

Exit status: 0.

```text
no-fit-block ('transmission', '<python-default>') True
kl-default-domain ('counts', 'kl') True
lm-explicit-counts ('counts', 'lm') True
kl-domain-typo ('transmission', 'kl') True
```

This confirms a count manifest with no fit block defaults to the transmission
fitter; `kl` changes the implicit domain to counts; explicitly requested raw
counts+LM reaches the information-losing fallback; and a misspelled domain
silently selects transmission.

Five selected binding/background route tests:

```text
pixi run pytest -q \
  tests/test_nereids.py::TestSpatialMapTransmission::test_spatial_map_back_d_f_requires_background_kwarg \
  tests/test_nereids.py::TestSpatialMapCounts::test_counts_with_nuisance_auto_dispatches_to_kl \
  tests/test_nereids.py::TestFitCountsSpectrumTyped::test_enable_polish_kwarg_accepted_and_toggles_behavior \
  tests/test_nereids.py::TestMultiplicativeBaseline::test_baseline_with_default_background_rejected_on_all_fitters \
  tests/test_nereids.py::TestMultiplicativeBaseline::test_fit_anorm_false_requires_background
```

`5 passed in 0.11s`.

Three narrow Rust rejection-contract commands each passed one test with zero
failures:

```text
cargo test -p nereids-pipeline \
  test_fit_energy_range_rejected_on_transmission_poisson_path -- --nocapture
cargo test -p nereids-pipeline \
  test_spatial_map_counts_with_nuisance_plus_lm_rejected_up_front -- --nocapture
cargo test -p nereids-pipeline \
  test_joint_poisson_rejects_nonzero_detector_background -- --nocapture
```
