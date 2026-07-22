# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed (breaking)

- **Removed URR (LRU=2) and R-Matrix Limited (LRF=7) cross-section
  evaluation.** The URR Hauser-Feshbach path lacked the ENDF
  width-fluctuation correction (a systematically wrong average), and the
  RML path's closed-channel treatment was incomplete (the Coulomb/SHF=1
  closed-channel shift was unimplemented) and never validated against SAMMY.
  Both range types are now parsed-and-skipped:
  the parser consumes their records for cursor alignment (all structural
  guards preserved) and tags them non-evaluable, so they contribute zero
  cross-section. A file whose every range is non-evaluable (e.g. a pure
  LRF=7 evaluation such as W-184) is rejected at load with an
  `UnsupportedFormat` error naming each skipped range; mixed evaluations
  (e.g. Ta-181, U-238) still load, with a Python `UserWarning` from
  `load_endf`/`load_endf_file` listing the skipped spans, a
  `ResonanceData.has_unevaluated_ranges` getter, and a GUI log warning.
- **LRU=0 scattering-radius-only evaluations are rejected at load.** ENDF-6
  §2.1 LRU=0 ranges (e.g. H-1) give a scattering radius but no resonance
  parameters, so they contribute zero cross-section; NEREIDS previously loaded
  them as zero-resonance data. They are now parsed-and-skipped placeholders,
  and a file whose only range is LRU=0 is rejected with an `UnsupportedFormat`
  error that names the LRU=0 span. Mixed files that also carry a resolved
  LRF=1/2/3 range still load and use it.
- **Removed the dead `alpha_1`/`alpha_2` counts-fit parameters** from the
  Python API (`fit_alpha_1`, `fit_alpha_2`, `alpha_1_init`,
  `alpha_2_init` kwargs). The MCP server now rejects manifests that still
  pass any of these keys with an explicit error instead of silently
  running with defaults.

## [0.3.0] - 2026-07-07

### Added

- **Beam-state event filtering for NXevent_data banks** (#637):
  `read_run_log(path, pv)` reads `/entry/DASlogs/<pv>` as a transition
  log (SNS logs record transitions, not uniform samples — the entry-mean
  of a real VENUS `pause` log read 0.43 while the time-weighted truth
  was 0.90); `intervals_where` derives `(t_start, t_end)` intervals with
  correct step-function semantics (last value persists to
  `/entry/duration`, pre-log time and NaN never match);
  `intervals_intersect` composes conditions across PVs.
  `load_nexus_bank_spectrum(path, bank, ..., keep_intervals=None)` loads
  one facility NXevent_data bank (e.g. `monitor1`) as a 1-D TOF spectrum
  keeping only pulses whose `event_time_zero` falls in the intervals
  (half-open, same run-start clock as DASlogs at SNS), with pulse/event
  retention statistics and clock-epoch attributes for cross-checking.
  `units` attributes are required on both `event_time_offset` and
  `event_time_zero` (#554: NXevent_data declares no defaults — never
  guess a scale); corrupt device-reconnect records in DASlogs (backward
  time jumps and subnormal uninitialized-memory payloads, both seen on
  real VENUS furnace channels) are dropped and counted in
  `RunLog.n_dropped_corrupt`; empty banks — the normal state of every VENUS
  imaging-detector bank, since tpx1 is frame-mode — load gracefully to a
  zero spectrum.  The rustpix `/entry/neutrons` event path is NOT
  filtered: that schema carries no per-event wall-clock time; filtering
  there stays out of scope until a producer demonstrably writes one.

- **Bounded multiplicative baseline `B(E)` in every fitter** (#635): a
  smooth, box-bounded normalization
  `B(E) = b0 + b1·ln(E/E_ref) + b2·ln²(E/E_ref)` (E_ref = geometric
  midpoint of the fit grid, reported on the result) applied OUTERMOST —
  `y = B(E)·[Anorm·T + additive background]` — across the LM,
  transmission-KL, and counts joint-Poisson paths, `spatial_map_typed`,
  the Python fitters (`baseline=True`, `fit_b0/b1/b2`, `b*_init`,
  `b*_bounds`, `fit_anorm`, spatial `baseline_global`), and the GUI
  (advanced-solver checkbox + overlay; baseline outputs persist in
  project files so a reloaded fit renders the same model). Real transmission ratios sit a
  few % off unity with smooth energy dependence (filters, holders,
  scattering). Resonance dips stay attributed to the physics model
  because a single global ln-E quadratic is too SMOOTH to produce a
  narrow local feature; the configurable coefficient boxes (default
  (0.9, 1.1) / ±0.05 per ln-E unit) additionally bound the baseline's
  tilt and curvature (the evaluated B(E) excursion grows with the
  grid's log-width; only positivity is enforced pointwise). A free SAMMY `Anorm`
  alongside the baseline is rejected (`b0`/`Anorm` are degenerate
  normalizations); the additive ABC background combines with the
  baseline when `Anorm` is held fixed. This is a documented NEREIDS
  extension: modern SAMMY normalizes with a scalar `Anorm` + additive
  backgrounds only (`cro/mnrm1.f90`); the nearest analogue is the
  dormant legacy power-law `Anrm(1)+Anrm(2)·E^Anrm(3)`
  (`acs/macs4.f90:440-450`).
- **Two-stage global baseline for spatial maps** (#635):
  `spatial_map_typed` fits the baseline ONCE on the aggregated mean
  spectrum, then freezes it for every pixel (default; per-pixel
  baselines at low counts biased fitted temperatures on real data).
  Stage-1 non-convergence is a hard error, never a silent per-pixel
  fallback. `SpatialResult` gains `baseline_global`,
  `baseline_e_ref_ev`, `baseline_maps` (per-pixel mode), `warnings`.
- **Structured fit warnings** (#635): `SpectrumFitResult::warnings` /
  `FitResult.warnings` / GUI fit feedback now flag the degenerate
  normalization trio (free `Anorm` + free temperature + ≥1 free
  density), which on real VENUS data converged to T = 4471 K at
  χ²/ν = 932 with no diagnostic.

- **Chunk-aware TIFF folder loading** (#636): folders following the
  chunked VENUS naming convention (`<prefix>_<chunk>_<frame>.tif`) are
  detected automatically — frames order by numeric index and chunks
  covering identical frame ranges are summed element-wise by default
  (`sum_chunks=False` / `TiffFolderOptions::sum_chunks` opts out; on the
  default summing path ragged chunks or duplicate (chunk, frame) pairs are
  a hard `ChunkMismatch` error, never a silent stack). New
  `load_tiff_folder_with_options` / `load_tiff_auto_with_options` /
  `load_tiff_stack_with_options` return a `TiffLoadInfo` provenance report
  (`n_files`, `n_chunks`, `chunk_ids`, `chunks_summed`,
  `n_clipped_pixels`); the GUI logs "summed k DAQ chunks" provenance. The Python loaders emit a `UserWarning` when
  chunks were summed (naming the chunk ids and the `sum_chunks=False`
  escape hatch) or when `pixel_policy="clip"` clamped pixels, and
  `load_tiff_folder(..., return_info=True)` returns the provenance dict
  alongside the array. The GUI's pre-normalized transmission mode loads
  chunked folders with `sum_chunks=False` — element-wise sums of
  transmissions are counts semantics and physically meaningless. The
  issue's coverage-mask item was dropped per its scope amendment: per
  the #646 masking policy, coverage/thickness is a model concern, not
  an I/O concern.
- **Mixed-folder chunk-detection observability** (#636): a folder where
  at least one file matches the chunked naming convention but others do
  not (a stray overview TIFF, a misnamed frame) still falls back to
  legacy lexicographic loading, but the fallback is now loud: the
  non-conforming files are counted in
  `TiffLoadInfo::n_unrecognized_files` (up to three named in
  `unrecognized_examples`), the Python loaders emit a new `UserWarning`
  class ("N file(s) did not match the chunk naming pattern..."), the
  `return_info` dict gains both keys, and the GUI logs the count in its
  load provenance. The GUI also logs "detected k DAQ chunks — NOT
  summed" when transmission mode loads a chunked folder with summing
  disabled, and `load_tiff_folder` on a nonexistent path now raises
  `FileNotFoundError` (`NotADirectoryError` is reserved for paths that
  exist but are not directories, preserving the real metadata-error kind
  so a permission-denied parent stays `OSError`), matching its documented
  contract.
- **`sum_chunks=False` honors its opt-out contract on inconsistent chunks**
  (#636): ragged chunks (differing frame counts/sets) or duplicate
  (chunk, frame) pairs are a hard `ChunkMismatch` error only on the
  default summing path — where summing them would corrupt counts. With
  `sum_chunks=False` there is nothing to corrupt, so the same folder now
  loads as the documented lexicographic concatenation of every file
  (frame count = the sum of all files) instead of erroring — inspecting
  raw frames of a ragged folder is exactly when `sum_chunks=False` is
  reached for. The irregularity is surfaced through a new
  `TiffLoadInfo::chunk_inconsistent` flag (the `return_info` dict gains a
  `chunk_inconsistent` key, the Python loader emits a `UserWarning`, and
  the GUI's transmission-mode provenance logs it), never silently
  swallowed. The default summing path is unchanged — inconsistent chunks
  that would be summed still fail loud.
- **`read_tof_sidecar`** (#636): reads a VENUS `*_Spectra.txt` sidecar
  (frame start times in seconds) into the N+1 ascending microsecond TOF
  bin edges `tof_to_energy_centers` expects, extrapolating the closing
  edge from the last frame width (shutter segments with different frame
  widths are valid). The GUI auto-detects `*_Spectra.txt` spectrum picks
  and never falls back to the verbatim-µs parser on sidecar failure. A
  VENUS run folder now loads to stack + energy axis in 3 Python calls.
- **`run_health`** (#636): DASlogs-based run-health summary
  (`pause_fraction`, `beam_dip_fraction`, `median_power`, `duration_s`)
  using last-value-held time-weighted integration (DASlogs PVs log
  transitions; entry means are wrong). SNS PV-name defaults (`pause`,
  `proton_charge`); other facilities pass their own. Absent PVs report
  `None`; present-but-malformed PVs are hard errors.

- **`fit_energy_scale` + `fit_temperature` jointly, in every fitter**
  (#634): the flag combination resonance thermometry needs — calibrate
  the SAMMY energy scale (t₀, L_scale) *and* fit temperature in one
  fit — is now supported across the LM, transmission-KL, and counts
  joint-Poisson paths and `spatial_map_typed` (the four guards that
  rejected it are gone). `EnergyScaleTransmissionModel` carries a fitted
  temperature column (central finite difference, validated against the
  analytic ∂σ/∂T column to <1e-4 relative; an independent (JᵀWJ)⁻¹
  reconstruction reproduces the reported temperature σ to <0.1 %).
- **`corrected_energies()` accessor** (#634): `SpectrumFitResult` (Rust)
  and `FitResult` (Python) expose the exact energy-scale transform the
  fit used — `corrected_energies(nominal_energies)` maps a nominal grid
  through the fitted `(t0_us, l_scale)`, using the flight path stored on
  the result at fit time, with the SAMMY
  `−t0` sign convention (`dat/mdat0.f90:189`), returning `None` when the
  energy scale was not fitted. Hand-deriving this transform (with a
  `+t0` slip) previously caused a silent +400 K temperature bias.

- **Per-parameter density freeze in every fitter** (#633): areal
  densities can now be held fixed while other parameters (temperature,
  energy scale, background) are fit. `UnifiedFitConfig` gains
  `with_fix_densities(bool)` (freeze all densities) and
  `with_density_free(Vec<bool>)` (per-density-parameter mask; `false`
  freezes density parameter *i* at its initial value, length must equal
  the density-parameter count — one entry per isotope for ungrouped
  fits, one per group for grouped fits). The Python fitters
  `fit_spectrum_typed`, `fit_counts_spectrum_typed`, and
  `spatial_map_typed` expose the same control via the new
  `fix_densities: bool = False` and `density_free: list[bool] | None =
  None` keyword arguments (mutually exclusive — supplying both is an
  error). Frozen densities no longer consume a free parameter, so
  reported degrees of freedom, reduced χ², and per-parameter
  uncertainties reflect only the parameters actually varied (a frozen
  density's reported 1-σ is `NaN`). This enables temperature-only
  thermometry fits against a known sample density.

- **Calibration degeneracy reporting** (#642): `calibrate_resolution`
  results now carry `n_free_params` and `bounds_hit` (parameters pinned at
  a box bound, e.g. `"r:lower"` on the β↔R ridge when the calibrant shows
  no storage tail), in Rust and Python.
- **Calibration simplex re-inflation** (#642): each `calibrate_resolution`
  restart now re-launches a fresh, larger Nelder–Mead simplex at the
  incumbent until it stops improving, escaping premature simplex collapse
  in the curved α↔β↔R valley of the 4-parameter IC family (a collapsed
  simplex once stalled a 300 K calibration Δχ² ≈ +130 above the noise
  floor, biasing the downstream pinned temperature fit by ~23 K). Applies
  to all families; results can only improve or stay put.
- **Closed-loop IC acceptance tests** (#642): synthetic Ta-181
  calibrate→pin→refit-temperature loops at 300 K and 1073 K
  (`crates/nereids-fitting/tests/ic_closed_loop.rs`) assert χ²/dof ≈ 1,
  interior recovery of the truth kernel, and temperature recovery within
  3σ with σ_T far below the old ~90 K degeneracy scale.

- **Hot/railed pixel detection** (#643): `detect_hot_pixels` — two-stage
  screen on per-pixel total counts: a robust one-sided log-space
  median + k·MAD cut (k = 6 default, with a Poisson floor on the robust
  scale so quantized low-count images never turn the screen into a
  low-count filter) confirmed by a 10× local-8-neighbor-median isolation
  test iterated to a fixpoint (already-flagged neighbors stop vouching),
  which erodes railed clusters up to 3 px wide from the boundary inward
  — a single local pass would miss the interior of clusters ≥2 px wide —
  provided the cluster exposes at least one end cap or convex corner to
  normal-scene neighbors (erosion must seed somewhere).  An edge-to-edge
  railed band ≥2 px wide (both ends off-detector) has no seed and is not
  caught — deliberate: a slit-aperture open beam produces a genuine
  full-width bright scene band indistinguishable from it, and a
  full-span row/column screen would mask that scene (the bimodal
  failure); declare such full-span detector pathologies in a file mask.
  A full-span width-1 railed line is caught.
  Bimodal scenes (dark-majority sample + bright open-beam region) never
  get their bright region masked — a contiguous bright region ≥2 px
  wide is scene, not a defect, and the erosion never seeds in it; a
  1-px-wide bright line at ≥10× local contrast is indistinguishable
  from a railed line and is masked by design (documented trade-off;
  real VENUS features are PSF-blurred over ≥2 px).  Requires raw
  (unscaled) counts.  In Rust (`nereids_io::normalization`) and Python.
- **Chunked dead-pixel detection** (#643):
  `detect_dead_pixels_chunked` — per-acquisition-chunk detection that
  flags pixels dead in *any* chunk, catching intermittent deadness
  invisible to the summed-stack test, in Rust and Python.
- **Unified bad-pixel entry point** (#643): `detect_bad_pixels` —
  validating detector computing dead ∪ hot over the sample and
  (optionally) the open-beam stack, in Rust and Python.
- `nereids-core` gains a `stats` module (`median`,
  `median_abs_deviation`, `MAD_TO_SIGMA`).
- **Opt-in χ²-scaled temperature uncertainty** (#638): `scale_by_chi2`
  (Rust `UnifiedFitConfig::with_scale_by_chi2`; Python `scale_by_chi2=`
  kwarg on `spatial_map_typed` / `fit_spectrum_typed` /
  `fit_counts_spectrum_typed`, default `False`) inflates the
  covariance-only uncertainties by `sqrt` of the goodness-of-fit the
  same result reports, turning the inverse-Fisher lower bound into a
  goodness-of-fit-scaled estimate. Self-consistent on every path: the
  transmission paths (LM and Poisson-KL) scale by the Gaussian
  `reduced_chi_squared`, and the counts joint-Poisson path scales by its
  `deviance_per_dof`. No-op on the already-χ²-scaled LM transmission path
  (Numerical Recipes §15.6). The transmission Poisson-KL path never scales
  by a Poisson deviance on transmission fractions (that would be a
  pseudo-Poisson statistic, not a valid reduced-χ²).
- **Ta-181 VIII.0 resonance-count regression guard** (#638): a vendored
  public-domain 73-Ta-181 ENDF/B-VIII.0 evaluation (MAT 7328) plus a
  pinning test confirm the parser faithfully reads every NER range —
  `total_resonance_count() == 76` reflects VIII.0's genuinely-sparse
  resolved region (MLBW RRR to 330 eV + an unresolved URR), not a
  dropped range (VIII.1 extended the RRR to 2554 eV / 565 resonances).

### Fixed

- **Python `n_resonances` under-counted LRF=7 evaluations as 0** (#638):
  the `ResonanceData.n_resonances` getter (and `__repr__`) summed only
  `l_groups`, so R-matrix-limited (LRF=7) evaluations — whose resonances
  live in `rml.spin_groups` — reported zero; both now delegate to the
  formalism-aware Rust `total_resonance_count()`. A `total_resonance_count`
  getter is exposed as an explicit alias.
- **2-chunk VENUS folders no longer load as doubled stacks** (#636):
  `load_tiff_folder` previously concatenated all files lexicographically,
  so a run split into k DAQ chunks silently produced a k× stack; chunked
  folders now sum element-wise (see Added).
- **Negative / non-finite pixels no longer import silently** (#636): TIFF
  loaders now reject them by default (`IoError::BadPixelValue` naming
  file/frame/index/value and pointing at `detect_bad_pixels()`), with
  `pixel_policy="clip"` (clamp negatives to zero, counted in
  `TiffLoadInfo::n_clipped_pixels`; NaN still errors) and
  `pixel_policy="allow"` (pre-normalized transmission, used by the MCP
  server and the GUI transmission tab) as explicit escape hatches.
- **Three real-VENUS defects from post-merge validation** (#676): fixes
  surfaced validating the merged build against IPTS-37432 data. (#648,
  fitting) the multiplicative-baseline `E_ref` was placed over the full
  cross-section grid instead of the active `fit_energy_range`, so on a VENUS
  Ta grid (4.5 eV–2.3 MeV) with an 8–45 eV fit window `E_ref` landed at
  ~3211 eV instead of ~19 eV — collapsing the `1, z, z²` basis's
  orthogonality and letting the baseline silently absorb Doppler broadening;
  `UnifiedFitConfig::baseline_reference_energy()` now folds the fit-range
  mask (all 7 construction sites updated; new
  `baseline_reference_energy_active`), a software fix with no
  temperature-correctness claim. (#653, io) duplicate DAQ chunks — every
  real VENUS multi-chunk folder is a byte-identical duplicate write of one
  exposure — no longer silently double-count: `load_chunked_sum` fingerprints
  each chunk (FNV-1a over the f64 bits) and refuses one identical to any
  earlier chunk (naming the `sum_chunks=false` escape), while distinct chunks
  still sum. (#652, io) `run_health` / `read_pv_series` now drop the corrupt
  B12 furnace-reconnect DASlogs records (`time=0.0` with a subnormal value —
  subnormals are `is_finite()`, so one otherwise entered the power median or
  tripped the ascending-time guard); genuine backward time jumps still error.
- **Analytic Jacobian for no-temperature fits without precomputed σ**
  (#635): `build_transmission_model` now precomputes working-grid σ for
  this case, so the model exposes its analytical Jacobian instead of
  silently degrading the LM path to finite differences and the counts
  joint-Poisson stage to an identity-Fisher gradient descent (which
  crawled through correlated-parameter valleys and, on the real-VENUS
  regression fixture, stopped 1.7 % short of the true deviance optimum).
  The model output is unchanged (bit-exact parity with the previous
  forward-model path, pinned by a committed test); only the optimizer
  quality improved.
- **Joint-Poisson deviance is clamped at its mathematical floor**
  (#635): per-bin xlogy round-off could report a total deviance of
  ~−1e-13 on machine-exact fits, violating the D ≥ 0 contract; D == 0
  (a perfect fit) is now reported as converged instead of inflating the
  damping to its ceiling and returning non-converged. The clamp is
  scoped to the accumulation round-off envelope — a deviance negative
  beyond it surfaces as an error instead of a silent perfect fit.

### Documentation

- **Temperature uncertainty documented as a covariance-only lower bound**
  (#638): `temperature_uncertainty_map` / `temperature_k_unc` (rustdoc,
  the PyO3 getters, the `.pyi` stubs, and the `docs/guide` SpatialResult
  table) now state that on the Poisson-KL / joint-Poisson paths they are
  inverse-Fisher lower bounds that omit baseline/model noise and can
  underestimate the observed per-superpixel scatter by ~3–4× on real
  data; `scale_by_chi2=True` gives the χ²-inflated estimate.

### Changed

- **rand 0.9→0.10 and egui 0.33→0.34 ecosystem upgrades** (#670, #671):
  coordinated bumps that supersede six Dependabot PRs (#660, #665, #661,
  #663, #664, #668) which each break the build in isolation.
  - `rand`/`rand_chacha` → 0.10, `rand_distr` → 0.6. Only `rand_core` 0.10
    is actually compiled (`cargo tree` shows a single built `rand_core`);
    the lockfile still lists older `rand_core` from optional/transitive deps
    (phf build-dep, quinn) that are not in the default build graph. No API
    changes at our one call site (the `ic_closed_loop` seeded-noise test):
    the seeded ChaCha12 stream is algorithm-stable, so closed-loop
    temperature recovery passes with no re-baseline or tolerance change.
  - `eframe`/`egui`/`egui_extras` → 0.34, `egui_plot` → 0.35,
    `egui-file-dialog` → 0.13. eframe 0.34 makes **wgpu** the default
    renderer; NEREIDS pins **glow** (`Renderer::Glow`) with eframe default
    features kept ON, preserving the `winit/default` wayland-dlopen display
    stack proven on the ORNL ThinLinc / manylinux_2_28 fleet. The egui 0.34
    render-model migration (`App::update`→`App::ui`, panels via
    `egui::Panel::{left,top,bottom}` + `show_inside`, `Context::style`→
    `global_style`) is internal to `apps/gui` — no user-facing behavior
    change.

- **Config-class Python errors now raise `ValueError` uniformly** (#635
  review): `fit_spectrum_typed`, `fit_counts_spectrum_typed`, and
  `compute_model_jacobian` previously stringified every `PipelineError`
  to `RuntimeError`; invalid-input errors
  (`PipelineError::InvalidParameter` — e.g. an init outside its bounds,
  `no free parameters`, a bad `fit_energy_range`) now raise `ValueError`,
  matching `spatial_map_typed`. Scripts catching `RuntimeError` for
  config-class errors must catch `ValueError` instead.

- **Physics-complete bounded Ikeda–Carpenter calibration family** (#642):
  `family="ic"` now fits the full moderator shape — the prompt law
  `α(E) = e^{c0}·√E + e^{c1}` is positive at every energy by construction
  (exp-encoded; a real calibration once returned `a1 = −0.396`, driving
  α(E) < 0 at low energy), the storage rate `β` and mixing fraction
  `R ∈ [0, 1]` are free within physics bounds (previously β was pinned at
  0.1 and R ≡ exp(−E_meV/25) ≈ 0, i.e. no storage freedom at all), and the
  kernel is folded with the SNS PSR channel triangle (`psr_fwhm_ns`,
  default 350 ns — the fold the VENUS FTS header records for the tabulated
  kernel; `0` disables; optional `fit_psr` adds the FWHM as a 5th
  parameter). The missing shape freedom previously re-expressed as a ~90 K
  temperature degeneracy on real data. The IC τ-grid is now anchored to the
  prompt core and refined to resolve any requested burst/channel fold, so
  low-β storage tails can neither undersample the pulse rise nor silently
  degenerate the fold to a delta; a parameter combination whose τ-grid
  cannot be resolved within the 8192-sample cap is now rejected with a
  descriptive error (`IkedaCarpenter::new` / `kernel_at`) instead of
  silently under-sampled — the calibrator treats such points as infeasible
  during its search. Synthesized kernels are bit-identical to earlier
  releases only when the storage tail does not extend past the prompt reach
  (`slow_reach ≤ fast_reach`, the R ≈ 0 eV-regime case), no fold is finer
  than the prompt design step, and `n_tau` ≤ 8192; direct-API kernels with
  constant R > 0 and long storage tails (or folds finer than the design
  step) change — finer, more accurate sampling. **Action required: earlier
  `family="ic"` calibrations (2-parameter, no PSR fold) are superseded —
  re-calibrate.** The raw `theta` for `"ic"` is now 4–5 ln/box-encoded
  values; read decoded physical parameters from `params()` (Python) or the
  returned `resolution` (Rust) instead of interpreting `theta` directly.

- **GUI normalization now masks dead ∪ hot on every raw-counts path**
  (#643): the TIFF-pair and HDF5-with-open-beam paths mask
  dead(sample) ∪ hot(sample) ∪ dead(OB) ∪ hot(OB) instead of
  dead(sample) only, and the HDF5-without-open-beam path (histogram or
  event data prepared with uniform weighting) now runs the same
  detection on the sample stack alone — masked-pixel counts may differ
  on re-normalization of existing data.  TransmissionTiff runs no
  detection: its data is a pre-normalized transmission ratio, not raw
  counts, so the hot screen's Poisson floor does not apply and an
  all-zero "dead" test would conflate opaque scene with a dead
  detector.  Mask provenance is explicit (#646): the file-declared mask
  (HDF5 dead-pixel dataset or a saved project's mask) is kept
  separately, and every normalization recomputes the effective mask
  from scratch as declared ∪ freshly-detected — detections never
  accumulate across open-beam swaps or re-normalizations.
- **Project files persist the pixel mask as two separate components**
  (#646): `/intermediate/dead_pixels` (path, dtype, and shape
  unchanged) now stores the declared mask rather than the effective
  (declared ∪ detected) mask, so save→load→save cycles no longer bake
  each session's detections into the next session's declared component;
  the detected component is persisted alongside it as a new
  session-scoped, versioned dataset
  (`/intermediate/detected_dead_pixels`, u8, `format_version` = 1).
  On restore the effective mask is rebuilt as declared ∪
  persisted-detected — a restored project carrying embedded normalized
  data without the raw stacks (where detection cannot re-run) refits
  with exactly the dead/hot exclusions active at save time instead of
  silently losing them.  When the raw stacks ARE present on restore,
  detection is recomputed purely as a drift check: a mismatch with the
  persisted detected mask is surfaced in the status bar and the
  provenance log (the saved mask stays active; re-running normalization
  adopts the fresh detection explicitly).  Neither component can grow
  across save/load cycles.
  Restore also validates both persisted components against the restored
  transmission geometry (when the project carries normalized data) and
  drops any component that cannot apply — a hand-corrupted mask can no
  longer become the effective mask (or crash a refit with restored
  ROIs).  Every mask drop on restore is surfaced in the status bar and
  the provenance log, and survives the restore replacing the session
  provenance with the project's own history.
  Backward/forward compatibility: files written by earlier versions
  stored the effective mask at the declared path and lack the detected
  dataset — loading one promotes the stored mask to declared once (the
  only lossless reading of a file that never recorded the split; such
  masks keep any detections they had already absorbed, but stop growing
  from now on) and restores with no detected component (declared-only),
  as does any file whose detected dataset carries an unrecognized
  `format_version` or a rank other than 2.
- Dead/hot detectors' validating entry points (`detect_bad_pixels`,
  `detect_hot_pixels`, `detect_dead_pixels_chunked`) now reject stacks
  and chunks with an empty TOF axis (`shape[0] == 0`), whose vacuous
  all-zero test would otherwise mask the whole detector (#643).
- Pixel-mask documentation now states the semantics explicitly (#643):
  masks are a pipeline-integrity screen only, never a
  data-quality/coverage filter — low-count pixels are alive and are
  kept.

- **`calibrate_energy` rebuilt on the `fit_energy_scale` LM path**
  (#634): the three-phase (L, t₀) grid + per-candidate golden-section
  density search (~35 000 forward evaluations; >10 min on production
  windows) is replaced by a staged search — coarse joint (t₀, L_scale)
  scan with exact per-candidate density, a plateau-robust dip-match
  anchor (handles saturated flat-bottom dips), fine joint pit-scans
  around each anchor, then a multi-start LM descent scored by the
  original valid-bins χ² (argmin). ~4× fewer forward evaluations on
  production windows, and the LM refinement removes the old grid's
  resolution floor (the optimum is continuous rather than quantized to
  0.001 % L / 0.05 µs t₀). New wide-offset round-trip regression tests
  pin recovery at production-scale offsets (0.3–1.2 % L, 1–6 µs t₀)
  across trace, mid-band, and saturated-foil densities. The public
  signature, `CalibrationResult` fields, density band `[1e-5, 1e-2]`,
  boundary-saturation and no-finite-χ² error contracts, and the
  `dof = n_valid − 3` reduced-χ² convention are unchanged; all existing
  calibration tests pass unmodified.


## [0.2.2] - 2026-07-03

### Fixed

- **Between-reference tabulated/IC kernel widths follow the physical
  power law** (#632): `interpolated_kernel` blended bracketing
  reference kernels element-wise — an arithmetic width chord over the
  convex σ_t ∝ ~E^(−1/2) law that systematically over-widened every
  between-reference energy (+7.8 % at a synthetic 10/50 eV midpoint;
  +4.1–7.2 % across the production VENUS 5→50 eV reference gap,
  spanning nearly all Ta-181 thermometry resonances — fitted
  temperatures read low as a result). Kernels are now interpolated as
  width-normalized shape blends with geometrically interpolated widths
  (exact for power-law files); the unequal-point-count
  nearest-reference fallback and its ±few-% IC width sawtooth are gone.
  This is a documented intentional departure from SAMMY, whose UDR
  interpolation takes the same arithmetic chord (`udr/mudr3.f90`).
  Fit-range support margins now track the true (narrower)
  between-reference kernel. **Action required: `calibrate_resolution`
  results and fits whose energy window lies between resolution-file
  reference energies, obtained with earlier versions under tabulated or
  IC resolution, are invalidated and must be re-fit**; externally
  maintained bit-exact broadener baselines must be regenerated.
  Gaussian-resolution and Doppler-only results are unaffected.

- **Tabulated/Ikeda–Carpenter resolution kernels were applied
  time-mirrored** (#631): the broadener gathered theory at `t + Δt`
  (a correlation) instead of `t − Δt` (the convolution — SAMMY
  `udr/mudr4.f90 Ud_Convolute`), putting every kernel's
  delayed-emission tail on the wrong (high-energy) side of every
  resonance. Consequences on strongly asymmetric kernels: mirrored
  model asymmetry in residuals, ~2–3.7 % inflated energy-space model
  width, fitted temperature biased low ~2.5 % at 300 K, and fitted
  `(t0, L_scale)` absorbing the kernel centroid lag with the wrong
  sign. **Action required: any fitted `(t0, L_scale)` energy scales and
  any `calibrate_resolution` results (α(E), width scales) obtained with
  earlier versions under tabulated or IC resolution are invalidated and
  must be re-fit**, and externally maintained bit-exact baselines of
  the broadener must be regenerated. Gaussian-resolution and
  Doppler-only results are unaffected. New sign-pinning regression
  tests cover both application paths in Rust and the
  `load_resolution`/`apply_resolution` Python bindings.
- **Fit-range resolution margin now uses the exact TOF→E map** (#626):
  `kernel_support_ev` previously used the linear chain-rule estimate,
  which under-covers on the high-energy side — the side the corrected
  convolution's delayed-emission tail actually reads from. Extreme
  tails that reach the nominal flight time now report an unbounded
  support, which the GUI clamps to the loaded grid.

- **Linux GUI wheel restored to `manylinux_2_28`** (glibc ≥ 2.28:
  RHEL/AlmaLinux 8+, Ubuntu 20.04+) — reverses 0.2.1's jump to
  `manylinux_2_34`, which made `pip install "nereids[gui]"` unresolvable
  on the ORNL RHEL 8 analysis fleet. The root cause — rfd's gtk3 dialog
  backend pinning `gtk+-3.0 >= 3.24` at build time and dragging a
  vendored 64-library GTK stack into the wheel — is retired for good:
  Linux file dialogs now use the XDG desktop portal (native dialogs on
  any desktop session, with automatic `zenity` fallback) and a pure-egui
  built-in file browser as the final tier. No GTK packages are needed at
  runtime or build time, and the wheel vendors no shared libraries
  (40 MB → ~13 MB).
- **File dialogs can no longer hang or fail silently on Linux** (#526):
  portal dialogs run on a worker thread (rfd 0.17's portal wait loop has
  no timeout and could freeze the UI thread indefinitely), a startup
  probe + portal canary select the built-in browser when no native chain
  works, an escape-hatch overlay offers the built-in browser while a
  native dialog is pending, and dialog-backend failures now surface as a
  visible in-app banner instead of dead buttons. `log`-crate diagnostics
  from dependencies (rfd, opener) are now bridged into the tracing log
  files instead of being discarded.

### Added

- **Wheel-policy CI gate** (`scripts/check_wheel_policy.sh` +
  `.github/workflows/wheel-policy.yml`): both published Linux wheels —
  the GUI wheel and the `nereids` bindings wheel — are built in the
  `manylinux_2_28` container at PR time and checked against the ORNL
  RHEL 8 ceiling (filename tag, `auditwheel` grade, no vendored
  libraries, max versioned-GLIBC symbol) — release-time-only breakage
  (the 0.2.0 yank) is now a PR failure. The publish workflow enforces
  the same policy on release artifacts.

## [0.2.1] - 2026-06-16

### Fixed

- **Release packaging** — the 0.2.0 tag published only partially before
  failing and was yanked; 0.2.1 re-publishes the complete 0.2.0 change set
  (below) with the publish pipeline repaired:
  - The `nereids-endf` and `nereids-physics` self dev-dependencies (which
    expose each crate's `test-support` feature to its own integration tests)
    are now path-only — `{ path = ".", features = [...] }` instead of carrying
    the workspace version. The versioned form made `cargo publish` try to
    resolve the crate's own not-yet-published current version from crates.io
    and abort, which is what halted the 0.2.0 crates publish after `endf-mat`
    and `nereids-core` had already gone live.
  - The Linux GUI wheel now builds on `manylinux_2_34` (AlmaLinux 9, GTK 3.24)
    rather than `manylinux_2_28` (AlmaLinux 8, GTK 3.22), satisfying the
    `gtk+-3.0 >= 3.24` requirement that `gtk-sys` 0.18 (rfd's gtk3 backend,
    added in 0.2.0) imposes. Consequently the Linux GUI wheel now requires
    glibc ≥ 2.34 (e.g. Ubuntu 22.04+, RHEL / AlmaLinux 9+).

## [0.2.0] - 2026-06-16 [YANKED]

> **Yanked.** The release pipeline failed partway through: `nereids` on PyPI
> and the `endf-mat` / `nereids-core` crates were published before a packaging
> bug aborted the remaining crates and the Linux GUI wheel. The 0.2.0
> artifacts that did publish have been yanked — **use 0.2.1**, which ships the
> same change set listed here.

### Changed

- License switched from BSD-3-Clause to MIT across the entire workspace —
  the root `LICENSE` text, the workspace `Cargo.toml` license field inherited
  by every crate, and `endf-mat`'s former `MIT OR Apache-2.0` dual license
  (now also inherits the workspace MIT field). Python wheel metadata
  (`pyproject.toml` license + classifier for both `nereids` and
  `nereids-gui`), `CITATION.cff`, the README badge/License section, and the
  CONTRIBUTING contributor-license statement all updated to match. Copyright
  years are now 2025-2026, and the copyright holder reads
  "UT-Battelle, LLC (Oak Ridge National Laboratory)" everywhere. Per the
  repository's bump-at-release convention, in-tree metadata (manifests,
  CITATION.cff) describes the current source tree — which is MIT — while
  still carrying the last released version number until the next release
  commit bumps it; the 0.1.8 artifacts already on crates.io, PyPI, and
  Zenodo were published under BSD-3-Clause (endf-mat: MIT OR Apache-2.0)
  and are unaffected. MIT applies to every version published from here on.
  Each publishable crate directory (and `apps/gui` for the wheel) now
  carries a copy of the LICENSE text, so published artifacts ship the MIT
  permission notice they are licensed under.
- **Detectability tool defaults** are less pessimistic: the default matrix
  areal density is now `0.005` at/barn (a realistic ~mm-scale sample rather
  than a vanishingly thin one) and the default expected counts/bin (I₀) is
  now `100_000`, so a typical first run lands near the detection boundary
  instead of always reading "NOT DETECTABLE". The verdict math is unchanged
  and both remain user-adjustable in the Advanced panel. (#620)
- Input validation hardened across the public API: energy grids,
  cross-section entry points, NeXus/IO inputs, and the spatial detector cube
  now reject NaN / non-finite / non-physical values with typed errors in
  release builds (not debug-assert-gated). (#559, #565, #592, #602)
- Internal architecture refactor (net ~−1600 LOC, no behavior change):
  dead-code and unused-API removal, test-fixture/oracle consolidation, and a
  shared `ResolutionPlan` helper extraction. (#580–#589)

### Changed (breaking — `nereids-io`)

- `NexusMetadata.tof_edges_ns: Option<Vec<f64>>` has been renamed to
  `NexusMetadata.tof_edges_us: Option<Vec<f64>>`.  The old field name was
  a misnomer: the values stored in it were already microseconds after
  `units`-attribute rescaling inside `probe_histogram_group` — they were
  never nanoseconds.  The field is now correctly named so callers do not
  need to second-guess the unit at the API boundary.  No data conversion
  is needed for downstream consumers; the numeric payload is unchanged.
  Per semantic versioning for `0.x` crates, this is an acceptable
  breaking change at this stage of the project; no external consumers
  of this field have been identified in the workspace or sibling repos.

### Fixed

- **Doppler broadening** now uses the exact SAMMY Free-Gas-Model kernel
  (manual Eq. III B1.7: a `w²`-weighted integrand divided by `E`) in both the
  forward pass and the analytical temperature-derivative path, replacing a
  kernel that omitted the `(w/v)` weight — a first-order antisymmetric skew on
  resonance flanks. Includes two latent edge-case fixes (always-on low-side
  velocity-grid padding; a SAMMY-faithful sparse-edge passthrough). SAMMY-oracle
  errors improved across the validation suite. (#611)
- **SLBW / MLBW cross-sections**: corrected the sign of the `Γ·sin²φ` elastic
  interference term (#549, #550) and removed an s-wave velocity-factor
  double-count (#577); both tighten agreement with SAMMY reference cases.
- **R-Matrix Limited (LRF=7) KRM=3**: SAMMY-parity fixes to the APE/APT radius
  roles, subthreshold channel widths, and the per-pair PNT penetrability flag.
  (#589)
- **Resolution broadening** is applied on the auxiliary (margin-extended) energy
  grid across all transmission-model fit paths, removing edge bias near
  fit-range boundaries. (#608)
- Further physics/robustness fixes: phase-shift continuity, RML closed-channel
  gating, and a sample-thickness guard (#599, #607); joint-Poisson no longer
  reports convergence on an all-fixed-parameter fit (#575); energy-calibration
  coverage and a zero-valid-bin guard (#563).
- **ENDF parsing**: fixed a URR Case-B (LFW=1 / LRF=1) resonance-stream
  misalignment (#606), and now hard-reject unsupported MF=2 NIS>1 and LRF=7
  layouts instead of mis-parsing them (#576).
- **NeXus I/O**: validate the `time_of_flight` `units` attribute, fixing a
  silent 1000× rescale on files written with `units = "us"` (#561).

### Security

- Bumped `pyo3` and `numpy` 0.28 → 0.29, clearing two Dependabot advisories:
  **GHSA-36hh-v3qg-5jq4** (HIGH — out-of-bounds read in `PyList`/`PyTuple`
  `nth`/`nth_back`) and **GHSA-chgr-c6px-7xpp** (MEDIUM — missing `Sync` bound
  on `PyCFunction::new_closure`). No binding code changes were required. (#615)

### Documentation

- Claim-accuracy pass: every README / guide / rustdoc capability claim and
  SAMMY citation verified against primary sources (#610). Review-process
  metadata scrubbed from production comments; `spatial_map_typed` and the
  crate module lists documented; research-only scripts moved out of the tree
  (#614). Guide screenshots refreshed for the current GUI and a stale
  solver-settings caption corrected (#617–#619).

## [0.1.8] - 2026-04-27

### Added

- TENDL-2023 ENDF library with full MAT-table coverage (#508)
- GUI: optional fitting of the energy scale — TZERO + flight-path L scale (#501)
- GUI: vertically stacked image + spectrum layout in the Analyze step (#506)
- Counts-domain joint-Poisson (KL) fitting with 2D spatial-map integration and
  Python API parity (#450)
- Sparse empirical-cubature forward-model surrogate for multi-isotope fits, plus
  a scalar k=1 Chebyshev surrogate (epic #472: #479, #481, #482)
- VENUS Hf 120-min NeXus test fixtures (via the PLEIADES submodule)

### Changed

- Resolution broadening rebuilt around a reusable `ResolutionPlan`, compiled to a
  CSR `ResolutionMatrix` (~4.2× faster apply, bit-exact) and wired through the
  fixed-grid fit models (#467, #468, #470, #478)
- Two-pointer walk in `broaden_presorted` (~4.2× on VENUS) (#464)
- Energy-scale (TZERO) Jacobian: density-column plan cache + partial-GAL
  default (#469, #484, #498)
- KL "polish" pass now off by default (~100× faster single-spectrum fits) (#487)

### Fixed

- MLBW batch-API correctness + dispatch consolidation, gated against VENUS (#466)
- GUI: invalidate the cached fit when solver controls change (#507)
- ENDF: use the OS trust roots for IAEA fetches (#505)
- Spatial map: NaN-on-failure handling, config guards, and binding
  validation (#461)
- Refuse to silently collapse multiple rotation angles in the NeXus histogram
  loader (#463)

### Build

- Bump `rustls-webpki` (#477) and `rand` (#449)

## [0.1.7] - 2026-04-04

### Added

- Uncertainty estimation and propagation, Phases 1-4 (#446)
- Python bindings for NeXus histogram and event loading (#447)
- Resolution-aware analytical Jacobians (#445)

### Fixed

- Apply resolution broadening *after* Beer-Lambert, including the temperature
  path of `TransmissionFitModel` (#442 → #443, #444)

## [0.1.6] - 2026-03-23

### Added

- Unified typed fitting pipeline: a solver-agnostic `ForwardModel` trait, typed
  `InputData` constructors (`from_counts` / `from_transmission`),
  `UnifiedFitConfig`, and the `fit_spectrum_typed` / `spatial_map_typed` entry
  points (Python + Rust)
- True multi-level Breit-Wigner (MLBW) with a coherent U-matrix, replacing the
  prior SLBW-dispatch approximation
- SAMMY-style normalization and background fitting for transmission, plus a
  KL-native background model for counts data
- Counts + KL temperature fitting with analytical Doppler temperature derivatives
- Unresolved-resonance region: LFW=1 (energy-dependent fission widths) and all
  five ENDF interpolation laws (INT=1..5)
- Energy calibration (`calibrate_energy`) with resolution broadening
- Constrained isotope-group fitting (core, GUI, and a tutorial notebook)
- Analytical Jacobian for the Poisson solver (incl. a temperature derivative)

### Changed

- Consolidated the fitting stack on a single optimizer and API: removed the
  external L-BFGS-B solver and the legacy pre-typed fitting/spatial API (the
  Poisson/KL path now uses an analytic Gauss-Newton step with an internal
  finite-difference L-BFGS fallback)
- ROI spectrum averaging now uses inverse-variance weighting instead of an
  arithmetic mean
- `spatial_map` / `sparse_reconstruct` `n_total` now count *attempted* pixels,
  not only successes
- Wrapped large shared buffers (`sample_data`, `open_beam_data`,
  `spectrum_values`) in `Arc` to avoid per-pixel clones (#374)
- Project files now persist normalized uncertainty, dead-pixel masks, and the
  anorm / background maps

### Fixed

- RML phase-convention correction (+ documented SAMMY truth sources); MLBW phase
  convention + total cross-section regression
- JENDL-5 Eu-151/153 parse failure: deduplicated URR energy grid (#402, #411)
- Detectability SNR computation and resolution-grid handling
- GUI ENDF-fetch defects and chip layout (#403, #410)
- Data-integrity hardening: reject synthetic energy axes, count load errors,
  eliminate silent pixel drops

### Project

- Added `CITATION.cff`, `CONTRIBUTING.md`, `CHANGELOG.md`, a Code of Conduct, and
  a Zenodo DOI badge

## [0.1.5] - 2026-03-11

### Fixed
- Switched HTTP backend from `native-tls` to `rustls-tls` to eliminate
  OpenSSL runtime dependency on Linux clusters
- Fixed crates.io publish detection for already-existing crates
- Fixed PyPI sdist LICENSE file path
- Built Python wheels for all supported versions (3.10-3.13) on all platforms

## [0.1.0] - 2026-03-11

### Added

#### Cross-Section Physics
- Reich-Moore R-matrix cross-section calculation
- Single-Level Breit-Wigner (SLBW) formalism
- R-Matrix Limited (LRF=7) with KRM=2 and KRM=3 support
- Unresolved Resonance Region (URR) via Hauser-Feshbach
- Coulomb wave functions (Steed's continued-fraction method)
- Energy-dependent scattering radius (NRO=1 TAB1 interpolation)

#### Broadening Models
- Free Gas Model Doppler broadening (O(N*W) optimized)
- Resolution broadening: Gaussian (channel width + flight path) and
  tabulated instrument functions
- Joint temperature + density fitting with bounds-based preconditioning

#### Data Processing
- ENDF/B resonance parameter loading from IAEA (all evaluated libraries)
- 535 built-in MAT numbers in `endf-mat` crate
- TIFF stack and folder I/O with TOF normalization
- NeXus/HDF5 histogram and event data loading
- Energy rebinning (sum for counts, average for transmission)

#### Fitting & Analysis
- Levenberg-Marquardt optimizer with analytical Beer-Lambert Jacobian
- Poisson/KL divergence optimizer (analytic path preferred; L-BFGS-B available) for low-count data
- Parallel per-pixel spatial mapping via rayon
- Region-of-interest (ROI) spectrum fitting
- Trace-element detectability analysis with energy-window optimization

#### Python Bindings
- Full PyO3 API: cross-sections, forward model, fitting, spatial mapping,
  I/O, detectability, element utilities
- Type stubs (PEP 561) for IDE support
- 25 Python tests

#### GUI Application
- egui desktop application with guided 5-step workflow
- Landing page with decision wizard (6 pipeline configurations)
- Studio mode: three-pane result explorer with density maps
- Forward Model, Detectability, and Periodic Table tools
- Project file save/load (.nrd.h5) with embedded and linked modes
- Session persistence via eframe::Storage
- macOS DMG and pip-installable wheel distribution

#### Notebooks
- 17 Jupyter tutorials across 4 tiers:
  - Foundations (6): cross-sections, SLBW, Doppler, resolution, URR, transmission
  - Building Blocks (6): ENDF loading, element utilities, fitting, multi-isotope,
    custom resolution, TIFF I/O
  - Workflows (4): enrichment analysis, trace analysis, forward model, spatial mapping
  - Applications (1): full 2D isotopic density mapping demo

#### Infrastructure
- CI/CD: cross-platform tests (Linux, macOS, Windows), coverage, rustdoc
- Publish pipeline: PyPI, crates.io, GitHub Releases, Homebrew cask
- Documentation site: mdBook user guide + rustdoc API reference on GitHub Pages
- SAMMY validation suite: 43 test cases validated against SAMMY reference code

[Unreleased]: https://github.com/ornlneutronimaging/NEREIDS/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/ornlneutronimaging/NEREIDS/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/ornlneutronimaging/NEREIDS/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/ornlneutronimaging/NEREIDS/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/ornlneutronimaging/NEREIDS/compare/v0.1.8...v0.2.0
[0.1.8]: https://github.com/ornlneutronimaging/NEREIDS/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/ornlneutronimaging/NEREIDS/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/ornlneutronimaging/NEREIDS/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/ornlneutronimaging/NEREIDS/compare/v0.1.0...v0.1.5
[0.1.0]: https://github.com/ornlneutronimaging/NEREIDS/releases/tag/v0.1.0
