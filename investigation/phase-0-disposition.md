# Phase 0 disposition recommendations

These are evidence-backed recommendations, not cleanup commits. Compatibility
removal follows warnings, migration tests, and a versioned deprecation window.
“Fit improved” is never an acceptance criterion.

## Fixed quantitative acceptance protocols

These thresholds are fixed in Phase 0 so implementation cannot choose a gate
after seeing its result. Every referenced test includes all generated cases and
optimizer failures.

- **Q1 — deterministic recovery:** on a noise-free matched spectrum/cube with
  at least 200 energy bins, every nonzero free density/nuisance coordinate has
  relative error <= `1e-6`, every injected zero coordinate has absolute error
  <= `1e-10`, and maximum modeled-transmission error is <= `1e-10`.
- **Q2 — stochastic inference:** use 1,000 fixed-seed replicates at expected
  open counts/bin 25, 250, and 2,500 and `c` in `{1, 5.982033}`. Absolute
  relative bias for every injected nonzero fitted coordinate is <= 10%, 2%,
  and 1%, respectively; Wilson 95% confidence intervals for each coordinate's
  empirical 68% and 95% coverage contain `0.6827` and `0.95`.
- **Q3 — numerical/operator parity:** maximum transmission difference is <=
  `1e-6` and <= `0.1` observed sigma, RMS difference is <= `0.01` sigma,
  fitted target shift is <= `0.1` reported statistical sigma, and deviance
  changes by <= `0.1%`.
- **Q4 — derivative parity:** central differences at relative steps `1e-6`,
  `3e-6`, and `1e-5` give median normalized Jacobian error <= `1e-5`, no
  component error > `1e-4`, and directional `Jv` error <= `1e-5`.
- **Q5 — identifiability/uncertainty:** the active Jacobian has
  `s_min/s_max >= 1e-8`; each interpreted coordinate has two-sided profiles
  crossing delta-objective `1.0` and `3.84` before either bound; simulated
  interval coverage passes Q2.
- **Q6 — response validity:** response entries are >= `-1e-14`, every column
  sums to one within `1e-12`, omitted tail mass is <= `1e-8`, and a 2x grid
  refinement passes Q3.
- **Q7 — background/nuisance recovery:** density and every injected
  background coordinate pass Q1 on noise-free data and Q2 on stochastic data;
  predicted open/sample arms pass Q3 against an independent generative oracle.
- **Q8 — real held-out transfer:** configuration and bins are frozen before
  unblinding, no residual-driven exclusion is allowed, 95% predictive-interval
  coverage is 93%-97%, standardized-residual absolute mean is <= `0.1`, and
  standardized-residual RMS is 0.9-1.1 on the held-out set.
- **Q9 — surface contract:** a table-driven golden test covers every supported
  and rejected domain/solver/background/resolution combination; allowed calls
  serialize the exact resolved config, invalid/unknown fields fail before the
  optimizer, no configured field is dropped, and spatial status counters sum
  exactly to the number of pixels.
- **Q10 — performance with parity:** after Q3 passes, median of 20 warmed
  release runs on fixed hardware is at least 5x faster for calibration and 3x
  faster for a fixed-response map than the recorded Phase 0 implementation.

## Canonical one-to-one disposition registry

This is the canonical R15 registry. Each normalized support cell from the
support matrix occurs exactly once. A gate is satisfied only by the stated
mechanism or contract check; a lower residual by itself never satisfies one.

| ID | Disposition | Compatibility and migration impact | Falsifiable acceptance gate |
|---|---|---|---|
| M01 | **keep+improve** | Preserve transmission LM as the default and keep existing result fields. | Matched F1 tests for R0, R1, and R2 recover every free density within `1e-6` relative and the E02 real anchor remains within its stored tolerances. |
| M02 | **keep+improve** | Keep native Rust IC but do not promise parity until an integration test exists. | On one fixed IC kernel/grid, native R3 and its R4 table give max absolute transmission error `<1e-10` and fitted-density relative difference `<1e-6`. |
| M03 | **keep** | Preserve `.as_tabulated()` as the Python compatibility route. | The E11 F1 recovery remains within `1e-6` relative; native R3 versus R4 gives max absolute transmission error `<1e-10` and fitted-density relative difference `<1e-6`. |
| M04 | **deprecate** | Warn F2 users and migrate normalized transmission to F1 or raw data to F3 before removal. | Every Rust/Python/MCP/GUI entry that can select F2 emits a tested deprecation diagnostic; negative T is rejected and no transmission call accepts `joint_poisson` as a count-likelihood claim. |
| M05 | **deprecate** | Same migration as the R0 F2 route; resolution choice must not suppress the warning. | One warning/rejection test passes for each R1-R4 representation and a repository usage search finds no undocumented internal F2 caller before deletion. |
| M06 | **keep+improve** | Preserve current F3 behind a versioned response-model selector while adding separate migrated sample/open-beam arms. | The new operator forms separate `R[Phi]` and `R[Phi*T]` arms and passes Q2 and Q6 on flat and structured flux. |
| M07 | **keep+improve** | No API removal; add explicit true-UDR and native-IC integrations before advertising them for counts. | Matched F3 integrations for R2 and R3 recover density within `1e-6` relative and agree with direct forward-arm oracles within `1e-10` absolute per bin. |
| M08 | **keep+improve** | Retain archived R4 workflows and add response-model provenance to their results. | E03 and E11 replay, the emitted result names the effective response operator, and R4 agrees with its source IC kernel under Q3. |
| M09 | **deprecate** | Reject `c != 1` immediately, warn all remaining F4 users, and direct them to F1 or F3. | A boundary test rejects unequal exposure, low-count inputs cannot silently enter F4, and at 2,500 expected open counts/bin its fitted density differs from F3 by <= 1%. |
| M10 | **deprecate** | Resolution selection does not preserve F4; migrate as for the R0 cell. | Each R1-R4 entry emits the same warning/rejection and, at 2,500 expected open counts/bin, differs from its F3 counterpart by <= 1% before removal. |
| M11 | **complete before exposure** | Rename the working zero-B route as paired open beam; hide Bdet/alpha controls until implemented. | Public type/stub/guide names agree on paired `(S,O,c)` semantics, zero-B recovery passes Q1, and unsupported B5 fields satisfy Q9 rejection parity. |
| M12 | **complete before exposure** | Do not advertise broadened F5 until each representation has a route-specific test. | R1-R4 paired-open-beam tests recover matched density within `1e-6` relative and return the effective resolution provenance. |
| M13 | **keep** | Preserve a stable explicit rejection; remove F6 from solver-choice documentation. | Rust and Python calls with every solver alias return the same typed “nuisance counts require counts-domain solver” error before model construction. |
| M14 | **keep+improve** | Preserve spatial transmission LM and existing maps. | R0-R2 spatial matched maps pass Q1, dead/cancelled pixels satisfy Q9, and a measured spatial fixture passes Q8. |
| M15 | **keep+improve** | Keep R3/R4 reachable but label them unverified until tested. | Native and tabulated IC spatial maps agree within `1e-6` relative in density and `1e-10` absolute in modeled transmission on a matched cube. |
| M16 | **deprecate** | Deprecate with F2 and migrate to S1 or paired/raw-count spatial routes. | Every spatial transmission-KL entry warns, negative T is rejected up front, and the migration documentation names S1/S3/S5 by input contract. |
| M17 | **deprecate** | Same migration as R0 S2 for every broadened representation. | R1-R4 route tests assert the deprecation/rejection before per-pixel work and no all-NaN map substitutes for an error. |
| M18 | **keep+improve** | Rename as explicit averaged-open-beam mode; make paired S5 the scientific default. | The result is flagged whenever nonzero-pixel total CV exceeds `sqrt(shot_CV^2 + 0.01^2)`; E05 must trigger that flag instead of returning an unqualified map. |
| M19 | **keep+improve** | Preserve only as the explicitly labeled approximation for all resolutions. | The R1-R4 result carries averaged-OB provenance and the E05 mechanism gate remains detectable with each operator. |
| M20 | **deprecate** | Reject unequal exposure and migrate spatial raw counts to S3/S5 or normalized data to S1. | `c != 1` fails before the pixel loop, every remaining S4 call warns, and no failed pixel map is reported as route success. |
| M21 | **deprecate** | Same migration as R0 S4; broadening does not grant continued support. | R1-R4 preflight tests assert the warning/rejection and, at 2,500 expected open counts/bin, each S4 density differs from S3 by <= 1% before removal. |
| M22 | **complete before exposure** | Preserve and rename paired per-pixel open beam; hide unfinished detector-background semantics. | E05 paired-gradient recovery remains `<1e-6` relative, the result names paired O rather than known flux, and unsupported B5 controls cannot be selected. |
| M23 | **complete before exposure** | Add resolution-specific paired-route coverage before public recommendation. | R1-R4 matched paired cubes recover every live pixel within `1e-6` relative and publish resolution plus pairing provenance. |
| M24 | **keep** | Preserve a single preflight error and omit S6 from solver choices. | Rust and Python spatial calls reject S6 before allocation/parallel fitting with one stable typed diagnostic. |
| M25 | **keep+improve** | Keep B1 for F1/S1; add rank diagnostics without changing default-off behavior. | Injected `Anorm+ABC` tests pass Q1 and Q5, including rejection of the rank-deficient Anorm/temperature/density construction. |
| M26 | **deprecate** | B1 users on F2/S2 migrate with the underlying objective. | B1 cannot suppress the F2/S2 deprecation, and warning tests pass for single and spatial APIs across each surface. |
| M27 | **keep+improve** | Preserve B1 on F3 while moving it into the versioned response-aware count model. | A nonzero-ABC counts ensemble with separate response arms passes Q2 and Q7. |
| M28 | **keep+improve** | Keep only within explicitly averaged-OB S3 and surface both approximation and background provenance. | The E05/B1 gradient exceeds the fixed M18 flatness threshold, while a flat-beam injected-ABC cube passes Q1 and Q7. |
| M29 | **deprecate** | B1 does not save F4/S4; migrate with those routes. | Background-enabled F4/S4 calls emit the same route warning and unequal-exposure rejection as their B0 cells. |
| M30 | **complete before exposure** | Retain internally for paired zero-Bdet development; do not advertise until independently recovered. | F5/S5 paired tests with injected ABC pass Q1 and Q7 for at least R0 and R1. |
| M31 | **keep+improve** | Keep BackD/F experimental and disabled by default; preserve paired-flag validation. | Synthetic D/F data pass Q1 and Q5, zero/nonfinite starts fail up front, and one held-out set passes Q8. |
| M32 | **deprecate** | D/F follows F2/S2 deprecation and must not imply statistical validity. | Single and spatial D/F calls on F2/S2 emit the objective warning before fitting for every exposed surface. |
| M33 | **keep** | Preserve the explicit joint-Poisson rejection; remove B2 controls from counts-only UI/schema choices. | F3/S3/F5/S5 reject either D/F flag before optimization with one consistent error and no result object. |
| M34 | **deprecate** | Remove B2 from F4/S4 documentation and migrate users with the fallback route. | B2-enabled fallback calls warn, reject `c != 1`, and have no undocumented surface entry before route removal. |
| M35 | **keep+improve** | Preserve B3 with bounded/global modes and add response-placement provenance. | Injected-baseline tests pass Q1 and Q5, positivity/rank guards fire, and the held-out set passes Q8. |
| M36 | **deprecate** | B3 users on F2/S2 migrate with the invalid objective. | Baseline configuration cannot suppress the F2/S2 warning and all surface aliases reach the tested deprecation path. |
| M37 | **keep+improve** | Keep the real archived baseline workflow while adding separate-arm response semantics. | E03 replays and a structured-flux injected-baseline test passes Q3 and Q7 against the independent two-arm oracle. |
| M38 | **keep+improve** | Keep global/per-pixel modes only with explicit averaged-OB labeling on S3. | Global and per-pixel injected-baseline maps pass Q1 and Q7 under a flat beam, while E05 exceeds the fixed M18 flatness threshold. |
| M39 | **deprecate** | Baseline-enabled F4/S4 migrates with the ratio fallback. | Calls warn and reject unequal exposure before fitting; no baseline option bypasses the route deprecation. |
| M40 | **complete before exposure** | Keep paired zero-Bdet development private until baseline placement is tested. | F5/S5 injected-baseline paired tests pass Q1 and Q7 and serialize pairing/response metadata. |
| M41 | **keep+improve** | Preserve the combined stack only with fixed Anorm and clear parameter-rank reporting. | Free Anorm+B3 is rejected on every entry; fixed-Anorm injected B1/B3 tests pass Q1 and Q5. |
| M42 | **deprecate** | Combined background users on F2/S2 migrate with the objective. | The valid rank configuration still emits F2/S2 deprecation on single and spatial surfaces before optimization. |
| M43 | **keep+improve** | Keep fixed-Anorm B1+B3 on F3 and continue rejecting B2. | F3 injected combined-stack tests pass Q1 and Q7, and every B2 combination rejects before fitting. |
| M44 | **deprecate** | Combined-stack F4/S4 users migrate to F1 or response-aware counts. | Combined-stack calls cannot bypass `c != 1` rejection or the F4/S4 deprecation diagnostic. |
| M45 | **complete before exposure** | Hide this F5/S5 combination until paired-background recovery is demonstrated. | Fixed-Anorm paired tests pass Q1 and Q7, while nonzero Bdet and B2 remain explicit preflight errors. |
| M46 | **complete before exposure** | Keep the alpha-fit rejection until a versioned likelihood implements it. | Alpha fitting remains a Q9 preflight error until its nonnegative two-arm model passes Q2 and Q7. |
| M47 | **remove** | Remove silent alpha-config acceptance from F4/S4; reject it before LM until a distinct likelihood exists. | Rust single/spatial Q9 tests prove every alpha flag fails before ratio conversion and no accepted result omits a requested alpha field. |
| M48 | **complete before exposure** | Preserve the single-spectrum B5 rejection until the versioned likelihood exists. | Nonzero-background and fitted-alpha fields remain Q9 preflight errors until single paired data pass Q2 and Q7. |
| M49 | **keep+improve** | Keep as an isolated research API; rename it and remove production-equivalence claims. | The helper passes Q4 and serializes fixed-flux objective, parameter layout, `c`, B7, and resolution metadata. |
| M50 | **keep+improve** | Preserve Gaussian transmission-WSSR calibration as the control family. | Gaussian recovery passes Q1; diagnostics pass Q5; high-count count/WSSR agreement passes Q3; one held-out calibrant passes Q8. |
| M51 | **keep+improve** | Keep B6 as legacy but return and bound its coefficients. | An injected Gaussian+B6 calibration passes Q1 and Q5 and serializes all three coefficients with the basis definition. |
| M52 | **keep+improve** | Require a checksummed base UDR and preserve empirical-family labeling. | Both E09 recoveries pass from a committed checksum, flight paths match exactly, and held-out kernel prediction passes Q3 and Q8. |
| M53 | **keep+improve** | Keep B6 optional but do not claim C2 coverage until tested. | A C2+B6 synthetic passes Q1 and Q5 for scale, exponent, and all profiled coefficients. |
| M54 | **keep+improve** | Preserve C3 as experimental; do not physically interpret parameters without identifiability gates. | C3 passes Q3, Q5, Q6, Q8, and Q10 on the fixed multi-standard protocol. |
| M55 | **keep+improve** | Keep B6 experimental and return its nuisance solution instead of discarding it. | C3+B6 passes Q1 and Q5, including a finite two-sided profile for every interpreted IC coordinate. |
| M56 | **remove** | Remove S5's accepted all-failed result for nonzero Bdet; reject before the pixel loop until B5 is implemented. | Q9 proves nonzero Bdet returns the same typed preflight error for F5 and S5, with no `SpatialResult`; a future enabled route must first pass Q2 and Q7. |
| M57 | **remove** | Reject counts-background config on F1 instead of silently ignoring it. | A Rust Q9 test attaches every B5 field to F1 and proves a typed error occurs before LM construction. |
| M58 | **deprecate** | Deprecate F2 and reject attached counts-background config during the migration window. | A Rust Q9 test proves B5 fails before F2 construction and every F2 call still emits its deprecation diagnostic. |
| M59 | **remove** | Reject counts-background config on S1 before the pixel loop instead of silently ignoring it. | A Rust spatial Q9 test proves every B5 field produces one typed preflight error and no map allocation/result. |
| M60 | **deprecate** | Deprecate S2 and reject attached counts-background config before the pixel loop. | A Rust spatial Q9 test proves B5 fails before S2 construction and the S2 deprecation diagnostic remains deterministic. |
| M61 | **keep+improve** | Keep fixed-Anorm B1+B3 on explicitly averaged-OB S3 and continue rejecting B2. | Flat-beam S3 combined-stack data pass Q1 and Q7; the E05 gradient exceeds the fixed M18 flatness threshold; every B2 combination rejects before fitting. |

## Orthogonal cross-surface issue dispositions

X decisions repair surface wrappers and do not replace or compete with the one
core M disposition selected through the P reachability mapping.

| ID | Disposition | Compatibility and migration impact | Falsifiable acceptance gate |
|---|---|---|---|
| X01 | **keep+improve** | Preserve typed Rust APIs while adding a resolved-route/provenance object to results. | Snapshot tests for all F/S/C dispatch classes serialize input domain, effective objective, B class, R class, exposure source, approximation flags, and warnings. |
| X02 | **keep+improve** | Keep Python's domain-sensitive defaults but make the resolved route observable and documented. | Golden Python calls for transmission, counts, and nuisance/spatial inputs assert the same effective route as Rust Auto and snapshot the resolved configuration. |
| X03 | **deprecate** | Warn on compatibility aliases; remove `joint_poisson` from transmission and retain canonical `kl` only where semantics are explicit. | Alias tests prove transmission cannot claim joint Poisson, counts aliases emit warnings during the window, and docs/stubs list one canonical objective name per route. |
| X04 | **keep+improve** | Preserve the C3 native-IC-to-table handoff and add durable checksummed provenance export; do not describe C1 or C2 as IC conversions. | A C3 export/import round trip preserves kernel values within `1e-12`, flight path/family/IC parameters are checksummed, and F1/F3 fitted results match within `1e-6` relative. |
| X05 | **complete before exposure** | Remove impossible detector-background promises until B5 exists; then expose one typed contract. | Binding, stub, and guide have identical fields; disabled B5 passes Q9 rejection parity, and any enabled B5 route first passes Q2 and Q7 for both single and spatial data. |
| X06 | **deprecate** | Migrate MCP count manifests from hidden transmission LM to typed counts Auto/F3 with a versioned manifest warning. | A no-fit-block count manifest resolves to F3, old manifests receive a deterministic migration diagnostic, and a golden summary records the effective route. |
| X07 | **remove** | Replace free-string fallback with strict enums; invalid manifests become errors. | Misspelled domain or solver fails validation before data loading/fitting, with golden errors for each invalid value and no transmission fallback. |
| X08 | **keep+improve** | Keep spatial counts Auto/F3 behavior but make its difference from legacy single manifests explicit. | A live density-map manifest and direct Python call produce the same resolved S3 config/result metadata; FastMCP tests execute rather than skip. |
| X09 | **remove** | Replace silent allowlists with a generated typed schema and reject unknown/unsupported fields. | Every supported core field round-trips through MCP, every unknown field fails validation, incomplete Gaussian/unsupported IC specs fail, and no configured key is silently dropped. |
| X10 | **remove** | Replace ambiguous `success` with separate execution and inference status; preserve warnings/provenance. | A forced nonconverged fit returns execution success plus `converged=false`, structured warnings and resolved route; thrown errors alone set execution failure. |
| X11 | **keep+improve** | Replace implicit GUI domain inference with explicit data-domain plus Auto/effective-objective display. | GUI state tests cover each OB/solver combination and snapshot the same resolved route as the Rust typed API; no hidden domain switch occurs. |
| X12 | **remove** | Block raw HDF5 fitting without open beam; retain only an explicitly typed pre-normalized transmission import. | A GUI regression proves raw HDF5 without OB cannot enable fit, while pre-normalized transmission remains usable and is labeled as such. |
| X13 | **remove** | Collapse proton-charge and `kl_c_ratio` into one checked, persisted exposure-ratio source. | Normalization, F3/S3 config, overlays, saved project, and provenance all read the same value in a state test with non-unit `c`. |
| X14 | **keep+improve** | Keep the overlay only after it renders the fitter's effective averaged-OB model and labels the approximation. | On the E05 gradient cube, plotted fitted counts equal the model-returned arm within `1e-12` per bin and differ from the local-OB counterexample as expected. |
| X15 | **complete before exposure** | Add durable resolution artifacts first; expose GUI/MCP calibration/native IC only after provenance parity. | Python exports a checksummed artifact that GUI and MCP reload with operator parity `<1e-12`; calibration config/result provenance round-trips and route tests execute on both surfaces. |
| X16 | **remove** | Remove overloaded chi-square labels and ambiguous counters; use typed GOF and explicit failure/convergence fields. | Counts results render `D/dof`, WLS renders `chi2/dof`, MCP/GUI distinguish error from nonconvergence, and golden tests cover single and spatial summaries. |
| X17 | **keep+improve** | Define one versioned default profile and always serialize its resolved values. | Q9 golden tests assert identical solver, temperature, library, polish, B, and R defaults for that profile on every surface and serialize every resolved value. |

## Detailed rationale (non-registry)

The original route/component discussion below is retained as rationale. The
table above supersedes it as the one-to-one disposition registry.

### Fit and spatial routes

| ID | Recommendation | Reason | Compatibility/migration | Acceptance before disposition is complete |
|---|---|---|---|---|
| F1 | **keep+improve** | Supplied-σ weighted LM is coherent for normalized transmission and has the strongest broad feature coverage. | Default transmission behavior remains LM. | Shared boundary validation; matched recovery for Gaussian, true tabulated UDR, and IC; background rank diagnostics; real route gates beyond no-background Hf. |
| F2 | **deprecate** | Fractional-transmission Poisson NLL ignores uncertainty, accepts negative observations in release mode, and `joint_poisson` is a false alias. | Warn now; direct users to F1 for normalized T or F3 for raw counts; reject `joint_poisson` alias on transmission first. | Deprecation warning/error tests across Rust/Python/MCP/GUI; usage search; migration note; eventual removal test. |
| F3 | **keep+improve** | Native count likelihood passes analytic, stochastic, and two real-data gates; it is the correct base for low-count inference. | Preserve current route as a compatibility baseline while adding a new response-aware objective behind an explicit model/version selector. | Column-conserving `R[j,i]`; separate `R[Φ]`/`R[ΦT]`; flat/structured-flux oracles; IC/UDR fit integration; background likelihood; real held-out comparison. |
| F4 | **deprecate** | Ignores `c`, omits OB variance, fails real unequal-exposure data, and has 52.8% low-count bias in the matched ensemble. | Immediately reject `c!=1` and counts-only nuisance options; direct low-count/raw users to F3 and normalized users to F1. | Warning/rejection tests, high-count parity envelope, usage search, and documented replacement before removal. |
| F5 | **complete before exposure** | Zero-background paired open-beam mode works, but `flux` uncertainty semantics are mislabeled and advertised detector background/alpha fits are rejected. | Split into an explicitly paired-open-beam input and, if needed, a separately specified known-flux/background likelihood. | Identifiable generative equations; nonzero-background synthetic recovery/coverage; result/covariance semantics; API/stub/guide parity. |
| F6 | **keep** (explicit rejection) | LM has no defined use for the nuisance input contract. | Reject at the boundary with one stable diagnostic; do not advertise as a solver choice. | Rust/Python/spatial/MCP parity tests for the rejection. |
| S1 | **keep+improve** | It reuses coherent F1 and has useful precompute/global-baseline infrastructure. | Preserve spatial transmission LM. | Resolution-matched spatial recovery for all families; real spatial gate; validation parity; cancellation/error semantics. |
| S2 | **deprecate** | It multiplies F2's invalid likelihood and lacks fit-range and independent validation. | Deprecate with F2; migrate to S1 or raw-count spatial routes. | Same surface-wide warning/migration gates as F2. |
| S3 | **keep+improve** | Useful as a variance-reducing approximation, but a gradient control produced fully converged false density structure, and the real VENUS OB has 4.55% spatial total CV versus 1.06% shot expectation. | Rename/document as averaged-open-beam mode; make paired S5-style input the scientific default. | Explicit measured-flatness threshold; account for averaging exposure/variance; return/flag the approximation; held-out spatial comparison. |
| S4 | **deprecate** | Inherits every F4 defect and has additional feature divergence. | Same migration as F4. | Same gates as F4 plus spatial warning/rejection tests. |
| S5 | **complete before exposure** | Paired zero-background route is valuable and recovered the gradient control exactly, but background/alpha contract is unfinished. | Preserve/rename paired open-beam functionality; hide unsupported detector-background controls until implemented. | Per-pixel matched recovery, varying beam/efficiency, nonzero-background likelihood tests, real small-ROI gate. |
| S6 | **keep** (explicit rejection) | Same reason as F6. | Stable boundary error. | Cross-surface rejection parity. |

### Calibration routes

| ID | Recommendation | Reason | Compatibility/migration | Acceptance before disposition is complete |
|---|---|---|---|---|
| C1 Gaussian | **keep+improve** | Matched synthetic recovery works and Gaussian remains a useful control family. | Preserve transmission-WSSR legacy calibration. | Return uncertainty/Jacobian/evaluation diagnostics; transfer t0/L consistently; add raw-count high-count agreement test and a real calibrant gate. |
| C2 UDR correction | **keep+improve** | Width scale/exponent recovery works without discarding the measured kernel shape. | Require a checksummed/provenanced base UDR and clearly label it empirical. | Real UDR fixture; edge/margin parity; held-out prediction; profile identifiability of scale/exponent. |
| C3 IC | **keep+improve** | Physics/kernel tests and synthetic calibrations support continued development, but no reproducible real calibration or count-aligned objective exists. | Preserve current calibrator as an experimental legacy route; do not interpret parameters physically without identifiability/held-out gates. | Continuous/converged kernel numerics; response-aware raw-count calibration; multi-standard profiles; covariance/bootstraps; real held-out transfer; performance target. |

### Model components and research helpers

| Component | Recommendation | Required action/gate |
|---|---|---|
| SAMMY Anorm+ABC | **keep+improve** | Expose selective Python controls; define placement in response-aware count arms; rank/profile diagnostics and synthetic recovery. |
| SAMMY BackD/F | **keep+improve** (experimental) | Harmonize starts/validation, add independent identifiability test and real evidence; otherwise freeze by default. |
| NEREIDS multiplicative log-E baseline | **keep+improve** | Preserve bounded/global modes; validate placement relative to `R[ΦT]/R[Φ]`; real held-out rather than training-GOF selection. |
| Detector-space alpha/background | **complete before exposure** | Specify and implement a nonnegative two-arm likelihood with background measured/placed in the correct arm(s); do not leave dead public flags. |
| Calibration index-linear nuisance | **keep+improve** as legacy | Return coefficients, bound/profile them, and add the same production nuisance option; supersede with aligned count calibration when available. |
| Fixed-flux Fisher helper | **keep+improve** as isolated research API | Rename, remove production-equivalence claim, add `c`/objective metadata; build a true production-objective Jacobian separately. |
| Gaussian resolution | **keep** | Maintain SAMMY/oracle and auxiliary-grid gates. |
| Tabulated UDR | **keep+improve** | Add a real, checksummed fixture and all-family auxiliary-grid/edge contract. |
| Native IC | **keep+improve** | Unify native/tabulated capability routing; add ordinary-fit integration and continuous, numerically converged kernel gates before optimization. |

### Surface-specific routes

| Surface behavior | Recommendation | Migration/acceptance gate |
|---|---|---|
| GUI raw HDF5 without open beam → transmission σ=1 → default KL | **remove** | Block fitting and request an OB; retain only a separately typed pre-normalized-transmission file workflow. Regression test must prove raw counts cannot reach F1/F2. |
| GUI separate proton-charge and `kl_c_ratio` state | **remove** duplicate state | One checked/provenanced ratio must drive normalization, count likelihood, overlays, persistence, and resolved-route metadata. |
| GUI local-OB overlay for averaged-OB S3 fit | **keep+improve** | Render the exact effective modeled arms returned by the fitter and label averaged-OB approximation. |
| MCP single-count hidden transmission/LM default | **deprecate** | Typed counts must default to F3 or require explicit objective; manifest migration and golden-route tests. |
| MCP free-string/typo fallback for domain and solver | **remove** | Strict enum/schema validation; unknown/incompatible values fail before loading/fitting. |
| MCP allowlist that silently drops fit options | **remove** | Generate schema from core typed config and reject unknown fields; route-provenance snapshot tests. |
| MCP `success=true` on unconverged fit | **remove** | Separate execution success from inference convergence and require clients to receive both plus warnings. |
| GUI/MCP lack of calibration/IC persistence | **complete before exposure** | Add checksummed embedded resolution artifact and full calibration provenance before adding workflow controls. |

## Dependency-ordered cleanup and repair sequence

1. **Reproduce and preserve behavior anchors first:** turn the current F1–F6,
   S1–S6, and C1–C3 mechanism probes, failures, rejections, and compatibility
   results into path-specific regression tests before changing any boundary.
2. **Truthful boundaries:** only after those anchors pass, correct
   docs/stubs/result layouts, add deprecation warnings, and reject currently
   ignored invalid combinations (`c!=1` on F4, negative observations on F2,
   bad D/F starts).
3. **Separate semantics:** split transmission likelihood names from count
   likelihood names; distinguish paired open-beam counts from known flux;
   isolate the research Fisher helper.
4. **Shared IC numerical correctness:** fix only kernel/grid/operator defects
   used by surviving routes, with analytic and convergence oracles.
5. **Domain-correct count response:** implement separate open/sample migration
   in F3, then S5; validate backgrounds in their physical locations.
6. **Calibration alignment:** add the same response-aware raw-count objective,
   nuisance contract, priors, diagnostics, and transfer outputs used in science.
7. **Measured validation:** multi-standard calibration, real open-beam flatness,
   and held-out spectra; no bin/resonance exclusion based on fit improvement.
8. **Performance:** optimize surviving continuous operators only after parity
   gates; benchmark fixed-response fits and full calibration separately.
9. **Removal:** delete deprecated F2/F4/S2/S4 and dead duplicate code only after
   the migration window and usage audit pass.

### Cleanup-ledger dependency map

Every CL item must first receive the step-1 pre-change reproduction/rejection
anchor. The table assigns exactly one later step as its primary implementation
owner, so no cleanup finding disappears between audit and Phase 1.

| Owner step | Cleanup IDs | Why this step owns them |
|---|---|---|
| 2 — truthful boundaries | CL-01, CL-02, CL-13, CL-15, CL-16, CL-17, CL-22, CL-25, CL-36, CL-39, CL-40, CL-42, CL-43, CL-44, CL-46, CL-47, CL-48, CL-49, CL-50 | Documentation, validation, GUI/MCP routing, result contracts, and ignored-config rejection must be truthful before model work. |
| 3 — separate semantics | CL-07, CL-12, CL-14 | Fixed-flux research, paired-open observations, and production Jacobians need distinct types/objective names. |
| 4 — IC/grid correctness | CL-10, CL-18, CL-19, CL-26, CL-32, CL-33, CL-35, CL-37 | Resolution representation, working grids, provenance, family scope, and parity are shared prerequisites. |
| 5 — count response/background | CL-05, CL-06, CL-09, CL-11, CL-20 | The surviving count arms, Bdet, averaged-open approximation, and SAMMY placement must be defined together. |
| 6 — calibration alignment | CL-08, CL-21, CL-24, CL-27, CL-28, CL-29, CL-30, CL-34, CL-38, CL-45 | Objective/nuisance alignment, identifiability, transfer, diagnostics, starts, and persistence form one calibration contract. |
| 7 — measured validation | CL-41 | The GUI overlay must be checked against the exact modeled arms on the measured-validation fixtures. |
| 8 — performance | CL-31 | Caching/optimization begins only after numerical parity and diagnostics exist. |
| 9 — removal | CL-03, CL-04, CL-23 | F4/S4, F2/S2, and their D/F behavior leave only after warnings, migration, and usage gates pass. |
