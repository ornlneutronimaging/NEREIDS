# Phase 0 path audit

## Verdict

The existing implementation is not one fitting pipeline with interchangeable
front ends. It contains three production objectives, one ratio fallback, a
separate calibration objective, a research-only fixed-flux/Fisher model, and
surface-specific domain/default conversions. Some paths are sound foundations,
some are explicit approximations, some violate their advertised contract, and
some expose functionality production immediately rejects.

The canonical capability table is
[`phase-0-support-matrix.md`](phase-0-support-matrix.md), exact run evidence is
[`evidence/phase-0/test-results.md`](evidence/phase-0/test-results.md), and every
confirmed stale/partial implementation is tracked in
[`phase-0-cleanup-ledger.md`](phase-0-cleanup-ledger.md).

## Audit method

For each public route, Phase 0 independently traced:

```text
surface input → typed data variant → requested/effective solver
→ model + resolution/background wrappers → actual objective → result semantics
```

Evidence was then layered as:

1. file/line dispatch trace;
2. existing route-specific unit/integration tests;
3. matched deterministic and stochastic synthetic controls;
4. committed or supplied real-data checks where provenance permits;
5. explicit blockers where real UDR/calibration/raw science inputs are absent.

No production code was modified, no residual-bearing data were masked or
reweighted to make a route pass, and regression-anchor stability was not
mistaken for physical correctness.

## Canonical Rust route audit

| ID | Public data and requested solver | Actual objective and code trace | Resolution/background/nuisance stack | Direct synthetic/contract evidence | Available real evidence and limitation | Status |
|---|---|---|---|---|---|---|
| F1 | `InputData::Transmission`; `Auto`→LM or explicit LM | supplied-σ WSSR optimized by LM (`pipeline.rs:1122-1133,1241-1249,1393-1549`) | Beer–Lambert → None/Gaussian/Tabulated/native-IC `R[T]` → SAMMY background → multiplicative baseline; fit range/energy scale supported | Exact matched Rust recovery and Python IC-as-tabulated recovery: [E14](evidence/phase-0/test-results.md#narrow-direct-r13-route-reruns), [E11](evidence/phase-0/test-results.md#ordinary-fit-ic-handoff) | Exact Hf+Gaussian selector passes, but χ²/ν≈219657 means stability, not model adequacy: [E14](evidence/phase-0/test-results.md#narrow-direct-r13-route-reruns), [E02](evidence/phase-0/test-results.md#committed-real-venus-single-spectrum-regression-routes) | native; keep+improve |
| F2 | `InputData::Transmission`; explicit `PoissonKL` and Python `kl`/`poisson`/`joint_poisson` aliases | single-arm Poisson NLL on fractional T; supplied σ is absent from optimization (`pipeline.rs:1251-1258,1551-1667`; `nereids-fitting/src/poisson.rs:1-22,101-157`) | Same configured `R[T]` and transmission backgrounds as F1; fit range is rejected | Exact matched Rust recovery passes, while the direct control accepts negative T: [E14](evidence/phase-0/test-results.md#narrow-direct-r13-route-reruns), [E04](evidence/phase-0/test-results.md#real-data-route-semantics-probe) | Real Hf is used only as a semantic invariance control: σ reweighting changes LM but leaves F2 bit-identical; there is no route-specific real validity gate ([E04](evidence/phase-0/test-results.md#real-data-route-semantics-probe)) | legacy-invalid; deprecate |
| F3 | `InputData::Counts`; `Auto`→PoissonKL or explicit KL alias | profiled paired-count conditional-binomial deviance with explicit `c` (`pipeline.rs:1260-1284,1695-1913`; `nereids-fitting/src/joint_poisson.rs:1-41`) | Same-bin Beer–Lambert `R[T]`; SAMMY Anorm+ABC and multiplicative baseline; BackD/F and nonzero detector B rejected | Exact `c=5.98` recovery, stochastic ensemble, and IC-as-tabulated recovery: [E14](evidence/phase-0/test-results.md#narrow-direct-r13-route-reruns), [E07](evidence/phase-0/test-results.md#matched-model-stochastic-counts-ensemble), [E11](evidence/phase-0/test-results.md#ordinary-fit-ic-handoff) | Exact Hf+Gaussian selector and archived Ta+IC/JENDL/baseline replay pass; neither validates the same-bin response model: [E14](evidence/phase-0/test-results.md#narrow-direct-r13-route-reruns), [E03](evidence/phase-0/test-results.md#archived-real-ta-icjendl-counts-replay) | primary native counts likelihood, response incomplete; keep+improve |
| F4 | `InputData::Counts`; explicit LM | converts `T=S/O`, `σ≈√max(S,1)/O`, then runs F1 LM; `c`, OB variance, and attached alpha-fit counts config are ignored (`pipeline.rs:1302-1320,1359-1389,2600-2613`) | Inherits the configured F1 resolution/background stack only after the lossy ratio conversion; a detector-background array changes the input to rejected F6, while alpha flags alone are silently dropped | Exact matched conversion passes; a 50-replicate low-count cell has +52.8% bias; direct `fit_alpha_1=True` returns convergence with no alpha fields: [E14](evidence/phase-0/test-results.md#narrow-direct-r13-route-reruns), [E07](evidence/phase-0/test-results.md#matched-model-stochastic-counts-ensemble), [E10](evidence/phase-0/test-results.md#remaining-public-route-cells) | Real Hf with `c=5.982` hits density 0/nonconvergence; manual OB prescaling converges ([E04](evidence/phase-0/test-results.md#real-data-route-semantics-probe)) | defective approximation with ignored config; deprecate |
| F5 | `InputData::CountsWithNuisance`; `Auto`→PoissonKL or explicit KL | caller `flux` is used as the Poisson open arm in the F3 objective (`pipeline.rs:1286-1300,1704-1736,1894-1900`) | Configured `R[T]`; only zero detector B succeeds; fitted alpha and BackD/F are rejected; SAMMY ABC/baseline otherwise follow F3 | Exact public-Python paired zero-B recovery plus exact nonzero-B/alpha rejection messages: [E10](evidence/phase-0/test-results.md#remaining-public-route-cells) | No dedicated real F5 gate is available; no sibling F3 real pass is credited | half-wired; split/complete before exposure |
| F6 | `InputData::CountsWithNuisance`; explicit LM | no objective is defined; dispatch returns `InvalidParameter` before optimization (`pipeline.rs:1322-1327`) | No resolution/background stack is entered | Exact public-Python rejection: [E10](evidence/phase-0/test-results.md#remaining-public-route-cells) | Unsupported by design; therefore no real-data fit exists or is claimed | explicit unsupported combination; keep rejection |

`Auto` is domain-dependent, not an objective: counts select F3 and transmission
selects F1 (`pipeline.rs:1122-1133`). `PoissonKL` is also not one engine: it
selects F2 or F3 based on the input variant.

The Rust builder can also attach `CountsBackgroundConfig` to any input
(`pipeline.rs:369-395,739-742`). F1/F2/S1/S2 silently ignore it; F4/S4 core
silently ignores its alpha flags. These are explicit ignored-config cells
M47/M57-M60, not supported background behavior. They are source-verified with
a direct Python F4 omission in E10; missing Rust rejection tests remain
recorded gaps.

## Spatial route audit

| ID | Public data and requested solver | Actual per-pixel objective and code trace | Resolution/background/nuisance stack | Direct synthetic/contract evidence | Available real evidence and limitation | Status |
|---|---|---|---|---|---|---|
| S1 | `InputData3D::Transmission`; `Auto`→LM or explicit LM | constructs each pixel's transmission+σ input and calls the supplied-σ LM objective (`spatial.rs:1818-1834`; `pipeline.rs:1241-1249`) | Shared cross-section/resolution plan; configured None/Gaussian/Tabulated/native-IC in Rust; SAMMY background and global/per-pixel multiplicative baseline | Exact public-Python 3×3 matched recovery: [E14](evidence/phase-0/test-results.md#narrow-direct-r13-route-reruns) | No measured spatial fit-truth gate; the real F1 spectrum is not credited as spatial validation | keep+improve |
| S2 | `InputData3D::Transmission`; explicit KL aliases | constructs each pixel's transmission+σ input and calls the single-arm fractional-T NLL (`spatial.rs:1818-1834`; `pipeline.rs:1251-1258`); fit range is rejected up front (`spatial.rs:330-346`) | Shared configured `R[T]`/transmission background stack; inherits F2's σ-blind objective | Exact matched public-Python recovery, alias parity, and accepted negative-T control: [E10](evidence/phase-0/test-results.md#remaining-public-route-cells) | No real spatial fit-truth or route-validity gate | legacy-invalid; deprecate with F2 |
| S3 | `InputData3D::Counts`; `Auto`→PoissonKL or explicit KL | replaces each local OB by the live-pixel spatial mean and constructs per-pixel F5-style input for the joint-Poisson deviance (`spatial.rs:1649-1704,1773-1805`) | Configured `R[T]`; zero detector B inserted; SAMMY ABC/baseline limits follow F3; optional global baseline stage | Exact 4:1 gradient control gives converged false structure while paired S5 recovers truth: [E05](evidence/phase-0/test-results.md#spatial-averaged-flux-semantic-control) | Real OB totals have 4.55% nonzero-pixel CV versus 1.06% shot CV, but there is no real density truth ([E12](evidence/phase-0/test-results.md#real-venus-spatial-open-beam-diagnostic)) | explicit averaged-OB approximation; keep+improve |
| S4 | `InputData3D::Counts`; explicit LM | retains local sample/OB, constructs per-pixel raw counts, then uses the F4 ratio/LM objective (`spatial.rs:1773-1805`; `pipeline.rs:1302-1320`); per-pixel energy scale is rejected | Rust core inherits F1 wrappers and ignores `c`, OB variance, and alpha flags; the Python spatial binding rejects alpha flags on ordinary counts instead of reaching that core state | Exact Rust execution/result-semantics, public c=2 nonconvergence, and Python alpha-flag rejection: [E14](evidence/phase-0/test-results.md#narrow-direct-r13-route-reruns), [E10](evidence/phase-0/test-results.md#remaining-public-route-cells) | No real spatial fit-truth gate; F4's single-spectrum real failure is mechanism context, not an S4 pass | defective approximation with surface-specific validation; deprecate with F4 |
| S5 | `InputData3D::CountsWithNuisance`; `Auto`→PoissonKL or explicit KL | retains caller sample/flux/background per pixel and invokes the F5 joint-Poisson objective; pixel errors are counted and swallowed into the returned map (`spatial.rs:1807-1839,1954-2042`; `pipeline.rs:1286-1300`) | Configured `R[T]`; zero detector B succeeds; nonzero B fails inside each pixel, while fitted alpha/BackD/F remain unavailable; global/per-pixel baseline otherwise available | Exact paired 4:1 gradient recovery to 1.56e-10 relative error, plus a nonzero-B 1x1 call that returns `n_failed=1`, nonconvergence, and NaN rather than a boundary error: [E05](evidence/phase-0/test-results.md#spatial-averaged-flux-semantic-control), [E10](evidence/phase-0/test-results.md#remaining-public-route-cells) | No real S5 fit truth or working nonzero-background gate; an accepted failure container is not credited as route success | half-wired paired-open route with swallowed pixel errors; complete before exposure |
| S6 | `InputData3D::CountsWithNuisance`; explicit LM | no per-pixel objective starts; spatial preflight rejects the combination (`spatial.rs:916-932`) | No resolution/background stack is entered | Exact public-Python preflight rejection: [E10](evidence/phase-0/test-results.md#remaining-public-route-cells) | Unsupported by design; therefore no real-data fit exists or is claimed | explicit unsupported combination; keep rejection |

The real open-beam check streamed
`tests/data/pleiades_data/venus_hf_open_beam.h5` without modifying it. Its
central 90% pixel-total range was 0.925–1.068 of the median, with systematic
half/quadrant differences. Beam profile and detector efficiency are not
separable from this file alone; both are effects that per-pixel open-beam
pairing retains and S3 discards.

## Resolution-calibration route audit

All calibration families use one separate normalized-transmission objective:
bounded Nelder–Mead outside, analytic profiling of `a·model` or
`a·model+b0+b1·normalized_bin_index` inside (`resolution_calib.rs:637-682,
799-1198`). They do not use LM, F2, or F3, and fitted nuisance coefficients are
not returned.

| ID | Public data and solver | Actual objective and code trace | Resolution/background/nuisance stack | Direct synthetic/contract evidence | Available real evidence and limitation | Status |
|---|---|---|---|---|---|---|
| C1 | `calibrate_resolution(Gaussian, transmission, uncertainty, ...)`; no F1–F6 solver selection | bounded Nelder–Mead minimizes normalized-transmission WSSR after analytic nuisance profiling (`resolution_calib.rs:637-682,799-1198`) | Gaussian Δt,ΔL; exponential tail fixed to zero; profiles normalization and optionally constant+linear-in-bin background; optional t0/L priors | Exact matched Gaussian recovery passed in 6.22 s: [E14](evidence/phase-0/test-results.md#narrow-direct-r13-route-reruns), [E09](evidence/phase-0/test-results.md#resolution-calibration-routes) | No real calibrant, count objective, covariance/transfer artifact, or full Gaussian-tail calibration surface | calibration-only; keep+improve |
| C2 | `calibrate_resolution(UdrCorr, transmission, uncertainty, ...)`; no F1–F6 solver selection | same profiled WSSR/Nelder–Mead calibration objective (`resolution_calib.rs:591-596,637-682,799-1198`) | base tabulated UDR with fitted width scale s0 and energy exponent p; same calibration-only nuisance and optional position priors | Exact scale/exponent recovery and independent-raw-kernel tests pass: [E14](evidence/phase-0/test-results.md#narrow-direct-r13-route-reruns), [E09](evidence/phase-0/test-results.md#resolution-calibration-routes) | No real UDR fixture/provenance; base/config flight paths are not cross-checked | calibration-only; conditional keep+improve |
| C3 | `calibrate_resolution(IkedaCarpenter, transmission, uncertainty, ...)`; no F1–F6 solver selection | same profiled WSSR/Nelder–Mead calibration objective (`resolution_calib.rs:598-632,637-682,799-1198`) | native IC sqrt-E alpha a0/a1, beta, constant R and optional PSR; burst Gaussian disabled; same calibration-only nuisance/position priors | Exact selected optimizer/convergence command passed in 115.81 s; fast bounds/validation commands also pass: [E09](evidence/phase-0/test-results.md#resolution-calibration-routes) | No fresh full recovery/PSR closed loop, real calibration, count-response objective, uncertainty/profile output, or persistence | calibration-only, experimental; keep+improve |

Position priors exist in calibration, but production cannot pin the returned
t0/L as the same constrained object and uses different L bounds. Calibration
returns no Jacobian, covariance, correlation, profiles, restart ensemble,
component timing, or `n_evals`; one small IC case taking nearly two minutes
confirms the performance issue independently.

## Background/nuisance audit

Four distinct concepts are present:

1. SAMMY apparent-transmission normalization/additive ABC(+DF), partially
   supported by F3/S3/F5/S5;
2. NEREIDS bounded multiplicative log-energy baseline, outermost across all
   executable production fits;
3. intended detector-space `α1 ΦT + α2 Bdet`, implemented only in a fixed-flux
   research helper; production variously rejects, ignores, or returns a failed
   spatial container and never completes this likelihood;
4. calibration-only profiled normalization+index-linear additive nuisance.

The research Fisher helper additionally uses its own legacy two-term background
and different flux convention. It is not the information geometry of F3 even
though Python stubs describe it as sharing production construction. The full
capability matrix and legal combinations are recorded in
`phase-0-support-matrix.md`.

## Public-surface audit

### Python

The typed single/spatial APIs reach all F/S routes, but IC fits require
`.as_tabulated()`. Python exposes detector-background/alpha controls that the
production solver rejects, and stubs/result comments retain obsolete counts
background layouts. Binding tests verify the current rejection behavior rather
than the advertised feature.

### MCP

Single raw-count manifests default to hidden transmission-domain LM; spatial
counts use Auto→F3. A direct injected probe also showed that `fit_domain="countz"`
silently selects transmission. Allowlists silently drop several valid fit
controls. Full MCP tests did not execute because FastMCP is unavailable; this
route is source- and helper-probe-verified, not server-verified.

### GUI

The default solver is KL. With both sample and OB, KL selects F3 and LM uses
normalized F1. However, sample-only raw HDF5 is explicitly copied into a
`Transmission` object with σ=1, so default KL can apply F2 to raw count
magnitudes (`guided/normalize.rs:525-617`, `state.rs:1615`,
`guided/analyze.rs:2422-2440`). This route must be blocked. GUI normalization
and counts likelihood also use separate exposure-ratio state, and spatial
overlays use local OB even when S3 fitted against averaged OB. GUI behavior is
source-verified; no interactive or completed GUI-suite result is claimed.

Neither GUI nor MCP exposes resolution calibration or an IC construction/
persistence workflow.

## Real-data coverage and blockers

Verified real evidence:

- aggregated VENUS Hf transmission+Gaussian+LM stability anchor;
- same Hf raw counts+Gaussian+joint-Poisson stability anchor;
- archived Ta raw counts+IC-as-tabulated+JENDL-5+baseline replay;
- raw VENUS HDF5 open-beam spatial-uniformity diagnostic.

Not available/verified:

- a reproducible real resolution-calibration fixture and full settings;
- the real VENUS UDR file;
- raw data/provenance needed to rebuild the archived Ta maps and calibration;
- real spatial fit truth or held-out calibrant transfer;
- interactive GUI and live MCP server behavior.

These gaps are blockers to physical validation, not reasons to substitute a
synthetic result or declare a path correct.
