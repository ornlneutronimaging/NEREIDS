# Improving VENUS IC instrument-resolution calibration

## Executive decision

Do **not** add more freely fitted IC parameters yet.

The current result has three distinct layers:

1. **The archived numerical result is reproducible.** On the synchronized
   v0.3.0 source, the supplied 1D counts and cached IC/JENDL-5 parameters
   reproduce 1058.697 ± 7.039 K and deviance/dof 28.6325.
2. **The calibration is practically underdetermined.** `a1` is at its lower
   bound; its local Jacobian column is 3,572× smaller than `a0` in the archived
   parameterization, and two JENDL-5 solutions with 48.9×-different `a1`
   produce only 0.149 RMS-σ prediction differences. The raw condition number
   13,206.8 is reported only with its coordinates and scaling; column
   normalization changes it to 8.65, so it is not an invariant quality score.
3. **The code has real defects and a consequential response approximation, but
   none is yet validated as the archive residual's cause.** The model broadens
   transmission directly instead of applying a true-energy-conditioned
   response separately to latent open and sample count spectra. Missing
   fine-grid convolution, non-unit-L timing, tau-cap accuracy, and an
   R-threshold discontinuity are also validated defects. Existing dense-grid
   A/Bs freeze the cached IC parameters and therefore do not eliminate bias in
   the calibration stage.

The achieved outcome is therefore the **unresolved-cause arm**: a
hypothesis-by-evidence [elimination ledger](elimination-ledger.md) plus a
discriminating next experiment. The right strategy is to fix count formation
and numerical correctness, reduce the parameterization to what the data
identify, and calibrate a shared instrument response on multiple non-Ta
standards with Ta held out and sample transport modeled per run.

## What the prior archive actually establishes

The [cold audit and full 71-member inventory](archive-inventory.md) finds the
package internally coherent but causally overconfident.

- Cached FTS/ENDF-B-VIII.1: 992.1 K, deviance/dof 50.0.
- Cached IC/ENDF-B-VIII.1: 1049.4 K, 44.8 (10.4% lower deviance).
- Cached FTS/JENDL-5: 1017.1 K, 37.4 (25.2% lower).
- Cached IC/JENDL-5: 1060.5 K, 29.7 (40.6% lower than the reference).
- Current-source replay of the notebook cache: 1058.697 K, 28.6325.
- Persistent features are residual-definition sensitive. Archive tables label
  roughly +69.8σ near 39.15 eV, -47.9σ near 13.92 eV, and -25.0σ near 23.95
  eV. The current reconstructed transmission-ratio diagnostic instead gives a
  +62.09σ maximum near 39.10 eV, -9.96σ at 13.92 eV, and -24.59σ at 23.95
  eV. This discrepancy is disclosed rather than treating cached residual
  definitions as interchangeable.

Thus JENDL-5 and IC both help, and their combination helps most. That does not
show that the UDR is the unique root cause: the exact UDR file, raw cubes,
NeXus timing data, and transport response are absent. Several archive claims
are sensitivity analyses mislabeled as causal decompositions; the 14 eV
contaminant scan was never run; and the map arrays needed to quantify the
column-256 detector-panel step were not supplied.

Independent new Ta-181 transmission/capture measurements using multiple sample
thicknesses report that JENDL-5 performs best among three major libraries over
their 0.15–100 keV measurement and recommend reevaluating ENDF/B
([Nuclear Science and Engineering](https://www.ans.org/pubs/journals/nse/article-55452/)).
That energy interval does not overlap this 8–45 eV fit, so the paper supports
JENDL-5 only as an external candidate, not as a low-eV oracle. The archive
itself favors JENDL-5 but is resolution-confounded; changing the library moves
IC parameters by 1.5× to 91×.

## Current implementation, in one page

The detailed trace is in the timestamped [code audit](code-audit.md).

The nuclear/transmission part of the forward chain is ordered correctly:

```text
resonance parameters → Reich–Moore cross sections → Doppler broadening
→ summed Beer–Lambert transmission
```

Calibration constructs Gaussian, corrected UDR, or IC candidates. IC uses

```text
alpha(E) = a0 sqrt(E) + a1
beta(E)  = beta
R(E)     = R
response = IC moderator pulse ⊗ inherited 350 ns PSR triangle
```

and bounded Nelder–Mead profiles a normalization plus optional additive
constant/linear baseline. Each candidate synthesizes a reference table, then
uses the same tabulated broadener as UDR. The returned object has convergence,
iterations, reduced chi-square, and bound hits, but no evaluation count,
Jacobian, covariance, parameter correlation, or profile likelihood.

The base IC pulse equation, units, normalization, and tail orientation pass
targeted tests. Recent commits corrected a time-mirrored correlation and width
interpolation; the changelog correctly invalidates older IC/UDR calibrations.
However, the complete measurement model is not a count-response convolution.
It computes `R[T]`, while a physical response matrix conditioned on true energy
requires

```text
open_j   = sum_i R[j,i] Phi_i
sample_j = sum_i R[j,i] Phi_i T_i
ratio_j  = sample_j / open_j
```

and generally `ratio != R[T]` when incident flux varies over a kernel. The
current count objective also profiles one independent same-bin flux per
measured bin, which cannot represent migration between true and measured bins.
The broadener chooses its energy-dependent kernel at the output bin and
renormalizes each gather row; the physical probability is instead conditioned
on the true neutron bin. This is the response convention used in the direct
resolution treatment of [Žugec et al.](https://arxiv.org/abs/1710.07443), and
transmission imaging likewise forms the transmitted and incident spectra
separately before taking their ratio
([Scientific Reports](https://www.nature.com/articles/s41598-020-71705-4)).
A synthetic probability-matrix control gives exact
agreement for flat flux but a 0.2106 transmission difference for structured
flux. This mechanism is validated; its magnitude on the archived data remains
unmeasured.

## Ranked changes

Every item below names the evidence and a falsifiable acceptance test.

### P0 — Correct the numerical model before recalibration

#### 0. Form open and sample counts through a true-energy response matrix

The present `R[T]` approximation is exact only for locally flat incident flux
and a compatible response normalization. Build a bin-integrated matrix
`R[j,i] = P(measured bin j | true bin i)` and evaluate open and sample arms
separately. The latent incident spectrum must be constrained by the open beam,
monitor data, or a regularized physical basis; it cannot be profiled as one
independent flux value per measured bin after migration. Apply background and
detector-efficiency terms to the count arms in their physical locations.

Acceptance: response columns conserve probability to the declared edge-loss
tolerance; a delta-function true-bin oracle lands at the predicted measured
TOF; flat-flux cases reproduce the normalized `R[T]` control; structured-flux
synthetics recover the injected response and transmission without bias. On
real data, compare current and matrix models at fixed parameters, then
recalibrate both IC and sample parameters and score the frozen response on the
held-out hot spectrum. The [count-response probe](evidence/count-response.md)
must remain a regression test.

#### 1. Replace the uniform tau cap and remove the R threshold discontinuity

Current support sizing couples a very slow storage tail to prompt-core
sampling. Inside accepted calibration bounds, the cap may leave only eight
prompt samples; an independent Gamma(3) oracle shows −26.226% variance error.
Crossing `R=1e-9` infinitesimally changes 1,004 grid points to 16 and makes
the objective discontinuous. See [code probes](evidence/code-probes.md).
The documented full box is not even feasible: the actual corner
`a0=5, a1=2, beta=.02, R=1, PSR=.35 us` fails synthesis at 24.178 eV because
the cap forces 0.0977 us steps above the 0.0967 us resolution floor. The test
named `ic_box_worst_corner_synthesizes_within_tau_cap` uses `a0=.5`, not the
declared maximum 5.0. The fitted PSR lower bound 0.05 us is likewise infeasible
at the default beta/R start (the feasible threshold is about 0.0586 us).

Implement a nonuniform/composite quadrature:

- allocate prompt-core nodes from alpha and tail nodes from beta separately;
- merge them on an ascending nonuniform offset grid, which the broadener
  already supports;
- choose tail extent from an integrated omitted-mass tolerance weighted by R,
  not a hard R threshold;
- preserve requested PSR/burst folds under an explicit error budget;
- use one synthesis policy for calibration and downstream science.
- derive a numerically feasible optimization domain from these error budgets
  instead of accepting a rectangular box with rejected interior/corner points.

Acceptance: across the revised feasible calibration domain, area error <1e-8, mean/variance
relative error <1e-3, no model jump larger than 1e-6 transmission under a
one-ULP parameter perturbation, and no silently accepted under-resolved
candidate. The archive-domain 400/500/600/1200 A/B must converge to
<0.01 RMS observational σ and <0.1σ maximum.

#### 2. Use a resonance-aware source grid for every resolution family

`build_aux_grid` currently densifies only Gaussian resolution. A synthetic
off-bin 0.005 eV dip disappears on the coarse IC/UDR input and changes the
convolved result by 0.047622. This is a correctness defect. The current
[archive dense-grid A/B](evidence/dense-grid.md) slightly worsens a sample-stage
WLS fit, but it freezes cached `a0`, `a1`, `beta`, and `R`; it therefore says
nothing about calibration-stage grid bias and does not eliminate this mechanism
as a cause of poor transfer.

Generalize the existing resonance/boundary grid construction to Tabulated and
IC: evaluate the unbroadened spectrum on a grid dense enough over the exact
kernel support, apply the resolution there, then sample to observed bins.

Acceptance: doubling the auxiliary density changes held-out transmission by
<1e-6 absolute, <0.01 RMS σ, and <0.1σ maximum. Add the off-bin probe as a
regression test, then recalibrate all IC parameters on the RT counts with both
grids and compare frozen-response hot predictions under the same likelihood.
Do not use fit improvement to choose grid density.

#### 3. Make `L_scale` and resolution timing physically consistent

The corrected energy uses `L_eff=L_nom*L_scale`, while the broadener converts
energy to TOF with the resolution object's fixed nominal L. The current
algebra scales every physical delay by `L_scale`. The defect is tiny in this
cache (maximum 0.078σ because scale=1.0008377) but grows linearly.

Pass the effective flight path into plan construction or formulate broadening
directly on the corrected physical TOF axis. Require a base UDR's stored L to
match calibration configuration.

Acceptance: injected `L_scale` values 0.98–1.02 recover the same physical
timing width to 1e-6 relative and yield finite-difference derivatives against
an independently transformed oracle.

#### 4. Make kernel origin, range, and validity explicit

IC is mode-centered; live SAMMY UDR code centroid-centers; raw NEREIDS UDR
loading preserves supplied offsets. Position is pinned by default, so this
convention matters. Separately, IC freezes at endpoint kernels outside its
synthesis range, and Rust/GUI UDR constructors accept invalid flight paths.

Add explicit `Mode | Centroid | Absolute` origin metadata, preserve it through
load/calibrate/export, and require the caller to choose any conversion. Reject
invalid L, all-zero kernels, and out-of-range IC use (or analytically extend it).

Acceptance: real-file round trips preserve origin and first moments; mode and
centroid A/B produce predicted t0 shifts; invalid inputs fail; no silent
endpoint freeze occurs.

### P1 — Make calibration identifiable and statistically consistent

#### 5. Reduce the eV IC parameterization before adding components

The archive's `a1=0.001000175` is effectively the lower bound. The local
Jacobian column norm for `log a1` is 0.1066 versus 380.7 for `log a0`, and
the least-informed singular vector is 0.9999999 `log a1`.

For 8–45 eV, start with `alpha(E)=a sqrt(E)` and no `a1`. The same conclusion
holds when the local probe retains all 4,280 bins selected on the archive's raw
4–120 eV axis: the `log a1` column remains about 4,235× smaller than `log a0`,
and its least-informed
direction remains effectively pure `a1`. The experimental KENS eV-region pulse
measurement found pulse width proportional to approximately E^-0.48,
consistent with a prompt rate close to sqrt(E)
([Kiyanagi et al.](https://doi.org/10.1080/18811248.2005.9726389)).
That measurement used a KENS H2O moderator, not VENUS's poisoned/decoupled H2
moderator, so it is a candidate prior rather than a VENUS law.

Only restore an intercept or alternate law if held-out likelihood and finite
profiles demand it. The official Mantid IC implementation uses
`alpha=1/(Alpha0+lambda Alpha1)` and warns that both alpha coordinates are
effectively 100% correlated over one peak
([Mantid documentation](https://docs.mantidproject.org/nightly/fitting/fitfunctions/IkedaCarpenterPV.html)).
That inverse-wavelength law is not generally `sqrt(E)` when `Alpha0` is
nonzero, and the correlation warning concerns a single peak; both are model
candidates, not VENUS validation.

Acceptance: every free coordinate has a finite 95% profile interval and stable
predictive estimates across starts and datasets. Report the Jacobian coordinate
definition, finite-difference step, raw singular values, column norms, and a
predeclared column scaling; never use a scale-dependent condition-number cutoff
as a substitute for profiles. The reduced model must not degrade held-out
per-bin Poisson deviance.

#### 6. Treat UDR as a comparator or weak prior, not the initializer/truth

Use physics- and data-derived starts:

- prompt width versus energy seeds `a`;
- measured tail area and time constant seed R/beta;
- SNS PSR width, L, and t0 come from metrology priors;
- transport or independently measured pulse kernels supply optional weak
  priors on remaining response components.

Run space-filling multi-starts in transformed coordinates and cluster solutions
by prediction, not parameter distance. Then compute one-dimensional profile
likelihoods by fixing each coordinate and reoptimizing all others. Profile
likelihood is designed to expose nonlinear practical non-identifiability and
guide new measurements
([Raue et al.](https://academic.oup.com/bioinformatics/article/25/15/1923/213246)).

Acceptance: all starts reach the same held-out prediction basin; profiles cross
predeclared confidence thresholds on both sides. If not, freeze the unsupported
coordinate or acquire a discriminating standard.

#### 7. Use the same response-aware raw-count likelihood in calibration and science

Current calibration fits Gaussian transmission errors with an additive
index-linear background; science uses raw sample/open-beam counts with Poisson
KL and a multiplicative log-energy baseline. Resolution parameters can absorb
that mismatch.

Add a response-matrix calibration entry point that predicts open and sample
count arms separately as described in P0.0, sharing the production flux,
background, open-beam uncertainty, and baseline model. Return latent-flux and
other nuisance estimates with covariance. Keep instrument parameters shared
across runs while normalization/background and sample transport remain per run.
Simply passing broadened `T_i` into the existing same-bin joint-Poisson
objective is not sufficient.

Acceptance: simulated counts with known response recover unbiased parameters;
Gaussian and counts objectives agree in the high-count limit; low-count tests
show correct coverage; real calibration and science residual definitions are
identical.

#### 8. Make uncertainty and transfer first-class outputs

Return objective evaluations, component timings, finite-difference/autodiff
Jacobian, singular values, covariance/correlation, profile intervals, and
bootstrap/replicate variability. Propagate calibrant density, temperature,
t0/L metrology, and nuclear resonance covariance into sample T.

Do model selection on held-out likelihood, not minimum training reduced
chi-square across families with different parameter counts.

Acceptance: nominal 68%/95% intervals achieve coverage in simulation; held-out
predictions contain replicate spectra at their nominal rate; the reported
sample-T uncertainty includes instrument and nuclear terms rather than scaling
only the local fit covariance.

### P2 — Upgrade the physics with independent information

#### 9. Separate moderator IC from the rest of the instrument response

The original Ikeda–Carpenter function models moderator emission
([Ikeda & Carpenter](https://doi.org/10.1016/0168-9002(85)90033-6)); it is not
the complete VENUS response. General TOF work treats resolution as a
site/setup-specific probability distribution and recommends Monte Carlo for the
actual spectrometer/detector
([Brusegan, Noguere & Gunsing](https://doi.org/10.1080/00223131.2002.10875192)).

Separate a shared instrument operator from per-run sample transport. A useful
starting decomposition is:

```text
shared instrument: moderator emission
                 ⊗ inherited PSR pulse (350 ns setting, acquisition tie unverified)
                 ⊗ surveyed source/path and panel-specific detector timing
                 ⊗ acquisition-bin integration
per-run physics:  incident flux → sample-specific geometry/transport → transmission
measured counts:  instrument operator applied to open and sample spectra separately
```

The direct IC struct already has an unused Gaussian burst field; calibration
hard-disables it and has no detector/path/bin coordinates. The 350 ns value is
documented by a code comment referring to the missing FTS header, not by an
input preserved in the archive, so treat it as an inherited setting until tied
to acquisition metadata. Multiple scattering, self-shielding corrections, and
finite sample geometry are material/thickness dependent and must not be folded
into a supposedly transferable convolution. Add components only with
independent measurements or tight priors. Never let all widths float on one Ta
spectrum.

Acceptance: shared moderator/PSR terms transfer across runs, surveyed/path and
detector terms transfer only over their declared geometry/panels, and each
sample's transport operator changes as predicted with material and thickness.
Removing a component must produce a predeclared residual signature; adding it
must improve held-out likelihood.

#### 10. Revisit R(E) and the meaning of the long tail

Mantid's standard parameterization uses
`R=exp[-81.799/(kappa lambda^2)]`. With
`lambda=0.285993/sqrt(E[eV])`, this implies, by substitution,
`R≈exp(-1000 E/kappa)`; conventional thermal-scale kappa makes R negligible
in the eV window. The current fit instead uses constant R≈0.15.

That constant may be a useful phenomenological tail, but it should not be
called moderator storage without independent evidence. Mantid's law is a
powder-peak model, not a VENUS moderator measurement, so all of these remain
candidate priors. Test:

- R=0 in the eV region;
- Mantid's candidate energy-dependent R law;
- a separately named geometry/detector/reflection tail.

Keep beta fixed or tightly prior-constrained unless multiple energies contain
tail information.

Acceptance: a tail component must predict held-out asymmetry across energy and
thickness with stable parameters. If only training fit improves, reject it as
nuisance absorption.

#### 11. Calibrate on non-Ta data and hold Ta out

Using Ta-181 nuclear data to calibrate the response and then infer Ta
temperature is intrinsically confounded. VENUS commissioning reports a
decoupled/poisoned H2 moderator, a ~25 m path, MCP Timepix detectors, an
existing 180 µm Ta measurement, and plans for copper, steel, silver, and cobalt
calibration samples
([VENUS commissioning report](https://link.springer.com/chapter/10.1007/978-3-032-15003-5_39)).

Run multiple known thin standards and at least two thicknesses, avoiding black
lines. Kiyanagi's neutron-resonance absorption method shows that eV moderator
pulses can be measured by fitting resonance-convolved capture timing. If a
capture detector is unavailable, use multiple transmission standards and
transport-informed priors.

Train the shared instrument response on non-Ta runs; keep Ta and one other
standard held out. Multiple thicknesses discriminate response from transport
only when each run has its own material/geometry-dependent transport operator;
self-shielding and multiple scattering cannot be absorbed into one frozen
instrument convolution.

Acceptance: one frozen response predicts held-out nuclides/thicknesses and the
Ta RT/hot spectra, with finite profiles and no panel-dependent response shift.

#### 12. Preserve transport and response provenance

Resolution can also be represented as a normalized, energy-dependent
probability matrix; direct inversion work stresses bin integration, banded
storage, uncertainty amplification, and multiple scattering
([Žugec et al.](https://arxiv.org/abs/1710.07443)).

Archive raw events/cubes, NeXus timing/metrology, open/black runs, detector
configuration, exact UDR/transport files, checksums, code revision, synthesis
settings, objective value, bound hits, profiles, and fitted response table.

Acceptance: a clean environment rebuilds region counts, calibration, maps, and
all report figures bit-for-bit where deterministic, or within declared numeric
tolerances.

### P3 — Optimize the measured bottleneck

The user's analytical-speed intuition is only partly applicable. IC pulse
generation is analytical; application to an arbitrary spectrum under nonlinear
TOF↔energy mapping is still numerical, and current IC becomes tabulated
immediately.

Release medians on 2,312 bins are:

| Component | Time |
|---|---:|
| synthesize 48×400 IC | 0.00753 s |
| synthesize 64×500 IC | 0.01429 s |
| forward model, no resolution | 0.01170 s |
| forward with reused IC table | 0.10019 s |
| fresh 48×400 IC + forward | 0.10911 s |

So synthesis is only ~7% of a candidate evaluation. The long interpolated IC
kernels contain about 2,100–4,600 points in the fit window. Optimize in this
order:

1. **Build compact error-controlled sparse rows.** Use nonuniform/adaptive
   quadrature and avoid allocating/merging two long vectors per target. Generate
   a true-energy-conditioned CSR response directly on the correct auxiliary
   grid; this attacks the measured 0.100 s application cost.
2. **Reuse the fixed plan for maps.** The pipeline has plan support; benchmark
   that the Python-converted IC table reaches it, measure NNZ per row against
   the real UDR, and batch sparse application across pixels.
3. **Cache fixed physics.** When position, calibrant T, and density are pinned,
   cache unbroadened transmission. This removes the measured ~0.012 s fixed
   nuclear/Doppler portion, useful but not dominant.
4. **Measure dimension/evaluation alternatives.** Removing unidentifiable
   `a1`, exposing `n_evals`, caching repeated simplex points, and testing a
   gradient/trust-region method after making support continuous are plausible
   savings, not measured ones. Mantid documents an analytical Jacobian for its
   isolated IC/pseudo-Voigt peak; it does not provide derivatives for the full
   NEREIDS nuclear, Doppler, Beer–Lambert, response-matrix chain.
5. **Parallelize synthesis only afterward.** Sixty-four reference kernels are
   serial, but their measured construction cost is secondary.
6. **Stream notebook data.** The IC notebook retains three
   5304×512×512 stacks—about 16.7 GiB at float32 or 33.4 GiB at float64—while
   the UDR notebook releases them. This is a separate memory/cache issue.

Optimization acceptance is two-dimensional:

- parity: max transmission difference ≤1e-6 and ≤0.1σ, RMS ≤0.01σ, fitted T
  shift ≤0.1 reported statistical σ, deviance change ≤0.1%;
- performance: median of at least 20 warmed release runs on fixed hardware.
  Treat ≥5× calibration and ≥3× fixed-response map speedups as engineering
  targets until an implementation benchmark demonstrates them.

No approximation may be accepted merely because the fit improves or a
residual disappears.

## Recommended calibration workflow

1. **Independent axis calibration:** determine t0/L from prompt timing,
   surveyed geometry, Bragg edges, or monitor data; carry Gaussian metrology
   priors.
2. **Per-line empirical diagnostics:** on non-black standards, estimate
   position, width, skew, and tail moments without forcing a global IC law.
3. **Reduced shared model:** fit `alpha=a sqrt(E)` first, fixed PSR, and
   measured/prior response components. Introduce R/beta only when line moments
   demand them.
4. **Joint multi-run count-response fit:** shared instrument parameters;
   separate open/sample response application; per-run latent flux, background,
   and sample transport; multiple nuclides/thicknesses; resonance covariance.
5. **Identifiability gate:** Jacobian SVD plus profile likelihood. Freeze or
   remove any coordinate without a finite profile.
6. **Held-out gate:** freeze the response and predict held-out non-Ta and Ta
   spectra. Report all predeclared residual features, not just aggregate
   deviance.
7. **Science propagation:** transfer the complete response posterior or
   covariance into temperature/density inference.

This replaces “UDR seed → one Ta RT fit → freeze” with independent timing,
physics-derived initialization, shared multi-standard inference, and a real
transfer test.

## Implementation and validation sequence

| Gate | Deliverable | Pass condition |
|---|---|---|
| A | True-energy count response, tau/R continuity, L-scale, all-family auxiliary grid, validation/range tests | Analytic/probe tolerances above; current relevant suites remain green |
| B | Reduced IC coordinates, counts calibration, `n_evals`/Jacobian/profiles | Simulated coverage; finite profiles; no hidden bound/ridge |
| C | Component provenance and multi-standard dataset | Raw/checksummed inputs; non-black lines; two thicknesses; Ta held out |
| D | IC-law/component comparison | Frozen held-out Poisson likelihood and predeclared residual improvement |
| E | Sparse/adaptive/gradient performance | Numerical parity plus benchmark targets |

Do not compare physical model families until Gate A passes. Do not interpret a
parameter physically until Gate B passes. Do not publish a transferable VENUS
response until Gate D passes.

## Data exclusions

No bins or resonances were newly excluded, masked, smoothed, reweighted, or
downweighted to improve any result during this investigation. The archive has
two inherited nested selections, both fixed before this audit:

1. The 5,304-bin cache spans 4.525 eV–2.2788 MeV, while the notebooks first
   select 4–120 eV (4,280 bins).
2. Their fit objective then activates only 8–45 eV (2,312 bins).

Their independent procedural justification here is exact reproduction, not fit
improvement. The archive contains no instrument record physically justifying
the exact 45 or 120 eV boundaries. With the same fixed 4–120 eV IC table:

- 8–45 eV: 1058.697 K, density 0.000632509, deviance/dof 28.6325;
- all 4,280 raw-selected 4–120 eV bins: 1051.672 K, density 0.000622891,
  deviance/dof 36.1516; seven corrected bins exceed 120 eV and use the current
  endpoint clamp;
- all 5,304 cache bins: 1038.948 K, density 0.000613725, deviance/dof 56.8114,
  but this is disclosure-only because 1,031 corrected bins use the endpoint
  clamp and one sparse Doppler edge passes through unbroadened.

A physically attempted full-domain IC table fails synthesis at 401 keV under
the 8,192-point cap, so the last numbers are not a valid full-domain IC result.
They show that the outer exclusion matters and identify the precise blocker.
The inner comparison shifts T -7.026 K and density -1.521%; the endpoint-clamped
all-cache diagnostic shifts them -19.749 K and -2.970%.

The identifiability conclusion was repeated after removing only the inner
8–45 exclusion: on all 4,280 raw-selected bins, `a1` remains least-informed and
the cache-versus-robust predictions differ by 0.121 RMS sigma. This report
treats both windows as reproduction scope, not evidence that excluded bins are
artifacts. A physically valid all-cache comparison requires the P0 adaptive
response and explicit high-energy nuclear/transport validity domains. See the
full [with/without and blocker record](evidence/fit-window.md).

## Final research conclusion

The current IC pulse function is not simply “wrong”: its core equation,
normalization, units, and tail orientation are well tested. The complete
measurement model is not yet physically sufficient. The real problems are a
combination of:

- a direct-transmission broadening approximation where open and sample count
  spectra should be formed through a true-energy response separately;
- validated numerical/cross-layer defects that must be corrected;
- a demonstrably underidentified four-parameter calibration;
- a moderator-only model being asked to absorb detector, geometry, baseline,
  nuclear-data, and sample-transport discrepancies, without separating shared
  instrument response from run-specific transport;
- a same-nuclide calibration/thermometry design without held-out transfer;
- a numerical tabulated application whose cost dominates analytical synthesis.

JENDL-5 is the strongest archived Ta candidate, and IC is a better fit than the
archived UDR, but neither fact identifies the residual's physical cause. The
independent JENDL-5 result is at non-overlapping higher energies and cannot
validate this 8–45 eV choice.
The most informative next result will come from an independent, multi-standard,
multi-thickness VENUS response measurement after the P0 correctness fixes—not
from another round of unconstrained IC fitting.
