# ADR 0002: Two-Tier Doppler Evaluation with Per-Isotope Route Disclosure

## Status

Accepted

## Context

The Free Gas Model Doppler broadener convolves the thermal kernel with a cross-section sampled on the caller's energy grid.
That is exact for the piecewise-linear table it is given, but a resonance narrower than the grid spacing loses area before the kernel ever sees it, so the broadened value at an energy depends on which other energies were requested.
On the real VENUS Hf-177 spectrum (3471 time-of-flight bins over 7–200 eV) the sampled route misses up to a factor of three at individual energies against a grid-independent evaluation, and the converged density shifts at the 0.1 % level.
SAMMY's own `Dopfgm` integrates over an auto-refined sampled grid, so agreement with SAMMY oracles is bounded by SAMMY's grid error.

The whitebox pipeline map (record R5·7, clause 3) declares Doppler evaluation two-tier and imposes two normative obligations the code did not meet: the route must be disclosed per isotope, and a File-3 background term must route an isotope to the sampled table.

## Decision

Doppler evaluation is two-tier, decided per isotope for the whole working grid, never mixed within one isotope.

- Tier 1, continuous: a resolved SLBW or MLBW source with `√E > 8u` (`u = √(k_B·T/A)`) whose thermal support window `[(√E−8u)², (√E+8u)²]` lies inside its resolved range, with no other evaluable range overlapping the window and no File-3 term, integrates the kernel over the resonance equation at error-controlled quadrature (`continuous_doppler`: Gauss–Kronrod G10/K21 panels split at resonance breakpoints and at the knots of an energy-dependent scattering radius AP(E), refined by global bisection, value and temperature derivative converged together, hard limits as errors).
  Both tiers apply SAMMY's rule to a negative broadened value (`fgm/mfgm4.f90` lines 83-101, `Dopfgm`): a value above −1e-15 barn is set to zero; otherwise it is set to zero when no contributing unbroadened point — a table node inside the kernel window on tier 2, a quadrature node on tier 1 — is positive, and kept when one is, where SAMMY prints "Negative cross section".
  An SLBW total that goes negative where two same-J interference terms outweigh the shared potential term is therefore kept negative next to its sign change and zeroed where the source is negative throughout the window; the count of kept negative values per isotope is disclosed as a `warnings` line.
- Tier 2, sampled table: every other case samples the equation on the working grid and convolves it (`doppler`); a caller-supplied zero-kelvin table is also tier 2 and is never re-derived from the source.
- Gate temperature: a fixed-temperature evaluation gates at that temperature; a free-temperature fit gates once at the fit's upper bound (`TEMPERATURE_FIT_UPPER_BOUND_K`, 5000 K).
  Every temperature-dependent tier-1 condition (fold-through-zero, window-in-range, overlapping-range) is monotone in the kernel width, and the formalism and File-3 conditions do not depend on temperature, so tier 1 at the bound is tier 1 at every temperature the optimizer can visit and the route cannot change between probes.
  A plan remembers a fingerprint of the data grid, of the working grid the routes were decided on and of the source, and refuses any other input, so a plan can never be applied to a same-length shifted grid, a different instrument or a same-count different source.
  The energy-scale model's working grid moves with every probe, so it decides the route per probe and refuses a change of route kind between probes as a hard error; the refusals are counted and disclosed as a `warnings` line, because each one is a rejected optimizer step at the resolved-range edge.
  Two continuous routes are the same kind whatever their formalism, so a probe whose corrected grid crosses an SLBW/MLBW boundary stays tier 1 and is not refused.
- The File-3 predicate `ResonanceRange::has_file3_background` exists before any parser can answer `true`; the gate consults it and a test forces it, so the change that adds MF=3 support cannot let File-3 data take the continuous route silently.

### Tier boundary

| Reason on the sampled-table route | Rendered as |
|---|---|
| Caller-supplied table | `sampled-table kernel-on-grid (caller-supplied zero-kelvin table)` |
| Formalism not resolved SLBW/MLBW | `sampled-table kernel-on-grid (Reich-Moore formalism)` |
| `√E ≤ 8u` | `sampled-table kernel-on-grid (√E ≤ 8u at 3.10e-3 eV (u = 1.20e-2 √eV))` |
| Window crosses the range boundary | `sampled-table kernel-on-grid (thermal window [9.81e3, 1.02e4] eV leaves the resolved range at 1.00e4 eV)` |
| Grid leaves the resolved range (the grid, or its auxiliary extension, jumps past an edge no window crossed) | `sampled-table kernel-on-grid (grid energy 2.55e2 eV lies outside the resolved MLBW range [1.00e-5, 2.50e2] eV)` |
| Another evaluable range overlaps the window | `sampled-table kernel-on-grid (resonance range 1 overlaps the thermal window at 5.00e0 eV)` |
| File-3 background term | `sampled-table kernel-on-grid (File-3 background term present at 7.00e0 eV)` |

### Disclosure

Every result that broadened something carries the route per isotope: `SpectrumFitResult.doppler_routes`, `SpatialResult.doppler_routes`, the Python `FitResult.doppler_routes` and `SpatialResult.doppler_routes`, the MCP result summaries, and the GUI fit feedback, convergence summary and provenance log.
`nereids.doppler_routes()` answers the same question for the same arguments as `forward_model()` without broadening, on the same working grid, so what is disclosed is what is executed by construction.
A `warnings` line is added only when a resolved SLBW/MLBW isotope is demoted for a window or grid-range reason, naming the reason and the gate temperature; a Reich-Moore isotope on the sampled table is the documented route and does not warn, because every Reich-Moore fit would otherwise warn and the field would become noise.
A caller-supplied zero-kelvin table is disclosed as such on every path, including a free-temperature spatial map, which builds no plan for it.

## Consequences

- Reich-Moore results are bit-identical to the previous pipeline, pinned with and without the auxiliary grid; the samtry suite is untouched.
- SLBW/MLBW numbers change wherever the grid under-resolved a line; the VENUS Hf-177 regression anchors were regenerated on this implementation, with the sampled route on 1×/4×/16×/64× the VENUS density converging monotonically onto the continuous value as the argument.
- Tier 1 costs about five times the sampled table for one broadening of Hf-177 on the VENUS grid; free-temperature fits pay it per probe, mitigated by deciding the route once per fit, converging a tier-1 isotope's temperature derivative on the same panels as its value so the Jacobian does no second integration (a sampled-table isotope's derivative stays on demand in the Jacobian, so a mixed fit pays no second convolution per trial point), sharing one plan across spatial pixels, and hoisting the Doppler + Beer–Lambert product out of the resolution-calibration objective when the energy scale is pinned.
- A spatial energy-scale map discloses the routes its pixels executed on their corrected grids (the first converged pixel's, when every converged pixel agrees by kind), and the GUI redraws a result through a plan gated the way the fit gated, so what is displayed is what was fitted.
- The change that adds MF=3 support must make `has_file3_background` observe the parsed data and update the paired test.
- The `ex001` fixture's mass ratio was corrected from 10.0 to 10.0 amu in neutron masses; the SAMMY ex001 curve is reproduced to 0.77 % over all 315 points through tier 1.
  The residual is confined to the Doppler core: ours/SAMMY is 1.0076 at the peak but 1 to 5e-6 at both far wings, and the E·σ area ratio is 1.0076, which is the signature of a 1.5 meV line sampled on a grid before convolution (area lost in the core, none in the smooth tails), as SAMMY's own `Dopfgm` does.
