# Phase 0 — path audit and disposition gate

## Purpose

Phase 0 precedes all IC response, calibration, background, and performance
changes. Its job is to establish what NEREIDS actually exposes, what each route
computes, which routes have independent evidence of correctness, and which
routes should survive. No production path is removed or repaired merely because
it is awkward or produces a poor fit; the audit must first reproduce its
behavior and identify the mechanism.

## Complete route inventory

The audit treats these as distinct routes rather than assuming sibling parity:

| ID | Data/API route | Solver request | Actual engine to verify |
|---|---|---|---|
| F1 | single transmission | `auto` / `lm` | weighted LM |
| F2 | single transmission | `kl` and aliases | legacy single-arm Poisson NLL |
| F3 | single raw counts | `auto` / `kl` and aliases | joint-Poisson conditional-binomial deviance |
| F4 | single raw counts | `lm` | counts-to-transmission conversion, then weighted LM |
| F5 | single counts with nuisance spectra | `auto` / `kl` | joint-Poisson compatibility route |
| F6 | single counts with nuisance spectra | `lm` | declared rejection |
| S1–S6 | spatial equivalents of F1–F6 | same | spatial precompute/global-baseline/per-pixel dispatch |
| C1 | resolution calibration | n/a | Gaussian family + profiled weighted least squares |
| C2 | resolution calibration | n/a | UDR width-correction family + profiled weighted least squares |
| C3 | resolution calibration | n/a | IC family + profiled weighted least squares |

For every route, the matrix separately records:

- Rust, Python, MCP, and GUI reachability;
- `None`, Gaussian, tabulated UDR, and IC resolution reachability (including
  Python's IC-to-tabulated conversion);
- no background, SAMMY `Anorm+ABC`, SAMMY `ABC+DF`, NEREIDS multiplicative
  baseline, permitted combined background/baseline, detector-space background,
  and calibration-only profiled nuisance;
- density, temperature, energy-scale, fit-window, grouping, and spatial-global
  baseline support;
- native, fallback, research-only, rejected, or calibration-only status.

## Evidence ladder

Each route is evaluated independently at the strongest available level:

1. **Static dispatch trace:** public API → input type → solver resolution →
   model wrappers → objective → result fields, with file/line evidence.
2. **Contract/rejection test:** aliases, unsupported combinations, validation,
   and result-field semantics are exercised rather than inferred.
3. **Deterministic matched-model synthetic:** injected parameters must be
   recovered within a predeclared numerical/statistical tolerance.
4. **Stochastic synthetic:** exposure scaling, low-count bias, uncertainty
   coverage, nuisance degeneracy, and high-count LM/Poisson agreement are
   measured where the route claims statistical inference.
5. **Real data:** the committed aggregated VENUS Hf spectrum and the archived
   Ta counts cache are used where their provenance supports the route. A real
   spectrum repeated into a small spatial cube is labeled as a dispatch test,
   not evidence of spatial-model validity. Missing UDR/raw calibration inputs
   remain an explicit blocker rather than being replaced silently.
6. **Cross-route invariants:** `auto` must equal its documented explicit
   solver; Rust/Python routes must agree; shared models must produce the same
   prediction before objective-specific weighting; rejected routes must fail
   before starting an optimizer.

Passing a regression anchor proves stability, not physical correctness.
Physical/statistical recommendations require an oracle, a matched-data recovery
test, or an independently supported mechanism.

## Deliverables

- `phase-0-path-audit.md`: one row per route with exact commands and results.
- `phase-0-support-matrix.md`: domain × solver × background × resolution matrix.
- `phase-0-cleanup-ledger.md`: stale, duplicate, misleading, research-only,
  exposed-but-rejected, and partially wired work.
- `phase-0-disposition.md`: exactly one disposition and acceptance gate per
  matrix cell.
- preserved scripts and outputs under `investigation/evidence/phase-0/`.

## Disposition rules

| Disposition | Required evidence |
|---|---|
| `keep` | coherent objective/data contract, reachable API, adequate independent tests, no material mismatch |
| `keep+improve` | scientifically defensible route with a bounded implementation, coverage, diagnostic, or performance gap |
| `complete before exposure` | intended route is defensible, but the public contract exceeds working production behavior |
| `deprecate` | compatibility use may remain temporarily, but the statistical/physical contract is misleading or a maintained route supersedes it |
| `remove` | unreachable/duplicate implementation with no distinct supported contract and migration evidence recorded |

Authorship is never evidence for a disposition. Every recommendation names the
compatibility impact, migration path, and a falsifiable acceptance test.

## Gate into Phase 1

Phase 1 begins only after every route has a disposition. Its implementation
sequence is then dependency-ordered:

1. shared IC/kernel numerical correctness used by surviving routes;
2. domain-correct response formation for the surviving counts route(s);
3. calibration/science objective alignment;
4. background placement and nuisance identifiability per surviving route;
5. diagnostics, uncertainty propagation, and performance optimization;
6. deprecation/removal only after migration tests and documentation exist.
