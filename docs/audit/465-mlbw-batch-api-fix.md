# Audit — Issue #465: `cross_sections_on_grid` MLBW divergence

## Scope of the fix

The batch Reich-Moore dispatcher (`cross_sections_on_grid` →
`precompute_range_data` → `evaluate_precomputed_range`) silently treated
MLBW (LRF=2) resonance ranges as SLBW: it lumped both formalisms into a
single `PrecomputedRangeKind::Slbw` variant and evaluated them with
`slbw_evaluate_with_cached_jgroups` (incoherent per-resonance sum),
where MLBW requires a coherent sum
`Σ_r Γ_n / (E_r − E − i Γ_tot / 2)` yielding
`σ_elastic = (π/k²) · g_J · |1 − U_nn|²`.

On real ENDF-B-VIII.1 Hf isotopes (LRF=2) this produced a **per-bin
relative error of up to 55 %**.  Every one of the 20 826 grid points
(Hf-174/176/177/178/179/180 × 3471 energies) differed from the
per-point reference.  U-238 (Reich-Moore, LRF=3) used a separate batch
path and was unaffected — which is why the existing
`test_grid_matches_per_point_u238_full` did not catch the bug.

## What was corrected

1. A dedicated MLBW evaluator
   (`slbw::mlbw_evaluate_with_cached_jgroups`) performs the coherent
   sum using the same precomputed-J-group layout as SLBW, so the plan
   struct need not be duplicated.
2. The batch dispatch enum now has **distinct `Slbw` and `Mlbw`
   variants**; the `evaluate_precomputed_range` match is exhaustive,
   so the compiler will refuse to let a future change lump the two
   again.
3. The per-point entry `cross_sections_at_energy` now shares the batch
   precompute+evaluate pipeline (no second dispatch table).  The old
   `cross_sections_for_range` (~210 lines) and
   `mlbw_cross_sections_for_range` (~60 lines) are deleted.
4. `slbw_cross_sections_for_range` was already code-duplicated with
   `slbw_evaluate_with_cached_jgroups`; it now delegates instead of
   inlining the same math.

The module-level comment in `crates/nereids-physics/src/reich_moore.rs`
lists one evaluator per formalism.  Adding a new formalism adds one
`match` arm in `precompute_range_data` plus one in
`evaluate_precomputed_range` — there is no second place to keep in
sync.

## Regression gates

| Gate | Fails on #465 before fix | Passes after fix |
|------|---|---|
| `test_batch_matches_per_point_mlbw_synthetic` (smallest MLBW case that differentiates coherent vs incoherent sums) | YES | YES |
| `test_batch_matches_per_point_hf177_real_endf` (500 energies, real ENDF fixture) | YES (500/500 points off) | YES |
| `test_grid_matches_per_point_u238_full` (existing Reich-Moore test) | N/A — never exercised the bug | YES |
| `TestVenusMlbwRegression::test_mlbw_lm_fit_is_bit_exact` — real aggregated VENUS Hf 120 min workload + real Hf-177 ENDF | N/A — new gate | YES |

The VENUS gate is a Python pytest over a committed 42 KB aggregated
spectrum fixture.  It asserts bit-exact (`==`) equality against a
baseline captured on the fixed code.  If any future change shifts any
of (density, χ²_r, iteration count, convergence flag) on this real
workload the PR cannot land without an explicit rebaseline.

## Python callers of `precompute_cross_sections` — impact

All 5 callers live under `.research/spatial-regularization/scripts/`
(gitignored R&D prototypes, not shipped):

| Script | Isotope set | Persistent output | Impact |
|--------|-------------|-------------------|--------|
| 60_joint_solver_prototype.py   | U-238 only (LRF=3) | printed table | **Unaffected** — U-238 is Reich-Moore, correct in both paths. |
| 71_patchwise_solver.py         | Hf-6 (MLBW) | printed table | Printed values were wrong by up to 55 % on σ.  No persistent artifact.  Next run regenerates correct values. |
| 75_venus_with_resolution.py    | Hf-6 (MLBW) | printed table + **inline cached values lines 299–325** | Printed values were wrong.  Inline cached values are SUPERSEDED — annotated in-script below. |
| 81_corrected_fisher_baseline.py| Hf-6 (MLBW) | **`results/81_corrected_fisher_baseline.json`** | Persistent artifact was wrong.  On attempted re-run it ran 50 minutes without completing stage A1 (64 per-pixel KL baselines is inherently slow on this grid) and was killed.  The stale JSON is superseded; re-run the script to regenerate.  Since `.research/results/` is gitignored, this affects only local workspaces that had run the script previously. |
| 82_exact_fisher_verification.py| Hf-6 (MLBW) | printed comparison | Was the script's primary consistency check.  **Re-ran on the fixed code: Case 1 (no resolution) now matches between Python-precompute and Rust-exact Fisher at machine precision** (`max\|diff\|=1.00e-30` on model μ, Fisher ratios 1.0000 to 16 digits), up from the earlier divergence that motivated Case 1. |

### Action per-script

- **Script 82** already refreshed.  See `.research/spatial-regularization/scripts/82_exact_fisher_verification.py` output above.
- **Script 81**: stale JSON in `results/81_corrected_fisher_baseline.json` is a pre-#465 artifact.  Re-run the script to refresh.  Takes ~1 hour on the current `GRID=8` configuration; users can reduce `GRID` or run overnight.
- **Script 75**: inline cached values at lines 299–325 are stale.  The scripts have been annotated at the top of the module to document that.  Full re-run depends on private VENUS HDF5 (41 GB, not redistributable) and the gitignored PLEIADES resolution file.
- **Script 71**: no persistent artifact; next run regenerates.  Annotated at the top.
- **Script 60**: no change (isotope set is unaffected).

### Why no aggressive cleanup of the stale `.research/` results

`.research/` is gitignored per project policy — it is research work
product, not a tracked reference.  The stale JSON, the stale inline
cached values in 75, and any downstream research memos that referenced
them are known-stale and annotated.  The fix itself guarantees that
any future run of these scripts produces correct values.  Scrubbing
every stale number inline is not part of this PR; that is work for the
research workstream as each script is next exercised.

## Production impact (shipped code)

The per-point `cross_sections_at_energy` API (used internally by
`forward_model` and thus every `fit_spectrum_typed` / `spatial_map_typed`
call) dispatched correctly for MLBW even before the fix — the bug was
confined to the batch path.  As a consequence:

- All committed regression baselines (`.research/.../results/baseline/section_*.json`)
  remain bit-exact to prior-main on every fit output (A.1, A.2, A.3,
  A.4, B.2) — verified end-to-end on real VENUS Hf 120 min data.
- No Rust callers of the batch API existed in the production pipeline
  prior to the forward_model closure sites in `transmission.rs`, which
  continued using the per-point API throughout.  They are unchanged by
  this PR.

In short: no shipped fit output moved.  The fix corrects σ values
returned to any caller that used the batch API directly (notably the
Python `precompute_cross_sections` helper), and it eliminates the
structural precondition that produced the bug in the first place.
