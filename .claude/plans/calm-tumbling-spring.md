# Plan: TV-ADMM Spatial Regularization Demo Notebook

**Issue**: #390
**File**: `examples/notebooks/applications/02_tv_regularization_demo.ipynb`
**Branch**: `feat/tv-regularization-notebook`

## Context

TV-ADMM spatial regularization just merged (PR #387). VENUS operates at low
photon counts where independent per-pixel fitting produces noisy density maps.
This notebook demonstrates that `spatial_map_tv()` preserves spatial structure
that vanilla `spatial_map()` cannot — serving as both a user tutorial and a
source of publication-ready figures for the SoftwareX paper.

## Key Design Decisions

1. **Phantom**: 20x20 pixel grid with two non-overlapping circular discs
   (U-235 enriched + Pu-241 inclusion) on a low-density background.
   Circular boundaries test edge preservation in all orientations.
   Uses U-235 + Pu-241 for continuity with `01_spatial_mapping_demo` and
   `04_spatial_mapping_synthetic`.

2. **Multi-zone noise**: `generate_noisy_cube()` can't be used (tiles a
   uniform 1D spectrum). Must use `rng.poisson(I0 * transmission_true)` per
   pixel, matching `04_spatial_mapping_synthetic.ipynb` pattern.

3. **Three photon tiers**: I0 = 5000 (clean baseline), 500 (VENUS-relevant),
   100 (very noisy — dramatic TV advantage).

4. **Lambda sweep**: 8 log-spaced values `[0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]`.
   Lambda=0 recovers vanilla as control.

5. **No temperature fitting**: `temperature_k=293.6` (room temp, matching
   `04_spatial_mapping_synthetic`). Focus is spatial regularization, not
   temperature physics.

6. **Publication figures**: `SAVE_FIGURES = False` flag; all figures use
   consistent rcParams, labeled axes, colorbars.

## Phantom Geometry

```
20x20 grid, 2 isotopes (U-235, Pu-241)

| Zone            | n(U-235)  | n(Pu-241) |
|-----------------|-----------|-----------|
| Background      | 0.0005    | 0.0002    |
| Left disc (r=5) | 0.002     | 0.0002    |  ← U-235 enriched
| Right disc (r=4)| 0.0005    | 0.001     |  ← Pu-241 inclusion

Disc centers: left=(10,6), right=(10,14). Non-overlapping.
```

## Cell-by-Cell Structure

### Section 1: Setup (Cells 1-4)

| Cell | Type | Content |
|------|------|---------|
| 1 | Markdown | Title, summary table, TOC, prerequisites |
| 2 | Code | Imports, rcParams, `SAVE_FIGURES`, `SEED` constants |
| 3 | Markdown | "Phantom Construction" — rationale for disc design |
| 4 | Code | Build `true_u235`, `true_pu241` arrays with `disc_mask()` helper. **Fig 1**: side-by-side ground truth maps (Blues/Oranges) |

### Section 2: Forward Model + Noise (Cells 5-8)

| Cell | Type | Content |
|------|------|---------|
| 5 | Markdown | Physics setup, `load_endf`, energy grid |
| 6 | Code | Load U-235/Pu-241, compute forward model per unique composition (4 unique), assemble `transmission_true` cube. **Fig 2**: three energy frames |
| 7 | Markdown | Poisson noise model, three count rate tiers |
| 8 | Code | Generate `noisy_data = {I0: (trans, unc)}` for I0 in [5000, 500, 100]. **Fig 3** (paper): 3x3 grid — rows=I0, cols=energy frames |

### Section 3: Vanilla vs TV-ADMM (Cells 9-13) — Hero Section

| Cell | Type | Content |
|------|------|---------|
| 9 | Code | `density_metrics()` helper + zone masks + true densities dict |
| 10 | Markdown | "Visual Comparison at Medium Noise (I0=500)" |
| 11 | Code | Run `spatial_map()` and `spatial_map_tv(tv_lambda=0.01)` at I0=500. Print timing + ADMM diagnostics |
| 12 | Code | **Fig 4** (paper hero): 2 rows (U-235, Pu-241) x 4 cols (Truth, Vanilla, TV-ADMM, Bias). Matched colorbars |
| 13 | Code | Quantitative table: zone × isotope × method → bias%, std%, RMSE |

### Section 4: Lambda Selection (Cells 14-17)

| Cell | Type | Content |
|------|------|---------|
| 14 | Markdown | Bias-variance tradeoff explanation |
| 15 | Code | Lambda sweep loop at I0=500. Collect per-zone metrics |
| 16 | Code | **Fig 5** (paper): RMSE vs lambda plot (log x-axis), optimal lambda marked |
| 17 | Code | **Fig 6** (paper): Lambda gallery — U-235 density maps at λ = [0, 0.01, 0.1, 1.0] |

### Section 5: Photon Count Sweep (Cells 18-21)

| Cell | Type | Content |
|------|------|---------|
| 18 | Markdown | "How TV advantage scales with noise" |
| 19 | Code | Run vanilla + TV at all 3 I0 levels. Collect metrics |
| 20 | Code | **Fig 7** (paper): 3 rows (I0) x 3 cols (Truth, Vanilla, TV) for U-235 |
| 21 | Code | **Fig 8**: Bar chart — std_vanilla / std_tv ratio per I0. Shows TV advantage grows at low counts |

### Section 6: Edge Preservation (Cells 22-25)

| Cell | Type | Content |
|------|------|---------|
| 22 | Markdown | TV preserves edges vs Gaussian smoothing |
| 23 | Code | Extract horizontal profile through U-235 disc center (row 10). **Fig 9** (paper): line plot — truth (step), vanilla (noisy), TV-ADMM (sharp), Gaussian-filtered (blurred) |
| 24 | Code | **Fig 10**: 1x3 maps — Vanilla, Gaussian-filtered (σ=1.5), TV-ADMM |
| 25 | Code | Transition width metric table (10%-90% step width in pixels) |

### Section 7: ADMM Diagnostics (Cells 26-27)

| Cell | Type | Content |
|------|------|---------|
| 26 | Markdown | ADMM convergence criteria (Boyd et al. 2011) |
| 27 | Code | Run TV at 3 lambdas with tv_max_iter=20. Summary table: outer_iters, primal/dual residual, converged flag. Note: per-iteration traces not exposed in Python API |

### Section 8: Summary (Cells 28-30)

| Cell | Type | Content |
|------|------|---------|
| 28 | Markdown | Key findings table + parameter selection guide |
| 29 | Code | Recommended VENUS workflow code snippet |
| 30 | Markdown | References, next steps, links to other notebooks |

## Reusable Patterns from Existing Notebooks

- **`density_metrics()`** from `01_spatial_mapping_demo.ipynb` cell 17
- **Composition-indexed cube assembly** from `04_spatial_mapping_synthetic.ipynb` cell 7
- **Noise generation**: `rng.poisson(I0 * T) / I0` from both notebooks
- **Figure styling**: `imshow(..., interpolation='nearest')`, colorbar with `fraction=0.046`
- **rcParams**: `figsize=(12, 5)`, `font.size=12`

## API Calls Used

```python
nereids.load_endf(Z, A)
nereids.forward_model(energies, [(isotope, density)], temperature_k=293.6)
nereids.spatial_map(trans, unc, energies, isotopes, temperature_k=293.6, max_iter=50)
nereids.spatial_map_tv(trans, unc, energies, isotopes,
    tv_lambda=0.01, temperature_k=293.6, max_iter=50, tv_max_iter=10, tv_rho=1.0)
# SpatialResult: density_maps, uncertainty_maps, chi_squared_map, converged_map
#   ADMM-only: admm_outer_iterations, admm_primal_residual, admm_dual_residual, admm_converged
```

## Timing Budget (20x20 grid, release build)

| Section | Est. Time |
|---------|-----------|
| Forward model (4 compositions) | < 1s |
| Noise generation (3 tiers) | < 1s |
| Vanilla + TV at I0=500 | ~3s |
| Lambda sweep (8 values) | ~20-30s |
| Photon count sweep (3×2) | ~10-15s |
| ADMM diagnostics (3 lambdas) | ~5-10s |
| **Total** | **~40-60s** |

## Verification

1. `pixi run build` — ensure release bindings
2. Run notebook end-to-end: `pixi run jupyter` → open → Kernel → Restart & Run All
3. Verify all figures render correctly
4. Verify TV std < vanilla std at I0=500 and I0=100
5. Verify lambda=0 matches vanilla results
6. Verify edge profile shows TV preserves step, Gaussian blurs it
7. Update `examples/notebooks/README.md` to include the new notebook

## Files Modified

| File | Change |
|------|--------|
| `examples/notebooks/applications/02_tv_regularization_demo.ipynb` | **NEW** — full notebook |
| `examples/notebooks/README.md` | Add entry for the new notebook |
