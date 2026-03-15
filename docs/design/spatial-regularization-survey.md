# Spatially Regularized Resonance Fitting for Neutron Imaging at VENUS

**Issue:** #395 (parent epic: #394)
**Date:** 2026-03-14
**Status:** Research & Design — no implementation yet

## 1. Problem Statement

NEREIDS performs per-pixel isotopic density mapping by fitting a nonlinear
transmission model to hyperspectral neutron data:

```
T(E) = exp(-Σᵢ nᵢ · σᵢ(E))
```

where nᵢ are areal densities (atoms/barn) and σᵢ(E) are energy-resolved
resonance cross-sections.  At the VENUS beamline (SNS, ORNL), photon
counts per energy bin are often 2–10, making per-pixel fits noisy.

Spatial regularization should exploit the physical prior that neighboring
pixels likely have similar compositions, reducing noise while preserving
material boundaries — **with every density value accompanied by a
physically meaningful uncertainty estimate.**

### VENUS Instrument Parameters

| Parameter | Current | Future |
|-----------|---------|--------|
| Detector | Timepix3, 512×512 (514×514 with gaps) | Timepix4, 2048×2048 |
| Pixels | 262,144 | ~4,000,000 |
| Energy bins | ~1500 (default, user-configurable) | Same |
| I₀ per bin | 2–10 (typical VENUS) | Similar |
| Isotopes | 1–5 per fit | Same |
| Key isotopes | U-235, U-238, Pu-241 (σ up to 25,000 barns) | Same |

### Why TV-ADMM Failed

The previous implementation (epic #341, stripped 2026-03-14) used TV-ADMM
with a proximal penalty in the LM/Poisson fitting objective.  It failed
because of a **structural mismatch**, not a scaling bug:

1. **Jacobian amplification**: The data Hessian includes (dT/dnᵢ)² =
   (σᵢ·T)², where σᵢ reaches 25,000 barns.  The proximal Jacobian is 1.
   The squared ratio is 10⁶–10⁸ — no scalar rho can bridge this gap.

2. **ADMM couples what shouldn't be coupled**: A single rho controls both
   the proximal strength (n-update) and the soft-threshold (z-update).
   Scaling rho for the n-update destroys the z-update, and vice versa.

3. **Lesson**: Standard image-processing algorithms (TV, ADMM) assume
   data and penalty have comparable curvature.  In spectroscopic fitting,
   the per-bin Jacobian amplifies the data curvature by orders of
   magnitude.  The method must account for this.

### Core Challenge

**How to couple neighboring pixels' physics fits without being drowned
out by the per-pixel data fidelity, while providing uncertainty estimates
at scale (512×512 now, 2048×2048 future).**

### Domain-Specific Structure to Exploit

Resonance neutron imaging has structure that generic methods don't use:

1. **Precomputed cross-sections**: σᵢ(E) are computed once from ENDF data
   and shared across all pixels.  The Jacobian structure is known a priori.

2. **Separable exponent**: T = exp(-(n₁σ₁ + n₂σ₂ + ...)).  The log-
   transmission is linear in the densities: -ln T = Σᵢ nᵢσᵢ(E).

3. **Sparse sensitivity**: Resonances are narrow — most energy bins have
   near-zero sensitivity to a given isotope.  Only bins near resonance
   peaks contribute significantly to the Hessian.

4. **Shared energy grid**: All pixels use the same energy binning and
   cross-sections.  Per-pixel variation is ONLY in the densities nᵢ.

5. **Known noise model**: Poisson counting statistics with known flux
   (open beam measurement).

## 2. Requirements

| Requirement | Priority | Detail |
|-------------|----------|--------|
| Physics-correct densities | Must | From the physical forward model, not image processing |
| Uncertainty quantification | Must (1 year) | Every density accompanied by uncertainty estimate |
| Ultra-low counts | Must | I₀ = 2–10 (Poisson regime, many zero-count bins) |
| Large cross-sections | Must | U-238 resonances at 25,000 barns |
| 512×512 × 1500 bins | Must | Current detector, ~seconds for interactive use |
| 2048×2048 × 1500 bins | Should | Future detector, ~minutes acceptable |
| User-friendly parameters | Must | Regularization strength ~O(1), isotope/bin-independent |
| Edge preservation | Should | Material boundaries should not blur |
| Parallelizable | Must | rayon-compatible, per-pixel or per-block |
| Convergence guarantee | Should | Monotonic decrease of a well-defined objective |
| Synthetic test data | Must | All tests on synthetic phantoms + TIFF stack, no proprietary data |

## 3. Literature Survey

### 3.1. Separable Paraboloidal Surrogates (SPS)

**Source:** Fessler (1994), Erdogan & Fessler (1999), Tilley et al.
(2019, PMB 64:035001)

The closest analogue: spectral CT material decomposition with
Beer-Lambert model, multiple materials, Poisson noise, spatial penalty.

**Key idea:** Construct a separable quadratic surrogate that
upper-bounds the negative log-likelihood.  The surrogate's curvature
matches the data Hessian at the current estimate, so the spatial
penalty is automatically balanced.  Optimize the surrogate with
coordinate descent (embarrassingly parallel per voxel).

**Strengths:** Monotonic convergence, automatic Hessian balancing,
proven on our exact problem structure.

**Limitations:** Point estimates only — no native uncertainty.
Surrogate curvatures must be derived for the specific forward model.

### 3.2. Penalized WLS with Hessian Preconditioning

**Source:** Standard PET/MRI reconstruction literature.

**Key idea:** Add β·Hₚ·Σ_q(nₚ-n_q)² to per-pixel chi², where Hₚ is
the Hessian diagonal from the initial vanilla fit.

**Strengths:** Simple, same parallelism as current spatial_map.

**Limitations:** No edge preservation (quadratic only), no convergence
guarantee, no uncertainty, requires initial fit + Hessian extraction.

### 3.3. BEETROOTS — Bayesian Hierarchical with MCMC

**Source:** Palud et al. (2025), A&A 698:A311.

**Key idea:** Full Bayesian inference with L2 spatial prior and
auto-estimated hyperparameters via MCMC sampling.  Provides posterior
distributions (uncertainty) and adapts regularization to data.

**Strengths:** Proper uncertainty quantification, auto-hyperparameters,
validated on hyperspectral astrophysical data (10K pixels, 200K channels).

**Limitations:** Computationally prohibitive at 512×512 (only validated
up to 10K pixels).  MCMC is inherently sequential.

### 3.4. Spectral CT KL-Divergence with Gauss-Newton

**Source:** Barber et al. (2016), Long & Fessler (2014).

**Key idea:** Minimize KL divergence + spatial penalty jointly using
Gauss-Newton, where JᵀWJ naturally scales the penalty.

**Strengths:** Poisson-native, Hessian balancing via normal equations.

**Limitations:** Requires solving a coupled linear system across ALL
pixels simultaneously — system size = pixels × materials (131K unknowns
at 256×256, 524K at 512×512).  Does not scale.

### 3.5. Laplace Approximation for Uncertainty

**Source:** Standard Bayesian statistics; widely used in PET (Qi &
Leahy 2006).

**Key idea:** After finding the MAP estimate (via SPS or any
optimizer), approximate the posterior as Gaussian centered at the MAP
with covariance = inverse Hessian of the penalized objective.

**Strengths:** Cheap (one Hessian inversion after convergence), gives
per-pixel uncertainty, accounts for spatial regularization bias.

**Limitations:** Gaussian approximation may underestimate uncertainty
in the extreme Poisson regime (I₀=2).  Only valid near the mode.

## 4. Comparison Summary

| Method | Hessian | Scale | Edges | Poisson | Uncertainty | Convergence |
|--------|---------|-------|-------|---------|-------------|-------------|
| SPS | ✅ auto | ✅ 2K² | ✅ Huber | ✅ | ❌ point only | ✅ monotonic |
| H-PWLS | ✅ manual | ✅ 2K² | ❌ | ⚠️ | ❌ | ⚠️ |
| BEETROOTS | ✅ MCMC | ❌ 10K px | ❌ L2 | ✅ | ✅ full posterior | ⚠️ |
| GN-KL | ✅ GN | ❌ 256² | ✅ | ✅ | ❌ | ✅ |
| Laplace | N/A (post-hoc) | ✅ | N/A | ⚠️ approx | ✅ approx | N/A |

## 5. Proposed Approach: SPS-Laplace for VENUS

### Design Rationale

No single existing method satisfies all requirements.  We propose a
**domain-specific two-phase approach** that combines the strengths of
SPS (scalable, physics-correct point estimates with automatic Hessian
balancing) and Laplace approximation (uncertainty quantification):

### Phase 1: SPS Point Estimates

Adapt Separable Paraboloidal Surrogates to the neutron transmission
model, exploiting domain-specific structure:

**Objective function:**

```
Φ(n) = Σₚ Lₚ(nₚ) + β · Σ_{edges} ψ(nₚ - n_q)
```

where Lₚ is the per-pixel Poisson negative log-likelihood:

```
Lₚ(nₚ) = Σ_E [ Φ_E · T(nₚ, E) - Yₚ(E) · ln(Φ_E · T(nₚ, E)) ]
```

and ψ is a Huber penalty for edge preservation.

**Domain-specific surrogate construction:**

Because -ln T = Σᵢ nᵢσᵢ(E) is **linear in n**, the log-likelihood
has a known convexity structure we can exploit:

```
∂²Lₚ/∂nᵢ∂nⱼ = Σ_E Φ_E · T(nₚ,E) · σᵢ(E) · σⱼ(E)
             - Σ_E (Yₚ(E)/T(nₚ,E)) · [σᵢ(E)·σⱼ(E)·T(nₚ,E)
                                        - σᵢ(E)·T(nₚ,E)·σⱼ(E)·T(nₚ,E)]
```

The key simplification: σᵢ(E) are **precomputed constants**, so the
surrogate curvatures can be computed efficiently using the current
cross-section arrays.  No finite-difference Jacobians needed.

**Per-iteration cost:**

1. Compute surrogate curvatures from current densities + precomputed σ:
   O(n_pixels × n_isotopes × n_energies) — same as current spatial_map
2. One coordinate descent step per pixel per isotope: O(n_pixels × n_isotopes)
3. Update neighbor differences for penalty: O(n_edges × n_isotopes)

Total: dominated by step 1, which is the same cost as the current
per-pixel fitting.  Each SPS iteration is ~1 spatial_map pass.

**Parallelism:** Step 1 is embarrassingly parallel over pixels (rayon).
Step 2 is per-pixel with read-only neighbor access (rayon with
red-black coloring or Jacobi-style parallel updates).

### Phase 2: Laplace Uncertainty

After SPS converges to the MAP estimate n*, compute:

```
Cov(nₚ) ≈ [Hₚ(n*) + β · Σ_q wₚq · ψ''(nₚ* - n_q*)]⁻¹
```

where Hₚ(n*) is the per-pixel data Hessian at convergence.

**Domain-specific efficiency:**

- For Huber penalty: ψ''(t) = 1 for |t| < δ, 0 otherwise.  The
  regularization Hessian is sparse (nonzero only for non-edge neighbors).
- The combined Hessian is block-diagonal with spatial coupling — each
  pixel's uncertainty depends only on its own data Hessian plus
  contributions from immediate neighbors.
- For a 4-connected grid with K isotopes, each block is K×K.  With
  K ≤ 5, the per-pixel matrix inversion is trivial.
- **Approximation:** Ignore inter-pixel covariance (treat pixels as
  conditionally independent given neighbors at convergence).  This gives
  per-pixel uncertainty that accounts for the regularization bias but
  not spatial correlation.

**Output:** For each pixel and each isotope:
- Point estimate: nᵢ (from SPS)
- Standard uncertainty: σ(nᵢ) = √(Cov(nᵢ, nᵢ))
- Optional: inter-isotope correlation within the pixel

### User-Facing API

```rust
pub fn spatial_map_regularized(
    transmission: ArrayView3<f64>,
    uncertainty: ArrayView3<f64>,
    config: &FitConfig,
    reg_config: &RegularizationConfig,
    dead_pixels: Option<&Array2<bool>>,
    cancel: Option<&AtomicBool>,
    progress: Option<&AtomicUsize>,
) -> Result<RegularizedResult, PipelineError>
```

```rust
pub struct RegularizationConfig {
    /// Regularization strength (dimensionless, default 1.0).
    /// Controls the tradeoff between data fidelity and spatial smoothness.
    pub beta: f64,
    /// Edge threshold for Huber penalty (default: auto from data).
    /// Density differences below delta are smoothed quadratically;
    /// above delta are penalized linearly (preserving edges).
    pub delta: Option<f64>,
    /// Maximum outer iterations (default: 20).
    pub max_iter: usize,
    /// Compute per-pixel uncertainty estimates (default: true).
    pub compute_uncertainty: bool,
}

pub struct RegularizedResult {
    /// Same as SpatialResult fields
    pub density_maps: Vec<Array2<f64>>,
    pub uncertainty_maps: Vec<Array2<f64>>,  // from Laplace, not LM
    pub chi_squared_map: Array2<f64>,
    pub converged_map: Array2<bool>,
    /// Regularization-specific
    pub objective_history: Vec<f64>,  // per-iteration objective value
    pub n_outer_iterations: usize,
}
```

**User parameters:**
- `beta = 1.0`: regularization strength.  Dimensionless because the SPS
  surrogate automatically scales the data term.  β=0 recovers vanilla.
  β=1 gives equal weight to data and spatial prior per surrogate step.
- `delta = auto`: edge threshold, auto-estimated from the initial fit
  density variance (median absolute difference between neighbors).

### Scalability Analysis

| Detector | Pixels | Bins | Isotopes | Per-iter cost | Est. 20 iters |
|----------|--------|------|----------|--------------|---------------|
| Timepix3 | 512² = 262K | 1500 | 2 | ~1 spatial_map | ~20× vanilla |
| Timepix4 | 2048² = 4.2M | 1500 | 2 | ~1 spatial_map | ~20× vanilla |

With current spatial_map running 262K pixels in <1s (release, rayon),
20 SPS iterations ≈ 20s.  Acceptable for interactive use.

For 4.2M pixels: 20 × ~15s = ~5 min.  Acceptable for batch processing.

## 6. Implementation Roadmap

### Phase 0: Python Prototype (validate concept)

- Implement SPS outer loop in Python using NEREIDS Python API
- Use `fit_spectrum` for surrogate curvature computation
- Validate on synthetic phantom: U-238, 512×512, I₀ = 2, 5, 10
- Measure: bias, std reduction, edge preservation, runtime
- Compare against vanilla spatial_map

### Phase 1: Rust Core

- Surrogate curvature computation using precomputed cross-sections
- Huber penalty with auto-delta
- Coordinate descent with rayon parallelism
- Laplace uncertainty (diagonal approximation)
- Comprehensive tests at realistic dimensions

### Phase 2: Integration

- Python bindings (`spatial_map_regularized`)
- GUI integration
- Demo notebook with publication-quality figures
- SoftwareX paper material

## 7. Open Questions

1. **Surrogate tightness**: How tight is the paraboloidal surrogate for
   the neutron transmission model?  Tighter surrogates → faster
   convergence.  May need numerical experiments.

2. **Laplace accuracy at I₀=2**: Is the Gaussian approximation adequate
   at extreme low counts?  May need comparison against bootstrap or
   profile likelihood for validation.

3. **Inter-pixel uncertainty**: The diagonal Laplace approximation
   ignores spatial covariance.  Is this sufficient for the imaging
   community, or do they need full covariance / credible regions?

4. **Auto-delta estimation**: What's the best estimator for the Huber
   threshold?  MAD of neighbor differences from the initial fit is a
   starting point.

5. **Ordered subsets**: For 2048×2048, can we accelerate SPS with
   ordered subsets (process a fraction of energy bins per iteration)?
   This is standard in PET (OS-SPS, Erdogan & Fessler 1999).

## 8. Key References

1. Erdogan, H. & Fessler, J.A. (1999). "Ordered subsets algorithms for
   transmission tomography." *Phys Med Biol*, 44:2835–2851.

2. Fessler, J.A. (1994). "Penalized weighted least-squares image
   reconstruction for positron emission tomography." *IEEE TMI*,
   13(2):290–300.

3. Tilley, S., Zbijewski, W. & Stayman, J.W. (2019). "Model-based
   material decomposition with a penalized nonlinear least-squares CT
   reconstruction algorithm." *Phys Med Biol*, 64:035001.

4. Palud, P. et al. (2025). "BEETROOTS: Spatially regularized Bayesian
   inference of physical parameter maps." *A&A*, 698:A311.

5. Barber, R.F. et al. (2016). "An algorithm for constrained one-step
   inversion of spectral CT data." *Phys Med Biol*, 61:3784–3818.

6. De Pierro, A.R. (1995). "A modified expectation maximization
   algorithm for penalized likelihood estimation in emission
   tomography." *IEEE TMI*, 14:132–137.

7. Qi, J. & Leahy, R.M. (2006). "Iterative reconstruction techniques
   in emission computed tomography." *Phys Med Biol*, 51:R541–R578.

8. Fessler, J.A. (2000). "Statistical Image Reconstruction Methods for
   Transmission Tomography." Chapter in *Handbook of Medical Imaging,
   Vol. 2*, SPIE Press.
