//! # nereids-python
//!
//! PyO3 Python bindings for the NEREIDS neutron resonance imaging library.
//!
//! Provides a Pythonic API for:
//! - Computing theoretical transmission spectra (`forward_model`)
//! - Spatial mapping across imaging data (`spatial_map_typed`)
//! - Trace-detectability analysis (`trace_detectability`)
//! - Energy calibration (`calibrate_energy`)
//!
//! ## Typed Input Data API
//!
//! Use `from_counts()`, `from_counts_with_nuisance()`, or `from_transmission()`
//! to create typed input data,
//! then pass to `spatial_map_typed()` for per-pixel fitting:
//!
//! - **Counts** → Poisson KL (statistically optimal for raw detector counts)
//! - **Transmission** → LM only; a count likelihood is not valid for ratios
//!
//! ## Usage
//! ```python
//! import nereids
//! import numpy as np
//!
//! # Load ENDF data for U-238
//! isotope = nereids.load_endf(92, 238)
//!
//! # Compute transmission spectrum
//! energies = np.linspace(1.0, 30.0, 1000)
//! transmission = nereids.forward_model(energies, [(isotope, 0.001)], temperature_k=293.6)
//!
//! # Spatial mapping with typed API
//! data = nereids.from_transmission(transmission_3d, uncertainty_3d)
//! result = nereids.spatial_map_typed(data, energies, [isotope])
//! ```

use std::sync::Arc;

use numpy::{
    PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;

use nereids_core::elements;
use nereids_core::types::{Isotope, IsotopeGroup};
use nereids_endf::parser::parse_endf_file2;
use nereids_endf::resonance::{
    LGroup, Resonance, ResonanceData, ResonanceFormalism, ResonanceRange,
};
use nereids_endf::retrieval::{EndfLibrary, EndfRetriever, mat_number};
use nereids_fitting::resolution_calib::{
    CalibrationConfig, DEFAULT_PSR_FWHM_NS, NS_TO_US, PSR_FWHM_PIN_CEILING_US, ResolutionFamily,
    UDR_S0_MAX, UDR_S0_MIN, calibrate_resolution as rust_calibrate_resolution,
};
use nereids_io::normalization::{self as norm, NormalizationParams};
use nereids_io::tof::BeamlineParams;
use nereids_physics::counts_response;
use nereids_physics::doppler::{self, DopplerParams};
use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid,
};
use nereids_physics::resolution::{
    self, ResolutionFunction, ResolutionParams, TabulatedResolution,
};
use nereids_physics::transmission::{self, InstrumentParams, SampleParams};
use nereids_pipeline::detectability;

// PyO3 needs a literal here to expose the real default through
// `inspect.signature`; keep that public literal locked to the core default.
const _: () = assert!(DEFAULT_PSR_FWHM_NS == 350.0);

/// Python wrapper for ENDF resonance data.
///
/// Uses `Arc` internally so that `.clone()` in `py.detach()` closures is O(1)
/// (refcount bump) instead of deep-copying the entire resonance dataset.
#[pyclass(name = "ResonanceData", from_py_object)]
#[derive(Clone)]
struct PyResonanceData {
    inner: Arc<ResonanceData>,
}

#[pymethods]
impl PyResonanceData {
    /// String representation.
    fn __repr__(&self) -> String {
        // Formalism-aware count (covers LRF=7 R-matrix-limited spin groups too).
        let n_res: usize = self.inner.total_resonance_count();
        format!(
            "ResonanceData(Z={}, A={}, AWR={:.3}, n_resonances={})",
            self.inner.isotope.z(),
            self.inner.isotope.a(),
            self.inner.awr,
            n_res
        )
    }

    /// Atomic number.
    #[getter]
    fn z(&self) -> u32 {
        self.inner.isotope.z()
    }

    /// Mass number.
    #[getter]
    fn a(&self) -> u32 {
        self.inner.isotope.a()
    }

    /// Atomic weight ratio.
    #[getter]
    fn awr(&self) -> f64 {
        self.inner.awr
    }

    /// Number of resonances (across all ranges and all formalisms).
    ///
    /// Delegates to the formalism-aware
    /// [`ResonanceData::total_resonance_count`], so LRF=7 R-matrix-limited
    /// evaluations (whose resonances live in `rml.spin_groups`, not
    /// `l_groups`) are counted correctly instead of reporting 0.
    #[getter]
    fn n_resonances(&self) -> usize {
        self.inner.total_resonance_count()
    }

    /// Total resonance count across all ranges and formalisms.
    ///
    /// Explicit alias for [`Self::n_resonances`]; both delegate to the
    /// formalism-aware Rust `ResonanceData::total_resonance_count()` (covers
    /// LRF=1/2/3 L-groups and LRF=7 R-matrix-limited spin groups).
    #[getter]
    fn total_resonance_count(&self) -> usize {
        self.inner.total_resonance_count()
    }

    /// Target spin (I) of the first resonance range.
    #[getter]
    fn target_spin(&self) -> f64 {
        self.inner
            .ranges
            .first()
            .map(|r| r.target_spin)
            .unwrap_or(0.0)
    }

    /// Effective scattering radius (fm).
    ///
    /// Returns the global AP from the first range. If AP=0 (common in
    /// ENDF Reich-Moore data that uses energy-dependent radii), falls back
    /// to the first L-group's channel radius APL.
    #[getter]
    fn scattering_radius(&self) -> f64 {
        self.inner
            .ranges
            .first()
            .map(|r| {
                if r.scattering_radius != 0.0 {
                    r.scattering_radius
                } else {
                    // Fall back to first L-group's channel radius
                    r.l_groups.first().map(|lg| lg.apl).unwrap_or(0.0)
                }
            })
            .unwrap_or(0.0)
    }

    /// Orbital angular momentum values (L) present in the data.
    #[getter]
    fn l_values(&self) -> Vec<u32> {
        let mut ls: Vec<u32> = self
            .inner
            .ranges
            .iter()
            .flat_map(|r| &r.l_groups)
            .map(|lg| lg.l)
            .collect();
        ls.sort();
        ls.dedup();
        ls
    }
}

/// Parse a library name string into an `EndfLibrary` enum variant.
fn parse_library_name(library: &str) -> PyResult<EndfLibrary> {
    match library {
        "endf8.0" | "endf/b-viii.0" => Ok(EndfLibrary::EndfB8_0),
        "endf8.1" | "endf/b-viii.1" => Ok(EndfLibrary::EndfB8_1),
        "jeff3.3" => Ok(EndfLibrary::Jeff3_3),
        "jendl5" => Ok(EndfLibrary::Jendl5),
        "tendl2023" | "tendl-2023" => Ok(EndfLibrary::Tendl2023),
        "cendl3.2" | "cendl-3.2" => Ok(EndfLibrary::Cendl3_2),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown library '{}'. Use one of: endf8.0, endf8.1, jeff3.3, jendl5, tendl2023, cendl3.2",
            library
        ))),
    }
}

/// Load and parse ENDF resonance data for a single isotope.
///
/// This helper encapsulates the retrieval + parse logic shared between
/// `load_endf` and `PyIsotopeGroup.load_endf`. It does NOT hold the GIL
/// and must be called from a `py.detach()` / `py.allow_threads()` closure.
fn load_and_parse_endf(
    isotope: &Isotope,
    lib: EndfLibrary,
    mat_num: u32,
) -> Result<ResonanceData, (bool, String)> {
    let retriever = EndfRetriever::new();
    let (_path, contents) = retriever
        .get_endf_file(isotope, lib, mat_num)
        .map_err(|e| (false, format!("{}", e)))?;
    let data =
        parse_endf_file2(&contents).map_err(|e| (true, format!("ENDF parse error: {}", e)))?;
    Ok(data)
}

/// Python wrapper for isotope groups.
///
/// An isotope group binds multiple isotopes with fixed fractional ratios
/// to a single fitted density parameter. The effective cross-section
/// `σ_eff(E) = Σ fᵢ · σᵢ(E)` reduces the group to a virtual isotope.
#[pyclass(name = "IsotopeGroup", from_py_object)]
#[derive(Clone)]
struct PyIsotopeGroup {
    inner: IsotopeGroup,
    /// Loaded ENDF resonance data for each member (indexed by position).
    resonance_data: Vec<Option<Arc<ResonanceData>>>,
}

#[pymethods]
impl PyIsotopeGroup {
    /// Create a group from all natural isotopes of element Z at IUPAC abundances.
    #[staticmethod]
    fn natural(z: u32) -> PyResult<Self> {
        let group = IsotopeGroup::natural(z)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let n = group.n_members();
        Ok(Self {
            inner: group,
            resonance_data: vec![None; n],
        })
    }

    /// Create a group from a subset of natural isotopes, re-normalized.
    #[staticmethod]
    fn subset(z: u32, mass_numbers: Vec<u32>) -> PyResult<Self> {
        let group = IsotopeGroup::subset(z, &mass_numbers)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let n = group.n_members();
        Ok(Self {
            inner: group,
            resonance_data: vec![None; n],
        })
    }

    /// Create a group with arbitrary isotope/ratio pairs.
    ///
    /// Args:
    ///     name: Display name for the group.
    ///     members: List of (z, a, ratio) tuples.
    #[staticmethod]
    fn custom(name: String, members: Vec<(u32, u32, f64)>) -> PyResult<Self> {
        let isotope_members: Vec<(Isotope, f64)> = members
            .into_iter()
            .map(|(z, a, ratio)| {
                let iso = Isotope::new(z, a).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Invalid isotope: {}", e))
                })?;
                Ok((iso, ratio))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let group = IsotopeGroup::custom(name, isotope_members)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let n = group.n_members();
        Ok(Self {
            inner: group,
            resonance_data: vec![None; n],
        })
    }

    /// Fetch ENDF data for all members.
    ///
    /// Args:
    ///     library: ENDF library name (default "endf8.1").
    #[pyo3(signature = (library=None))]
    fn load_endf(&mut self, py: Python<'_>, library: Option<&str>) -> PyResult<()> {
        let lib = parse_library_name(library.unwrap_or("endf8.1"))?;

        // Collect (isotope, mat) pairs for all members up front.
        let members: Vec<(Isotope, u32)> = self
            .inner
            .members()
            .iter()
            .map(|(iso, _)| {
                let mat_num = mat_number(iso, lib).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "MAT number not found for Z={} A={}; cannot fetch ENDF data",
                        iso.z(),
                        iso.a(),
                    ))
                })?;
                Ok((*iso, mat_num))
            })
            .collect::<PyResult<Vec<_>>>()?;

        // Release the GIL for the network I/O + parsing.
        let results: Vec<Result<ResonanceData, (bool, String)>> = py.detach(move || {
            members
                .iter()
                .map(|(iso, mat)| load_and_parse_endf(iso, lib, *mat))
                .collect()
        });

        // Stage all results first — if any failed, return error without
        // modifying self.resonance_data (atomic update).
        let staged: Vec<Arc<ResonanceData>> = results
            .into_iter()
            .enumerate()
            .map(|(i, result)| {
                result.map(|d| Arc::new(d)).map_err(|(is_parse, msg)| {
                    let member = &self.inner.members()[i];
                    let prefix = format!("Z={} A={}: ", member.0.z(), member.0.a());
                    if is_parse {
                        pyo3::exceptions::PyValueError::new_err(prefix + &msg)
                    } else {
                        pyo3::exceptions::PyRuntimeError::new_err(prefix + &msg)
                    }
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        // All succeeded — swap in atomically.
        for (i, data) in staged.into_iter().enumerate() {
            self.resonance_data[i] = Some(data);
        }

        Ok(())
    }

    /// Group display name.
    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Number of member isotopes.
    #[getter]
    fn n_members(&self) -> usize {
        self.inner.n_members()
    }

    /// Member isotopes with their fractional ratios.
    ///
    /// Returns a list of ((z, a), ratio) tuples.
    #[getter]
    fn members(&self) -> Vec<((u32, u32), f64)> {
        self.inner
            .members()
            .iter()
            .map(|(iso, ratio)| ((iso.z(), iso.a()), *ratio))
            .collect()
    }

    /// Whether ENDF data has been loaded for all members.
    #[getter]
    fn is_loaded(&self) -> bool {
        self.resonance_data.iter().all(|d| d.is_some())
    }

    /// Get loaded resonance data for all members.
    ///
    /// Returns a list of ResonanceData objects, one per member.
    ///
    /// Raises:
    ///     ValueError: If not all members have loaded ENDF data.
    #[getter]
    fn resonance_data(&self) -> PyResult<Vec<PyResonanceData>> {
        if !self.resonance_data.iter().all(|d| d.is_some()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Not all group members have loaded ENDF data. Call load_endf() first.",
            ));
        }
        Ok(self
            .resonance_data
            .iter()
            .map(|d| PyResonanceData {
                inner: d.clone().unwrap(),
            })
            .collect())
    }

    fn __repr__(&self) -> String {
        format!(
            "IsotopeGroup(name='{}', n_members={}, loaded={})",
            self.inner.name(),
            self.inner.n_members(),
            self.is_loaded(),
        )
    }
}

/// Result of fitting a spectrum.
#[pyclass(name = "FitResult")]
struct PyFitResult {
    densities: Vec<f64>,
    /// `None` when covariance computation was skipped.
    uncertainties: Option<Vec<f64>>,
    reduced_chi_squared: f64,
    converged: bool,
    iterations: usize,
    /// Fitted temperature in Kelvin (only meaningful when `fit_temperature=True`).
    temperature_k: Option<f64>,
    /// 1-sigma uncertainty on fitted temperature (K).
    temperature_k_unc: Option<f64>,
    /// Fitted normalization / signal-scale parameter.
    /// Transmission LM uses `Anorm`; counts background scaling uses `alpha_1`.
    anorm: f64,
    /// Fitted background parameter triplet.
    /// Transmission LM uses `[BackA, BackB, BackC]`.
    /// Counts KL background uses `[b0, b1, alpha_2]`.
    background: [f64; 3],
    /// Fitted exponential background amplitude (SAMMY BackD).
    /// `None` whenever the polynomial background model was not active
    /// — i.e. either ``background=False`` (the bg model is never
    /// attached, so the inner Rust ``FitResult.back_d`` is the
    /// sentinel `0.0`) or ``background=True`` with ``fit_back_d=False``
    /// (bg model attached but BackD was held at its initial value).
    /// We surface that as `None` at the Python boundary so MCP
    /// consumers can distinguish "fit was active and recovered 0.0"
    /// from "exponential background never engaged" — mirroring the
    /// `t0_us` / `l_scale` convention.
    back_d: Option<f64>,
    /// Fitted exponential background decay constant (SAMMY BackF).
    /// `None` when ``background=False`` OR ``fit_back_f=False``;
    /// see `back_d` for the full rationale.
    back_f: Option<f64>,
    /// Fitted TOF offset in microseconds (SAMMY TZERO t₀).
    /// None when energy-scale fitting is not enabled.
    t0_us: Option<f64>,
    /// Fitted flight-path scale factor (SAMMY TZERO L₀, dimensionless).
    /// None when energy-scale fitting is not enabled.
    l_scale: Option<f64>,
    /// Nominal flight path (m) the energy-scale fit was configured with;
    /// consumed by `corrected_energies` so the transform is reproduced with
    /// the SAME flight path the fit used (issue #634).
    energy_scale_flight_path_m: Option<f64>,
    /// Conditional binomial deviance / (n − k).  `Some(...)` only for the
    /// counts-KL dispatch (`solver="kl"` on counts input).
    deviance_per_dof: Option<f64>,
    /// Fitted multiplicative-baseline coefficients `[b0, b1, b2]` (issue
    /// #635); `None` when ``baseline=False``.
    baseline: Option<[f64; 3]>,
    /// Reference energy E_ref (eV) of the baseline's centered ln(E/E_ref)
    /// basis; `None` when ``baseline=False``.
    baseline_e_ref_ev: Option<f64>,
    /// Structured fit-configuration warnings (e.g. the degenerate
    /// free-Anorm + free-temperature + free-density trio).
    warnings: Vec<String>,
}

#[pymethods]
impl PyFitResult {
    /// Fitted areal densities (atoms/barn).
    #[getter]
    fn densities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.densities.clone())
    }

    /// Uncertainties on fitted densities.
    ///
    /// Returns NaN-filled array when covariance computation was skipped.
    /// Uncertainty values are NaN when covariance is not available
    /// (e.g., Poisson fits via `poisson_fit`, which does not
    /// compute an analytic Hessian for uncertainty estimation).
    #[getter]
    fn uncertainties<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let unc = self
            .uncertainties
            .clone()
            .unwrap_or_else(|| vec![f64::NAN; self.densities.len()]);
        PyArray1::from_vec(py, unc)
    }

    /// Reduced chi-squared of the fit.
    #[getter]
    fn reduced_chi_squared(&self) -> f64 {
        self.reduced_chi_squared
    }

    /// Whether the fit converged.
    #[getter]
    fn converged(&self) -> bool {
        self.converged
    }

    /// Number of iterations.
    #[getter]
    fn iterations(&self) -> usize {
        self.iterations
    }

    /// Fitted sample temperature in Kelvin (``None`` when ``fit_temperature=False``).
    #[getter]
    fn temperature_k(&self) -> Option<f64> {
        self.temperature_k
    }

    /// 1-sigma uncertainty on fitted temperature in Kelvin (``None`` when
    /// ``fit_temperature=False``).
    ///
    /// For the raw-count joint-Poisson path this is a
    /// **covariance-only lower bound**: it is `sqrt` of the temperature diagonal
    /// of the inverse Fisher matrix and omits baseline/model noise, so on real
    /// data it can underestimate the observed per-superpixel scatter by ~3–4×.
    /// Pass ``scale_by_chi2=True`` for a `sqrt(χ²/dof)`-inflated estimate: σ is
    /// scaled by `sqrt` of the goodness-of-fit this result reports (Gaussian
    /// reduced-χ² on the transmission paths, deviance-per-dof on the counts
    /// joint-Poisson path). No-op on the already-χ²-scaled LM transmission path.
    #[getter]
    fn temperature_k_unc(&self) -> Option<f64> {
        self.temperature_k_unc
    }

    /// Fitted normalization / signal-scale parameter.
    #[getter]
    fn anorm(&self) -> f64 {
        self.anorm
    }

    /// Fitted background parameter triplet.
    #[getter]
    fn background(&self) -> [f64; 3] {
        self.background
    }

    /// Fitted exponential background amplitude (SAMMY BackD).
    /// `None` when ``background=False`` OR ``fit_back_d=False``.
    #[getter]
    fn back_d(&self) -> Option<f64> {
        self.back_d
    }

    /// Fitted exponential background decay constant (SAMMY BackF).
    /// `None` when ``background=False`` OR ``fit_back_f=False``.
    #[getter]
    fn back_f(&self) -> Option<f64> {
        self.back_f
    }

    /// Fitted TOF offset in microseconds (SAMMY TZERO t₀).
    /// None when energy-scale fitting is not enabled.
    #[getter]
    fn t0_us(&self) -> Option<f64> {
        self.t0_us
    }

    /// Fitted flight-path scale factor (SAMMY TZERO L₀).
    /// None when energy-scale fitting is not enabled.
    #[getter]
    fn l_scale(&self) -> Option<f64> {
        self.l_scale
    }

    /// Map a nominal energy grid through the fitted ``(t0_us, l_scale)`` energy
    /// scale to the corrected (calibrated) energies the fit evaluated the
    /// physics on (issue #634). Reuses the exact SAMMY-convention transform
    /// (``dat/mdat0.f90:189``, −t0 sign) with the SAME flight path the fit
    /// was configured with (stored on the result), so the corrected axis is
    /// never re-derived by hand (a +t0 slip caused a silent +400 K
    /// temperature bias in the field) and a mismatched caller-supplied
    /// flight path cannot silently skew the t₀ term.
    ///
    /// Returns ``None`` when energy-scale fitting was not enabled. Raises
    /// ``ValueError`` on an invalid nominal grid (non-finite / non-positive /
    /// non-ascending — the binding's standard energy-grid validation) or a
    /// degenerate calibration (a ``t0`` past the shortest flight time).
    #[pyo3(signature = (nominal_energies))]
    fn corrected_energies<'py>(
        &self,
        py: Python<'py>,
        nominal_energies: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        match (self.t0_us, self.l_scale, self.energy_scale_flight_path_m) {
            (Some(t0), Some(l_scale), Some(flight_path_m)) => {
                let e = nominal_energies.as_slice()?;
                require_non_empty_energy_grid(e)?;
                validate_energy_grid(e)?;
                let corr = nereids_fitting::resolution_calib::corrected_energy_grid(
                    e,
                    t0,
                    l_scale,
                    flight_path_m,
                )
                .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
                Ok(Some(PyArray1::from_vec(py, corr)))
            }
            _ => Ok(None),
        }
    }

    /// Conditional binomial deviance divided by (n − k) from the counts-KL
    /// dispatch (joint-Poisson profile-deviance fitter).
    ///
    /// Primary goodness-of-fit statistic for ``solver="kl"`` on counts
    /// data — replaces the fixed-flux Pearson χ² that scaled with ``c``.
    /// Returns ``None`` for LM transmission fits;
    /// those populate ``reduced_chi_squared`` with Pearson χ² / (n − k).
    #[getter]
    fn deviance_per_dof(&self) -> Option<f64> {
        self.deviance_per_dof
    }

    /// Fitted multiplicative-baseline coefficients ``[b0, b1, b2]`` for
    /// ``B(E) = b0 + b1·ln(E/E_ref) + b2·ln²(E/E_ref)`` applied OUTERMOST
    /// (issue #635).  ``None`` when ``baseline=False``.  Coefficients that
    /// were configured but frozen (``fit_b0=False`` …) still report their
    /// values — they are part of the model that produced the fit.
    #[getter]
    fn baseline(&self) -> Option<[f64; 3]> {
        self.baseline
    }

    /// Reference energy E_ref (eV) the baseline's ``ln(E/E_ref)`` basis was
    /// centered on — the geometric midpoint ``sqrt(E_min·E_max)`` of the fit
    /// grid, stored so ``B(E)`` can be reconstructed with the EXACT
    /// reference the fit used.  ``None`` when ``baseline=False``.
    #[getter]
    fn baseline_e_ref_ev(&self) -> Option<f64> {
        self.baseline_e_ref_ev
    }

    /// Structured fit-configuration warnings.  Currently flags the
    /// degenerate normalization trio (free ``Anorm`` + free temperature +
    /// ≥1 free density), which on real VENUS data ran to T = 4471 K with
    /// χ²/ν = 932 and no diagnostic.  Empty list when nothing is flagged.
    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }

    fn __repr__(&self) -> String {
        if let Some(t) = self.temperature_k {
            format!(
                "FitResult(converged={}, chi2_red={:.4}, densities={:?}, temperature_k={:.1})",
                self.converged, self.reduced_chi_squared, self.densities, t
            )
        } else {
            format!(
                "FitResult(converged={}, chi2_red={:.4}, densities={:?})",
                self.converged, self.reduced_chi_squared, self.densities
            )
        }
    }
}

/// Python wrapper for tabulated resolution function.
///
/// Uses `Arc` internally so that `.clone()` in `py.detach()` closures is O(1).
#[pyclass(name = "TabulatedResolution", from_py_object)]
#[derive(Clone)]
struct PyTabulatedResolution {
    inner: Arc<TabulatedResolution>,
}

#[pymethods]
impl PyTabulatedResolution {
    /// Number of reference energies.
    #[getter]
    fn n_energies(&self) -> usize {
        self.inner.ref_energies().len()
    }

    /// Energy range (min, max) of the reference kernels in eV.
    #[getter]
    fn energy_range(&self) -> (f64, f64) {
        let e = self.inner.ref_energies();
        if e.is_empty() {
            (0.0, 0.0)
        } else {
            (e[0], e[e.len() - 1])
        }
    }

    /// Flight path length in meters.
    #[getter]
    fn flight_path_m(&self) -> f64 {
        self.inner.flight_path_m()
    }

    /// Number of points per kernel.
    #[getter]
    fn points_per_kernel(&self) -> usize {
        self.inner
            .kernels()
            .first()
            .map(|(o, _)| o.len())
            .unwrap_or(0)
    }

    /// Probability that a neutron at one true energy is recorded in each
    /// adjacent detector-time bin. The tabulated UDR is evaluated directly;
    /// the result is not renormalized when the supplied time window omits part
    /// of the pulse.
    fn detector_bin_probabilities(
        &self,
        true_energy_ev: f64,
        detector_time_edges_us: Vec<f64>,
        timing_offset_us: f64,
    ) -> PyResult<Vec<f64>> {
        self.inner
            .detector_bin_probabilities(true_energy_ev, &detector_time_edges_us, timing_offset_us)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
    }

    fn __repr__(&self) -> String {
        let (lo, hi) = self.energy_range();
        format!(
            "TabulatedResolution(n_energies={}, range=[{:.4e}, {:.4e}] eV, flight_path={:.1} m)",
            self.n_energies(),
            lo,
            hi,
            self.inner.flight_path_m(),
        )
    }
}

/// Energy-dependence law for an Ikeda–Carpenter parameter.
///
/// Build via the static constructors:
/// `EnergyLaw.const(c)`, `EnergyLaw.sqrt_e(a0, a1)`,
/// `EnergyLaw.inverse_lambda(a0, a1)`, `EnergyLaw.exp_mev(kappa)`.
#[pyclass(name = "EnergyLaw", from_py_object)]
#[derive(Clone)]
struct PyEnergyLaw {
    inner: EnergyLaw,
}

#[pymethods]
impl PyEnergyLaw {
    /// Energy-independent constant value.
    #[staticmethod]
    #[pyo3(name = "const")]
    fn const_law(value: f64) -> Self {
        Self {
            inner: EnergyLaw::Const(value),
        }
    }

    /// `a0·√(E[eV]) + a1` — leading epithermal scaling of the fast rate α(E).
    #[staticmethod]
    fn sqrt_e(a0: f64, a1: f64) -> Self {
        Self {
            inner: EnergyLaw::SqrtE { a0, a1 },
        }
    }

    /// Mantid IC form `1/(a0 + a1·λ)`, λ[Å] ∝ 1/√E (α ∝ √E low-E, → 1/a0 high-E).
    #[staticmethod]
    fn inverse_lambda(a0: f64, a1: f64) -> Self {
        Self {
            inner: EnergyLaw::InverseLambda { a0, a1 },
        }
    }

    /// `exp(−E[meV]/kappa)` — storage fraction R(E), → 0 in the eV regime.
    #[staticmethod]
    fn exp_mev(kappa: f64) -> Self {
        Self {
            inner: EnergyLaw::ExpMilliEv { kappa },
        }
    }

    /// Evaluate the law at `energy_ev` (eV).
    fn eval(&self, energy_ev: f64) -> f64 {
        self.inner.eval(energy_ev)
    }

    fn __repr__(&self) -> String {
        format!("EnergyLaw({:?})", self.inner)
    }
}

/// Analytical Ikeda–Carpenter instrument-resolution model.
///
/// Synthesizes a dense tabulated kernel at construction; pass
/// [`IkedaCarpenter.as_tabulated`] anywhere a loaded resolution file is accepted
/// (`forward_model`, `fit_spectrum_typed`, `calibrate_resolution`). Note that
/// `precompute_cross_sections` does NOT take a resolution — broadening is applied
/// after Beer–Lambert, not on the cross-sections. The synthesized kernel rides
/// the *same* broadening path as a Monte-Carlo file, so the IC-vs-tabulated
/// comparison differs only in kernel source.
///
/// Parameters: `alpha`/`r` are [`EnergyLaw`]s (fixed-or-fit general case).
/// `beta` keeps the original constant-rate API; the optional trailing
/// `beta_law` overrides it with an energy-dependent rate. Optional
/// `burst_sigma_us` (Gaussian) and `channel_fwhm_us` (triangle) fold in the
/// proton-burst and chopper terms.
#[pyclass(name = "IkedaCarpenter", skip_from_py_object)]
#[derive(Clone)]
struct PyIkedaCarpenter {
    inner: Arc<IkedaCarpenter>,
}

#[pymethods]
impl PyIkedaCarpenter {
    #[new]
    #[pyo3(signature = (
        flight_path_m,
        e_min_ev,
        e_max_ev,
        alpha,
        beta,
        r,
        n_energies = 64,
        n_tau = 600,
        burst_sigma_us = None,
        channel_fwhm_us = None,
        beta_law = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        flight_path_m: f64,
        e_min_ev: f64,
        e_max_ev: f64,
        alpha: PyEnergyLaw,
        beta: f64,
        r: PyEnergyLaw,
        n_energies: usize,
        n_tau: usize,
        burst_sigma_us: Option<f64>,
        channel_fwhm_us: Option<f64>,
        beta_law: Option<PyEnergyLaw>,
    ) -> PyResult<Self> {
        for (name, v) in [
            ("burst_sigma_us", burst_sigma_us),
            ("channel_fwhm_us", channel_fwhm_us),
        ] {
            if let Some(x) = v {
                if !x.is_finite() || x < 0.0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "{name} must be finite and >= 0, got {x}"
                    )));
                }
            }
        }
        let params = IkedaCarpenterParams {
            alpha: alpha.inner,
            beta: beta_law.map_or(EnergyLaw::Const(beta), |law| law.inner),
            r: r.inner,
            burst_sigma_us,
            channel_fwhm_us,
        };
        let grid = SynthesisGrid {
            e_min_ev,
            e_max_ev,
            n_energies,
            n_tau,
        };
        let ic = IkedaCarpenter::new(params, flight_path_m, &grid)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(ic),
        })
    }

    /// The synthesized tabulated kernel — pass this anywhere a loaded
    /// resolution file (`TabulatedResolution`) is accepted.
    fn as_tabulated(&self) -> PyTabulatedResolution {
        PyTabulatedResolution {
            inner: Arc::new(self.inner.tabulated().clone()),
        }
    }

    /// `(tof_offsets_us, weights)` kernel at a single energy (eV); offsets
    /// ascending with the mode at 0, weights peak-normalized.
    ///
    /// Raises ``ValueError`` when the τ-grid cannot resolve the prompt core
    /// and requested folds within the sample cap at this energy (construction
    /// validates the reference energies; a probe energy outside their range
    /// can still be unresolvable).
    fn kernel_at(&self, energy_ev: f64) -> PyResult<(Vec<f64>, Vec<f64>)> {
        self.inner
            .kernel_at(energy_ev)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// `(moderator_delay_us, density)` for the physical source pulse at one
    /// true energy. Unlike `kernel_at`, this keeps the pulse's time origin.
    fn source_pulse_at(&self, true_energy_ev: f64) -> PyResult<(Vec<f64>, Vec<f64>)> {
        self.inner
            .source_pulse_at(true_energy_ev)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Probability that a neutron at one true energy is recorded in each
    /// adjacent detector-time bin. The result is not renormalized when the
    /// supplied time window omits part of the pulse.
    fn detector_bin_probabilities(
        &self,
        true_energy_ev: f64,
        detector_time_edges_us: Vec<f64>,
        timing_offset_us: f64,
    ) -> PyResult<Vec<f64>> {
        self.inner
            .detector_bin_probabilities(true_energy_ev, &detector_time_edges_us, timing_offset_us)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Flight path length in meters.
    #[getter]
    fn flight_path_m(&self) -> f64 {
        self.inner.flight_path_m()
    }

    /// Number of synthesized reference energies.
    #[getter]
    fn n_energies(&self) -> usize {
        self.inner.ref_energies().len()
    }

    fn __repr__(&self) -> String {
        let e = self.inner.ref_energies();
        format!(
            "IkedaCarpenter(n_energies={}, range=[{:.4e}, {:.4e}] eV, flight_path={:.1} m)",
            e.len(),
            e.first().copied().unwrap_or(0.0),
            e.last().copied().unwrap_or(0.0),
            self.inner.flight_path_m(),
        )
    }
}

/// Result of spatial (per-pixel) mapping.
///
/// Numpy arrays are constructed once and cached; property access returns
/// cheap references (refcount bump) rather than copying data.
#[pyclass(name = "SpatialResult")]
struct PySpatialResult {
    density_maps: Vec<Py<PyArray2<f64>>>,
    uncertainty_maps: Vec<Py<PyArray2<f64>>>,
    chi_squared_map: Py<PyArray2<f64>>,
    /// Counts-KL conditional binomial deviance / (n − k) per pixel.
    /// None for transmission-only and LM-only runs.
    deviance_per_dof_map: Option<Py<PyArray2<f64>>>,
    converged_map: Py<PyArray2<bool>>,
    n_converged: usize,
    n_total: usize,
    n_failed: usize,
    isotope_names: Vec<String>,
    shape: (usize, usize),
    /// Per-pixel fitted temperature (None when fit_temperature=False).
    temperature_map: Option<Py<PyArray2<f64>>>,
    /// Per-pixel temperature uncertainty (None when fit_temperature=False).
    temperature_uncertainty_map: Option<Py<PyArray2<f64>>>,
    /// Per-pixel normalization / signal-scale map (None when background=False).
    anorm_map: Option<Py<PyArray2<f64>>>,
    /// Per-pixel background parameter maps (None when background=False).
    background_maps: Option<[Py<PyArray2<f64>>; 3]>,
    /// Per-pixel SAMMY BackD exponential-amplitude map
    /// (None unless ``background=True`` AND ``fit_back_d=True``).
    back_d_map: Option<Py<PyArray2<f64>>>,
    /// Per-pixel SAMMY BackF exponential-decay-constant map
    /// (None unless ``background=True`` AND ``fit_back_f=True``).
    back_f_map: Option<Py<PyArray2<f64>>>,
    /// Per-pixel fitted TZERO t0 (µs) map (None when fit_energy_scale=False).
    t0_us_map: Option<Py<PyArray2<f64>>>,
    /// Per-pixel fitted TZERO L_scale map (None when fit_energy_scale=False).
    l_scale_map: Option<Py<PyArray2<f64>>>,
    /// Global multiplicative-baseline coefficients (issue #635);
    /// None when baseline=False or baseline_global=False.
    baseline_global: Option<[f64; 3]>,
    /// Baseline reference energy E_ref (eV); None when baseline=False.
    baseline_e_ref_ev: Option<f64>,
    /// Per-pixel baseline coefficient maps; None unless baseline=True with
    /// baseline_global=False.
    baseline_maps: Option<[Py<PyArray2<f64>>; 3]>,
    /// Structured fit-configuration warnings.
    warnings: Vec<String>,
}

#[pymethods]
impl PySpatialResult {
    /// Density maps as a list of 2D numpy arrays, one per isotope.
    #[getter]
    fn density_maps<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f64>>> {
        self.density_maps
            .iter()
            .map(|m| m.bind(py).clone())
            .collect()
    }

    /// Uncertainty maps as a list of 2D numpy arrays.
    #[getter]
    fn uncertainty_maps<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f64>>> {
        self.uncertainty_maps
            .iter()
            .map(|m| m.bind(py).clone())
            .collect()
    }

    /// Reduced chi-squared map.
    #[getter]
    fn chi_squared_map<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.chi_squared_map.bind(py).clone()
    }

    /// Counts-KL conditional binomial deviance per degree of freedom.
    ///
    /// Primary goodness-of-fit for ``solver="kl"`` on counts data
    /// (replaces the fixed-flux Pearson that scaled
    /// with ``c``).  Returns ``None`` for LM fits and transmission +
    /// LM transmission fits; those populate ``chi_squared_map`` with Pearson χ² /
    /// (n − k) instead.
    #[getter]
    fn deviance_per_dof_map<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.deviance_per_dof_map
            .as_ref()
            .map(|m| m.bind(py).clone())
    }

    /// Convergence map (True = converged).
    #[getter]
    fn converged_map<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<bool>> {
        self.converged_map.bind(py).clone()
    }

    /// Number of converged pixels.
    #[getter]
    fn n_converged(&self) -> usize {
        self.n_converged
    }

    /// Total number of fitted pixels.
    #[getter]
    fn n_total(&self) -> usize {
        self.n_total
    }

    /// Number of pixels where the fitter returned a hard error (NaN density).
    #[getter]
    fn n_failed(&self) -> usize {
        self.n_failed
    }

    /// Isotope names.
    #[getter]
    fn isotope_names(&self) -> Vec<String> {
        self.isotope_names.clone()
    }

    /// Per-pixel fitted temperature map (None when fit_temperature=False).
    #[getter]
    fn temperature_map<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.temperature_map.as_ref().map(|m| m.bind(py).clone())
    }

    /// Per-pixel temperature uncertainty map (None when fit_temperature=False).
    /// Entries are NaN where uncertainty was unavailable for that pixel.
    ///
    /// For the raw-count joint-Poisson path each σ_T
    /// is a **covariance-only lower bound** (`sqrt` of the temperature diagonal
    /// of the inverse Fisher matrix): it omits baseline/model noise and on real
    /// data can underestimate the observed per-superpixel scatter by ~3–4×.
    /// Pass ``scale_by_chi2=True`` to ``spatial_map*`` for a `sqrt(χ²/dof)`-
    /// inflated estimate: σ is scaled by `sqrt` of the goodness-of-fit each
    /// pixel's result reports (Gaussian reduced-χ² on the transmission paths,
    /// deviance-per-dof on the counts joint-Poisson path). No-op on the
    /// already-χ²-scaled LM transmission path.
    #[getter]
    fn temperature_uncertainty_map<'py>(
        &self,
        py: Python<'py>,
    ) -> Option<Bound<'py, PyArray2<f64>>> {
        self.temperature_uncertainty_map
            .as_ref()
            .map(|m| m.bind(py).clone())
    }

    /// Per-pixel normalization factor Anorm (None when background fitting was disabled).
    #[getter]
    fn anorm_map<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.anorm_map.as_ref().map(|m| m.bind(py).clone())
    }

    /// Per-pixel background parameter maps
    /// (transmission LM: `[BackA, BackB, BackC]`; counts KL: `[b0, b1, alpha_2]`).
    #[getter]
    fn background_maps<'py>(&self, py: Python<'py>) -> Option<Vec<Bound<'py, PyArray2<f64>>>> {
        self.background_maps
            .as_ref()
            .map(|maps| maps.iter().map(|m| m.bind(py).clone()).collect())
    }

    /// Per-pixel SAMMY exponential background amplitude (``BackD``) map.
    /// ``None`` whenever the LM transmission background was not active
    /// (``background=False``) OR the exponential tail was not fit
    /// (``fit_back_d=False``).  Counts-KL runs are always ``None``
    /// because the joint-Poisson dispatch never fits ``BackD``/``BackF``.
    /// Mirrors the per-spectrum ``FitResult.back_d`` convention from
    /// issue #537.
    #[getter]
    fn back_d_map<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.back_d_map.as_ref().map(|m| m.bind(py).clone())
    }

    /// Per-pixel SAMMY exponential background decay constant (``BackF``) map.
    /// ``None`` under the same conditions as :py:attr:`back_d_map`.
    #[getter]
    fn back_f_map<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.back_f_map.as_ref().map(|m| m.bind(py).clone())
    }

    /// Per-pixel SAMMY TZERO offset t0 (µs) map.
    /// `None` when the run did not fit energy scale.
    #[getter]
    fn t0_us_map<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.t0_us_map.as_ref().map(|m| m.bind(py).clone())
    }

    /// Per-pixel SAMMY TZERO flight-path scale factor map.
    /// `None` when the run did not fit energy scale.
    #[getter]
    fn l_scale_map<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.l_scale_map.as_ref().map(|m| m.bind(py).clone())
    }

    /// Global multiplicative-baseline coefficients ``[b0, b1, b2]``
    /// (issue #635): the stage-1 fit on the aggregated mean spectrum that
    /// every pixel then used frozen.  ``None`` when ``baseline=False`` or
    /// in per-pixel mode (``baseline_global=False`` — see
    /// :py:attr:`baseline_maps`).
    #[getter]
    fn baseline_global(&self) -> Option<[f64; 3]> {
        self.baseline_global
    }

    /// Baseline reference energy E_ref (eV) of the centered ``ln(E/E_ref)``
    /// basis — reconstruct ``B(E)`` with exactly this reference.
    /// ``None`` when ``baseline=False``.
    #[getter]
    fn baseline_e_ref_ev(&self) -> Option<f64> {
        self.baseline_e_ref_ev
    }

    /// Per-pixel baseline coefficient maps ``[b0, b1, b2]``.  ``None``
    /// unless ``baseline=True`` with ``baseline_global=False``.
    /// NaN at pixels that did not converge.
    #[getter]
    fn baseline_maps<'py>(&self, py: Python<'py>) -> Option<Vec<Bound<'py, PyArray2<f64>>>> {
        self.baseline_maps
            .as_ref()
            .map(|maps| maps.iter().map(|m| m.bind(py).clone()).collect())
    }

    /// Structured fit-configuration warnings (also printed once to stderr
    /// by the spatial engine).  Empty list when nothing is flagged.
    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "SpatialResult(shape={}x{}, isotopes={}, converged={}/{})",
            self.shape.0,
            self.shape.1,
            self.isotope_names.len(),
            self.n_converged,
            self.n_total,
        )
    }
}

/// Compute cross-sections at given energies for an isotope.
///
/// Args:
///     energies: Energy grid in eV (1D numpy array).  Must be **strictly
///         ascending**, with every value **finite and positive** — these
///         constraints are enforced by ``validate_energy_grid`` before any
///         physics is evaluated.  Empty grids are accepted and return
///         empty arrays.
///     data: ResonanceData for the isotope.
///
/// Returns:
///     Dictionary with 'total', 'elastic', 'capture', 'fission' arrays.
///
/// Raises:
///     ValueError: If the grid contains NaN/∞, non-positive entries, or
///         is not strictly ascending.
#[pyfunction]
fn cross_sections<'py>(
    py: Python<'py>,
    energies: PyReadonlyArray1<f64>,
    data: &PyResonanceData,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    // Validate the full grid up front so invalid input surfaces as ValueError
    // rather than a release-mode `PanicException` from the per-point
    // `assert!(energy_ev.is_finite() && energy_ev > 0.0)` guards added to
    // the SLBW / RML / URR leaf paths.  Sibling PyO3 entries
    // (`doppler_broaden`, `resolution_broaden`, `forward_model`, etc.)
    // already validate via this same helper; this entry was the lone gap.
    let e_slice = energies.as_slice()?;
    validate_energy_grid(e_slice)?;
    let e_owned = e_slice.to_vec();
    let res_data = data.inner.clone();

    // Release the GIL for the cross-section computation.
    // Use cross_sections_on_grid() which precomputes J-group data once,
    // rather than recomputing per energy point via cross_sections_at_energy().
    let (total, elastic, capture, fission) = py.detach(move || {
        let results = nereids_physics::reich_moore::cross_sections_on_grid(&res_data, &e_owned);
        let mut total = Vec::with_capacity(results.len());
        let mut elastic = Vec::with_capacity(results.len());
        let mut capture = Vec::with_capacity(results.len());
        let mut fission = Vec::with_capacity(results.len());
        for xs in results {
            total.push(xs.total);
            elastic.push(xs.elastic);
            capture.push(xs.capture);
            fission.push(xs.fission);
        }
        (total, elastic, capture, fission)
    });

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("total", PyArray1::from_vec(py, total))?;
    dict.set_item("elastic", PyArray1::from_vec(py, elastic))?;
    dict.set_item("capture", PyArray1::from_vec(py, capture))?;
    dict.set_item("fission", PyArray1::from_vec(py, fission))?;
    Ok(dict)
}

/// Compute theoretical transmission spectrum.
///
/// Resolution broadening can be applied via either Gaussian parameters
/// (``flight_path_m``, ``delta_t_us``, ``delta_l_m``) or a tabulated
/// resolution function (``resolution``). Providing both is an error.
///
/// Either ``isotopes`` or ``groups`` must be provided, but not both.
/// When ``groups`` is provided, each group is expanded into its members with
/// effective densities = group_density × member_ratio.
///
/// Args:
///     energies: Energy grid in eV (1D numpy array).
///     isotopes: List of (ResonanceData, areal_density) tuples (mutually exclusive with groups).
///     temperature_k: Sample temperature in Kelvin (default 293.6).
///     flight_path_m: Flight path in meters for Gaussian resolution (optional).
///     delta_t_us: Timing uncertainty in microseconds (optional).
///     delta_l_m: Path length uncertainty in meters (optional).
///     delta_e_us: Exponential tail parameter in SAMMY Deltae units (optional,
///         default None/0.0). When non-zero, adds an exponential tail to the
///         resolution kernel (SAMMY Iesopr=3).
///     resolution: TabulatedResolution from ``load_resolution()`` (optional).
///     groups: List of (IsotopeGroup, group_density) tuples (mutually exclusive with isotopes).
///
/// Returns:
///     1D numpy array of transmission values.
#[pyfunction]
#[pyo3(signature = (energies, isotopes=None, temperature_k=293.6, flight_path_m=None, delta_t_us=None, delta_l_m=None, resolution=None, delta_e_us=None, groups=None))]
fn forward_model<'py>(
    py: Python<'py>,
    energies: PyReadonlyArray1<f64>,
    isotopes: Option<Vec<(PyResonanceData, f64)>>,
    temperature_k: f64,
    flight_path_m: Option<f64>,
    delta_t_us: Option<f64>,
    delta_l_m: Option<f64>,
    resolution: Option<PyTabulatedResolution>,
    delta_e_us: Option<f64>,
    groups: Option<Vec<(PyIsotopeGroup, f64)>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let has_isotopes = isotopes.is_some();
    let has_groups = groups.is_some();
    if has_isotopes && has_groups {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Provide either 'isotopes' or 'groups', not both.",
        ));
    }
    if !has_isotopes && !has_groups {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Must provide either 'isotopes' or 'groups'.",
        ));
    }

    // Validate the energy grid up front so invalid input surfaces as
    // ValueError rather than a release-mode `PanicException` from the
    // per-point `assert!(energy_ev.is_finite() && energy_ev > 0.0)` guards
    // added to the SLBW / RML / URR / Reich-Moore leaf paths.  Empty
    // grids are accepted (the downstream `transmission::forward_model`
    // returns an empty vector for an empty grid).
    let e_slice = energies.as_slice()?;
    validate_energy_grid(e_slice)?;
    let e_owned = e_slice.to_vec();

    // Build sample isotopes list from either isotopes or groups
    let sample_isotopes: Vec<(ResonanceData, f64)> = if let Some(isotopes) = isotopes {
        isotopes
            .into_iter()
            .map(|(d, thick)| (Arc::unwrap_or_clone(d.inner), thick))
            .collect()
    } else {
        let groups = groups.unwrap();
        let mut expanded = Vec::new();
        for (group, group_density) in &groups {
            if !group.is_loaded() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "IsotopeGroup '{}' has not been fully loaded. Call load_endf() first.",
                    group.inner.name(),
                )));
            }
            for (i, (_iso, ratio)) in group.inner.members().iter().enumerate() {
                let rd = Arc::unwrap_or_clone(group.resonance_data[i].clone().unwrap());
                expanded.push((rd, group_density * ratio));
            }
        }
        expanded
    };

    let sample = SampleParams::new(temperature_k, sample_isotopes)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let res_fn = build_resolution(flight_path_m, delta_t_us, delta_l_m, resolution, delta_e_us)?;
    let instrument = res_fn.map(|r| InstrumentParams { resolution: r });

    // Release the GIL for the forward model computation.
    let t = py.detach(move || transmission::forward_model(&e_owned, &sample, instrument.as_ref()));
    let t = t.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, t))
}

/// Result of an instrument-resolution calibration ([`calibrate_resolution`]).
#[pyclass(name = "ResolutionCalibration", skip_from_py_object)]
struct PyResolutionCalibration {
    inner: nereids_fitting::resolution_calib::CalibrationResult,
}

#[pymethods]
impl PyResolutionCalibration {
    /// Family label (`"gaussian"` | `"udr_corr"` | `"ic"`).
    #[getter]
    fn family(&self) -> String {
        self.inner.family.clone()
    }

    /// Raw fitted parameter vector (optimizer space).
    #[getter]
    fn theta(&self) -> Vec<f64> {
        self.inner.theta.clone()
    }

    /// χ²/dof of the calibration fit.
    #[getter]
    fn chi2(&self) -> f64 {
        self.inner.chi2_dof
    }

    /// Whether the optimizer self-converged.
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    /// Optimizer iterations.
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    /// Fitted (or pinned) SAMMY energy-scale TOF zero ``t0`` (µs). Equals
    /// ``t0_center_us`` when ``fit_t0=False`` (the default — position pinned).
    /// When fit, this is a SHARED energy-scale parameter under a metrology prior,
    /// not a per-family nuisance (the asymmetric-kernel lag is confounded with
    /// flight-path ``L_scale``).
    #[getter]
    fn position_t0_us(&self) -> f64 {
        self.inner.position_t0_us
    }

    /// Fitted (or pinned) flight-path scale ``L_scale``. Equals ``l_scale_center``
    /// when ``fit_l_scale=False`` (the default).
    #[getter]
    fn position_l_scale(&self) -> f64 {
        self.inner.position_l_scale
    }

    /// Gaussian-prior penalty on the fitted ``(t0, L_scale)`` at the solution
    /// (0 when position is pinned or has no prior). ``objective = chi2_data +
    /// prior_penalty``; a large value flags a family that needed a big position
    /// move (e.g. ΔL/L ≫ the metrology σ) to fit.
    #[getter]
    fn prior_penalty(&self) -> f64 {
        self.inner.prior_penalty
    }

    /// Decoded, human-readable fitted parameters.
    ///
    /// For ``family="ic"`` (#642) the keys are ``a0``/``a1`` (α(E) = a0·√E +
    /// a1, positive by construction), ``beta``, ``r`` and ``psr_fwhm_us`` —
    /// decoded from the calibrated resolution itself (the raw ``theta`` is
    /// ln/box-encoded optimizer space).
    fn params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let d = pyo3::types::PyDict::new(py);
        match self.inner.family.as_str() {
            "udr_corr" => {
                // Decode against the SAME clamp bounds the Rust optimizer used
                // (resolution_calib::UDR_S0_MIN/MAX), not duplicated literals.
                let s0 = self.inner.theta[0].exp().clamp(UDR_S0_MIN, UDR_S0_MAX);
                d.set_item("s0", s0)?;
                d.set_item("p", self.inner.theta[1])?;
            }
            "gaussian" => {
                d.set_item("delta_t_us", self.inner.theta[0].abs())?;
                d.set_item("delta_l_m", self.inner.theta[1].abs())?;
            }
            _ => {
                // IC: decode off the calibrated resolution — the single source
                // of truth (never re-derive from the encoded theta by hand).
                // The a0/a1 keys keep their pre-#642 meaning for back-compat.
                if let ResolutionFunction::IkedaCarpenter(ic) = &self.inner.resolution {
                    let p = ic.params();
                    if let EnergyLaw::SqrtE { a0, a1 } = &p.alpha {
                        d.set_item("a0", *a0)?;
                        d.set_item("a1", *a1)?;
                    }
                    if let EnergyLaw::Const(beta) = p.beta {
                        d.set_item("beta", beta)?;
                    }
                    if let EnergyLaw::Const(r) = &p.r {
                        d.set_item("r", *r)?;
                    }
                    d.set_item("psr_fwhm_us", p.channel_fwhm_us.unwrap_or(0.0))?;
                }
            }
        }
        Ok(d)
    }

    /// Number of outer-loop free parameters: resolution θ plus any fitted
    /// position coordinates (4–5 for ``"ic"``, 2 for the other families).
    #[getter]
    fn n_free_params(&self) -> usize {
        self.inner.n_free_params
    }

    /// Coordinates pinned at a box bound at the solution, as
    /// ``"name:lower"`` / ``"name:upper"`` strings (empty = interior
    /// solution). E.g. ``"r:lower"`` flags the β↔R ridge: the calibrant shows
    /// no storage tail, so the reported β carries no information.
    #[getter]
    fn bounds_hit(&self) -> Vec<String> {
        self.inner.bounds_hit.clone()
    }

    /// The calibrated resolution as a [`TabulatedResolution`] — pass to
    /// `resolution=` in the fitters. `None` for the Gaussian family (use
    /// [`Self::gaussian_params`] there).
    fn as_tabulated(&self) -> Option<PyTabulatedResolution> {
        match &self.inner.resolution {
            ResolutionFunction::Tabulated(t) => Some(PyTabulatedResolution { inner: t.clone() }),
            ResolutionFunction::IkedaCarpenter(ic) => Some(PyTabulatedResolution {
                inner: Arc::new(ic.tabulated().clone()),
            }),
            ResolutionFunction::Gaussian(_) => None,
        }
    }

    /// `(delta_t_us, delta_l_m)` for the Gaussian family; `None` otherwise.
    fn gaussian_params(&self) -> Option<(f64, f64)> {
        match &self.inner.resolution {
            ResolutionFunction::Gaussian(p) => Some((p.delta_t_us(), p.delta_l_m())),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ResolutionCalibration(family={}, chi2/dof={:.4}, converged={}, n_free_params={}, bounds_hit={:?})",
            self.inner.family,
            self.inner.chi2_dof,
            self.inner.converged,
            self.inner.n_free_params,
            self.inner.bounds_hit
        )
    }
}

/// Calibrate instrument-resolution parameters against a known-(ρ,T) calibrant.
///
/// Fits the resolution parameters of `family` while holding the calibrant's
/// density (the values in `isotopes`/`groups`) and `temperature_k` FIXED. Pin
/// the returned resolution into a subsequent sample fit.
///
/// Args:
///     energies, data, uncertainty: calibrant transmission spectrum.
///     family: ``"gaussian"`` | ``"udr_corr"`` | ``"ic"``. The ``"ic"`` family
///         (#642) fits the full bounded moderator shape — ``α(E) = a0·√E + a1``
///         positive by construction, free bounded ``beta`` and storage
///         fraction ``r`` — folded with the SNS PSR channel triangle.
///     isotopes / groups: known calibrant composition + density (exactly one).
///     temperature_k: known calibrant temperature.
///     base_udr: base UDR kernel (required for ``family="udr_corr"``).
///     fit_background: also fit anorm + linear baseline (default anorm only).
///     restarts: optimizer restarts (keep the best).
///     fit_t0, fit_l_scale: also fit the SHARED SAMMY energy-scale ``(t0,
///         L_scale)``. Default ``False`` — position is PINNED at its center (a pure
///         shape/width calibration on the already energy-calibrated grid). Opt in
///         only WITH a prior: the asymmetric-kernel mode→centroid lag is the same
///         ``1/√E`` basis as ``L_scale``, so a free ``L_scale`` absorbs the lag and
///         corrupts the calibrated width. Use this for joint energy-scale / cross-
///         family identifiability work, with ``t0_prior_us`` / ``l_scale_prior``
///         set from the instrument's flight-path / timing metrology.
///     t0_center_us, l_scale_center: prior means / pinned values (default 0.0, 1.0).
///     t0_prior_us, l_scale_prior: Gaussian prior σ (``None`` = flat/bounded only).
///     psr_fwhm_ns: SNS PSR channel-triangle FWHM in ns, folded into the
///         ``"ic"`` family's kernel only (default 350 — the VENUS FTS header
///         value; ``0`` disables). Tabulated/UDR kernels already carry the
///         fold in the file and are never re-folded. NANOSECONDS: values
///         above 10_000 ns (10 µs) are rejected as a µs-as-ns unit slip
///         (synthesis cost is quadratic in the fold width).
///     fit_psr: also FIT the PSR FWHM (``"ic"`` only; appends a 5th
///         parameter, box-bounded 0.05–1 µs, started at ``psr_fwhm_ns``
///         clamped into that box — an out-of-box start, legal as a pin up
///         to 10 µs, starts at the nearer box edge with a stderr warning,
///         and a fit that stays there reports ``psr_fwhm_us:lower`` /
///         ``:upper`` in ``bounds_hit``; ``psr_fwhm_ns`` must then be > 0:
///         a zero start contradicts "0 disables").
///
/// Returns:
///     ResolutionCalibration with the fitted params, data χ²/dof, the fitted (or
///     pinned) ``position_t0_us`` / ``position_l_scale`` / ``prior_penalty``, and
///     the calibrated resolution (``.as_tabulated()`` / ``.gaussian_params()``).
// `psr_fwhm_ns` / `fit_psr` sit at the END of the signature (after every
// parameter that predates them): inserting them mid-signature would silently
// shift the meaning of existing ≥ 14-positional-argument calls (review #645
// F7). Keep any future additions at the end for the same reason.
#[pyfunction]
#[pyo3(signature = (
    energies, data, uncertainty, family, isotopes=None, groups=None,
    temperature_k=293.6, base_udr=None, flight_path_m=25.0, fit_background=false,
    restarts=1, ic_n_energies=64, ic_n_tau=500,
    fit_t0=false, fit_l_scale=false, t0_center_us=0.0, l_scale_center=1.0,
    t0_prior_us=None, l_scale_prior=None,
    psr_fwhm_ns=350.0, fit_psr=false
))]
#[allow(clippy::too_many_arguments)]
fn calibrate_resolution(
    py: Python<'_>,
    energies: PyReadonlyArray1<f64>,
    data: PyReadonlyArray1<f64>,
    uncertainty: PyReadonlyArray1<f64>,
    family: &str,
    isotopes: Option<Vec<(PyResonanceData, f64)>>,
    groups: Option<Vec<(PyIsotopeGroup, f64)>>,
    temperature_k: f64,
    base_udr: Option<PyTabulatedResolution>,
    flight_path_m: f64,
    fit_background: bool,
    restarts: usize,
    ic_n_energies: usize,
    ic_n_tau: usize,
    fit_t0: bool,
    fit_l_scale: bool,
    t0_center_us: f64,
    l_scale_center: f64,
    t0_prior_us: Option<f64>,
    l_scale_prior: Option<f64>,
    psr_fwhm_ns: f64,
    fit_psr: bool,
) -> PyResult<PyResolutionCalibration> {
    if isotopes.is_some() == groups.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Provide exactly one of 'isotopes' or 'groups'.",
        ));
    }
    let e = energies.as_slice()?;
    // Reject an empty grid up front with a precise message (validate_energy_grid
    // tolerates empty; the Rust calibrator would otherwise reject it later as a
    // generic EmptyData). Matches the other non-empty entry points.
    require_non_empty_energy_grid(e)?;
    let d = data.as_slice()?;
    let u = uncertainty.as_slice()?;
    if d.len() != e.len() || u.len() != e.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "energies, data, uncertainty must have equal length",
        ));
    }

    let sample_isotopes: Vec<(ResonanceData, f64)> = if let Some(isotopes) = isotopes {
        isotopes
            .into_iter()
            .map(|(rd, thick)| (Arc::unwrap_or_clone(rd.inner), thick))
            .collect()
    } else {
        let groups = groups.unwrap();
        let mut expanded = Vec::new();
        for (group, group_density) in &groups {
            if !group.is_loaded() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "IsotopeGroup '{}' has not been fully loaded. Call load_endf() first.",
                    group.inner.name(),
                )));
            }
            for (i, (_iso, ratio)) in group.inner.members().iter().enumerate() {
                let rd = Arc::unwrap_or_clone(group.resonance_data[i].clone().unwrap());
                expanded.push((rd, group_density * ratio));
            }
        }
        expanded
    };
    // Validate the calibrant composition: at least one isotope with a finite,
    // positive areal density (otherwise the calibrant has no resonances to fit
    // the resolution against, and the optimization is degenerate).
    if sample_isotopes.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "calibrant has no isotopes/groups; provide a known composition with densities > 0",
        ));
    }
    if !sample_isotopes
        .iter()
        .any(|(_, d)| d.is_finite() && *d > 0.0)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "calibrant densities must include at least one finite, positive value",
        ));
    }
    let sample = SampleParams::new(temperature_k, sample_isotopes)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let fam = match family {
        "gaussian" => ResolutionFamily::Gaussian,
        "ic" => ResolutionFamily::IkedaCarpenter { fit_psr },
        "udr_corr" => {
            let base = base_udr.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "family='udr_corr' requires base_udr (a TabulatedResolution).",
                )
            })?;
            ResolutionFamily::UdrCorr { base: base.inner }
        }
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown family '{other}'; expected 'gaussian', 'udr_corr', or 'ic'"
            )));
        }
    };

    // fit_psr appends the PSR-FWHM fit coordinate, which only the IC family
    // carries — reject the flag on other families instead of silently ignoring it.
    if fit_psr && !matches!(fam, ResolutionFamily::IkedaCarpenter { .. }) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "fit_psr=True requires family='ic', got family='{family}'"
        )));
    }
    // For family='ic' these size the kernel-synthesis grid; validate up front so an
    // out-of-range value gives a precise error instead of the generic "no finite-χ²
    // resolution" (every IkedaCarpenter::new eval would otherwise fail). They are
    // inert for the gaussian/udr_corr families.
    if matches!(fam, ResolutionFamily::IkedaCarpenter { .. }) {
        if ic_n_energies < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "ic_n_energies must be >= 2 for family='ic', got {ic_n_energies}"
            )));
        }
        if ic_n_tau < 8 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "ic_n_tau must be >= 8 for family='ic', got {ic_n_tau}"
            )));
        }
        // PSR triangle FWHM: finite and >= 0 ns (0 disables the fold). Also
        // enforced by the Rust calibrator; rejected here with the precise
        // Python-facing message.
        if !psr_fwhm_ns.is_finite() || psr_fwhm_ns < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "psr_fwhm_ns must be finite and >= 0 (0 disables the PSR fold), got {psr_fwhm_ns}"
            )));
        }
        // Sanity ceiling (mirrors the Rust calibrator; see
        // PSR_FWHM_PIN_CEILING_US): psr_fwhm_ns is NANOSECONDS and kernel-
        // synthesis cost is quadratic in the fold width, so a µs-as-ns unit
        // slip (350 meaning µs) would hang the calibration for hours behind
        // a fictitious fold. Reject with the precise Python-facing message.
        if psr_fwhm_ns * NS_TO_US > PSR_FWHM_PIN_CEILING_US {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "psr_fwhm_ns={psr_fwhm_ns} ns (= {} us) exceeds the {PSR_FWHM_PIN_CEILING_US} \
                 us sanity ceiling: psr_fwhm_ns is in NANOSECONDS (the SNS/VENUS FTS \
                 convention is 350 ns) and kernel-synthesis cost grows quadratically with \
                 the fold width, so a us-as-ns unit slip would hang the calibration. Pass \
                 the width in ns, or 0 to disable the PSR fold",
                psr_fwhm_ns * NS_TO_US
            )));
        }
        // fit_psr fits the PSR FWHM from the psr_fwhm_ns starting value, but 0
        // means "no fold" — a zero start would be silently clamped into the
        // [0.05, 1] us fit box, contradicting the documented "0 disables".
        // Also enforced by the Rust calibrator.
        if fit_psr && psr_fwhm_ns == 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fit_psr=True requires a positive psr_fwhm_ns starting value (psr_fwhm_ns=0 \
                 disables the PSR fold; use fit_psr=False to calibrate without one)",
            ));
        }
    }

    let cfg = CalibrationConfig {
        flight_path_m,
        fit_background,
        restarts,
        ic_n_energies,
        ic_n_tau,
        psr_fwhm_ns,
        fit_t0,
        fit_l_scale,
        position_t0_center_us: t0_center_us,
        position_l_scale_center: l_scale_center,
        position_t0_prior_us: t0_prior_us,
        position_l_scale_prior: l_scale_prior,
        ..Default::default()
    };
    let (e_owned, d_owned, u_owned) = (e.to_vec(), d.to_vec(), u.to_vec());
    let result = py
        .detach(move || rust_calibrate_resolution(fam, &e_owned, &d_owned, &u_owned, &sample, &cfg))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyResolutionCalibration { inner: result })
}

/// Convert time-of-flight (μs) to energy (eV).
///
/// Args:
///     tof_us: Time-of-flight in microseconds (must be positive and finite).
///     flight_path_m: Flight path in meters (must be positive and finite).
///
/// Returns:
///     Energy in eV.
///
/// Raises:
///     ValueError: If ``tof_us`` or ``flight_path_m`` is non-positive or
///         non-finite. The underlying conversion returns NaN for such input;
///         the binding rejects it explicitly so a bad TOF axis surfaces as an
///         error here rather than silently poisoning a downstream energy grid.
#[pyfunction]
fn tof_to_energy(tof_us: f64, flight_path_m: f64) -> PyResult<f64> {
    if !tof_us.is_finite() || tof_us <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "tof_us must be positive and finite, got {tof_us}"
        )));
    }
    if !flight_path_m.is_finite() || flight_path_m <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "flight_path_m must be positive and finite, got {flight_path_m}"
        )));
    }
    Ok(nereids_core::constants::tof_to_energy(
        tof_us,
        flight_path_m,
    ))
}

/// Convert energy (eV) to time-of-flight (μs).
///
/// Args:
///     energy_ev: Energy in eV (must be positive and finite).
///     flight_path_m: Flight path in meters (must be positive and finite).
///
/// Returns:
///     Time-of-flight in microseconds.
///
/// Raises:
///     ValueError: If ``energy_ev`` or ``flight_path_m`` is non-positive or
///         non-finite (mirrors ``tof_to_energy``).
#[pyfunction]
fn energy_to_tof(energy_ev: f64, flight_path_m: f64) -> PyResult<f64> {
    if !energy_ev.is_finite() || energy_ev <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "energy_ev must be positive and finite, got {energy_ev}"
        )));
    }
    if !flight_path_m.is_finite() || flight_path_m <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "flight_path_m must be positive and finite, got {flight_path_m}"
        )));
    }
    Ok(nereids_core::constants::energy_to_tof(
        energy_ev,
        flight_path_m,
    ))
}

/// Load ENDF resonance data for an isotope from the IAEA database.
///
/// Downloads and parses the ENDF file, caching it locally at
/// ``~/.cache/nereids/endf/`` for subsequent calls.
///
/// Args:
///     z: Atomic number (e.g. 92 for uranium).
///     a: Mass number (e.g. 238).
///     library: ENDF library name. One of "endf8.0", "endf8.1" (default),
///              "jeff3.3", "jendl5", "tendl2023", "cendl3.2".
///     mat: ENDF MAT (material) number. If None, looks up from built-in table
///          (~40 common isotopes). Provide explicitly for uncommon isotopes.
///
/// Returns:
///     ResonanceData parsed from the ENDF file.
#[pyfunction]
#[pyo3(signature = (z, a, library="endf8.1", mat=None))]
fn load_endf(
    py: Python<'_>,
    z: u32,
    a: u32,
    library: &str,
    mat: Option<u32>,
) -> PyResult<PyResonanceData> {
    let lib = parse_library_name(library)?;

    let isotope = Isotope::new(z, a)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid isotope: {}", e)))?;

    let mat_num = match mat {
        Some(m) => m,
        None => mat_number(&isotope, lib).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "MAT number not found for Z={} A={}; provide mat= explicitly",
                z, a
            ))
        })?,
    };

    // Release the GIL for the network I/O (download / cache lookup) and
    // ENDF file parsing.  All types captured by the closure are Send.
    //
    // We tag errors so we can map retrieval failures → PyRuntimeError and
    // parse failures → PyValueError (preserving the pre-GIL-release contract).
    let result: Result<ResonanceData, (bool, String)> =
        py.detach(move || load_and_parse_endf(&isotope, lib, mat_num));

    let data = result.map_err(|(is_parse, msg)| {
        if is_parse {
            pyo3::exceptions::PyValueError::new_err(msg)
        } else {
            pyo3::exceptions::PyRuntimeError::new_err(msg)
        }
    })?;

    // Validate that the parsed ENDF data matches the requested isotope.
    if data.isotope.z() != z || data.isotope.a() != a {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "ENDF data mismatch: requested Z={} A={} but file contains Z={} A={}",
            z,
            a,
            data.isotope.z(),
            data.isotope.a()
        )));
    }

    Ok(PyResonanceData {
        inner: Arc::new(data),
    })
}

/// Load ENDF resonance data from a local file.
///
/// Args:
///     path: Path to an ENDF-format file on disk.
///
/// Returns:
///     ResonanceData parsed from the file.
#[pyfunction]
fn load_endf_file(py: Python<'_>, path: &str) -> PyResult<PyResonanceData> {
    // Release the GIL for the file I/O and ENDF parsing.
    // Tag errors: false = I/O, true = parse.
    let owned_path = path.to_owned();
    let result: Result<ResonanceData, (bool, String)> = py.detach(move || {
        let contents = std::fs::read_to_string(&owned_path)
            .map_err(|e| (false, format!("Cannot read '{}': {}", owned_path, e)))?;

        let data =
            parse_endf_file2(&contents).map_err(|e| (true, format!("ENDF parse error: {}", e)))?;

        Ok(data)
    });

    let data = result.map_err(|(is_parse, msg)| {
        if is_parse {
            pyo3::exceptions::PyValueError::new_err(msg)
        } else {
            pyo3::exceptions::PyIOError::new_err(msg)
        }
    })?;

    Ok(PyResonanceData {
        inner: Arc::new(data),
    })
}

/// Create ResonanceData from parameters (for testing/custom isotopes).
///
/// Args:
///     z: Atomic number.
///     a: Mass number.
///     awr: Atomic weight ratio.
///     scattering_radius: Scattering radius in fm.
///     resonances: List of (energy_eV, j, gn, gg) tuples for L=0.
///     target_spin: Target nuclear spin (default 0.0).
///     l_groups: Optional list of (l_value, [(energy, j, gn, gg), ...]) tuples
///               for multiple L-groups. If provided, the ``resonances`` parameter
///               is ignored.
///     formalism: Resonance formalism to use. Accepted values:
///                - ``None`` or ``"reich_moore"`` (also ``"ReichMoore"``, ``"rm"``,
///                  ``"RM"``, ``"reich-moore"``) — Reich-Moore R-matrix (default).
///                - ``"slbw"`` or ``"SLBW"`` — Single-Level Breit-Wigner.
///
/// Returns:
///     ResonanceData object.
#[pyfunction]
#[pyo3(signature = (z, a, awr, scattering_radius, resonances, target_spin=0.0, l_groups=None, formalism=None))]
fn create_resonance_data(
    z: u32,
    a: u32,
    awr: f64,
    scattering_radius: f64,
    resonances: Vec<(f64, f64, f64, f64)>,
    target_spin: f64,
    l_groups: Option<Vec<(u32, Vec<(f64, f64, f64, f64)>)>>,
    formalism: Option<&str>,
) -> PyResult<PyResonanceData> {
    let res_formalism = match formalism {
        Some("slbw" | "SLBW") => ResonanceFormalism::SLBW,
        Some("reich_moore" | "ReichMoore" | "reich-moore" | "rm" | "RM") | None => {
            ResonanceFormalism::ReichMoore
        }
        Some(other) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown formalism '{other}'. Use 'slbw' or 'reich_moore'."
            )));
        }
    };
    let groups = match l_groups {
        Some(lg) => lg
            .into_iter()
            .map(|(l_val, res_list)| LGroup {
                l: l_val,
                awr,
                apl: 0.0,
                qx: 0.0,
                lrx: 0,
                resonances: res_list
                    .into_iter()
                    .map(|(energy, j, gn, gg)| Resonance {
                        energy,
                        j,
                        gn,
                        gg,
                        gfa: 0.0,
                        gfb: 0.0,
                    })
                    .collect(),
            })
            .collect(),
        None => {
            let res: Vec<Resonance> = resonances
                .into_iter()
                .map(|(energy, j, gn, gg)| Resonance {
                    energy,
                    j,
                    gn,
                    gg,
                    gfa: 0.0,
                    gfb: 0.0,
                })
                .collect();
            vec![LGroup {
                l: 0,
                awr,
                apl: 0.0,
                qx: 0.0,
                lrx: 0,
                resonances: res,
            }]
        }
    };

    Ok(PyResonanceData {
        inner: Arc::new(ResonanceData {
            isotope: Isotope::new(z, a).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Invalid isotope: {}", e))
            })?,
            za: z * 1000 + a,
            awr,
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high: 1e6,
                resolved: true,
                formalism: res_formalism,
                target_spin,
                scattering_radius,
                naps: 1,
                l_groups: groups,
                rml: None,
                urr: None,
                ap_table: None,
                r_external: vec![],
            }],
        }),
    })
}

/// Beer-Lambert transmission: T = exp(-thickness * sigma).
///
/// Args:
///     cross_sections: Cross-sections in barns (1D numpy array).
///     thickness: Areal density in atoms/barn.
///
/// Returns:
///     1D numpy array of transmission values.
#[pyfunction]
fn beer_lambert<'py>(
    py: Python<'py>,
    cross_sections: PyReadonlyArray1<f64>,
    thickness: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = cross_sections.as_slice()?;
    let t = transmission::beer_lambert(xs, thickness);
    Ok(PyArray1::from_vec(py, t))
}

/// Validate that an energy grid is finite, positive, and sorted ascending.
/// Empty grids are accepted (callers that need non-empty should use
/// `require_non_empty_energy_grid` instead).
fn validate_energy_grid(e: &[f64]) -> PyResult<()> {
    if e.is_empty() {
        return Ok(());
    }
    if !e[0].is_finite() || e[0] <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "energies must be finite and positive",
        ));
    }
    for i in 1..e.len() {
        if !e[i].is_finite() || e[i] <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "energies must be finite and positive",
            ));
        }
        if e[i] <= e[i - 1] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "energies must be sorted in strictly ascending order",
            ));
        }
    }
    Ok(())
}

/// Validate that an energy grid is **non-empty**, finite, positive, and sorted.
fn require_non_empty_energy_grid(e: &[f64]) -> PyResult<()> {
    if e.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "energies must not be empty",
        ));
    }
    validate_energy_grid(e)
}

/// Build a `ResolutionFunction` from Python arguments.
///
/// Validates mutual exclusivity (Gaussian vs. tabulated) and completeness
/// of Gaussian parameters. Returns `None` when no resolution is requested.
fn build_resolution(
    flight_path_m: Option<f64>,
    delta_t_us: Option<f64>,
    delta_l_m: Option<f64>,
    resolution: Option<PyTabulatedResolution>,
    delta_e_us: Option<f64>,
) -> PyResult<Option<ResolutionFunction>> {
    let has_gaussian = flight_path_m.is_some() || delta_t_us.is_some() || delta_l_m.is_some();
    if has_gaussian && resolution.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot specify both Gaussian resolution parameters and tabulated resolution",
        ));
    }
    let all_gaussian = flight_path_m.is_some() && delta_t_us.is_some() && delta_l_m.is_some();
    if has_gaussian && !all_gaussian {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Gaussian resolution requires all three parameters: flight_path_m, delta_t_us, and delta_l_m",
        ));
    }
    if let Some(tab) = resolution {
        Ok(Some(ResolutionFunction::Tabulated(tab.inner)))
    } else if let (Some(fp), Some(dt), Some(dl)) = (flight_path_m, delta_t_us, delta_l_m) {
        let de = delta_e_us.unwrap_or(0.0);
        let rp = ResolutionParams::new(fp, dt, dl, de)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Some(ResolutionFunction::Gaussian(rp)))
    } else {
        Ok(None)
    }
}

/// Extract a detector-time response that has exact bin probabilities.
fn extract_detector_time_resolution(resolution: &Bound<'_, PyAny>) -> PyResult<ResolutionFunction> {
    if let Ok(tabulated) = resolution.extract::<PyRef<'_, PyTabulatedResolution>>() {
        Ok(ResolutionFunction::Tabulated(Arc::clone(&tabulated.inner)))
    } else if let Ok(ic) = resolution.extract::<PyRef<'_, PyIkedaCarpenter>>() {
        Ok(ResolutionFunction::IkedaCarpenter(Arc::clone(&ic.inner)))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "resolution must be a TabulatedResolution or IkedaCarpenter",
        ))
    }
}

/// Apply Free Gas Model (FGM) Doppler broadening to a cross-section array.
///
/// Convolves the input cross-sections with a Gaussian kernel whose width
/// depends on the sample temperature and atomic weight ratio. This is the
/// same broadening applied internally by `forward_model()`, but exposed here
/// so users can broaden individual components (capture, elastic, fission)
/// independently.
///
/// Args:
///     energies: Energy grid in eV (1D numpy array, sorted ascending).
///     cross_sections: Cross-sections in barns (1D numpy array, same length).
///     awr: Atomic weight ratio (target mass / neutron mass).
///     temperature_k: Sample temperature in Kelvin.
///
/// Returns:
///     1D numpy array of Doppler-broadened cross-sections in barns.
///
/// Reference:
///     SAMMY Manual Section III.B.1 (Free-Gas Model of Doppler Broadening).
///     Exact FGM kernel: Eq. III B1.7 with the w²-weighted integrand —
///     the same weighting as SAMMY's Dopfgm (the numerical quadrature
///     differs).
///
/// Edge behavior:
///     Near both grid edges, sigma beyond the supplied grid is
///     extrapolated by the 1/v law; edge points whose Doppler window is
///     both grid-truncated and under-resolved (fewer than 3 nodes) are
///     returned unbroadened, matching SAMMY.
#[pyfunction]
#[pyo3(signature = (energies, cross_sections, awr, temperature_k))]
fn doppler_broaden<'py>(
    py: Python<'py>,
    energies: PyReadonlyArray1<f64>,
    cross_sections: PyReadonlyArray1<f64>,
    awr: f64,
    temperature_k: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let e = energies.as_slice()?;
    let xs = cross_sections.as_slice()?;

    if e.len() != xs.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "energies length ({}) must match cross_sections length ({})",
            e.len(),
            xs.len(),
        )));
    }
    validate_energy_grid(e)?;

    if temperature_k == 0.0 {
        return Ok(PyArray1::from_vec(py, xs.to_vec()));
    }

    let params = DopplerParams::new(temperature_k, awr).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("invalid DopplerParams: {e}"))
    })?;

    // Copy numpy slices to owned vectors so we can release the GIL.
    let e_owned = e.to_vec();
    let xs_owned = xs.to_vec();

    // Release the GIL for the Doppler broadening convolution.
    let result = py.detach(move || {
        doppler::doppler_broaden(&e_owned, &xs_owned, &params)
            .map_err(|e| format!("doppler_broaden failed: {e}"))
    });
    let result = result.map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    Ok(PyArray1::from_vec(py, result))
}

/// Apply Gaussian resolution broadening to a cross-section or spectrum array.
///
/// Convolves the input with an energy-dependent Gaussian kernel derived from
/// the instrument's timing uncertainty and flight path length uncertainty.
/// This is the same broadening applied internally by `forward_model()`, but
/// exposed here for independent use on arbitrary arrays.
///
/// Args:
///     energies: Energy grid in eV (1D numpy array, sorted ascending).
///     cross_sections: Values to broaden (1D numpy array, same length).
///     flight_path_m: Flight path length in meters (source to detector).
///     delta_t_us: Total timing uncertainty (1σ Gaussian) in microseconds.
///     delta_l_m: Flight path uncertainty (1σ Gaussian) in meters.
///
/// Returns:
///     1D numpy array of resolution-broadened values.
///
/// Reference:
///     SAMMY Manual Section 3.2 (Resolution Broadening).
#[pyfunction]
#[pyo3(signature = (energies, cross_sections, flight_path_m, delta_t_us, delta_l_m, delta_e_us=0.0))]
fn resolution_broaden<'py>(
    py: Python<'py>,
    energies: PyReadonlyArray1<f64>,
    cross_sections: PyReadonlyArray1<f64>,
    flight_path_m: f64,
    delta_t_us: f64,
    delta_l_m: f64,
    delta_e_us: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let e = energies.as_slice()?;
    let xs = cross_sections.as_slice()?;

    if e.len() != xs.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "energies length ({}) must match cross_sections length ({})",
            e.len(),
            xs.len(),
        )));
    }
    validate_energy_grid(e)?;
    let params = ResolutionParams::new(flight_path_m, delta_t_us, delta_l_m, delta_e_us)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    if delta_t_us == 0.0 && delta_l_m == 0.0 {
        return Ok(PyArray1::from_vec(py, xs.to_vec()));
    }

    // Copy numpy slices to owned vectors so we can release the GIL.
    let e_owned = e.to_vec();
    let xs_owned = xs.to_vec();

    // Release the GIL for the resolution broadening convolution.
    let result = py.detach(move || resolution::resolution_broaden(&e_owned, &xs_owned, &params));
    let result = result.map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    Ok(PyArray1::from_vec(py, result))
}

/// Load a tabulated resolution function from a VENUS/FTS-format file.
///
/// The file contains reference kernels R(Δt; E_ref) at discrete energies,
/// stored as (TOF_offset_μs, weight) pairs. Kernels are interpolated between
/// reference energies and converted from TOF to energy space during broadening.
///
/// Args:
///     path: Path to the resolution file.
///     flight_path_m: Flight path length in meters (source to detector).
///
/// Returns:
///     TabulatedResolution object for use with ``forward_model()`` or
///     ``fit_spectrum()``.
#[pyfunction]
fn load_resolution(path: &str, flight_path_m: f64) -> PyResult<PyTabulatedResolution> {
    if !flight_path_m.is_finite() || flight_path_m <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "flight_path_m must be finite and positive",
        ));
    }

    let tab = TabulatedResolution::from_file(path, flight_path_m)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;

    Ok(PyTabulatedResolution {
        inner: Arc::new(tab),
    })
}

/// Apply tabulated resolution broadening to a spectrum.
///
/// Convolves the input spectrum with the tabulated instrument resolution
/// function. For each energy point, the kernel is interpolated between
/// reference energies and converted from TOF-offset space to energy space.
///
/// Args:
///     energies: Energy grid in eV (1D numpy array, sorted ascending).
///     spectrum: Values to broaden (1D numpy array, same length).
///     resolution: TabulatedResolution from ``load_resolution()``.
///
/// Returns:
///     1D numpy array of resolution-broadened values.
#[pyfunction]
#[pyo3(name = "apply_resolution")]
fn py_apply_resolution<'py>(
    py: Python<'py>,
    energies: PyReadonlyArray1<f64>,
    spectrum: PyReadonlyArray1<f64>,
    resolution: PyTabulatedResolution,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let e = energies.as_slice()?;
    let s = spectrum.as_slice()?;

    if e.len() != s.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "energies length ({}) must match spectrum length ({})",
            e.len(),
            s.len(),
        )));
    }
    validate_energy_grid(e)?;

    let res_fn = ResolutionFunction::Tabulated(resolution.inner);

    // Copy numpy slices to owned vectors so we can release the GIL.
    let e_owned = e.to_vec();
    let s_owned = s.to_vec();

    // Release the GIL for the tabulated resolution broadening.
    let result = py.detach(move || resolution::apply_resolution(&e_owned, &s_owned, &res_fn));
    let result = result.map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    Ok(PyArray1::from_vec(py, result))
}

/// Predict separate open-beam and sample counts on actual detector-time bins.
///
/// ``incident_fluence_weights[j]`` is the incident flux density at true energy
/// ``j`` multiplied by the caller's energy-integration weight. The response is
/// applied to the open and attenuated sample arms separately; this function
/// never broadens a transmission ratio.
#[pyfunction]
#[pyo3(name = "two_arm_count_response", signature = (
    true_energies_ev,
    incident_fluence_weights,
    transmission,
    detector_time_edges_us,
    resolution,
    timing_offset_us = 0.0,
))]
fn py_two_arm_count_response<'py>(
    py: Python<'py>,
    true_energies_ev: PyReadonlyArray1<f64>,
    incident_fluence_weights: PyReadonlyArray1<f64>,
    transmission: PyReadonlyArray1<f64>,
    detector_time_edges_us: PyReadonlyArray1<f64>,
    resolution: &Bound<'_, PyAny>,
    timing_offset_us: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let response = extract_detector_time_resolution(resolution)?;

    let energies = true_energies_ev.as_slice()?.to_vec();
    let fluence = incident_fluence_weights.as_slice()?.to_vec();
    let transmission = transmission.as_slice()?.to_vec();
    let detector_edges = detector_time_edges_us.as_slice()?.to_vec();
    let result = py.detach(move || {
        counts_response::two_arm_count_response(
            &energies,
            &fluence,
            &transmission,
            &detector_edges,
            timing_offset_us,
            &response,
        )
    });
    let result = result.map_err(|error| {
        pyo3::exceptions::PyValueError::new_err(format!("two-arm count response: {error}"))
    })?;
    Ok((
        PyArray1::from_vec(py, result.open_beam),
        PyArray1::from_vec(py, result.sample),
    ))
}

/// Python result for a fixed-signal, measured-template count-background fit.
#[pyclass(name = "TwoArmBackgroundFitResult")]
struct PyTwoArmBackgroundFitResult {
    names: Vec<String>,
    amplitudes: Vec<f64>,
    amplitude_uncertainties: Option<Vec<f64>>,
    amplitudes_identifiable: bool,
    open_neutron_signal: Vec<f64>,
    open_background: Vec<f64>,
    open_total: Vec<f64>,
    sample_neutron_signal: Vec<f64>,
    sample_background: Vec<f64>,
    sample_total: Vec<f64>,
    poisson_deviance: f64,
    deviance_per_dof: f64,
    converged: bool,
    iterations: usize,
}

#[pymethods]
impl PyTwoArmBackgroundFitResult {
    #[getter]
    fn names(&self) -> Vec<String> {
        self.names.clone()
    }

    #[getter]
    fn amplitudes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.amplitudes.clone())
    }

    #[getter]
    fn amplitude_uncertainties<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(
            py,
            self.amplitude_uncertainties
                .clone()
                .unwrap_or_else(|| vec![f64::NAN; self.amplitudes.len()]),
        )
    }

    #[getter]
    fn amplitudes_identifiable(&self) -> bool {
        self.amplitudes_identifiable
    }

    #[getter]
    fn open_neutron_signal<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.open_neutron_signal.clone())
    }

    #[getter]
    fn open_background<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.open_background.clone())
    }

    #[getter]
    fn open_total<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.open_total.clone())
    }

    #[getter]
    fn sample_neutron_signal<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.sample_neutron_signal.clone())
    }

    #[getter]
    fn sample_background<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.sample_background.clone())
    }

    #[getter]
    fn sample_total<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.sample_total.clone())
    }

    #[getter]
    fn poisson_deviance(&self) -> f64 {
        self.poisson_deviance
    }

    #[getter]
    fn deviance_per_dof(&self) -> f64 {
        self.deviance_per_dof
    }

    #[getter]
    fn converged(&self) -> bool {
        self.converged
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.iterations
    }
}

/// Fit non-negative amplitudes for independently measured count backgrounds.
///
/// Each row of the two template matrices is one named component. Templates
/// must already be normalized into the detector bins of the corresponding
/// complete acquisition. Only amplitudes are fitted; shapes and neutron
/// signals remain fixed. The two required exposure scales convert the common
/// reference signal into expected counts for each complete acquisition.
#[pyfunction]
#[pyo3(signature = (
    observed_open_counts,
    observed_sample_counts,
    open_neutron_signal,
    sample_neutron_signal,
    open_exposure_scale,
    sample_exposure_scale,
    template_names,
    open_background_templates,
    sample_background_templates,
    initial_amplitudes,
    max_iter = 200,
))]
#[allow(clippy::too_many_arguments)]
fn fit_two_arm_background_templates<'py>(
    py: Python<'py>,
    observed_open_counts: PyReadonlyArray1<'py, f64>,
    observed_sample_counts: PyReadonlyArray1<'py, f64>,
    open_neutron_signal: PyReadonlyArray1<'py, f64>,
    sample_neutron_signal: PyReadonlyArray1<'py, f64>,
    open_exposure_scale: f64,
    sample_exposure_scale: f64,
    template_names: Vec<String>,
    open_background_templates: PyReadonlyArray2<'py, f64>,
    sample_background_templates: PyReadonlyArray2<'py, f64>,
    initial_amplitudes: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
) -> PyResult<PyTwoArmBackgroundFitResult> {
    use nereids_fitting::count_background::{
        TwoArmBackgroundTemplate, fit_two_arm_background_templates as rust_fit_background,
    };
    use nereids_fitting::poisson::PoissonConfig;
    use nereids_physics::counts_response::TwoArmCounts;

    let open_template_array = open_background_templates.as_array();
    let sample_template_array = sample_background_templates.as_array();
    let open_shape = open_template_array.shape();
    let sample_shape = sample_template_array.shape();
    if open_shape != sample_shape {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "open_background_templates shape {:?} must match sample_background_templates shape {:?}",
            open_shape, sample_shape
        )));
    }
    if template_names.len() != open_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "template_names length ({}) must match template rows ({})",
            template_names.len(),
            open_shape[0]
        )));
    }
    let initial = initial_amplitudes.as_slice()?.to_vec();
    if initial.len() != open_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "initial_amplitudes length ({}) must match template rows ({})",
            initial.len(),
            open_shape[0]
        )));
    }

    let templates: Vec<TwoArmBackgroundTemplate> = template_names
        .into_iter()
        .zip(open_template_array.outer_iter())
        .zip(sample_template_array.outer_iter())
        .map(|((name, open_beam), sample)| TwoArmBackgroundTemplate {
            name,
            open_beam: open_beam.to_vec(),
            sample: sample.to_vec(),
        })
        .collect();
    let observed = TwoArmCounts {
        open_beam: observed_open_counts.as_slice()?.to_vec(),
        sample: observed_sample_counts.as_slice()?.to_vec(),
    };
    let signal = TwoArmCounts {
        open_beam: open_neutron_signal.as_slice()?.to_vec(),
        sample: sample_neutron_signal.as_slice()?.to_vec(),
    };
    let config = PoissonConfig {
        max_iter,
        ..PoissonConfig::default()
    };
    let result = py.detach(move || {
        rust_fit_background(
            &observed,
            signal,
            open_exposure_scale,
            sample_exposure_scale,
            &templates,
            &initial,
            &config,
        )
    });
    let result = result.map_err(|error| match error {
        nereids_fitting::error::FittingError::EvaluationFailed(_) => {
            pyo3::exceptions::PyRuntimeError::new_err(error.to_string())
        }
        _ => pyo3::exceptions::PyValueError::new_err(error.to_string()),
    })?;

    Ok(PyTwoArmBackgroundFitResult {
        names: result.names,
        amplitudes: result.amplitudes,
        amplitude_uncertainties: result.amplitude_uncertainties,
        amplitudes_identifiable: result.amplitudes_identifiable,
        open_neutron_signal: result.prediction.open_beam.neutron_signal,
        open_background: result.prediction.open_beam.background,
        open_total: result.prediction.open_beam.total,
        sample_neutron_signal: result.prediction.sample.neutron_signal,
        sample_background: result.prediction.sample.background,
        sample_total: result.prediction.sample.total,
        poisson_deviance: result.poisson_deviance,
        deviance_per_dof: result.deviance_per_dof,
        converged: result.converged,
        iterations: result.iterations,
    })
}

/// Parse a Python-facing pixel-value policy string.
///
/// Accepted values: ``"reject"`` (default — negative or non-finite pixels
/// are an error), ``"clip"`` (clamp negatives to 0.0; NaN still errors),
/// ``"allow"`` (verbatim pass-through for pre-normalized transmission).
fn parse_pixel_policy(policy: &str) -> PyResult<nereids_io::tiff_stack::PixelValuePolicy> {
    use nereids_io::tiff_stack::PixelValuePolicy;
    match policy {
        "reject" => Ok(PixelValuePolicy::Reject),
        "clip" => Ok(PixelValuePolicy::ClipToZero),
        "allow" => Ok(PixelValuePolicy::Allow),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid pixel_policy '{}': expected \"reject\", \"clip\", or \"allow\"",
            other
        ))),
    }
}

/// True when `e` is an [`IoError::FileNotFound`] whose underlying OS error
/// kind is genuinely `NotFound`.
///
/// `IoError::FileNotFound` wraps *any* `std::io::Error` raised while
/// opening a path (permission denied, `InvalidInput`, ...), so mapping the
/// variant unconditionally to Python ``FileNotFoundError`` would mislabel
/// e.g. a permission-denied file as missing.  Callers use this guard and
/// fall through to ``OSError`` for every other kind.
///
/// [`IoError::FileNotFound`]: nereids_io::error::IoError::FileNotFound
fn is_genuine_not_found(e: &nereids_io::error::IoError) -> bool {
    matches!(
        e,
        nereids_io::error::IoError::FileNotFound(_, source)
            if source.kind() == std::io::ErrorKind::NotFound
    )
}

/// Emit Python ``UserWarning``s for semantically significant TIFF-load
/// events recorded in a [`nereids_io::tiff_stack::TiffLoadInfo`].
///
/// Python callers otherwise have zero observability of chunk summing (a
/// semantic change versus the old concatenate behavior: the returned stack
/// is the element-wise *sum* of k DAQ chunks), of a *mixed* folder whose
/// non-conforming files disabled chunk detection (silently reinstating the
/// legacy concatenated load — potentially a k× stack), and of
/// negative-pixel clipping under ``pixel_policy="clip"`` — all alter the
/// data relative to the caller's expectation, so they are surfaced as
/// warnings the caller can catch, filter, or escalate with the stdlib
/// ``warnings`` module (mirroring the GUI's provenance-log entries for the
/// same events).
///
/// ``stacklevel=1`` attributes the warning to the frame that invoked this
/// pyo3 function — which *is* the Python call site: extension functions
/// execute inside the caller's Python frame without pushing a frame of
/// their own, so there is no "extension frame" to skip.
fn emit_tiff_load_warnings(
    py: Python<'_>,
    info: &nereids_io::tiff_stack::TiffLoadInfo,
) -> PyResult<()> {
    /// ``warnings.warn`` stacklevel.  pyo3 ``#[pyfunction]``s run in the
    /// *caller's* Python frame (no extension frame exists), so level 1 is
    /// already the Python call site; 2 would blame the caller's caller
    /// (empirically: pytest internals when called from the test suite).
    const WARN_STACKLEVEL: i32 = 1;
    let mut messages: Vec<String> = Vec::new();
    if info.chunks_summed {
        let ids: Vec<String> = info.chunk_ids.iter().map(|id| id.to_string()).collect();
        messages.push(format!(
            "summed {} DAQ chunks ({}) element-wise into one stack; \
             pass sum_chunks=False to load the raw per-file stack instead",
            info.n_chunks,
            ids.join(", "),
        ));
    }
    if info.n_unrecognized_files > 0 {
        messages.push(format!(
            "{} file(s) did not match the chunk naming pattern (e.g. {}); \
             chunk detection disabled, frames loaded in lexicographic order \
             — check for stray TIFFs or pass pattern=... to exclude them",
            info.n_unrecognized_files,
            info.unrecognized_examples.join(", "),
        ));
    }
    if info.chunk_inconsistent {
        // Chunk-patterned files that are internally inconsistent (ragged
        // frame counts/sets or a duplicate (chunk, frame) pair).  With
        // sum_chunks=True this is a ValueError; the caller passed
        // sum_chunks=False, so the files were concatenated lexicographically
        // verbatim instead of raising — surface that so a ragged folder that
        // would otherwise error is not silently accepted.
        messages.push(
            "DAQ chunks are internally inconsistent (ragged frame counts/sets \
             or duplicate (chunk, frame) pairs); loaded as the lexicographic \
             concatenation of all files because sum_chunks=False — pass \
             sum_chunks=True to make this a hard error instead"
                .to_string(),
        );
    }
    if info.n_clipped_pixels > 0 {
        messages.push(format!(
            "{} negative pixel(s) clipped to 0.0 under pixel_policy=\"clip\"",
            info.n_clipped_pixels,
        ));
    }
    let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
    for msg in messages {
        // The messages above are built from numeric fields and contain no
        // NUL bytes, but map the error instead of unwrapping on principle.
        let c_msg = std::ffi::CString::new(msg).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "internal TIFF-load warning message contained a NUL byte: {e}"
            ))
        })?;
        PyErr::warn(py, category.as_any(), &c_msg, WARN_STACKLEVEL)?;
    }
    Ok(())
}

/// Load a multi-frame TIFF file into a 3D numpy array.
///
/// Each TIFF frame becomes one slice along the first axis.
/// Data is converted to float64 regardless of the source pixel type.
///
/// Args:
///     path: Path to the multi-frame TIFF file.
///     pixel_policy: ``"reject"`` (default) errors on negative or
///         non-finite pixels — raw counts are non-negative by
///         construction, so a violation signals corruption (mask corrupt
///         readout pixels per acquisition with ``detect_bad_pixels()``
///         instead).  ``"clip"`` clamps negatives to 0.0 (NaN still
///         errors).  ``"allow"`` passes all values through verbatim —
///         use for pre-normalized transmission stacks.
///
/// Returns:
///     3D numpy array with shape (n_frames, height, width).
///
/// Warns:
///     UserWarning: When ``pixel_policy="clip"`` clamped one or more
///         negative pixels (the message reports the count).
///
/// Raises:
///     FileNotFoundError: If the file does not exist.
///     ValueError: For bad pixel values under the active policy, or an
///         invalid ``pixel_policy`` string.
///     IOError: For TIFF decoding errors or other I/O failures
///         (e.g. permission denied).
#[pyfunction]
#[pyo3(signature = (path, pixel_policy="reject"))]
fn load_tiff_stack<'py>(
    py: Python<'py>,
    path: &str,
    pixel_policy: &str,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let policy = parse_pixel_policy(pixel_policy)?;
    let (arr, info) =
        nereids_io::tiff_stack::load_tiff_stack_with_options(std::path::Path::new(path), policy)
            .map_err(|e| match &e {
                nereids_io::error::IoError::BadPixelValue { .. } => {
                    pyo3::exceptions::PyValueError::new_err(format!("{}", e))
                }
                err if is_genuine_not_found(err) => {
                    pyo3::exceptions::PyFileNotFoundError::new_err(format!("{}", e))
                }
                _ => pyo3::exceptions::PyIOError::new_err(format!("{}", e)),
            })?;
    emit_tiff_load_warnings(py, &info)?;
    Ok(PyArray3::from_owned_array(py, arr))
}

/// Load a folder of single-frame TIFFs into a 3D numpy array.
///
/// Chunked VENUS folders (files named ``<prefix>_<chunk>_<frame>.tif``)
/// are detected automatically: when every filename follows the convention
/// with one common prefix, frames are ordered by numeric frame index and
/// chunks covering identical frame ranges are summed element-wise
/// (``sum_chunks=True``, the default).  Ragged chunks or duplicate
/// (chunk, frame) pairs are an error under the default summing path — never
/// a silent stack; with ``sum_chunks=False`` they are instead loaded as the
/// lexicographic concatenation (there is nothing to corrupt when not
/// summing) and flagged via ``chunk_inconsistent`` (see below).  Folders not
/// following the convention (or with several distinct prefixes) load in
/// lexicographic filename order, so name legacy files with zero-padded
/// indices (e.g., ``frame_0001.tif``, ``frame_0002.tif``, ...).  A *mixed*
/// folder — at least one chunk-patterned name alongside files that do not
/// match (a stray overview TIFF, a misnamed frame) — also loads
/// lexicographically, with a ``UserWarning`` counting the non-conforming
/// files (naming up to three).
///
/// The chunk heuristic assumes **one acquisition per folder** (the VENUS
/// autoreduce layout: one run folder per directory).  It cannot
/// distinguish same-prefix sibling *runs* co-located in one folder from
/// DAQ chunks — they would be summed.  Use ``pattern`` to select one run,
/// or ``sum_chunks=False``, when a folder may hold multiple runs.
///
/// Args:
///     folder: Path to the directory containing TIFF files.
///     pattern: Optional glob pattern matched against each filename (not the
///              full path).  Supports ``*`` and ``?`` wildcards
///              (case-insensitive).  Only files with ``.tif`` or ``.tiff``
///              extensions are ever loaded; the pattern adds an additional
///              filename filter on top of that.
///     sum_chunks: Sum DAQ chunks element-wise when a chunked folder is
///                 detected (default ``True``).  ``False`` loads the legacy
///                 lexicographic concatenation of all files.  The flag only
///                 affects folders with **two or more** chunks: a
///                 *consistent* single-chunk (or non-chunk-patterned)
///                 folder loads identically either way — chunk-patterned
///                 names in numeric frame order, others lexicographically.
///                 It also
///                 decides how *inconsistent* chunks are handled: ragged or
///                 duplicated chunks raise ``ValueError`` with
///                 ``sum_chunks=True`` but, with ``sum_chunks=False``, load
///                 as the lexicographic concatenation and set
///                 ``chunk_inconsistent`` (a ``UserWarning`` is emitted).
///     pixel_policy: ``"reject"`` (default) errors on negative or
///         non-finite pixels; ``"clip"`` clamps negatives to 0.0 (NaN
///         still errors); ``"allow"`` passes values through verbatim (for
///         pre-normalized transmission stacks).
///     return_info: Keyword-only.  When ``True``, return ``(array, info)``
///         where ``info`` is a dict with keys ``n_files``, ``n_chunks``,
///         ``chunk_ids``, ``chunks_summed``, ``n_clipped_pixels``,
///         ``n_unrecognized_files`` (files that broke chunk detection in a
///         mixed folder; 0 otherwise), ``unrecognized_examples`` (up to
///         three of their names), and ``chunk_inconsistent`` (``True`` when
///         inconsistent chunks were concatenated under ``sum_chunks=False``
///         instead of raising) (default ``False`` — return just the array).
///
/// Returns:
///     3D numpy array with shape (n_frames, height, width), dtype float64;
///     or an ``(array, info)`` tuple when ``return_info=True``.
///
/// Warns:
///     UserWarning: When chunks were summed element-wise (the message
///         names the chunk count and ids and the ``sum_chunks=False``
///         escape hatch), when a mixed folder disabled chunk detection
///         (the message counts the non-conforming files and names up to
///         three), when inconsistent chunks were concatenated under
///         ``sum_chunks=False`` instead of raising, and when
///         ``pixel_policy="clip"`` clamped one or more negative pixels (the
///         message reports the count).
///
/// Raises:
///     FileNotFoundError: If the folder does not exist, or no files match
///         the pattern.
///     NotADirectoryError: If the provided path exists but is not a
///         directory.
///     ValueError: If matched frames have inconsistent dimensions, the
///         chunked layout is internally inconsistent *and*
///         ``sum_chunks=True`` (with ``sum_chunks=False`` an inconsistent
///         layout is concatenated and flagged, not raised), a pixel value
///         violates the active policy, or ``pixel_policy`` is invalid.
///     IOError: For TIFF decoding errors or other I/O failures.
#[pyfunction]
#[pyo3(signature = (folder, pattern=None, sum_chunks=true, pixel_policy="reject", *, return_info=false))]
fn load_tiff_folder<'py>(
    py: Python<'py>,
    folder: &str,
    pattern: Option<&str>,
    sum_chunks: bool,
    pixel_policy: &str,
    return_info: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let options = nereids_io::tiff_stack::TiffFolderOptions {
        sum_chunks,
        pixel_policy: parse_pixel_policy(pixel_policy)?,
    };
    let (arr, info) = nereids_io::tiff_stack::load_tiff_folder_with_options(
        std::path::Path::new(folder),
        pattern,
        &options,
    )
    .map_err(|e| match &e {
        nereids_io::error::IoError::NoMatchingFiles { .. } => {
            pyo3::exceptions::PyFileNotFoundError::new_err(format!("{}", e))
        }
        nereids_io::error::IoError::NotADirectory(_) => {
            pyo3::exceptions::PyNotADirectoryError::new_err(format!("{}", e))
        }
        err if is_genuine_not_found(err) => {
            pyo3::exceptions::PyFileNotFoundError::new_err(format!("{}", e))
        }
        nereids_io::error::IoError::DimensionMismatch { .. } => {
            pyo3::exceptions::PyValueError::new_err(format!("{}", e))
        }
        nereids_io::error::IoError::ChunkMismatch { .. } => {
            pyo3::exceptions::PyValueError::new_err(format!("{}", e))
        }
        nereids_io::error::IoError::BadPixelValue { .. } => {
            pyo3::exceptions::PyValueError::new_err(format!("{}", e))
        }
        _ => pyo3::exceptions::PyIOError::new_err(format!("{}", e)),
    })?;
    emit_tiff_load_warnings(py, &info)?;
    let arr = PyArray3::from_owned_array(py, arr);
    if return_info {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("n_files", info.n_files)?;
        d.set_item("n_chunks", info.n_chunks)?;
        d.set_item("chunk_ids", info.chunk_ids)?;
        d.set_item("chunks_summed", info.chunks_summed)?;
        d.set_item("n_clipped_pixels", info.n_clipped_pixels)?;
        d.set_item("n_unrecognized_files", info.n_unrecognized_files)?;
        d.set_item("unrecognized_examples", info.unrecognized_examples)?;
        d.set_item("chunk_inconsistent", info.chunk_inconsistent)?;
        Ok((arr, d).into_pyobject(py)?.into_any())
    } else {
        Ok(arr.into_any())
    }
}

/// Read a VENUS ``*_Spectra.txt`` TOF sidecar into bin edges (µs).
///
/// The sidecar's first CSV column is each frame's start time in SECONDS
/// (one row per TOF frame; the second column is counts).  Values are
/// converted to microseconds and the closing edge of the last frame is
/// synthesized by extrapolating the last frame width, yielding N+1
/// ascending edges for N rows — exactly what ``tof_to_energy_centers``
/// expects.  Bin uniformity is not enforced (MCP shutter segments change
/// the frame width mid-run).
///
/// The start-time = left-bin-edge semantics is verified on measured
/// VENUS autoreduce output: every ``shutter_time`` value is an exact
/// integer multiple of the bin width, which only edges (not centers)
/// satisfy.  Note PLEIADES uses these values directly as frame TOFs,
/// which differs by half a bin width; see the NEREIDS data-io guide.
///
/// A sidecar whose first start time is exactly 0 s parses fine, but the
/// t = 0 edge cannot be energy-converted (E is undefined at t = 0) —
/// crop the first frame from BOTH the stack and the edges
/// (``stack[1:]``, ``edges[1:]``) before conversion.  Real autoreduce
/// sidecars start after the pre-trigger bins (e.g. at 1.12 µs), so this
/// only arises for hand-made files.
///
/// Args:
///     path: Path to the ``*_Spectra.txt`` sidecar file.
///     n_frames: When given, validate the edge count against the TIFF
///         stack's frame count (``n_frames + 1`` edges).
///
/// Returns:
///     1D numpy array of N+1 ascending TOF bin edges in microseconds.
///
/// Raises:
///     FileNotFoundError: If the sidecar file does not exist (genuinely
///         missing — other open failures such as permission denied raise
///         ``OSError`` instead).
///     ValueError: For malformed content (fewer than 2 rows, non-finite
///         or non-increasing start times, a negative first start) or an
///         edge/frame count mismatch.
///     IOError: For other I/O failures (e.g. permission denied).
#[pyfunction]
#[pyo3(signature = (path, n_frames=None))]
fn read_tof_sidecar<'py>(
    py: Python<'py>,
    path: &str,
    n_frames: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let edges = nereids_io::spectrum::read_tof_sidecar(std::path::Path::new(path), n_frames)
        .map_err(|e| match &e {
            nereids_io::error::IoError::InvalidParameter(_) => {
                pyo3::exceptions::PyValueError::new_err(format!("{}", e))
            }
            err if is_genuine_not_found(err) => {
                pyo3::exceptions::PyFileNotFoundError::new_err(format!("{}", e))
            }
            _ => pyo3::exceptions::PyIOError::new_err(format!("{}", e)),
        })?;
    Ok(PyArray1::from_vec(py, edges))
}

/// Normalize raw sample and open-beam data to transmission.
///
/// Computes T = (C_sample / C_ob) × (PC_ob / PC_sample) with Poisson
/// uncertainty propagation.
///
/// Args:
///     sample: 3D numpy array of raw sample counts (n_tof, height, width).
///     open_beam: 3D numpy array of open-beam counts (same shape).
///     pc_sample: Proton charge for the sample measurement.
///     pc_ob: Proton charge for the open-beam measurement.
///     dark_current: Optional 2D numpy array (height, width) to subtract.
///
/// Returns:
///     Tuple of (transmission, uncertainty) as 3D numpy arrays.
#[pyfunction]
#[pyo3(signature = (sample, open_beam, pc_sample, pc_ob, dark_current=None))]
fn normalize<'py>(
    py: Python<'py>,
    sample: PyReadonlyArray3<f64>,
    open_beam: PyReadonlyArray3<f64>,
    pc_sample: f64,
    pc_ob: f64,
    dark_current: Option<PyReadonlyArray2<f64>>,
) -> PyResult<(Bound<'py, PyArray3<f64>>, Bound<'py, PyArray3<f64>>)> {
    // Validate shapes using cheap PyReadonly views before cloning
    let s_shape = sample.shape();
    let ob_shape = open_beam.shape();
    if s_shape != ob_shape {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "sample shape {:?} must match open_beam shape {:?}",
            s_shape, ob_shape,
        )));
    }

    if let Some(ref dc_arr) = dark_current {
        let dc_shape = dc_arr.shape();
        if dc_shape[0] != s_shape[1] || dc_shape[1] != s_shape[2] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dark_current shape ({}, {}) must match spatial dimensions ({}, {}) of sample",
                dc_shape[0], dc_shape[1], s_shape[1], s_shape[2],
            )));
        }
    }

    // Clone arrays only after all validation passes
    let s = sample.as_array().to_owned();
    let ob = open_beam.as_array().to_owned();
    let dc = dark_current.map(|d| d.as_array().to_owned());

    let params = NormalizationParams {
        proton_charge_sample: pc_sample,
        proton_charge_ob: pc_ob,
    };

    // Release the GIL for the normalization computation.
    let result = py.detach(move || norm::normalize(&s, &ob, &params, dc.as_ref()));
    let result = result.map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;

    Ok((
        PyArray3::from_owned_array(py, result.transmission),
        PyArray3::from_owned_array(py, result.uncertainty),
    ))
}

/// Convert TOF bin edges to energy bin centers.
///
/// Returns the geometric mean of adjacent energy bin edges (ascending order).
/// This is the standard energy grid for neutron resonance analysis.
///
/// Args:
///     tof_edges: 1D numpy array of TOF bin edges in microseconds (ascending).
///     flight_path_m: Total flight path in meters.
///     delay_us: Electronic/moderator delay in microseconds (default 0.0).
///
/// Returns:
///     1D numpy array of energy bin centers in eV (ascending).
///     Length = len(tof_edges) - 1.
#[pyfunction]
#[pyo3(signature = (tof_edges, flight_path_m, delay_us=0.0))]
fn tof_to_energy_centers<'py>(
    py: Python<'py>,
    tof_edges: PyReadonlyArray1<f64>,
    flight_path_m: f64,
    delay_us: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let edges = tof_edges.as_slice()?;
    let params = BeamlineParams {
        flight_path_m,
        delay_us,
    };

    let centers = nereids_io::tof::tof_edges_to_energy_centers(edges, &params)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;

    Ok(PyArray1::from_owned_array(py, centers))
}

// ── Element / isotope utilities ──────────────────────────────────────

/// Get the element symbol for a given atomic number Z.
///
/// Args:
///     z: Atomic number (e.g. 92 for uranium).
///
/// Returns:
///     Element symbol (e.g. "U"), or None if Z is out of range.
#[pyfunction]
#[pyo3(name = "element_symbol")]
fn py_element_symbol(z: u32) -> Option<String> {
    elements::element_symbol(z).map(|s| s.to_string())
}

/// Get the element name for a given atomic number Z.
///
/// Args:
///     z: Atomic number (e.g. 92 for uranium).
///
/// Returns:
///     Element name (e.g. "Uranium"), or None if Z is out of range.
#[pyfunction]
#[pyo3(name = "element_name")]
fn py_element_name(z: u32) -> Option<String> {
    elements::element_name(z).map(|s| s.to_string())
}

/// Parse an isotope string like "U-238" into (Z, A).
///
/// Args:
///     s: Isotope string in "Symbol-A" format (e.g. "U-238", "Fe-56").
///
/// Returns:
///     Tuple (z, a) or None if the string cannot be parsed.
#[pyfunction]
#[pyo3(name = "parse_isotope_str")]
fn py_parse_isotope_str(s: &str) -> Option<(u32, u32)> {
    elements::parse_isotope_str(s).map(|iso| (iso.z(), iso.a()))
}

/// Get the natural isotopic abundance for a specific isotope.
///
/// Args:
///     z: Atomic number.
///     a: Mass number.
///
/// Returns:
///     Abundance as a fraction (0.0 to 1.0), or None for synthetic isotopes.
#[pyfunction]
#[pyo3(name = "natural_abundance")]
fn py_natural_abundance(z: u32, a: u32) -> Option<f64> {
    Isotope::new(z, a)
        .ok()
        .and_then(|iso| elements::natural_abundance(&iso))
}

/// Get all naturally occurring isotopes for an element.
///
/// Args:
///     z: Atomic number (e.g. 74 for tungsten).
///
/// Returns:
///     List of ((z, a), abundance) tuples for all stable isotopes.
#[pyfunction]
#[pyo3(name = "natural_isotopes")]
fn py_natural_isotopes(z: u32) -> Vec<((u32, u32), f64)> {
    elements::natural_isotopes(z)
        .into_iter()
        .map(|(iso, frac)| ((iso.z(), iso.a()), frac))
        .collect()
}

/// Result of a trace-detectability analysis.
///
/// Returned by ``trace_detectability()`` and ``trace_detectability_survey()``.
/// Contains the peak SNR, the energy at which peak contrast occurs, and the
/// full |ΔT| spectrum for plotting.
#[pyclass(name = "TraceDetectabilityReport")]
struct PyTraceDetectabilityReport {
    inner: detectability::TraceDetectabilityReport,
}

#[pymethods]
impl PyTraceDetectabilityReport {
    /// Peak |ΔT| per ppm concentration at the most sensitive energy.
    #[getter]
    fn peak_delta_t_per_ppm(&self) -> f64 {
        self.inner.peak_delta_t_per_ppm
    }

    /// Energy at which peak contrast occurs (eV).
    #[getter]
    fn peak_energy_ev(&self) -> f64 {
        self.inner.peak_energy_ev
    }

    /// Estimated peak SNR at the given concentration and I₀.
    #[getter]
    fn peak_snr(&self) -> f64 {
        self.inner.peak_snr
    }

    /// Whether the combination is detectable (SNR > threshold).
    #[getter]
    fn detectable(&self) -> bool {
        self.inner.detectable
    }

    /// Energy-resolved |ΔT| spectrum for the given concentration.
    #[getter]
    fn delta_t_spectrum<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.delta_t_spectrum.clone())
    }

    /// Energies used (eV).
    #[getter]
    fn energies<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.energies.clone())
    }

    /// Fraction of energy bins where the matrix baseline is opaque (T < 1e-15).
    #[getter]
    fn opaque_fraction(&self) -> f64 {
        self.inner.opaque_fraction
    }

    fn __repr__(&self) -> String {
        format!(
            "TraceDetectabilityReport(detectable={}, peak_snr={:.2}, peak_energy_ev={:.2}, opaque_fraction={:.2})",
            self.inner.detectable,
            self.inner.peak_snr,
            self.inner.peak_energy_ev,
            self.inner.opaque_fraction,
        )
    }
}

/// Compute trace-detectability for a matrix + trace isotope pair.
///
/// Determines whether a trace isotope is detectable at a given concentration
/// (in ppm) within a matrix, by computing the peak spectral SNR over the
/// supplied energy window.
///
/// Resolution broadening can be applied via either Gaussian parameters
/// (``flight_path_m``, ``delta_t_us``, ``delta_l_m``) or a tabulated
/// resolution function (``resolution``). Providing both is an error.
///
/// Args:
///     matrix: ResonanceData for the matrix isotope.
///     matrix_density: Matrix areal density in atoms/barn.
///     trace: ResonanceData for the trace isotope.
///     trace_ppm: Trace concentration in ppm by atom.
///     energies: Energy grid in eV (1D numpy array, sorted ascending).
///     i0: Expected counts per energy bin (for Poisson noise estimate).
///     temperature_k: Sample temperature in Kelvin (default 293.6).
///     flight_path_m: Flight path for Gaussian resolution (optional).
///     delta_t_us: Timing uncertainty for Gaussian resolution (optional).
///     delta_l_m: Path length uncertainty for Gaussian resolution (optional).
///     resolution: TabulatedResolution for tabulated broadening (optional).
///     snr_threshold: Detection threshold in σ (default 3.0).
///
/// Returns:
///     TraceDetectabilityReport with peak SNR, peak energy, and |ΔT| spectrum.
#[pyfunction]
#[pyo3(name = "trace_detectability", signature = (matrix, matrix_density, trace, trace_ppm, energies, i0, temperature_k=293.6, flight_path_m=None, delta_t_us=None, delta_l_m=None, resolution=None, delta_e_us=None, snr_threshold=3.0))]
fn py_trace_detectability(
    py: Python<'_>,
    matrix: &PyResonanceData,
    matrix_density: f64,
    trace: &PyResonanceData,
    trace_ppm: f64,
    energies: PyReadonlyArray1<f64>,
    i0: f64,
    temperature_k: f64,
    flight_path_m: Option<f64>,
    delta_t_us: Option<f64>,
    delta_l_m: Option<f64>,
    resolution: Option<PyTabulatedResolution>,
    delta_e_us: Option<f64>,
    snr_threshold: f64,
) -> PyResult<PyTraceDetectabilityReport> {
    let e = energies.as_slice()?;
    require_non_empty_energy_grid(e)?;

    if matrix_density <= 0.0 || !matrix_density.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "matrix_density must be finite and positive",
        ));
    }
    if trace_ppm < 0.0 || !trace_ppm.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "trace_ppm must be finite and non-negative",
        ));
    }
    if i0 <= 0.0 || !i0.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "i0 must be finite and positive",
        ));
    }
    if !temperature_k.is_finite() || temperature_k < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "temperature_k must be finite and non-negative",
        ));
    }
    if snr_threshold < 0.0 || !snr_threshold.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "snr_threshold must be finite and non-negative",
        ));
    }

    let res_fn = build_resolution(flight_path_m, delta_t_us, delta_l_m, resolution, delta_e_us)?;

    // Clone data to owned types so we can release the GIL.
    let e_owned = e.to_vec();
    let matrix_data = matrix.inner.clone();
    let trace_data = trace.inner.clone();

    // Release the GIL for the detectability computation.
    // Wrap single matrix in a vec — Rust API supports multi-matrix but Python
    // API preserves backward compatibility with a single matrix argument.
    let report = py.detach(move || {
        let matrix_isotopes = vec![(Arc::unwrap_or_clone(matrix_data), matrix_density)];
        let config = detectability::TraceDetectabilityConfig {
            matrix_isotopes: &matrix_isotopes,
            energies: &e_owned,
            i0,
            temperature_k,
            resolution: res_fn.as_ref(),
            snr_threshold,
        };
        detectability::trace_detectability(&config, &trace_data, trace_ppm)
            .map_err(|e| format!("trace_detectability failed: {e}"))
    });
    let report = report.map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    Ok(PyTraceDetectabilityReport { inner: report })
}

/// Survey multiple trace candidates against a single matrix.
///
/// Parallelises over candidates with rayon. Returns a list of
/// ``(isotope_name, TraceDetectabilityReport)`` tuples sorted by
/// ``peak_snr`` descending.
///
/// Resolution broadening can be applied via either Gaussian parameters
/// (``flight_path_m``, ``delta_t_us``, ``delta_l_m``) or a tabulated
/// resolution function (``resolution``). Providing both is an error.
///
/// Args:
///     matrix: ResonanceData for the matrix isotope.
///     matrix_density: Matrix areal density in atoms/barn.
///     trace_candidates: List of ResonanceData for candidate trace isotopes.
///     trace_ppm: Trace concentration in ppm by atom.
///     energies: Energy grid in eV (1D numpy array, sorted ascending).
///     i0: Expected counts per energy bin (for Poisson noise estimate).
///     temperature_k: Sample temperature in Kelvin (default 293.6).
///     flight_path_m: Flight path for Gaussian resolution (optional).
///     delta_t_us: Timing uncertainty for Gaussian resolution (optional).
///     delta_l_m: Path length uncertainty for Gaussian resolution (optional).
///     resolution: TabulatedResolution for tabulated broadening (optional).
///     snr_threshold: Detection threshold in σ (default 3.0).
///
/// Returns:
///     List of (isotope_name, TraceDetectabilityReport) sorted by peak_snr descending.
#[pyfunction]
#[pyo3(name = "trace_detectability_survey", signature = (matrix, matrix_density, trace_candidates, trace_ppm, energies, i0, temperature_k=293.6, flight_path_m=None, delta_t_us=None, delta_l_m=None, resolution=None, delta_e_us=None, snr_threshold=3.0))]
fn py_trace_detectability_survey(
    py: Python<'_>,
    matrix: &PyResonanceData,
    matrix_density: f64,
    trace_candidates: Vec<PyResonanceData>,
    trace_ppm: f64,
    energies: PyReadonlyArray1<f64>,
    i0: f64,
    temperature_k: f64,
    flight_path_m: Option<f64>,
    delta_t_us: Option<f64>,
    delta_l_m: Option<f64>,
    resolution: Option<PyTabulatedResolution>,
    delta_e_us: Option<f64>,
    snr_threshold: f64,
) -> PyResult<Vec<(String, PyTraceDetectabilityReport)>> {
    let e = energies.as_slice()?;
    require_non_empty_energy_grid(e)?;

    if matrix_density <= 0.0 || !matrix_density.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "matrix_density must be finite and positive",
        ));
    }
    if trace_ppm < 0.0 || !trace_ppm.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "trace_ppm must be finite and non-negative",
        ));
    }
    if i0 <= 0.0 || !i0.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "i0 must be finite and positive",
        ));
    }
    if !temperature_k.is_finite() || temperature_k < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "temperature_k must be finite and non-negative",
        ));
    }
    if snr_threshold < 0.0 || !snr_threshold.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "snr_threshold must be finite and non-negative",
        ));
    }
    if trace_candidates.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "trace_candidates must not be empty",
        ));
    }

    let res_fn = build_resolution(flight_path_m, delta_t_us, delta_l_m, resolution, delta_e_us)?;

    let candidates: Vec<ResonanceData> = trace_candidates
        .into_iter()
        .map(|d| Arc::unwrap_or_clone(d.inner))
        .collect();

    // Clone data to owned types so we can release the GIL.
    let e_owned = e.to_vec();
    let matrix_data = matrix.inner.clone();

    // Release the GIL for the parallelised detectability survey.
    // Wrap single matrix in a vec — Rust API supports multi-matrix but Python
    // API preserves backward compatibility with a single matrix argument.
    let results = py.detach(move || {
        let matrix_isotopes = vec![(Arc::unwrap_or_clone(matrix_data), matrix_density)];
        let config = detectability::TraceDetectabilityConfig {
            matrix_isotopes: &matrix_isotopes,
            energies: &e_owned,
            i0,
            temperature_k,
            resolution: res_fn.as_ref(),
            snr_threshold,
        };
        detectability::trace_detectability_survey(&config, &candidates, trace_ppm)
            .map_err(|e| format!("trace_detectability_survey failed: {e}"))
    });
    let results = results.map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    Ok(results
        .into_iter()
        .map(|(name, report)| (name, PyTraceDetectabilityReport { inner: report }))
        .collect())
}

/// Precompute Doppler- and resolution-broadened total cross-sections.
///
/// Returns one broadened total cross-section array per isotope.  This is the
/// expensive physics step (Doppler FGM + resolution convolution); calling it
/// once and caching the result avoids redundant computation when the same
/// isotopes and energy grid are reused across many fits or forward-model
/// evaluations.
///
/// Resolution broadening can be applied via either Gaussian parameters
/// (``flight_path_m``, ``delta_t_us``, ``delta_l_m``) or a tabulated
/// resolution function (``resolution``). Providing both is an error.
///
/// Args:
///     energies: Energy grid in eV (1D numpy array, sorted ascending).
///     isotopes: List of ResonanceData objects.
///     temperature_k: Sample temperature in Kelvin (default 0.0).
///     flight_path_m: Flight path in meters for Gaussian resolution (optional).
///     delta_t_us: Timing uncertainty in microseconds (optional).
///     delta_l_m: Path length uncertainty in meters (optional).
///     resolution: TabulatedResolution from ``load_resolution()`` (optional).
///
/// Returns:
///     List of 1D numpy arrays (one per isotope), each containing the broadened
///     total cross-section in barns on the supplied energy grid.
#[pyfunction]
#[pyo3(signature = (energies, isotopes, temperature_k=293.6, flight_path_m=None, delta_t_us=None, delta_l_m=None, resolution=None, delta_e_us=None))]
fn precompute_cross_sections<'py>(
    py: Python<'py>,
    energies: PyReadonlyArray1<f64>,
    isotopes: Vec<PyResonanceData>,
    temperature_k: f64,
    flight_path_m: Option<f64>,
    delta_t_us: Option<f64>,
    delta_l_m: Option<f64>,
    resolution: Option<PyTabulatedResolution>,
    delta_e_us: Option<f64>,
) -> PyResult<Vec<Bound<'py, PyArray1<f64>>>> {
    let e = energies.as_slice()?;
    require_non_empty_energy_grid(e)?;

    if isotopes.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "isotopes list must not be empty",
        ));
    }
    if !temperature_k.is_finite() || temperature_k < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "temperature_k must be finite and non-negative",
        ));
    }

    // Issue #442: resolution-broadened cross-sections are not physically
    // meaningful for transmission fitting.  Resolution broadening must be
    // applied after Beer-Lambert on the total transmission, which depends
    // on per-pixel densities and cannot be precomputed as broadened σ.
    let res_fn = build_resolution(flight_path_m, delta_t_us, delta_l_m, resolution, delta_e_us)?;
    if res_fn.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "precompute_cross_sections() cannot apply resolution broadening to \
             cross-sections.  For transmission data, resolution broadening must \
             be applied after Beer-Lambert on the total transmission T(E), not \
             to individual cross-sections σ(E).  Use forward_model() instead, \
             which applies resolution in the correct order.  \
             To get Doppler-only cross-sections, omit the resolution parameters.",
        ));
    }

    let res_data: Vec<ResonanceData> = isotopes
        .into_iter()
        .map(|d| Arc::unwrap_or_clone(d.inner))
        .collect();

    // Copy numpy slice to owned Vec so we can release the GIL.
    let e_owned = e.to_vec();

    // Release the GIL for the heavy Doppler broadening.
    let xs = py.detach(move || {
        transmission::broadened_cross_sections(&e_owned, &res_data, temperature_k, None, None)
    });

    // GIL is re-acquired after detach returns — use `py` directly.
    let xs = xs.map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("broadened_cross_sections failed: {}", e))
    })?;

    Ok(xs.into_iter().map(|v| PyArray1::from_vec(py, v)).collect())
}

/// Detect dead pixels in a 3D image stack.
///
/// A pixel is marked as "dead" when all its counts across the spectral/TOF
/// axis are exactly zero.  The returned mask can be passed directly to
/// ``spatial_map_typed(dead_pixels=...)``.
///
/// Pixel masks are a **pipeline-integrity screen only** — they exclude
/// pixels whose data stream is broken (dead, hot/railed), never low-count
/// or poorly covered pixels.  Low-count pixels are alive and must be kept.
/// Prefer ``detect_bad_pixels()`` (validating, unions sample and open-beam,
/// optional hot screen); see also ``detect_hot_pixels()`` and
/// ``detect_dead_pixels_chunked()`` for intermittent deadness across
/// acquisition chunks.
///
/// Args:
///     data: 3D numpy array with shape ``(n_frames, height, width)``.
///         Typically an open-beam stack or raw sample stack.
///
/// Returns:
///     2D boolean numpy array with shape ``(height, width)``.
///     ``True`` marks a dead pixel.
#[pyfunction]
fn detect_dead_pixels<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<f64>,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let arr = data.as_array().to_owned();
    let mask = py.detach(move || norm::detect_dead_pixels(&arr));
    Ok(PyArray2::from_owned_array(py, mask))
}

/// Detect hot (railed / runaway) pixels — two-stage screen.
///
/// Stage 1 (global): robust one-sided cut on per-pixel total counts — a
/// pixel is a candidate when ``ln(total) > median + k_mad * sigma``, where
/// median and MAD are computed over the live (``total > 0``) pixels only
/// and ``sigma`` is the MAD-based robust scale floored by the Poisson
/// counting noise of the median total.  Stage 2 (local), iterated to a
/// fixpoint: a candidate is flagged only if its total also exceeds 10x the
/// median of its 8-neighborhood reference sample — live unflagged
/// neighbors contribute their totals, already-flagged neighbors contribute
/// 0 (a known defect cannot vouch for its neighbors), dead neighbors are
/// omitted; edge pixels use whatever neighbors exist, and a candidate with
/// no live neighbor keeps the global verdict.  Passes repeat until no new
/// flag is added (bounded by ``height * width`` passes; in practice ~the
/// defect-cluster radius), eroding railed CLUSTERS from the boundary
/// inward — a single pass would miss the interior of clusters >= 2 px
/// wide, whose neighbors are railed too.  Clusters up to 3 px wide are
/// fully consumed PROVIDED they expose at least one end cap or convex
/// corner to normal-scene neighbors (erosion must seed somewhere): an
/// EDGE-TO-EDGE railed band >= 2 px wide, spanning the full detector
/// width or height with both ends off-detector, has no seed and is NOT
/// caught — deliberately, because a slit-aperture open beam produces a
/// genuine full-width bright scene band pixel-for-pixel
/// indistinguishable from it, and a full-span screen would mask that
/// scene (the bimodal failure).  Declare such full-span detector
/// pathologies in a file mask.  A full-span width-1 railed line IS
/// caught (each pixel keeps >= 4 normal neighbors).  The local
/// confirmation keeps bimodal scenes honest:
/// with a dark majority holding the median, the global statistics describe
/// only the dark population and the entire bright region would otherwise
/// be masked — a contiguous bright region is scene, not a defect.  Upper
/// tail only — stuck-low pixels are indistinguishable from low-count-alive
/// pixels and are kept (masks are pipeline-integrity only, never a
/// low-count screen).
///
/// Bright SCENE regions never erode: a boundary pixel of a contiguous
/// bright region >= 2 px wide keeps >= 4 same-side neighbors for any
/// straight or diagonal edge, so its reference median stays bright and
/// scene gradients (<= 2-3x across real edges) never reach the 10x factor
/// — the erosion has no seed.  Documented width-1 limitation (accepted
/// trade-off): a 1-px-wide bright scene line at >= 10x local contrast is
/// spatially indistinguishable from a railed line and IS masked; real
/// scene features on VENUS are PSF-blurred over >= 2 px, so >= 10x
/// single-pixel scene contrast is physically rare, and contiguous bright
/// regions of width >= 2 are safe from the local stage.
///
/// ``data`` must be RAW detected counts (unscaled): the Poisson floor
/// assumes ``Var[N] = N``, so scaled inputs silently distort it —
/// down-scaling (proton-charge-normalized rates << 1, gain division)
/// inflates the floor and can suppress real flags; up-scaling (event
/// weights > 1) deflates it.  Detect on raw counts, normalize afterwards.
///
/// Args:
///     data: 3D numpy array of raw counts with shape
///         ``(n_frames, height, width)``.
///     k_mad: Robust-sigma multiplier for the stage-1 upper-tail cut
///         (default 6.0: one-sided Gaussian tail ~1e-9, on a unimodal
///         image essentially never flags a statistically plausible pixel).
///
/// Returns:
///     2D boolean numpy array with shape ``(height, width)``.
///     ``True`` marks a hot pixel.
///
/// Raises:
///     ValueError: If ``data`` contains non-finite or negative values or
///         has zero frames (``shape[0] == 0``), or ``k_mad`` is not finite
///         and positive.
#[pyfunction]
#[pyo3(signature = (data, k_mad = norm::HOT_PIXEL_K_MAD))]
fn detect_hot_pixels<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<f64>,
    k_mad: f64,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let arr = data.as_array().to_owned();
    let mask = py.detach(move || norm::detect_hot_pixels(&arr, k_mad));
    let mask = mask.map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    Ok(PyArray2::from_owned_array(py, mask))
}

/// Detect dead pixels across acquisition chunks (dead in ANY chunk).
///
/// Catches intermittent deadness that ``detect_dead_pixels()`` on the
/// summed stack cannot see: a pixel dead for one acquisition chunk but
/// alive in another has nonzero summed counts, yet its dead-chunk data
/// corrupts the combined spectrum.  Chunk the acquisition so each live
/// pixel has an expected >= 20 total counts per chunk (misflag probability
/// per live pixel is ``m * exp(-lambda)`` over ``m`` chunks).
///
/// Chunks may have different numbers of frames (ragged event
/// re-histogramming is fine); spatial dimensions must agree.
///
/// Args:
///     chunks: List of 3D numpy arrays, one per acquisition chunk, each
///         with shape ``(n_frames_i, height, width)``.
///
/// Returns:
///     2D boolean numpy array with shape ``(height, width)``.
///     ``True`` marks a pixel that is all-zero in at least one chunk.
///
/// Raises:
///     ValueError: If ``chunks`` is empty, any chunk has zero frames
///         (``shape[0] == 0`` — its all-zero test would vacuously mark
///         every pixel dead), any chunk contains non-finite or negative
///         values, or the spatial dimensions differ.
#[pyfunction]
fn detect_dead_pixels_chunked<'py>(
    py: Python<'py>,
    chunks: Vec<PyReadonlyArray3<f64>>,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    // Copy to owned arrays before releasing the GIL.
    let owned: Vec<_> = chunks.iter().map(|c| c.as_array().to_owned()).collect();
    let mask = py.detach(move || norm::detect_dead_pixels_chunked(&owned));
    let mask = mask.map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    Ok(PyArray2::from_owned_array(py, mask))
}

/// Detect all pipeline-corrupting pixels: dead + hot over sample and
/// (optionally) open beam.
///
/// This is the validating entry point.  Deadness/hotness is
/// per-acquisition — a pixel dead only in the open-beam run still corrupts
/// every transmission ratio computed from it — so the masks of both stacks
/// are unioned: ``dead(sample) | hot(sample) [| dead(ob) | hot(ob)]``.
/// The stacks' frame counts may differ; spatial dimensions must agree.
///
/// Both stacks must be RAW detected counts (unscaled) — see
/// ``detect_hot_pixels()``: scaling distorts the Poisson floor of the hot
/// screen.  Detect on raw counts, before any normalization.
///
/// Args:
///     sample: 3D numpy array of raw counts with shape
///         ``(n_frames, height, width)``.
///     open_beam: Optional 3D numpy array of raw counts with shape
///         ``(n_frames2, height, width)``.
///     hot_k_mad: Robust-sigma multiplier for the hot-pixel screen
///         (default 6.0), or ``None`` to disable it (dead-only detection).
///
/// Returns:
///     2D boolean numpy array with shape ``(height, width)``.
///     ``True`` marks a pixel to exclude.
///
/// Raises:
///     ValueError: If either stack contains non-finite or negative values
///         or has zero frames (``shape[0] == 0``), the spatial dimensions
///         differ, or ``hot_k_mad`` is not finite and positive.
#[pyfunction]
#[pyo3(signature = (sample, open_beam = None, hot_k_mad = Some(norm::HOT_PIXEL_K_MAD)))]
fn detect_bad_pixels<'py>(
    py: Python<'py>,
    sample: PyReadonlyArray3<f64>,
    open_beam: Option<PyReadonlyArray3<f64>>,
    hot_k_mad: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let s = sample.as_array().to_owned();
    let ob = open_beam.map(|o| o.as_array().to_owned());
    let mask = py.detach(move || norm::detect_bad_pixels(&s, ob.as_ref(), hot_k_mad));
    let mask = mask.map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    Ok(PyArray2::from_owned_array(py, mask))
}

/// Result of energy calibration.
#[pyclass(name = "CalibrationResult")]
#[derive(Debug)]
struct PyCalibrationResult {
    /// Fitted flight path length in metres.
    flight_path_m: f64,
    /// Fitted TOF delay in microseconds.
    t0_us: f64,
    /// Fitted total areal density in atoms/barn.
    total_density: f64,
    /// Reduced chi-squared at the best parameters.
    reduced_chi_squared: f64,
    /// Corrected energy grid.
    energies_corrected: Py<PyArray1<f64>>,
}

#[pymethods]
impl PyCalibrationResult {
    #[getter]
    fn flight_path_m(&self) -> f64 {
        self.flight_path_m
    }
    #[getter]
    fn t0_us(&self) -> f64 {
        self.t0_us
    }
    #[getter]
    fn total_density(&self) -> f64 {
        self.total_density
    }
    #[getter]
    fn reduced_chi_squared(&self) -> f64 {
        self.reduced_chi_squared
    }
    #[getter]
    fn energies_corrected<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.energies_corrected.bind(py).clone()
    }
    fn __repr__(&self) -> String {
        format!(
            "CalibrationResult(L={:.4}m, t0={:.2}µs, n={:.2e}, chi2r={:.4})",
            self.flight_path_m, self.t0_us, self.total_density, self.reduced_chi_squared
        )
    }
}

/// Calibrate the energy axis by fitting flight path and TOF delay.
///
/// Finds the (L, t₀, n_total) that best align the ENDF resonance model
/// with measured transmission data from a known-composition reference sample.
///
/// Args:
///     energies_nominal: 1D ascending energy grid (eV) computed with assumed L.
///     transmission: 1D measured transmission values.
///     uncertainty: 1D per-bin uncertainty.
///     isotopes: List of ResonanceData for the reference sample.
///     abundances: Natural abundance fractions (same length as isotopes).
///     assumed_flight_path_m: Flight path used to compute energies_nominal.
///     temperature_k: Sample temperature in Kelvin (default 293.6).
///
/// Returns:
///     CalibrationResult with fitted (L, t₀, n_total) and corrected energies.
#[pyfunction]
#[pyo3(name = "calibrate_energy", signature = (energies_nominal, transmission, uncertainty, isotopes, abundances, assumed_flight_path_m, temperature_k=293.6, resolution=None))]
fn py_calibrate_energy(
    py: Python<'_>,
    energies_nominal: PyReadonlyArray1<f64>,
    transmission: PyReadonlyArray1<f64>,
    uncertainty: PyReadonlyArray1<f64>,
    isotopes: Vec<PyResonanceData>,
    abundances: Vec<f64>,
    assumed_flight_path_m: f64,
    temperature_k: f64,
    resolution: Option<PyTabulatedResolution>,
) -> PyResult<PyCalibrationResult> {
    // Copy NumPy slices to owned `Vec<f64>` *before* `py.detach` so the
    // closure does not hold borrows into NumPy-owned memory across the GIL
    // release.  rust-numpy only guards borrows while the GIL is held;
    // once detached another Python thread could mutate/reallocate the
    // arrays and the inner Rust slices would dangle.  Every other
    // `py.detach` site in this file follows the same `.as_slice()?.to_vec()`
    // pattern — `calibrate_energy` was the lone outlier.
    let e_owned = energies_nominal.as_slice()?.to_vec();
    let t_owned = transmission.as_slice()?.to_vec();
    let s_owned = uncertainty.as_slice()?.to_vec();

    // Validate the nominal energy grid up front so malformed energies
    // surface as ValueError rather than a release-mode `PanicException`
    // from the per-point `assert!(energy_ev.is_finite() && energy_ev > 0.0)`
    // guards inside `transmission::forward_model` → SLBW / RML / URR
    // leaves.  Calibration requires at least one data point to fit
    // (L, t₀, n_total), so an empty grid is also rejected.
    require_non_empty_energy_grid(&e_owned)?;
    if t_owned.len() != e_owned.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "transmission length ({}) must match energies_nominal length ({})",
            t_owned.len(),
            e_owned.len(),
        )));
    }
    if s_owned.len() != e_owned.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "uncertainty length ({}) must match energies_nominal length ({})",
            s_owned.len(),
            e_owned.len(),
        )));
    }

    let res_data: Vec<nereids_endf::resonance::ResonanceData> = isotopes
        .into_iter()
        .map(|d| Arc::unwrap_or_clone(d.inner))
        .collect();

    let instrument = resolution.map(|r| nereids_physics::transmission::InstrumentParams {
        resolution: nereids_physics::resolution::ResolutionFunction::Tabulated(r.inner.clone()),
    });

    let result = py.detach(move || {
        nereids_pipeline::calibration::calibrate_energy(
            &e_owned,
            &t_owned,
            &s_owned,
            &res_data,
            &abundances,
            assumed_flight_path_m,
            temperature_k,
            instrument.as_ref(),
        )
    });
    let result = result.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(PyCalibrationResult {
        flight_path_m: result.flight_path_m,
        t0_us: result.t0_us,
        total_density: result.total_density,
        reduced_chi_squared: result.reduced_chi_squared,
        energies_corrected: PyArray1::from_vec(py, result.energies_corrected).unbind(),
    })
}

// ── NeXus I/O Bindings ──────────────────────────────────────────────────

/// Result of probing a NeXus file for available data.
#[pyclass(name = "NexusMetadata")]
struct PyNexusMetadata {
    inner: nereids_io::nexus::NexusMetadata,
}

#[pymethods]
impl PyNexusMetadata {
    /// Whether the file contains a pre-histogrammed dataset.
    #[getter]
    fn has_histogram(&self) -> bool {
        self.inner.has_histogram
    }

    /// Whether the file contains event data.
    #[getter]
    fn has_events(&self) -> bool {
        self.inner.has_events
    }

    /// Shape of the histogram dataset as (rotation, y, x, tof), if present.
    #[getter]
    fn histogram_shape(&self) -> Option<[usize; 4]> {
        self.inner.histogram_shape
    }

    /// Number of neutron events, if present.
    #[getter]
    fn n_events(&self) -> Option<usize> {
        self.inner.n_events
    }

    /// Flight path in metres from file metadata, if present.
    #[getter]
    fn flight_path_m(&self) -> Option<f64> {
        self.inner.flight_path_m
    }

    /// TOF offset in nanoseconds, if present.
    #[getter]
    fn tof_offset_ns(&self) -> Option<f64> {
        self.inner.tof_offset_ns
    }

    fn __repr__(&self) -> String {
        format!(
            "NexusMetadata(histogram={}, events={}, n_events={:?}, flight_path={:?})",
            self.inner.has_histogram,
            self.inner.has_events,
            self.inner.n_events,
            self.inner.flight_path_m,
        )
    }
}

/// Result of loading NeXus histogram or event data.
#[pyclass(name = "NexusData")]
struct PyNexusData {
    counts: Py<PyArray3<f64>>,
    tof_edges_us: Py<PyArray1<f64>>,
    flight_path_m: Option<f64>,
    dead_pixels: Option<Py<PyArray2<bool>>>,
    n_rotation_angles: usize,
    event_total: Option<usize>,
    event_kept: Option<usize>,
}

#[pymethods]
impl PyNexusData {
    /// 3D counts array with shape (n_tof, height, width).
    #[getter]
    fn counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.counts.bind(py).clone()
    }

    /// TOF bin edges in microseconds (length = n_tof + 1).
    #[getter]
    fn tof_edges_us<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.tof_edges_us.bind(py).clone()
    }

    /// Flight path in metres from file metadata, if present.
    #[getter]
    fn flight_path_m(&self) -> Option<f64> {
        self.flight_path_m
    }

    /// Dead pixel mask (height, width), if present.  True = dead.
    #[getter]
    fn dead_pixels<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<bool>>> {
        self.dead_pixels.as_ref().map(|m| m.bind(py).clone())
    }

    /// Number of rotation angles summed (1 for single-angle data).
    #[getter]
    fn n_rotation_angles(&self) -> usize {
        self.n_rotation_angles
    }

    /// Total events before filtering (event data only).
    #[getter]
    fn event_total(&self) -> Option<usize> {
        self.event_total
    }

    /// Events kept after filtering (event data only).
    #[getter]
    fn event_kept(&self) -> Option<usize> {
        self.event_kept
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let shape = self.counts.bind(py).shape();
        format!(
            "NexusData(shape=({}, {}, {}), tof_bins={}, flight_path={:?})",
            shape[0], shape[1], shape[2], shape[0], self.flight_path_m,
        )
    }
}

/// Probe a NeXus/HDF5 file for available data without loading it.
///
/// Returns metadata about what the file contains (histogram, events,
/// flight path, etc.) without reading the full dataset.
///
/// Args:
///     path: Path to the NeXus/HDF5 file.
///
/// Returns:
///     NexusMetadata with has_histogram, has_events, n_events, etc.
/// Map nereids_io::IoError to appropriate Python exception.
fn map_io_error(e: nereids_io::error::IoError) -> pyo3::PyErr {
    use nereids_io::error::IoError;
    match e {
        IoError::FileNotFound(..) => pyo3::exceptions::PyFileNotFoundError::new_err(format!("{e}")),
        IoError::InvalidParameter(..) | IoError::ShapeMismatch(..) => {
            pyo3::exceptions::PyValueError::new_err(format!("{e}"))
        }
        _ => pyo3::exceptions::PyIOError::new_err(format!("{e}")),
    }
}

#[pyfunction]
fn probe_nexus(path: &str) -> PyResult<PyNexusMetadata> {
    let meta = nereids_io::nexus::probe_nexus(std::path::Path::new(path)).map_err(map_io_error)?;
    Ok(PyNexusMetadata { inner: meta })
}

/// DASlogs-based run-health summary.
#[pyclass(name = "RunHealth")]
struct PyRunHealth {
    inner: nereids_io::daslogs::RunHealth,
}

#[pymethods]
impl PyRunHealth {
    /// Time-weighted fraction of the run spent paused, if the pause PV
    /// is present.
    #[getter]
    fn pause_fraction(&self) -> Option<f64> {
        self.inner.pause_fraction
    }

    /// Time-weighted fraction of the run with power below
    /// `power_dip_fraction × median(power)`.  ``None`` when the power PV
    /// is absent or empty, or when the dip threshold is undefined because
    /// the sample median of the power entries is non-positive (e.g. the
    /// beam was off for at least half the entries) — check
    /// ``median_power``, which is co-reported.
    #[getter]
    fn beam_dip_fraction(&self) -> Option<f64> {
        self.inner.beam_dip_fraction
    }

    /// Sample median of the power PV entries (not time-weighted), if
    /// present.
    #[getter]
    fn median_power(&self) -> Option<f64> {
        self.inner.median_power
    }

    /// Run duration in seconds (`/entry/duration`, or the latest log
    /// timestamp as a lower bound), if determinable.
    #[getter]
    fn duration_s(&self) -> Option<f64> {
        self.inner.duration_s
    }

    /// Number of pause-PV log entries read (0 when absent).
    #[getter]
    fn n_pause_entries(&self) -> usize {
        self.inner.n_pause_entries
    }

    /// Number of power-PV log entries read (0 when absent).
    #[getter]
    fn n_power_entries(&self) -> usize {
        self.inner.n_power_entries
    }

    fn __repr__(&self) -> String {
        format!(
            "RunHealth(pause_fraction={:?}, beam_dip_fraction={:?}, median_power={:?}, duration_s={:?})",
            self.inner.pause_fraction,
            self.inner.beam_dip_fraction,
            self.inner.median_power,
            self.inner.duration_s,
        )
    }
}

/// Compute a run-health summary from /entry/DASlogs of a NeXus file.
///
/// DASlogs PVs log transitions, not regular samples, so entry means are
/// wrong; this uses last-value-held time-weighted integration over the
/// run window (``/entry/duration`` when present, else the latest log
/// timestamp — a lower bound).  Absent PVs (or a missing DASlogs group)
/// and present-but-empty PVs (zero entries logged) yield ``None`` fields,
/// not errors.  ``beam_dip_fraction`` is additionally ``None`` when the
/// sample median of the power entries is non-positive (dip threshold
/// undefined); ``median_power`` is co-reported so callers can see why.
///
/// Args:
///     path: Path to the NeXus/HDF5 file.
///     pause_pv: DASlogs PV nonzero while the DAQ is paused
///         (SNS default ``"pause"``).
///     power_pv: DASlogs PV proxying beam power (SNS default
///         ``"proton_charge"``).  Other facilities pass their own names.
///     power_dip_fraction: Beam-dip threshold as a fraction of the
///         median power (default 0.5 — between nominal source jitter
///         and true beam-off dips).
///
/// Returns:
///     RunHealth with pause_fraction, beam_dip_fraction, median_power,
///     duration_s, n_pause_entries, n_power_entries.
///
/// Raises:
///     ValueError: For a present-but-malformed PV (length mismatch,
///         non-finite entries, negative power values, decreasing
///         timestamps), a non-positive run window, or an invalid
///         power_dip_fraction.
///     IOError: If the file is missing/unreadable or /entry is absent
///         (HDF5 access failures).
#[pyfunction]
#[pyo3(signature = (path, pause_pv="pause", power_pv="proton_charge", power_dip_fraction=nereids_io::daslogs::DEFAULT_POWER_DIP_FRACTION))]
fn run_health(
    path: &str,
    pause_pv: &str,
    power_pv: &str,
    power_dip_fraction: f64,
) -> PyResult<PyRunHealth> {
    let options = nereids_io::daslogs::RunHealthOptions {
        pause_pv: pause_pv.to_string(),
        power_pv: power_pv.to_string(),
        power_dip_fraction,
    };
    let health = nereids_io::daslogs::run_health(std::path::Path::new(path), &options)
        .map_err(map_io_error)?;
    Ok(PyRunHealth { inner: health })
}

/// Load pre-histogrammed counts from a NeXus/HDF5 file.
///
/// Reads `/entry/histogram/counts` (4D: rotation × y × x × tof) and
/// transposes the caller-selected single-angle slice to
/// `(tof, y, x)`.
///
/// **Issue #430**: by default this function refuses multi-angle files
/// (more than one rotation angle) because silently summing them
/// destroys projection-resolved information on import.  Callers with
/// a multi-angle file must choose a policy via `multi_angle_mode`.
///
/// Args:
///     path: Path to the NeXus/HDF5 file.
///     multi_angle_mode: one of:
///         - ``"error"`` (default): reject multi-angle files with a clear error.
///         - ``"sum"``: sum over all rotation angles into one volume
///           (legacy behaviour, now opt-in).
///         - ``"select"``: extract a single rotation angle by
///           ``angle_index`` (default 0).
///     angle_index: index of the rotation angle to extract when
///         ``multi_angle_mode="select"``.  Ignored otherwise.
///
/// Returns:
///     NexusData with counts, tof_edges_us, flight_path_m, dead_pixels.
#[pyfunction]
#[pyo3(signature = (path, multi_angle_mode="error", angle_index=0))]
fn load_nexus_histogram(
    py: Python<'_>,
    path: &str,
    multi_angle_mode: &str,
    angle_index: usize,
) -> PyResult<PyNexusData> {
    use nereids_io::nexus::MultiAngleMode;
    let mode = match multi_angle_mode {
        "error" => MultiAngleMode::Error,
        "sum" => MultiAngleMode::Sum,
        "select" => MultiAngleMode::SelectAngle(angle_index),
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "multi_angle_mode must be 'error', 'sum', or 'select', got {other:?}"
            )));
        }
    };
    let data = nereids_io::nexus::load_nexus_histogram_with_mode(std::path::Path::new(path), mode)
        .map_err(map_io_error)?;
    Ok(nexus_data_to_py(py, data))
}

/// Load event data from a NeXus/HDF5 file, histogramming into TOF bins.
///
/// Reads `/entry/neutrons/event_time_offset`, `/x`, `/y` and bins
/// events into a linear TOF grid with the specified parameters.
///
/// Args:
///     path: Path to the NeXus/HDF5 file.
///     n_bins: Number of TOF bins.
///     tof_min_us: Minimum TOF in microseconds.
///     tof_max_us: Maximum TOF in microseconds.
///     height: Detector height in pixels.
///     width: Detector width in pixels.
///
/// Returns:
///     NexusData with counts, tof_edges_us, flight_path_m, and event stats.
#[pyfunction]
#[pyo3(signature = (path, n_bins, tof_min_us, tof_max_us, height, width))]
fn load_nexus_events(
    py: Python<'_>,
    path: &str,
    n_bins: usize,
    tof_min_us: f64,
    tof_max_us: f64,
    height: usize,
    width: usize,
) -> PyResult<PyNexusData> {
    let params = nereids_io::nexus::EventBinningParams {
        n_bins,
        tof_min_us,
        tof_max_us,
        height,
        width,
    };
    let data = nereids_io::nexus::load_nexus_events(std::path::Path::new(path), &params)
        .map_err(map_io_error)?;
    Ok(nexus_data_to_py(py, data))
}

/// A slow-control PV read from `/entry/DASlogs/<pv>` as a transition log.
#[pyclass(name = "RunLog")]
struct PyRunLog {
    times: Py<PyArray1<f64>>,
    values: Py<PyArray1<f64>>,
    duration_s: f64,
    offset_iso: Option<String>,
    n_dropped_corrupt: usize,
}

#[pymethods]
impl PyRunLog {
    /// Transition times in seconds relative to run start (ascending).
    #[getter]
    fn times<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.times.bind(py).clone()
    }

    /// Value taking effect at the matching `times` entry.
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.values.bind(py).clone()
    }

    /// Total run duration in seconds (`/entry/duration`).
    #[getter]
    fn duration_s(&self) -> f64 {
        self.duration_s
    }

    /// ISO-8601 epoch of the time axis, when recorded.  Compare with
    /// `BankSpectrum.pulse_time_offset_iso` to confirm both clocks share
    /// a zero point.
    #[getter]
    fn offset_iso(&self) -> Option<String> {
        self.offset_iso.clone()
    }

    /// Entries dropped as corrupt device-reconnect records (backward time
    /// jump or subnormal value payload, both seen in real SNS files).
    /// Non-zero is worth a mention in run-health screens.
    #[getter]
    fn n_dropped_corrupt(&self) -> usize {
        self.n_dropped_corrupt
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let offset = match &self.offset_iso {
            Some(o) => format!("{o:?}"),
            None => "None".to_string(),
        };
        format!(
            "RunLog(n_entries={}, duration_s={}, offset={}, n_dropped_corrupt={})",
            self.times.bind(py).len(),
            self.duration_s,
            offset,
            self.n_dropped_corrupt,
        )
    }
}

/// Read a DASlogs PV from a NeXus file as a transition log (issue #637).
///
/// SNS/HFIR DASlogs are **transition logs**, not uniformly-sampled time
/// series: `values[i]` takes effect at `times[i]` and persists until the
/// next entry (the last value persists to the end of the run).  Averaging
/// `values` directly is therefore wrong whenever entries are unevenly
/// spaced — on a real VENUS run the entry-mean of the `pause` log read
/// 0.43 while the time-weighted truth was 0.90.  Use `intervals_where`
/// to derive time intervals with the correct step-function semantics.
///
/// Args:
///     path: Path to the NeXus/HDF5 file.
///     pv: PV name under `/entry/DASlogs/` (e.g. "pause",
///         "BL10:Det:rtdl:BeamPowerAvg").
///
/// Returns:
///     RunLog with times (s, relative to run start), values, duration_s,
///     the ISO-8601 epoch of the time axis when recorded, and
///     n_dropped_corrupt — the number of corrupt device-reconnect
///     records (backward time jump or subnormal value payload) dropped.
#[pyfunction]
#[pyo3(signature = (path, pv))]
fn read_run_log(py: Python<'_>, path: &str, pv: &str) -> PyResult<PyRunLog> {
    let log =
        nereids_io::runlog::read_run_log(std::path::Path::new(path), pv).map_err(map_io_error)?;
    Ok(PyRunLog {
        times: PyArray1::from_vec(py, log.times).unbind(),
        values: PyArray1::from_vec(py, log.values).unbind(),
        duration_s: log.duration_s,
        offset_iso: log.offset_iso,
        n_dropped_corrupt: log.n_dropped_corrupt,
    })
}

/// Derive run-time intervals where a transition-log PV satisfies
/// `min_value <= value <= max_value` (issue #637).
///
/// Uses the correct step-function semantics: `values[i]` holds on
/// `[times[i], times[i+1])` and the last value holds to `duration_s`.
/// Time before the first log entry is treated as NOT matching (the state
/// is unrecorded — the conservative choice for a keep-filter), and NaN
/// values never match.  Adjacent matching segments are merged.
///
/// Example — beam-state filtering for a paused run::
///
///     pause = read_run_log(path, "pause")
///     live = intervals_where(pause.times, pause.values,
///                            pause.duration_s, max_value=0.5)
///     power = read_run_log(path, "BL10:Det:rtdl:BeamPowerAvg")
///     stable = intervals_where(power.times, power.values,
///                              power.duration_s, min_value=1.5)
///     keep = intervals_intersect(live, stable)
///
/// Args:
///     times: Transition times in seconds (ascending).
///     values: PV values taking effect at each time.
///     duration_s: Total run duration in seconds.
///     min_value: Keep intervals where value >= min_value (optional).
///     max_value: Keep intervals where value <= max_value (optional).
///
/// Returns:
///     List of (t_start, t_end) tuples in seconds, sorted, non-overlapping.
#[pyfunction]
#[pyo3(signature = (times, values, duration_s, min_value=None, max_value=None))]
fn intervals_where(
    times: Vec<f64>,
    values: Vec<f64>,
    duration_s: f64,
    min_value: Option<f64>,
    max_value: Option<f64>,
) -> PyResult<Vec<(f64, f64)>> {
    nereids_io::runlog::intervals_where(&times, &values, duration_s, min_value, max_value)
        .map_err(map_io_error)
}

/// Intersect two interval lists (issue #637).
///
/// Composes conditions across PVs, e.g. `pause == 0` AND
/// `beam_power > 1.5 MW`.  Inputs are lists of (t_start, t_end) tuples;
/// unsorted or overlapping lists are normalised first (every pair must be
/// finite with t_end > t_start).  The output is sorted, non-overlapping,
/// and drops empty intersections.
#[pyfunction]
#[pyo3(signature = (a, b))]
fn intervals_intersect(a: Vec<(f64, f64)>, b: Vec<(f64, f64)>) -> PyResult<Vec<(f64, f64)>> {
    nereids_io::runlog::intervals_intersect(&a, &b).map_err(map_io_error)
}

/// A 1-D TOF spectrum from one NXevent_data bank, with pulse/event
/// retention statistics.
#[pyclass(name = "BankSpectrum")]
struct PyBankSpectrum {
    tof_edges_us: Py<PyArray1<f64>>,
    counts: Py<PyArray1<u64>>,
    pulses_total: usize,
    pulses_kept: usize,
    events_total: usize,
    events_kept: usize,
    dropped_tof_range: usize,
    dropped_non_finite: usize,
    pulse_time_offset_iso: Option<String>,
}

#[pymethods]
impl PyBankSpectrum {
    /// TOF bin edges in microseconds (length = n_bins + 1).
    #[getter]
    fn tof_edges_us<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.tof_edges_us.bind(py).clone()
    }

    /// Event counts per TOF bin (length = n_bins).
    #[getter]
    fn counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        self.counts.bind(py).clone()
    }

    /// Total pulses recorded in the bank.
    #[getter]
    fn pulses_total(&self) -> usize {
        self.pulses_total
    }

    /// Pulses inside keep_intervals (= pulses_total when unfiltered).
    #[getter]
    fn pulses_kept(&self) -> usize {
        self.pulses_kept
    }

    /// Total events recorded in the bank.
    #[getter]
    fn events_total(&self) -> usize {
        self.events_total
    }

    /// Events on kept pulses inside the TOF window.
    #[getter]
    fn events_kept(&self) -> usize {
        self.events_kept
    }

    /// Events on kept pulses dropped for TOF outside the window.
    #[getter]
    fn dropped_tof_range(&self) -> usize {
        self.dropped_tof_range
    }

    /// Events on kept pulses dropped for non-finite TOF.
    #[getter]
    fn dropped_non_finite(&self) -> usize {
        self.dropped_non_finite
    }

    /// ISO-8601 epoch of the pulse clock, when recorded.  Compare with
    /// `RunLog.offset_iso` to confirm both clocks share a zero point.
    #[getter]
    fn pulse_time_offset_iso(&self) -> Option<String> {
        self.pulse_time_offset_iso.clone()
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "BankSpectrum(n_bins={}, pulses={}/{}, events={}/{})",
            self.counts.bind(py).len(),
            self.pulses_kept,
            self.pulses_total,
            self.events_kept,
            self.events_total,
        )
    }
}

/// Load one NXevent_data bank (e.g. a beam monitor) as a 1-D TOF spectrum,
/// optionally keeping only pulses inside wall-clock intervals (issue #637).
///
/// Reads `/entry/<bank>/{event_time_offset, event_index, event_time_zero}`
/// (the facility NeXus convention).  The `units` attributes on
/// `event_time_offset` AND `event_time_zero` are required — NXevent_data
/// specifies no defaults, and refusing to guess closes the #554
/// silent-rescale class.
/// A bank with zero events loads to an all-zero spectrum with correct
/// pulse statistics — it never errors (on VENUS every imaging-detector
/// bank is empty because tpx1 is frame-mode; only monitors carry events).
///
/// Args:
///     path: Path to the NeXus/HDF5 file.
///     bank: Bank group name under `/entry` (e.g. "monitor1",
///         "bank100_events").
///     n_bins: Number of TOF bins.
///     tof_min_us: Minimum TOF in microseconds (inclusive).
///     tof_max_us: Maximum TOF in microseconds (exclusive).
///     keep_intervals: Optional list of (t_start, t_end) pairs in seconds
///         on the pulse clock (at SNS: seconds since run start, the same
///         clock as DASlogs times — pass the output of `intervals_where`
///         / `intervals_intersect` directly).  A pulse is kept iff
///         t_start <= event_time_zero < t_end for some interval.
///
/// Returns:
///     BankSpectrum with tof_edges_us, counts, and pulse/event stats.
#[pyfunction]
#[pyo3(signature = (path, bank, n_bins, tof_min_us, tof_max_us, keep_intervals=None))]
fn load_nexus_bank_spectrum(
    py: Python<'_>,
    path: &str,
    bank: &str,
    n_bins: usize,
    tof_min_us: f64,
    tof_max_us: f64,
    keep_intervals: Option<Vec<(f64, f64)>>,
) -> PyResult<PyBankSpectrum> {
    let params = nereids_io::nexus::BankBinningParams {
        n_bins,
        tof_min_us,
        tof_max_us,
    };
    let s = nereids_io::nexus::load_nexus_bank_spectrum(
        std::path::Path::new(path),
        bank,
        &params,
        keep_intervals.as_deref(),
    )
    .map_err(map_io_error)?;
    Ok(PyBankSpectrum {
        tof_edges_us: PyArray1::from_vec(py, s.tof_edges_us).unbind(),
        counts: PyArray1::from_vec(py, s.counts).unbind(),
        pulses_total: s.pulses_total,
        pulses_kept: s.pulses_kept,
        events_total: s.events_total,
        events_kept: s.events_kept,
        dropped_tof_range: s.dropped_tof_range,
        dropped_non_finite: s.dropped_non_finite,
        pulse_time_offset_iso: s.pulse_time_offset_iso,
    })
}

/// Convert Rust NexusHistogramData to Python PyNexusData.
fn nexus_data_to_py(py: Python<'_>, data: nereids_io::nexus::NexusHistogramData) -> PyNexusData {
    let (event_total, event_kept) = data
        .event_stats
        .as_ref()
        .map(|s| (Some(s.total), Some(s.kept)))
        .unwrap_or((None, None));
    PyNexusData {
        counts: PyArray3::from_owned_array(py, data.counts).unbind(),
        tof_edges_us: PyArray1::from_vec(py, data.tof_edges_us).unbind(),
        flight_path_m: data.flight_path_m,
        dead_pixels: data
            .dead_pixels
            .map(|dp| PyArray2::from_owned_array(py, dp).unbind()),
        n_rotation_angles: data.n_rotation_angles,
        event_total,
        event_kept,
    }
}

/// NEREIDS Python module.
#[pymodule]
fn nereids(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyResonanceData>()?;
    m.add_class::<PyFitResult>()?;
    m.add_class::<PyTabulatedResolution>()?;
    m.add_class::<PyEnergyLaw>()?;
    m.add_class::<PyIkedaCarpenter>()?;
    m.add_class::<PyTwoArmBackgroundFitResult>()?;
    m.add_class::<PyResolutionCalibration>()?;
    m.add_class::<PySpatialResult>()?;
    m.add_class::<PyTraceDetectabilityReport>()?;
    m.add_function(wrap_pyfunction!(cross_sections, m)?)?;
    m.add_function(wrap_pyfunction!(forward_model, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_resolution, m)?)?;
    m.add_function(wrap_pyfunction!(tof_to_energy, m)?)?;
    m.add_function(wrap_pyfunction!(energy_to_tof, m)?)?;
    m.add_function(wrap_pyfunction!(load_endf, m)?)?;
    m.add_function(wrap_pyfunction!(load_endf_file, m)?)?;
    m.add_function(wrap_pyfunction!(create_resonance_data, m)?)?;
    m.add_function(wrap_pyfunction!(beer_lambert, m)?)?;
    m.add_function(wrap_pyfunction!(doppler_broaden, m)?)?;
    m.add_function(wrap_pyfunction!(resolution_broaden, m)?)?;
    m.add_function(wrap_pyfunction!(load_resolution, m)?)?;
    m.add_function(wrap_pyfunction!(py_apply_resolution, m)?)?;
    m.add_function(wrap_pyfunction!(py_two_arm_count_response, m)?)?;
    m.add_function(wrap_pyfunction!(fit_two_arm_background_templates, m)?)?;
    m.add_function(wrap_pyfunction!(load_tiff_stack, m)?)?;
    m.add_function(wrap_pyfunction!(load_tiff_folder, m)?)?;
    m.add_function(wrap_pyfunction!(read_tof_sidecar, m)?)?;
    m.add_class::<PyNexusMetadata>()?;
    m.add_class::<PyNexusData>()?;
    m.add_class::<PyRunHealth>()?;
    m.add_function(wrap_pyfunction!(probe_nexus, m)?)?;
    m.add_function(wrap_pyfunction!(run_health, m)?)?;
    m.add_function(wrap_pyfunction!(load_nexus_histogram, m)?)?;
    m.add_function(wrap_pyfunction!(load_nexus_events, m)?)?;
    m.add_class::<PyRunLog>()?;
    m.add_class::<PyBankSpectrum>()?;
    m.add_function(wrap_pyfunction!(read_run_log, m)?)?;
    m.add_function(wrap_pyfunction!(intervals_where, m)?)?;
    m.add_function(wrap_pyfunction!(intervals_intersect, m)?)?;
    m.add_function(wrap_pyfunction!(load_nexus_bank_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(tof_to_energy_centers, m)?)?;
    m.add_function(wrap_pyfunction!(py_element_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(py_element_name, m)?)?;
    m.add_function(wrap_pyfunction!(py_parse_isotope_str, m)?)?;
    m.add_function(wrap_pyfunction!(py_natural_abundance, m)?)?;
    m.add_function(wrap_pyfunction!(py_natural_isotopes, m)?)?;
    m.add_function(wrap_pyfunction!(py_trace_detectability, m)?)?;
    m.add_function(wrap_pyfunction!(py_trace_detectability_survey, m)?)?;
    m.add_function(wrap_pyfunction!(precompute_cross_sections, m)?)?;
    m.add_function(wrap_pyfunction!(detect_dead_pixels, m)?)?;
    m.add_function(wrap_pyfunction!(detect_hot_pixels, m)?)?;
    m.add_function(wrap_pyfunction!(detect_dead_pixels_chunked, m)?)?;
    m.add_function(wrap_pyfunction!(detect_bad_pixels, m)?)?;
    m.add_function(wrap_pyfunction!(py_calibrate_energy, m)?)?;
    m.add_class::<PyCalibrationResult>()?;
    // Phase 5: Typed API
    m.add_class::<PyInputData>()?;
    m.add_class::<PyIsotopeGroup>()?;
    m.add_function(wrap_pyfunction!(py_from_counts, m)?)?;
    m.add_function(wrap_pyfunction!(py_from_counts_with_nuisance, m)?)?;
    m.add_function(wrap_pyfunction!(py_from_transmission, m)?)?;
    m.add_function(wrap_pyfunction!(py_spatial_map_typed, m)?)?;
    m.add_function(wrap_pyfunction!(py_fit_spectrum_typed, m)?)?;
    m.add_function(wrap_pyfunction!(py_fit_counts_spectrum_typed, m)?)?;
    m.add_class::<PyModelJacobianResult>()?;
    m.add_function(wrap_pyfunction!(py_compute_model_jacobian, m)?)?;
    Ok(())
}

// ── Phase 5: Typed Python API ────────────────────────────────────────────

use nereids_pipeline::pipeline::UnifiedFitConfig;
use nereids_pipeline::spatial::{InputData3D, spatial_map_typed};

/// Opaque wrapper around InputData3D for Python.
///
/// Created via `from_counts()`, `from_counts_with_nuisance()`, or `from_transmission()`.
/// Passed to `spatial_map_typed()`.
#[pyclass(name = "InputData")]
struct PyInputData {
    /// We store owned 3D arrays (ndarray::Array3) so the data lives
    /// as long as the Python object.
    kind: String, // "counts" or "transmission"
    data_a: ndarray::Array3<f64>,
    data_b: ndarray::Array3<f64>,
    data_c: Option<ndarray::Array3<f64>>,
}

#[pymethods]
impl PyInputData {
    fn __repr__(&self) -> String {
        let s = self.data_a.shape();
        format!(
            "InputData(kind={}, shape=({}, {}, {}))",
            self.kind, s[0], s[1], s[2]
        )
    }

    #[getter]
    fn kind(&self) -> &str {
        &self.kind
    }

    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        let s = self.data_a.shape();
        (s[0], s[1], s[2])
    }
}

/// Create InputData from raw detector counts and open beam.
///
/// The fitting engine will use Poisson KL by default (statistically
/// optimal for count data).
///
/// **Note:** Both arrays must have dtype `np.float64`. Neutron event histograms
/// are naturally `int64`; call `.astype(np.float64)` before passing them here.
///
/// Args:
///     sample_counts: 3D float64 array (n_energies, height, width) of sample counts.
///     open_beam_counts: 3D float64 array (n_energies, height, width) of open beam counts.
///
/// Returns:
///     InputData object to pass to spatial_map_typed().
#[pyfunction]
#[pyo3(name = "from_counts")]
fn py_from_counts<'py>(
    sample_counts: PyReadonlyArray3<'py, f64>,
    open_beam_counts: PyReadonlyArray3<'py, f64>,
) -> PyResult<PyInputData> {
    let sample = sample_counts.as_array().to_owned();
    let ob = open_beam_counts.as_array().to_owned();
    if sample.shape() != ob.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "sample shape {:?} != open_beam shape {:?}",
            sample.shape(),
            ob.shape()
        )));
    }
    if sample.shape()[0] == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "spectral axis (dimension 0) must have at least 1 element",
        ));
    }
    Ok(PyInputData {
        kind: "counts".into(),
        data_a: sample,
        data_b: ob,
        data_c: None,
    })
}

/// Legacy raw-count nuisance wrapper retained for compatibility.
///
/// Production fitting rejects a nonzero background because this legacy array
/// is not connected to the physical two-arm likelihood. New code should use
/// `from_counts` and fit independently measured detector-bin shapes with
/// `fit_two_arm_background_templates`.
#[pyfunction]
#[pyo3(name = "from_counts_with_nuisance")]
fn py_from_counts_with_nuisance<'py>(
    sample_counts: PyReadonlyArray3<'py, f64>,
    flux: PyReadonlyArray3<'py, f64>,
    background: PyReadonlyArray3<'py, f64>,
) -> PyResult<PyInputData> {
    let sample = sample_counts.as_array().to_owned();
    let flux_arr = flux.as_array().to_owned();
    let background_arr = background.as_array().to_owned();
    if sample.shape() != flux_arr.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "sample shape {:?} != flux shape {:?}",
            sample.shape(),
            flux_arr.shape()
        )));
    }
    if sample.shape() != background_arr.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "sample shape {:?} != background shape {:?}",
            sample.shape(),
            background_arr.shape()
        )));
    }
    if sample.shape()[0] == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "spectral axis (dimension 0) must have at least 1 element",
        ));
    }
    Ok(PyInputData {
        kind: "counts_with_nuisance".into(),
        data_a: sample,
        data_b: flux_arr,
        data_c: Some(background_arr),
    })
}

/// Create InputData from normalized transmission and uncertainty.
///
/// The fitting engine uses LM. A Poisson/KL count likelihood is rejected for
/// normalized transmission because the separate count arms are no longer
/// available.
///
/// **Note:** Both arrays must have dtype `np.float64`. Call `.astype(np.float64)`
/// if your arrays are a different type.
///
/// Args:
///     transmission: 3D float64 array (n_energies, height, width) of transmission values.
///     uncertainty: 3D float64 array (n_energies, height, width) of uncertainties.
///
/// Returns:
///     InputData object to pass to spatial_map_typed().
#[pyfunction]
#[pyo3(name = "from_transmission")]
fn py_from_transmission<'py>(
    transmission: PyReadonlyArray3<'py, f64>,
    uncertainty: PyReadonlyArray3<'py, f64>,
) -> PyResult<PyInputData> {
    let t = transmission.as_array().to_owned();
    let u = uncertainty.as_array().to_owned();
    if t.shape() != u.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "transmission shape {:?} != uncertainty shape {:?}",
            t.shape(),
            u.shape()
        )));
    }
    if t.shape()[0] == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "spectral axis (dimension 0) must have at least 1 element",
        ));
    }
    Ok(PyInputData {
        kind: "transmission".into(),
        data_a: t,
        data_b: u,
        data_c: None,
    })
}

/// Parse a solver string into SolverConfig, resolving "auto" eagerly.
fn parse_solver_config(
    solver: &str,
    is_counts: bool,
    max_iter: usize,
) -> PyResult<nereids_pipeline::pipeline::SolverConfig> {
    match solver {
        "auto" => {
            if is_counts {
                Ok(nereids_pipeline::pipeline::SolverConfig::PoissonKL(
                    nereids_fitting::poisson::PoissonConfig {
                        max_iter,
                        ..Default::default()
                    },
                ))
            } else {
                Ok(
                    nereids_pipeline::pipeline::SolverConfig::LevenbergMarquardt(
                        nereids_fitting::lm::LmConfig {
                            max_iter,
                            ..Default::default()
                        },
                    ),
                )
            }
        }
        "lm" if is_counts => Err(pyo3::exceptions::PyValueError::new_err(
            "raw sample/open-beam counts cannot use solver='lm': dividing the \
             count arms into transmission loses count statistics; use \
             solver='auto' or solver='kl'",
        )),
        "lm" => Ok(
            nereids_pipeline::pipeline::SolverConfig::LevenbergMarquardt(
                nereids_fitting::lm::LmConfig {
                    max_iter,
                    ..Default::default()
                },
            ),
        ),
        // Counts-KL dispatch.  "kl" is the canonical name;  "poisson" and
        // "joint_poisson" are compatibility aliases that resolve to the
        // same path — the joint-Poisson / conditional-binomial-deviance
        // implementation IS the KL solver.  No runtime deprecation
        // warning is emitted; the aliases simply accept the older name
        // strings so existing user scripts keep working.
        "kl" | "poisson" | "joint_poisson" if !is_counts => {
            Err(pyo3::exceptions::PyValueError::new_err(
                "normalized transmission cannot use a Poisson/KL count \
                 likelihood because the separate open/sample count arms are \
                 unavailable; use solver='auto' or solver='lm'",
            ))
        }
        "kl" | "poisson" | "joint_poisson" => {
            Ok(nereids_pipeline::pipeline::SolverConfig::PoissonKL(
                nereids_fitting::poisson::PoissonConfig {
                    max_iter,
                    ..Default::default()
                },
            ))
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown solver: '{other}'. Use 'auto', 'lm', or 'kl'."
        ))),
    }
}

/// Validate the SAMMY TZERO kwargs before they reach
/// `UnifiedFitConfig::with_energy_scale`.  Shared across
/// `py_spatial_map_typed`, `py_fit_spectrum_typed`, and
/// `py_fit_counts_spectrum_typed`.
///
/// Issue #458: without these checks, NaN /
/// Inf / non-positive values flowed into
/// `EnergyScaleTransmissionModel::corrected_energies`, which divides
/// by TOF values derived from `flight_path_m`.  Garbage inputs yielded
/// NaN grids and confusing `PyRuntimeError`s from the solver rather
/// than an actionable `PyValueError` at the binding boundary.
fn validate_energy_scale_params(
    t0_init_us: f64,
    l_scale_init: f64,
    energy_scale_flight_path_m: f64,
) -> PyResult<()> {
    if !t0_init_us.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "t0_init_us must be finite when fit_energy_scale=True, got {t0_init_us}"
        )));
    }
    if !l_scale_init.is_finite() || l_scale_init <= 0.0 {
        // Must be positive: l_scale is a multiplicative flight-path scale, and a
        // non-positive value drives the corrected energies to 0 / non-finite,
        // which the true-σ energy-scale model (issue #608) cannot evaluate
        // (reich_moore requires positive energy).  Mirrors the flight_path check
        // below.
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "l_scale_init must be finite and positive when fit_energy_scale=True, \
             got {l_scale_init}"
        )));
    }
    if !energy_scale_flight_path_m.is_finite() || energy_scale_flight_path_m <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "energy_scale_flight_path_m must be finite and positive when \
             fit_energy_scale=True, got {energy_scale_flight_path_m}"
        )));
    }
    Ok(())
}

/// Parse the optional `tzero_jacobian` Python kwarg into the Rust
/// `EnergyScaleJacobianMethod` enum.
///
/// Recognised values (case-insensitive):
/// - `"fd2"`, `"finite-difference"`, `"finite_difference"` → FiniteDifference.
/// - `"partial-gal"`, `"partial_gal"`                      → PartialGal.
///
/// The legacy `"chain"` / `"frozen-r"` FrozenResolutionChainRule method was
/// removed in #608 (it interpolated a precomputed σ on the data grid,
/// incompatible with the true-σ aux-grid `evaluate`); use `"partial-gal"`.
///
/// `None` returns `Ok(None)`, deferring to the Rust model's
/// `EnergyScaleJacobianMethod::from_env`: the `NEREIDS_TZERO_JACOBIAN`
/// env var when set, otherwise `PartialGal` (default since issue #489).
fn parse_tzero_jacobian(
    s: Option<&str>,
) -> PyResult<Option<nereids_fitting::transmission_model::EnergyScaleJacobianMethod>> {
    use nereids_fitting::transmission_model::EnergyScaleJacobianMethod;
    let Some(name) = s else {
        return Ok(None);
    };
    let m = if name.eq_ignore_ascii_case("fd2")
        || name.eq_ignore_ascii_case("finite-difference")
        || name.eq_ignore_ascii_case("finite_difference")
    {
        EnergyScaleJacobianMethod::FiniteDifference
    } else if name.eq_ignore_ascii_case("partial-gal") || name.eq_ignore_ascii_case("partial_gal") {
        EnergyScaleJacobianMethod::PartialGal
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "tzero_jacobian must be one of: \
             \"fd2\", \"finite-difference\", \"finite_difference\"; \
             \"partial-gal\", \"partial_gal\"; got {name:?}"
        )));
    };
    Ok(Some(m))
}

/// Build `UnifiedFitConfig` from groups, returning the config and the number of
/// density parameters (one per group) for initial_densities default.
fn build_config_from_groups(
    groups: &[PyIsotopeGroup],
    energies_vec: Vec<f64>,
    temperature_k: f64,
    res_fn: Option<ResolutionFunction>,
    initial_densities: Option<Vec<f64>>,
) -> PyResult<UnifiedFitConfig> {
    if groups.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "groups must not be empty",
        ));
    }
    // Validate all groups are loaded
    for g in groups {
        if !g.is_loaded() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "IsotopeGroup '{}' has not been fully loaded. Call load_endf() first.",
                g.inner.name(),
            )));
        }
    }

    let n_groups = groups.len();
    let init_densities = initial_densities.unwrap_or_else(|| vec![0.001; n_groups]);

    // Build the groups slice for with_groups: &[(&IsotopeGroup, &[ResonanceData])]
    let group_rd: Vec<Vec<ResonanceData>> = groups
        .iter()
        .map(|g| {
            g.resonance_data
                .iter()
                .map(|d| Arc::unwrap_or_clone(d.clone().unwrap()))
                .collect()
        })
        .collect();

    let group_pairs: Vec<(&IsotopeGroup, &[ResonanceData])> = groups
        .iter()
        .zip(group_rd.iter())
        .map(|(g, rd)| (&g.inner, rd.as_slice()))
        .collect();

    // Create a placeholder config first (with_groups requires a valid base config)
    // We use the first member's data as placeholder — with_groups replaces everything.
    let first_rd = Arc::unwrap_or_clone(groups[0].resonance_data[0].clone().unwrap());
    let placeholder_name = groups[0].inner.name().to_string();
    let base_config = UnifiedFitConfig::new(
        energies_vec,
        vec![first_rd],
        vec![placeholder_name],
        temperature_k,
        res_fn,
        vec![0.001],
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let config = base_config
        .with_groups(&group_pairs, init_densities)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(config)
}

/// Convert a pipeline `SpatialResult` to the Python `PySpatialResult`.
fn spatial_result_to_py(
    py: Python<'_>,
    result: &nereids_pipeline::spatial::SpatialResult,
) -> PySpatialResult {
    let density_maps: Vec<Py<PyArray2<f64>>> = result
        .density_maps
        .iter()
        .map(|m| PyArray2::from_array(py, m).into())
        .collect();
    let uncertainty_maps: Vec<Py<PyArray2<f64>>> = result
        .uncertainty_maps
        .iter()
        .map(|m| PyArray2::from_array(py, m).into())
        .collect();
    let shape = (
        result.converged_map.shape()[0],
        result.converged_map.shape()[1],
    );
    let anorm_map = result
        .anorm_map
        .as_ref()
        .map(|m| PyArray2::from_array(py, m).into());
    let background_maps = result.background_maps.as_ref().map(|maps| {
        [
            PyArray2::from_array(py, &maps[0]).into(),
            PyArray2::from_array(py, &maps[1]).into(),
            PyArray2::from_array(py, &maps[2]).into(),
        ]
    });
    let back_d_map = result
        .back_d_map
        .as_ref()
        .map(|m| PyArray2::from_array(py, m).into());
    let back_f_map = result
        .back_f_map
        .as_ref()
        .map(|m| PyArray2::from_array(py, m).into());
    let temperature_map = result
        .temperature_map
        .as_ref()
        .map(|m| PyArray2::from_array(py, m).into());
    let temperature_uncertainty_map = result
        .temperature_uncertainty_map
        .as_ref()
        .map(|m| PyArray2::from_array(py, m).into());
    let deviance_per_dof_map = result
        .deviance_per_dof_map
        .as_ref()
        .map(|m| PyArray2::from_array(py, m).into());
    let t0_us_map = result
        .t0_us_map
        .as_ref()
        .map(|m| PyArray2::from_array(py, m).into());
    let l_scale_map = result
        .l_scale_map
        .as_ref()
        .map(|m| PyArray2::from_array(py, m).into());
    let baseline_maps = result.baseline_maps.as_ref().map(|maps| {
        [
            PyArray2::from_array(py, &maps[0]).into(),
            PyArray2::from_array(py, &maps[1]).into(),
            PyArray2::from_array(py, &maps[2]).into(),
        ]
    });

    PySpatialResult {
        density_maps,
        uncertainty_maps,
        chi_squared_map: PyArray2::from_array(py, &result.chi_squared_map).into(),
        deviance_per_dof_map,
        converged_map: PyArray2::from_array(py, &result.converged_map).into(),
        n_converged: result.n_converged,
        n_total: result.n_total,
        n_failed: result.n_failed,
        isotope_names: result.isotope_labels.clone(),
        shape,
        temperature_map,
        temperature_uncertainty_map,
        anorm_map,
        background_maps,
        back_d_map,
        back_f_map,
        t0_us_map,
        l_scale_map,
        baseline_global: result.baseline_global,
        baseline_e_ref_ev: result.baseline_e_ref_ev,
        baseline_maps,
        warnings: result.warnings.clone(),
    }
}

/// Spatial mapping using the typed input data API.
///
/// Dispatches per-pixel fitting based on the InputData type:
///   - from_counts → Poisson KL on raw counts (statistically optimal)
///   - from_transmission → LM; Poisson/KL is rejected for ratios
///
/// Either `isotopes` or `groups` must be provided, but not both.
/// When `groups` is provided, each group maps to one fitted density parameter.
///
/// Always returns SpatialResult.
///
/// Args:
///     data: InputData from `from_counts()`, `from_counts_with_nuisance()`,
///         or `from_transmission()`.
///     energies: 1D energy grid in eV (ascending).
///     isotopes: list of ResonanceData objects (mutually exclusive with groups).
///     temperature_k: Sample temperature in Kelvin (default 293.6).
///     fit_temperature: Whether to fit temperature per pixel (default False).
///     initial_densities: Initial density guesses (default 0.001 each).
///     dead_pixels: Optional 2D boolean dead pixel mask.
///     max_iter: Maximum iterations per pixel (default 200).
///     solver: "auto" (default), "lm", or "kl".
///     background: Enable transmission-background fitting.
///         For transmission data this uses the transmission-domain background model.
///         For counts data this enables the same transmission background inside the
///         count-domain KL/LM pipelines.
///     fit_alpha_1: Fit counts nuisance flux scale `alpha_1` when using
///         `from_counts_with_nuisance()`.
///     fit_alpha_2: Fit detector-background scale `alpha_2` when using
///         `from_counts_with_nuisance()`.
///     alpha_1_init: Initial value for `alpha_1` (default 1.0).
///     alpha_2_init: Initial value for `alpha_2` (default 1.0).
///     fit_energy_scale: Fit per-pixel SAMMY TZERO calibration (t0, L_scale).
///         Required for real VENUS counts data to match SAMMY chi2 performance.
///     t0_init_us: Initial TOF offset in microseconds (default 0.0).
///     l_scale_init: Initial flight-path scale factor (default 1.0).
///     energy_scale_flight_path_m: Nominal flight path (m) for the
///         energy-scale model. Must match the grid used to compute `energies`.
///     resolution: Optional resolution function.
///     groups: list of IsotopeGroup objects (mutually exclusive with isotopes).
///     fit_anorm: Whether Anorm is free when ``background=True`` (default
///         True).  Must be False to combine ``background=True`` with
///         ``baseline=True`` (b0 and Anorm are degenerate normalizations).
///     baseline: Enable the bounded multiplicative baseline
///         ``B(E) = b0 + b1·ln(E/E_ref) + b2·ln²(E/E_ref)`` applied
///         OUTERMOST (issue #635).  E_ref is the geometric midpoint of the
///         energy grid and is reported as ``FitResult.baseline_e_ref_ev``.
///     fit_b0, fit_b1, fit_b2: Per-coefficient fit flags (default True).
///         All-False freezes the baseline at its inits.
///     b0_init, b1_init, b2_init: Initial coefficients (default 1, 0, 0 —
///         the identity baseline).
///     b0_bounds, b1_bounds, b2_bounds: Optional (lower, upper) optimizer
///         boxes; defaults (0.9, 1.1) / (−0.05, 0.05) / (−0.05, 0.05).
///         The boxes bound the coefficients (tilt/curvature per ln-E
///         unit), not the evaluated B(E); dip protection comes from the
///         quadratic's smoothness, and B(E) > 0 is enforced pointwise.
///     baseline_global: With ``baseline=True``: True (default) fits the
///         baseline ONCE on the aggregated mean spectrum and freezes it for
///         every pixel (stage-1 non-convergence is a hard error); False
///         fits a baseline per pixel and populates ``baseline_maps``.
///     scale_by_chi2: When True, inflate the covariance-only uncertainties
///         (incl. ``temperature_uncertainty_map``) by ``sqrt(chi2/dof)`` at
///         convergence — the inverse-Fisher lower bound becomes a
///         goodness-of-fit-scaled estimate, scaled by the goodness-of-fit each
///         pixel's result reports (Gaussian reduced-chi2 on the transmission
///         paths incl. Poisson-KL, deviance-per-dof on the counts joint-Poisson
///         path). No-op on the already-chi2-scaled LM transmission path.
///         Default False (issue #638).
///
/// Returns:
///     SpatialResult with density_maps, chi_squared_map, converged_map, etc.
#[pyfunction]
#[pyo3(name = "spatial_map_typed", signature = (
    data, energies, isotopes=None, *,
    temperature_k = 293.6,
    fit_temperature = false,
    initial_densities = None,
    fix_densities = false,
    density_free = None,
    dead_pixels = None,
    max_iter = 200,
    solver = "auto",
    background = false,
    fit_back_d = false,
    fit_back_f = false,
    back_d_init = 0.01,
    back_f_init = 1.0,
    fit_alpha_1 = false,
    fit_alpha_2 = false,
    alpha_1_init = 1.0,
    alpha_2_init = 1.0,
    c = 1.0,
    enable_polish = None,
    fit_energy_scale = false,
    t0_init_us = 0.0,
    l_scale_init = 1.0,
    energy_scale_flight_path_m = 25.0,
    resolution = None,
    flight_path_m = None,
    delta_t_us = None,
    delta_l_m = None,
    groups = None,
    tzero_jacobian = None,
    fit_energy_range = None,
    fit_anorm = true,
    baseline = false,
    fit_b0 = true,
    fit_b1 = true,
    fit_b2 = true,
    b0_init = 1.0,
    b1_init = 0.0,
    b2_init = 0.0,
    b0_bounds = None,
    b1_bounds = None,
    b2_bounds = None,
    baseline_global = true,
    scale_by_chi2 = false,
))]
#[allow(clippy::too_many_arguments)]
fn py_spatial_map_typed<'py>(
    py: Python<'py>,
    data: &PyInputData,
    energies: PyReadonlyArray1<'py, f64>,
    isotopes: Option<Vec<PyResonanceData>>,
    temperature_k: f64,
    fit_temperature: bool,
    initial_densities: Option<Vec<f64>>,
    fix_densities: bool,
    density_free: Option<Vec<bool>>,
    dead_pixels: Option<PyReadonlyArray2<'py, bool>>,
    max_iter: usize,
    solver: &str,
    background: bool,
    fit_back_d: bool,
    fit_back_f: bool,
    back_d_init: f64,
    back_f_init: f64,
    fit_alpha_1: bool,
    fit_alpha_2: bool,
    alpha_1_init: f64,
    alpha_2_init: f64,
    c: f64,
    enable_polish: Option<bool>,
    fit_energy_scale: bool,
    t0_init_us: f64,
    l_scale_init: f64,
    energy_scale_flight_path_m: f64,
    resolution: Option<PyTabulatedResolution>,
    flight_path_m: Option<f64>,
    delta_t_us: Option<f64>,
    delta_l_m: Option<f64>,
    groups: Option<Vec<PyIsotopeGroup>>,
    tzero_jacobian: Option<&str>,
    fit_energy_range: Option<(f64, f64)>,
    fit_anorm: bool,
    baseline: bool,
    fit_b0: bool,
    fit_b1: bool,
    fit_b2: bool,
    b0_init: f64,
    b1_init: f64,
    b2_init: f64,
    b0_bounds: Option<(f64, f64)>,
    b1_bounds: Option<(f64, f64)>,
    b2_bounds: Option<(f64, f64)>,
    baseline_global: bool,
    scale_by_chi2: bool,
) -> PyResult<PySpatialResult> {
    // Validate mutual exclusivity
    let has_isotopes = isotopes.is_some();
    let has_groups = groups.is_some();
    if has_isotopes && has_groups {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Provide either 'isotopes' or 'groups', not both.",
        ));
    }
    if !has_isotopes && !has_groups {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Must provide either 'isotopes' or 'groups'.",
        ));
    }

    // ── Issue #458 V1-V3: input validation at the binding boundary ──

    // V1: proton-charge ratio `c` is used by the counts-KL dispatch to
    // relate sample to open-beam flux.  It is ignored on the
    // transmission path, so we only validate it when the input is a
    // counts variant — otherwise a user passing (say) `c=0.0` with
    // transmission data would see a misleading error about a value
    // that was never consulted.  Non-positive or non-finite values
    // produce garbage fits deep in `joint_poisson_fit`; reject here
    // with a clear message instead of letting a PyRuntimeError
    // bubble up from the solver.
    let is_counts_input = data.kind == "counts" || data.kind == "counts_with_nuisance";
    if is_counts_input && (!c.is_finite() || c <= 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "c (proton-charge ratio Q_s/Q_ob) must be positive and finite, got {c}",
        )));
    }

    // V2: `initial_densities`, when supplied, must all be finite and
    // non-negative.  NaN/Inf propagates through the solver and
    // produces meaningless output; negative densities are non-physical
    // (and LM's analytical Jacobian for exp(-n·σ) assumes n ≥ 0).
    if let Some(ref init_d) = initial_densities {
        for (i, &d) in init_d.iter().enumerate() {
            if !d.is_finite() || d < 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "initial_densities[{i}] must be finite and non-negative, got {d}",
                )));
            }
        }
    }

    // V3: shape validation against the input data cube.
    let data_shape = data.data_a.shape();
    let (data_n_e, data_h, data_w) = (data_shape[0], data_shape[1], data_shape[2]);
    let e_slice = energies.as_slice()?;
    let n_e_supplied = e_slice.len();
    if n_e_supplied != data_n_e {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "energies length ({n_e_supplied}) != data spectral axis length ({data_n_e})",
        )));
    }
    if let Some(ref dp) = dead_pixels {
        let dp_shape = dp.as_array().shape().to_vec();
        if dp_shape != [data_h, data_w] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dead_pixels shape {dp_shape:?} != data spatial dims ({data_h}, {data_w})",
            )));
        }
    }

    // Validate the energy grid up front so malformed energies surface as
    // ValueError rather than a release-mode `PanicException` from the
    // per-point `assert!(energy_ev.is_finite() && energy_ev > 0.0)`
    // guards inside the rayon-parallelised pipeline precompute → SLBW /
    // RML / URR leaves.  Empty grids are accepted and yield an empty
    // result; per-pixel failures still degrade gracefully via
    // `filter_map(Err(_) => None)`.
    validate_energy_grid(e_slice)?;

    let energies_vec = e_slice.to_vec();

    // Build resolution
    let res_fn = build_resolution(flight_path_m, delta_t_us, delta_l_m, resolution, None)?;

    // Build config based on isotopes or groups
    let mut config = if let Some(isotopes) = isotopes {
        let n_iso = isotopes.len();
        let iso_names: Vec<String> = isotopes
            .iter()
            .map(|i| {
                let sym =
                    nereids_core::elements::element_symbol(i.inner.isotope.z()).unwrap_or("?");
                format!("{}-{}", sym, i.inner.isotope.a())
            })
            .collect();
        let resonance_data: Vec<ResonanceData> = isotopes
            .into_iter()
            .map(|d| Arc::unwrap_or_clone(d.inner))
            .collect();
        let init_densities = initial_densities.unwrap_or_else(|| vec![0.001; n_iso]);

        UnifiedFitConfig::new(
            energies_vec,
            resonance_data,
            iso_names,
            temperature_k,
            res_fn,
            init_densities,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    } else {
        let groups = groups.unwrap();
        if groups.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "groups list must not be empty",
            ));
        }
        build_config_from_groups(
            &groups,
            energies_vec,
            temperature_k,
            res_fn,
            initial_densities,
        )?
    };

    // Solver — resolve "auto" eagerly so max_iter is always propagated.
    // Issue #458 B4: `data.kind` is one of `"counts"`, `"counts_with_nuisance"`,
    // or the transmission string produced by `from_transmission()`.  Both
    // counts variants should route `solver="auto"` to the counts-KL
    // (joint-Poisson) dispatch — `data.kind == "counts"` alone misses
    // `counts_with_nuisance`, which silently fell through to LM before.
    // (`is_counts_input` computed earlier for V1 validation.)
    let solver_config = parse_solver_config(solver, is_counts_input, max_iter)?;
    config = config.with_solver(solver_config);

    // Issue #638: χ²-scaled uncertainties (no-op on the LM transmission path).
    config = config.with_scale_by_chi2(scale_by_chi2);

    // Temperature fitting
    if fit_temperature {
        config = config.with_fit_temperature(true);
    }

    // Background
    if background {
        // Plumb `fit_back_d` / `fit_back_f` / `back_d_init` /
        // `back_f_init` through to `BackgroundConfig` so the LM
        // transmission per-pixel fit can actually fit the exponential
        // tail.  Without this, the spatial pipeline would attach only
        // the default config (both flags `false`) and the exposed
        // `back_d_map` / `back_f_map` would never be `Some`.  Mirrors
        // the single-spectrum `py_fit_spectrum_typed` wiring above.
        let bg = nereids_pipeline::pipeline::BackgroundConfig {
            fit_anorm,
            fit_back_d,
            fit_back_f,
            back_d_init,
            back_f_init,
            ..nereids_pipeline::pipeline::BackgroundConfig::default()
        };
        config = config.with_transmission_background(bg);
    } else if fit_back_d || fit_back_f {
        // The exponential tail can only be fit when the background
        // model is active.  Reject the silent-noop combination at the
        // binding boundary so the user gets a clear error rather than
        // a `back_d_map` / `back_f_map` of all None.
        return Err(pyo3::exceptions::PyValueError::new_err(
            "fit_back_d / fit_back_f require background=True: the \
             exponential tail of the SAMMY background only exists when \
             the polynomial background model is attached.",
        ));
    } else if !fit_anorm {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "fit_anorm=False requires background=True: Anorm is a parameter \
             of the SAMMY background model.",
        ));
    }

    // Bounded multiplicative baseline (issue #635).  `baseline_global`
    // selects the two-stage global mode (fit once on the aggregated mean
    // spectrum, freeze per-pixel) vs per-pixel baseline maps.
    config = apply_baseline(
        config,
        baseline,
        fit_b0,
        fit_b1,
        fit_b2,
        b0_init,
        b1_init,
        b2_init,
        b0_bounds,
        b1_bounds,
        b2_bounds,
        baseline_global,
        background,
        fit_anorm,
    )?;
    if fit_alpha_1 || fit_alpha_2 || alpha_1_init != 1.0 || alpha_2_init != 1.0 {
        if data.kind != "counts_with_nuisance" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "counts background scaling requires from_counts_with_nuisance() input",
            ));
        }
        config =
            config.with_counts_background(nereids_pipeline::pipeline::CountsBackgroundConfig {
                alpha_1_init,
                alpha_2_init,
                fit_alpha_1,
                fit_alpha_2,
                c,
            });
    } else if c != 1.0 && (data.kind == "counts" || data.kind == "counts_with_nuisance") {
        // Caller provided `c` without alpha fitting — attach a minimal
        // CountsBackgroundConfig carrying just the proton-charge ratio.
        config =
            config.with_counts_background(nereids_pipeline::pipeline::CountsBackgroundConfig {
                c,
                ..Default::default()
            });
    }

    // Polish override.  None = auto-disable
    // when n_pixels > 1 inside spatial_map_typed.
    if let Some(v) = enable_polish {
        config = config.with_counts_enable_polish(Some(v));
    }

    // Energy-scale calibration (SAMMY TZERO equivalent).  Required for
    // real VENUS data — without it, sharp resonances are offset ~0.5 us
    // in TOF and per-pixel chi2 explodes (observed on the VENUS Hf
    // 120min NEREIDS↔SAMMY parity comparison).
    if fit_energy_scale {
        validate_energy_scale_params(t0_init_us, l_scale_init, energy_scale_flight_path_m)?;
        config = config.with_energy_scale(t0_init_us, l_scale_init, energy_scale_flight_path_m);
    }
    let tzero_method = parse_tzero_jacobian(tzero_jacobian)?;
    if tzero_method.is_some() {
        config = config.with_tzero_jacobian_method(tzero_method);
    }

    // Build InputData3D from the PyInputData
    let input = match data.kind.as_str() {
        "counts" => InputData3D::Counts {
            sample_counts: data.data_a.view(),
            open_beam_counts: data.data_b.view(),
        },
        "counts_with_nuisance" => InputData3D::CountsWithNuisance {
            sample_counts: data.data_a.view(),
            flux: data.data_b.view(),
            background: data
                .data_c
                .as_ref()
                .expect("counts_with_nuisance requires background data")
                .view(),
        },
        _ => InputData3D::Transmission {
            transmission: data.data_a.view(),
            uncertainty: data.data_b.view(),
        },
    };

    // Dead pixels
    let dead_arr = dead_pixels.map(|dp| dp.as_array().to_owned());

    // SAMMY EMIN/EMAX-equivalent fit-energy-range (#514): mask residuals
    // to bins inside [E_min, E_max] in both LM and joint-Poisson per-
    // pixel cost paths.  The model is evaluated on the full grid so
    // resolution broadening at the boundaries is correct.
    // No Python-side data slicing required.
    config = config
        .with_fit_energy_range(fit_energy_range)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    // Issue #633: freeze known densities across every pixel — per-pixel
    // T-only (or T + energy-scale) fits with a calibration-foil density.
    config = apply_density_freeze(config, fix_densities, density_free)?;

    // GIL held during computation.  InputData3D borrows PyInputData arrays
    // which are not Send, so we cannot use py.allow_threads().  The existing
    // py_spatial_map has the same limitation.  Rayon still parallelizes the
    // per-pixel fitting within the GIL.
    // Map user-input configuration errors (unpaired
    // `fit_back_d`/`fit_back_f`, non-finite/non-positive `back_*_init`,
    // counts-KL + exponential tail) to `PyValueError`, matching the
    // up-front `PyValueError` at the `background=False +
    // fit_back_d=True` boundary above.  Other `PipelineError`
    // variants stay as `PyRuntimeError` — they signal solver /
    // numeric failures, not bad input.
    let result =
        spatial_map_typed(&input, &config, dead_arr.as_ref(), None, None).map_err(|e| {
            let msg = e.to_string();
            if matches!(
                e,
                nereids_pipeline::error::PipelineError::InvalidParameter(_)
            ) {
                pyo3::exceptions::PyValueError::new_err(msg)
            } else {
                pyo3::exceptions::PyRuntimeError::new_err(msg)
            }
        })?;

    Ok(spatial_result_to_py(py, &result))
}

/// Fit a single raw-count spectrum using the typed input data API.
///
/// Either `isotopes` or `groups` must be provided, but not both.
///
/// Args:
///     sample_counts: 1D sample counts spectrum.
///     open_beam_counts: 1D open-beam counts reference.
///     energies: 1D energy grid in eV (ascending).
///     isotopes: list of (ResonanceData, initial_density) tuples (mutually exclusive with groups).
///     temperature_k: Sample temperature in Kelvin (default 293.6).
///     fit_temperature: Whether to fit temperature (default False).
///     max_iter: Maximum iterations (default 200).
///     solver: "auto" (default), "kl", "poisson", or "joint_poisson".
///         "lm" is rejected for raw counts.
///     background: Enable the SAMMY-style transmission background inside the
///         counts likelihood. This is not detector-count background.
///     detector_background: Reserved detector/counts background reference.
///         Production fitting currently rejects every non-zero value; use
///         `fit_two_arm_background_templates` against a fixed exact two-arm
///         neutron signal instead.
///     fit_alpha_1: Research-only flux-scale parameter; rejected by production
///         counts fitting.
///     fit_alpha_2: Research-only detector-background scale; rejected by
///         production counts fitting.
///     alpha_1_init: Initial value for `alpha_1` (default 1.0).
///     alpha_2_init: Initial value for `alpha_2` (default 1.0).
///     resolution: Exact detector-time response. Resolved raw-count fitting
///         accepts a TabulatedResolution or IkedaCarpenter.
///     incident_fluence_weights: Incident fluence integrated over each point
///         of the true-energy quadrature. Required with detector_time_edges_us.
///     detector_time_edges_us: Actual measured detector-time bin edges. Its
///         length must be one greater than the sample/open count arrays.
///     timing_offset_us: Fixed detector-clock offset applied by the response.
///     groups: list of IsotopeGroup objects (mutually exclusive with isotopes).
///     initial_densities: Initial density guesses when using groups (default 0.001 each).
///     enable_polish: Override the Nelder-Mead polish phase on the
///         counts-KL solver (default ``None`` → use the library default,
///         which is ``False`` as of #486 because polish's absolute
///         ``fatol = 1e-10`` is sub-f64-ULP on real-data deviance scales
///         where ``D ≈ 10⁴``–``10⁵``, so polish hits ``max_iter = 5000``
///         every fit at 70-260× wall cost for ≤ 0.35 Fisher σ parameter
///         shift).  Pass ``True`` to opt in for clean / synthetic fits
///         where ``D → 0`` is achievable and the polish tolerances are
///         physically meaningful.  See ``JointPoissonFitConfig``
///         ``enable_polish`` field doc for details.
///
///     fit_anorm: Whether Anorm is free when ``background=True`` (default
///         True).  Must be False to combine ``background=True`` with
///         ``baseline=True`` (b0 and Anorm are degenerate normalizations).
///     baseline: Enable the bounded multiplicative baseline
///         ``B(E) = b0 + b1·ln(E/E_ref) + b2·ln²(E/E_ref)`` applied
///         OUTERMOST (issue #635).  E_ref is the geometric midpoint of the
///         energy grid and is reported as ``FitResult.baseline_e_ref_ev``.
///     fit_b0, fit_b1, fit_b2: Per-coefficient fit flags (default True).
///         All-False freezes the baseline at its inits.
///     b0_init, b1_init, b2_init: Initial coefficients (default 1, 0, 0 —
///         the identity baseline).
///     b0_bounds, b1_bounds, b2_bounds: Optional (lower, upper) optimizer
///         boxes; defaults (0.9, 1.1) / (−0.05, 0.05) / (−0.05, 0.05).
///         The boxes bound the coefficients (tilt/curvature per ln-E
///         unit), not the evaluated B(E); dip protection comes from the
///         quadratic's smoothness, and B(E) > 0 is enforced pointwise.
///
/// Returns:
///     FitResult with densities, uncertainties, chi2, etc.
///
/// For pre-normalized transmission data, use `fit_spectrum_typed(...)`.
#[pyfunction]
#[pyo3(name = "fit_counts_spectrum_typed", signature = (
    sample_counts, open_beam_counts, energies, isotopes=None, *,
    temperature_k = 293.6,
    fit_temperature = false,
    max_iter = 200,
    solver = "auto",
    background = false,
    fit_back_d = false,
    fit_back_f = false,
    back_d_init = 0.01,
    back_f_init = 1.0,
    fit_energy_scale = false,
    t0_init_us = 0.0,
    l_scale_init = 1.0,
    energy_scale_flight_path_m = 25.0,
    detector_background = None,
    fit_alpha_1 = false,
    fit_alpha_2 = false,
    alpha_1_init = 1.0,
    alpha_2_init = 1.0,
    c = 1.0,
    resolution = None,
    incident_fluence_weights = None,
    detector_time_edges_us = None,
    timing_offset_us = 0.0,
    flight_path_m = None,
    delta_t_us = None,
    delta_l_m = None,
    groups = None,
    initial_densities = None,
    fix_densities = false,
    density_free = None,
    enable_polish = None,
    tzero_jacobian = None,
    fit_energy_range = None,
    fit_anorm = true,
    baseline = false,
    fit_b0 = true,
    fit_b1 = true,
    fit_b2 = true,
    b0_init = 1.0,
    b1_init = 0.0,
    b2_init = 0.0,
    b0_bounds = None,
    b1_bounds = None,
    b2_bounds = None,
    scale_by_chi2 = false,
))]
fn py_fit_counts_spectrum_typed<'py>(
    py: Python<'py>,
    sample_counts: PyReadonlyArray1<'py, f64>,
    open_beam_counts: PyReadonlyArray1<'py, f64>,
    energies: PyReadonlyArray1<'py, f64>,
    isotopes: Option<Vec<(PyResonanceData, f64)>>,
    temperature_k: f64,
    fit_temperature: bool,
    max_iter: usize,
    solver: &str,
    background: bool,
    fit_back_d: bool,
    fit_back_f: bool,
    back_d_init: f64,
    back_f_init: f64,
    fit_energy_scale: bool,
    t0_init_us: f64,
    l_scale_init: f64,
    energy_scale_flight_path_m: f64,
    detector_background: Option<PyReadonlyArray1<'py, f64>>,
    fit_alpha_1: bool,
    fit_alpha_2: bool,
    alpha_1_init: f64,
    alpha_2_init: f64,
    c: f64,
    resolution: Option<&Bound<'py, PyAny>>,
    incident_fluence_weights: Option<PyReadonlyArray1<'py, f64>>,
    detector_time_edges_us: Option<PyReadonlyArray1<'py, f64>>,
    timing_offset_us: f64,
    flight_path_m: Option<f64>,
    delta_t_us: Option<f64>,
    delta_l_m: Option<f64>,
    groups: Option<Vec<PyIsotopeGroup>>,
    initial_densities: Option<Vec<f64>>,
    fix_densities: bool,
    density_free: Option<Vec<bool>>,
    enable_polish: Option<bool>,
    tzero_jacobian: Option<&str>,
    fit_energy_range: Option<(f64, f64)>,
    fit_anorm: bool,
    baseline: bool,
    fit_b0: bool,
    fit_b1: bool,
    fit_b2: bool,
    b0_init: f64,
    b1_init: f64,
    b2_init: f64,
    b0_bounds: Option<(f64, f64)>,
    b1_bounds: Option<(f64, f64)>,
    b2_bounds: Option<(f64, f64)>,
    scale_by_chi2: bool,
) -> PyResult<PyFitResult> {
    use nereids_pipeline::pipeline::{
        CountsBackgroundConfig, ExactCountResponseConfig, InputData, UnifiedFitConfig,
        fit_spectrum_typed,
    };

    let has_isotopes = isotopes.is_some();
    let has_groups = groups.is_some();
    if has_isotopes && has_groups {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Provide either 'isotopes' or 'groups', not both.",
        ));
    }
    if !has_isotopes && !has_groups {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Must provide either 'isotopes' or 'groups'.",
        ));
    }

    let sample_slice = sample_counts.as_slice()?;
    let ob_slice = open_beam_counts.as_slice()?;
    let e_slice = energies.as_slice()?;
    if sample_slice.len() != ob_slice.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "sample_counts length ({}) must match open_beam_counts length ({})",
            sample_slice.len(),
            ob_slice.len(),
        )));
    }
    let exact_source = incident_fluence_weights
        .map(|values| values.as_slice().map(<[f64]>::to_vec))
        .transpose()?;
    let exact_edges = detector_time_edges_us
        .map(|values| values.as_slice().map(<[f64]>::to_vec))
        .transpose()?;
    let exact_requested = exact_source.is_some() || exact_edges.is_some();
    if exact_source.is_some() != exact_edges.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "incident_fluence_weights and detector_time_edges_us must be supplied together",
        ));
    }
    if !exact_requested && timing_offset_us != 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "timing_offset_us requires incident_fluence_weights and detector_time_edges_us",
        ));
    }
    if !exact_requested && sample_slice.len() != e_slice.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "sample_counts length ({}) must match energies length ({})",
            sample_slice.len(),
            e_slice.len(),
        )));
    }
    if let Some(source) = exact_source.as_ref()
        && source.len() != e_slice.len()
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "incident_fluence_weights length ({}) must match the true-energy grid length ({})",
            source.len(),
            e_slice.len(),
        )));
    }
    if let Some(edges) = exact_edges.as_ref()
        && edges.len() != sample_slice.len() + 1
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "detector_time_edges_us length ({}) must be one greater than the measured count-bin length ({})",
            edges.len(),
            sample_slice.len(),
        )));
    }
    if exact_requested && !timing_offset_us.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "timing_offset_us must be finite, got {timing_offset_us}",
        )));
    }
    require_non_empty_energy_grid(e_slice)?;

    // Issue #458 V1: reject non-positive / non-finite `c` (Q_s / Q_ob).
    if !c.is_finite() || c <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "c (proton-charge ratio Q_s/Q_ob) must be positive and finite, got {c}",
        )));
    }
    // Issue #458 V2: reject non-finite / negative initial densities —
    // both in `initial_densities` kwarg and in `isotopes: list[(rd, d)]`.
    if let Some(ref init_d) = initial_densities {
        for (i, &d) in init_d.iter().enumerate() {
            if !d.is_finite() || d < 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "initial_densities[{i}] must be finite and non-negative, got {d}",
                )));
            }
        }
    }
    if let Some(ref iso) = isotopes {
        for (i, (_, d)) in iso.iter().enumerate() {
            if !d.is_finite() || *d < 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "isotopes[{i}] initial density must be finite and non-negative, got {d}",
                )));
            }
        }
    }

    let detector_background_vec = if let Some(bg) = detector_background {
        let bg_slice = bg.as_slice()?;
        if bg_slice.len() != sample_slice.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "detector_background length ({}) must match sample_counts length ({})",
                bg_slice.len(),
                sample_slice.len(),
            )));
        }
        Some(bg_slice.to_vec())
    } else {
        None
    };
    if fit_alpha_2 && detector_background_vec.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "fit_alpha_2 requires detector_background to be provided",
        ));
    }

    let energies_vec = e_slice.to_vec();
    let has_gaussian = flight_path_m.is_some() || delta_t_us.is_some() || delta_l_m.is_some();
    if has_gaussian && resolution.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot specify both Gaussian resolution parameters and resolution",
        ));
    }
    let res_fn = if let Some(response) = resolution {
        Some(extract_detector_time_resolution(response)?)
    } else {
        build_resolution(flight_path_m, delta_t_us, delta_l_m, None, None)?
    };

    let mut config = if let Some(isotopes) = isotopes {
        let iso_names: Vec<String> = isotopes
            .iter()
            .map(|(d, _)| {
                let sym =
                    nereids_core::elements::element_symbol(d.inner.isotope.z()).unwrap_or("?");
                format!("{}-{}", sym, d.inner.isotope.a())
            })
            .collect();
        let init_densities: Vec<f64> = isotopes.iter().map(|(_, d)| *d).collect();
        let resonance_data: Vec<ResonanceData> = isotopes
            .into_iter()
            .map(|(d, _)| Arc::unwrap_or_clone(d.inner))
            .collect();

        UnifiedFitConfig::new(
            energies_vec,
            resonance_data,
            iso_names,
            temperature_k,
            res_fn,
            init_densities,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    } else {
        let groups = groups.unwrap();
        if groups.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "groups list must not be empty",
            ));
        }
        build_config_from_groups(
            &groups,
            energies_vec,
            temperature_k,
            res_fn,
            initial_densities,
        )?
    };

    config = config.with_solver(parse_solver_config(solver, true, max_iter)?);
    if let (Some(incident_fluence_weights), Some(detector_time_edges_us)) =
        (exact_source, exact_edges)
    {
        if config.resolution().is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "exact resolved counts require resolution=TabulatedResolution or IkedaCarpenter",
            ));
        }
        config = config.with_exact_count_response(ExactCountResponseConfig {
            incident_fluence_weights,
            detector_time_edges_us,
            timing_offset_us,
        });
    }
    // Issue #638: χ²-scaled uncertainties (no-op on the LM transmission path).
    config = config.with_scale_by_chi2(scale_by_chi2);
    if fit_temperature {
        config = config.with_fit_temperature(true);
    }
    if background {
        let mut bg = nereids_pipeline::pipeline::BackgroundConfig::default();
        bg.fit_anorm = fit_anorm;
        bg.fit_back_d = fit_back_d;
        bg.fit_back_f = fit_back_f;
        bg.back_d_init = back_d_init;
        bg.back_f_init = back_f_init;
        config = config.with_transmission_background(bg);
    } else if !fit_anorm {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "fit_anorm=False requires background=True: Anorm is a parameter \
             of the SAMMY background model.",
        ));
    }
    // Bounded multiplicative baseline (issue #635).
    config = apply_baseline(
        config, baseline, fit_b0, fit_b1, fit_b2, b0_init, b1_init, b2_init, b0_bounds, b1_bounds,
        b2_bounds, true, background, fit_anorm,
    )?;
    if fit_energy_scale {
        validate_energy_scale_params(t0_init_us, l_scale_init, energy_scale_flight_path_m)?;
        config = config.with_energy_scale(t0_init_us, l_scale_init, energy_scale_flight_path_m);
    }
    let tzero_method = parse_tzero_jacobian(tzero_jacobian)?;
    if tzero_method.is_some() {
        config = config.with_tzero_jacobian_method(tzero_method);
    }
    // Attach CountsBackgroundConfig whenever any of its fields deviates from
    // the default — including c, which is the explicit proton-charge ratio
    // consumed by the counts-KL (joint-Poisson) dispatch.
    if fit_alpha_1
        || fit_alpha_2
        || alpha_1_init != 1.0
        || alpha_2_init != 1.0
        || c != 1.0
        || solver == "kl"
        || solver == "poisson"
        || solver == "joint_poisson"
    {
        config = config.with_counts_background(CountsBackgroundConfig {
            alpha_1_init,
            alpha_2_init,
            fit_alpha_1,
            fit_alpha_2,
            c,
        });
    }

    let input = if let Some(bg) = detector_background_vec {
        InputData::CountsWithNuisance {
            sample_counts: sample_slice.to_vec(),
            flux: ob_slice.to_vec(),
            background: bg,
        }
    } else {
        InputData::Counts {
            sample_counts: sample_slice.to_vec(),
            open_beam_counts: ob_slice.to_vec(),
        }
    };

    // #486: counts-KL polish override.  Default `None` falls through to
    // the Rust `JointPoissonFitConfig::default().enable_polish` (now
    // `false`).  Pass `enable_polish=True` to opt back in for clean /
    // synthetic fits where the polish tolerances are physically
    // meaningful (see the field doc on `JointPoissonFitConfig::enable_polish`).
    if let Some(v) = enable_polish {
        config = config.with_counts_enable_polish(Some(v));
    }

    // SAMMY EMIN/EMAX-equivalent fit-energy-range (#514): mask residuals
    // to bins inside [E_min, E_max]; the joint-Poisson cost path
    // honours the mask in deviance / gradient / Fisher loops.  No
    // Python-side data slicing required — the model is evaluated on
    // the full grid so resolution broadening at the boundaries is
    // correct.
    config = config
        .with_fit_energy_range(fit_energy_range)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    // Issue #633: freeze known densities (calibration-foil thermometry).
    config = apply_density_freeze(config, fix_densities, density_free)?;

    let result = py.detach(move || fit_spectrum_typed(&input, &config));
    // Config-class errors (InvalidParameter) surface as ValueError —
    // matching py_spatial_map_typed, which established the convention for
    // PipelineError at this boundary (review R2: identical invalid input,
    // e.g. a baseline init outside its bounds, previously raised ValueError
    // from the spatial API but RuntimeError here because the error variant
    // was stringified inside py.detach).  Other variants stay RuntimeError —
    // they signal solver / numeric failures, not bad input.
    let result = result.map_err(|e| {
        let msg = e.to_string();
        if matches!(
            e,
            nereids_pipeline::error::PipelineError::InvalidParameter(_)
        ) {
            pyo3::exceptions::PyValueError::new_err(msg)
        } else {
            pyo3::exceptions::PyRuntimeError::new_err(msg)
        }
    })?;

    Ok(PyFitResult {
        densities: result.densities,
        uncertainties: result.uncertainties,
        reduced_chi_squared: result.reduced_chi_squared,
        converged: result.converged,
        iterations: result.iterations,
        temperature_k: result.temperature_k,
        temperature_k_unc: result.temperature_k_unc,
        anorm: result.anorm,
        background: result.background,
        // BackD / BackF are only meaningful when the polynomial
        // background model was actually attached.  `result.back_d` /
        // `result.back_f` are `Option<f64>` (None when the inner
        // Rust fit didn't fit the exponential tail).  The outer gate
        // on `background && fit_back_d` is defensive — it ensures
        // PyFitResult.back_d is None whenever the caller didn't
        // request the tail, regardless of any future change to the
        // inner Rust contract.
        back_d: if background && fit_back_d {
            result.back_d
        } else {
            None
        },
        back_f: if background && fit_back_f {
            result.back_f
        } else {
            None
        },
        t0_us: result.t0_us,
        l_scale: result.l_scale,
        energy_scale_flight_path_m: result.energy_scale_flight_path_m,
        deviance_per_dof: result.deviance_per_dof,
        baseline: result.baseline,
        baseline_e_ref_ev: result.baseline_e_ref_ev,
        warnings: result.warnings,
    })
}

// ── Research: exact Jacobian/Fisher at arbitrary parameters ──────────────

/// Result of exact Jacobian/Fisher evaluation from the Rust engine.
#[pyclass(name = "ModelJacobianResult")]
struct PyModelJacobianResult {
    jacobian_data: Vec<f64>,
    jacobian_nrows: usize,
    jacobian_ncols: usize,
    fisher_data: Vec<f64>,
    fisher_n: usize,
    model_prediction: Vec<f64>,
    param_names: Vec<String>,
}

#[pymethods]
impl PyModelJacobianResult {
    /// Analytical Jacobian J (n_energy × n_free_params), row-major.
    #[getter]
    fn jacobian<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let shape = [self.jacobian_nrows, self.jacobian_ncols];
        let arr =
            ndarray::Array2::from_shape_vec(shape, self.jacobian_data.clone()).expect("shape ok");
        PyArray2::from_owned_array(py, arr)
    }

    /// Expected Poisson Fisher F = Jᵀ diag(1/μ) J (n_free × n_free).
    #[getter]
    fn fisher<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let shape = [self.fisher_n, self.fisher_n];
        let arr =
            ndarray::Array2::from_shape_vec(shape, self.fisher_data.clone()).expect("shape ok");
        PyArray2::from_owned_array(py, arr)
    }

    /// Model prediction μ(E) at the evaluation point.
    #[getter]
    fn model_prediction<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        numpy::PyArray1::from_vec(py, self.model_prediction.clone())
    }

    /// Names of free parameters in Jacobian column order.
    #[getter]
    fn param_names(&self) -> Vec<String> {
        self.param_names.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelJacobianResult(n_data={}, n_free={}, params={:?})",
            self.jacobian_nrows, self.jacobian_ncols, self.param_names
        )
    }
}

/// Compute the exact resolved analytical Jacobian and expected Fisher at given
/// parameter values.
///
/// Uses the same model construction as ``fit_counts_spectrum_typed()`` but does
/// **not** optimise — evaluates once at the provided densities/temperature and
/// returns the exact Jacobian and Fisher from the Rust engine.
///
/// This is a research-oriented function for Fisher-based regularisation studies.
///
/// Either ``isotopes`` or ``groups`` must be provided, but not both.
/// When ``groups`` is provided, each group maps to one density parameter
/// (same semantics as ``fit_counts_spectrum_typed``).
///
/// Args:
///     open_beam_counts: Incident flux Φ(E) (1D array, length n_energy).
///     energies: Energy grid in eV (1D array, sorted ascending).
///     isotopes: List of (ResonanceData, density_at_eval_point) tuples.
///     temperature_k: Temperature at which to evaluate (default 293.6 K).
///     fit_temperature: If True, include temperature as a free parameter
///         in the Jacobian.
///     flight_path_m, delta_t_us, delta_l_m: Gaussian resolution parameters.
///     resolution: Tabulated resolution object.
///     detector_background: Detector background B(E) for counts background model.
///     fit_alpha_1: If True, include signal scale α₁ as free parameter.
///     fit_alpha_2: If True, include background scale α₂ as free parameter.
///     alpha_1: Signal scale evaluation value (default 1.0).
///     alpha_2: Background scale evaluation value (default 1.0).
///     groups: List of IsotopeGroup objects (mutually exclusive with isotopes).
///     initial_densities: Initial/evaluation densities when using groups.
///
/// Returns:
///     ModelJacobianResult with jacobian, fisher, model_prediction, param_names.
#[pyfunction]
#[pyo3(name = "compute_model_jacobian", signature = (
    open_beam_counts, energies, isotopes=None, *,
    temperature_k = 293.6,
    fit_temperature = false,
    flight_path_m = None,
    delta_t_us = None,
    delta_l_m = None,
    resolution = None,
    detector_background = None,
    fit_alpha_1 = false,
    fit_alpha_2 = false,
    alpha_1 = 1.0,
    alpha_2 = 1.0,
    groups = None,
    initial_densities = None,
))]
fn py_compute_model_jacobian<'py>(
    py: Python<'py>,
    open_beam_counts: PyReadonlyArray1<'py, f64>,
    energies: PyReadonlyArray1<'py, f64>,
    isotopes: Option<Vec<(PyResonanceData, f64)>>,
    temperature_k: f64,
    fit_temperature: bool,
    flight_path_m: Option<f64>,
    delta_t_us: Option<f64>,
    delta_l_m: Option<f64>,
    resolution: Option<PyTabulatedResolution>,
    detector_background: Option<PyReadonlyArray1<'py, f64>>,
    fit_alpha_1: bool,
    fit_alpha_2: bool,
    alpha_1: f64,
    alpha_2: f64,
    groups: Option<Vec<PyIsotopeGroup>>,
    initial_densities: Option<Vec<f64>>,
) -> PyResult<PyModelJacobianResult> {
    use nereids_pipeline::pipeline::{CountsBackgroundConfig, evaluate_jacobian_and_fisher};

    let has_isotopes = isotopes.is_some();
    let has_groups = groups.is_some();
    if has_isotopes && has_groups {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Provide either 'isotopes' or 'groups', not both.",
        ));
    }
    if !has_isotopes && !has_groups {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Must provide either 'isotopes' or 'groups'.",
        ));
    }

    let ob_slice = open_beam_counts.as_slice()?;
    let e_slice = energies.as_slice()?;
    require_non_empty_energy_grid(e_slice)?;

    if ob_slice.len() != e_slice.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "open_beam_counts length ({}) must match energies length ({})",
            ob_slice.len(),
            e_slice.len(),
        )));
    }

    if fit_alpha_2 && detector_background.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "fit_alpha_2 requires detector_background to be provided",
        ));
    }

    let det_bg_vec = if let Some(ref bg) = detector_background {
        let bg_s = bg.as_slice()?;
        if bg_s.len() != e_slice.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "detector_background length ({}) must match energies length ({})",
                bg_s.len(),
                e_slice.len(),
            )));
        }
        bg_s.to_vec()
    } else {
        vec![0.0; e_slice.len()]
    };

    let res_fn = build_resolution(flight_path_m, delta_t_us, delta_l_m, resolution, None)?;
    let energies_vec = e_slice.to_vec();

    let mut config = if let Some(isotopes) = isotopes {
        let iso_names: Vec<String> = isotopes
            .iter()
            .map(|(d, _)| {
                let sym =
                    nereids_core::elements::element_symbol(d.inner.isotope.z()).unwrap_or("?");
                format!("{}-{}", sym, d.inner.isotope.a())
            })
            .collect();
        let init_densities: Vec<f64> = isotopes.iter().map(|(_, d)| *d).collect();
        let resonance_data: Vec<ResonanceData> = isotopes
            .into_iter()
            .map(|(d, _)| Arc::unwrap_or_clone(d.inner))
            .collect();

        UnifiedFitConfig::new(
            energies_vec,
            resonance_data,
            iso_names,
            temperature_k,
            res_fn,
            init_densities,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    } else {
        let groups = groups.unwrap();
        if groups.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "groups list must not be empty",
            ));
        }
        build_config_from_groups(
            &groups,
            energies_vec,
            temperature_k,
            res_fn,
            initial_densities,
        )?
    };

    if fit_temperature {
        config = config.with_fit_temperature(true);
    }
    if fit_alpha_1 || fit_alpha_2 || alpha_1 != 1.0 || alpha_2 != 1.0 {
        config = config.with_counts_background(CountsBackgroundConfig {
            alpha_1_init: alpha_1,
            alpha_2_init: alpha_2,
            fit_alpha_1,
            fit_alpha_2,
            c: 1.0,
        });
    }

    let flux = ob_slice.to_vec();
    let background = det_bg_vec;

    let result = py.detach(move || evaluate_jacobian_and_fisher(&config, &flux, &background));
    // InvalidParameter → ValueError, other variants → RuntimeError — the
    // same PipelineError convention as the three fitters (review R2).
    let result = result.map_err(|e| {
        let msg = e.to_string();
        if matches!(
            e,
            nereids_pipeline::error::PipelineError::InvalidParameter(_)
        ) {
            pyo3::exceptions::PyValueError::new_err(msg)
        } else {
            pyo3::exceptions::PyRuntimeError::new_err(msg)
        }
    })?;

    Ok(PyModelJacobianResult {
        jacobian_data: result.jacobian.data,
        jacobian_nrows: result.jacobian.nrows,
        jacobian_ncols: result.jacobian.ncols,
        fisher_data: result.fisher.data,
        fisher_n: result.fisher.nrows,
        model_prediction: result.model_prediction,
        param_names: result.param_names,
    })
}

/// Apply the issue-#633 density freeze to a config: an explicit
/// per-density `density_free` mask takes precedence; otherwise
/// `fix_densities=true` freezes all densities. `density_free` and
/// `fix_densities` are mutually exclusive — supplying both is an error
/// (the mask is unambiguous, the bool would be redundant or conflicting).
fn apply_density_freeze(
    config: nereids_pipeline::pipeline::UnifiedFitConfig,
    fix_densities: bool,
    density_free: Option<Vec<bool>>,
) -> PyResult<nereids_pipeline::pipeline::UnifiedFitConfig> {
    match density_free {
        Some(free) => {
            if fix_densities {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Provide either 'fix_densities' or 'density_free', not both.",
                ));
            }
            config
                .with_density_free(free)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
        }
        None => Ok(config.with_fix_densities(fix_densities)),
    }
}

/// Shared kwarg plumbing for the bounded multiplicative baseline (issue
/// #635), mirroring `apply_density_freeze`.  Rejects silent-noop
/// combinations (baseline sub-options without `baseline=True`) and the
/// degenerate free-Anorm pairing at the binding boundary (ValueError with
/// the fix) rather than letting the core rejection surface as a
/// RuntimeError after config assembly.
#[allow(clippy::too_many_arguments)]
fn apply_baseline(
    config: nereids_pipeline::pipeline::UnifiedFitConfig,
    baseline: bool,
    fit_b0: bool,
    fit_b1: bool,
    fit_b2: bool,
    b0_init: f64,
    b1_init: f64,
    b2_init: f64,
    b0_bounds: Option<(f64, f64)>,
    b1_bounds: Option<(f64, f64)>,
    b2_bounds: Option<(f64, f64)>,
    baseline_global: bool,
    background: bool,
    fit_anorm: bool,
) -> PyResult<nereids_pipeline::pipeline::UnifiedFitConfig> {
    if !baseline {
        let sub_options_at_default = fit_b0
            && fit_b1
            && fit_b2
            && b0_init == 1.0
            && b1_init == 0.0
            && b2_init == 0.0
            && b0_bounds.is_none()
            && b1_bounds.is_none()
            && b2_bounds.is_none()
            && baseline_global;
        if !sub_options_at_default {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "baseline sub-options (fit_b0/fit_b1/fit_b2, b*_init, b*_bounds, \
                 baseline_global) require baseline=True — without it they would \
                 silently do nothing.",
            ));
        }
        return Ok(config);
    }
    if background && fit_anorm {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "baseline=True with background=True requires fit_anorm=False: the \
             baseline's b0 and the SAMMY Anorm are degenerate normalizations \
             (never free two of them). Pass fit_anorm=False to combine the \
             additive ABC background with the multiplicative baseline.",
        ));
    }
    let mut bl = nereids_pipeline::pipeline::MultiplicativeBaselineConfig {
        b0_init,
        b1_init,
        b2_init,
        fit_b0,
        fit_b1,
        fit_b2,
        spatial_global: baseline_global,
        ..Default::default()
    };
    if let Some(b) = b0_bounds {
        bl.b0_bounds = b;
    }
    if let Some(b) = b1_bounds {
        bl.b1_bounds = b;
    }
    if let Some(b) = b2_bounds {
        bl.b2_bounds = b;
    }
    Ok(config.with_multiplicative_baseline(bl))
}

/// Fit a single pre-normalized transmission spectrum.
///
/// This function accepts **transmission** data only (T = sample/open-beam).
/// For raw-count fitting, use `fit_counts_spectrum_typed(...)`.
///
/// Either `isotopes` or `groups` must be provided, but not both.
///
/// Args:
///     transmission: 1D transmission spectrum (pre-normalized).
///     uncertainty: 1D uncertainty (same length as transmission).
///     energies: 1D energy grid in eV (ascending).
///     isotopes: list of (ResonanceData, initial_density) tuples (mutually exclusive with groups).
///     temperature_k: Sample temperature in Kelvin (default 293.6).
///     fit_temperature: Whether to fit temperature (default False).
///     max_iter: Maximum iterations (default 200).
///     solver: "lm" (default) or "auto". Count-likelihood names are rejected
///         for normalized transmission.
///     background: Enable SAMMY transmission background.
///     resolution: Optional resolution function.
///     groups: list of IsotopeGroup objects (mutually exclusive with isotopes).
///     initial_densities: Initial density guesses when using groups (default 0.001 each).
///     fit_anorm: Whether Anorm is free when ``background=True`` (default
///         True).  Must be False to combine ``background=True`` with
///         ``baseline=True`` (b0 and Anorm are degenerate normalizations).
///     baseline: Enable the bounded multiplicative baseline
///         ``B(E) = b0 + b1·ln(E/E_ref) + b2·ln²(E/E_ref)`` applied
///         OUTERMOST (issue #635).  E_ref is the geometric midpoint of the
///         energy grid and is reported as ``FitResult.baseline_e_ref_ev``.
///     fit_b0, fit_b1, fit_b2: Per-coefficient fit flags (default True).
///         All-False freezes the baseline at its inits.
///     b0_init, b1_init, b2_init: Initial coefficients (default 1, 0, 0 —
///         the identity baseline).
///     b0_bounds, b1_bounds, b2_bounds: Optional (lower, upper) optimizer
///         boxes; defaults (0.9, 1.1) / (−0.05, 0.05) / (−0.05, 0.05).
///         The boxes bound the coefficients (tilt/curvature per ln-E
///         unit), not the evaluated B(E); dip protection comes from the
///         quadratic's smoothness, and B(E) > 0 is enforced pointwise.
///
/// Returns:
///     FitResult with densities, uncertainties, chi2, etc.
#[pyfunction]
#[pyo3(name = "fit_spectrum_typed", signature = (
    transmission, uncertainty, energies, isotopes=None, *,
    temperature_k = 293.6,
    fit_temperature = false,
    max_iter = 200,
    solver = "lm",
    background = false,
    fit_back_d = false,
    fit_back_f = false,
    back_d_init = 0.01,
    back_f_init = 1.0,
    fit_energy_scale = false,
    t0_init_us = 0.0,
    l_scale_init = 1.0,
    energy_scale_flight_path_m = 25.0,
    resolution = None,
    flight_path_m = None,
    delta_t_us = None,
    delta_l_m = None,
    groups = None,
    initial_densities = None,
    fix_densities = false,
    density_free = None,
    tzero_jacobian = None,
    fit_energy_range = None,
    fit_anorm = true,
    baseline = false,
    fit_b0 = true,
    fit_b1 = true,
    fit_b2 = true,
    b0_init = 1.0,
    b1_init = 0.0,
    b2_init = 0.0,
    b0_bounds = None,
    b1_bounds = None,
    b2_bounds = None,
    scale_by_chi2 = false,
))]
fn py_fit_spectrum_typed<'py>(
    py: Python<'py>,
    transmission: PyReadonlyArray1<'py, f64>,
    uncertainty: PyReadonlyArray1<'py, f64>,
    energies: PyReadonlyArray1<'py, f64>,
    isotopes: Option<Vec<(PyResonanceData, f64)>>,
    temperature_k: f64,
    fit_temperature: bool,
    max_iter: usize,
    solver: &str,
    background: bool,
    fit_back_d: bool,
    fit_back_f: bool,
    back_d_init: f64,
    back_f_init: f64,
    fit_energy_scale: bool,
    t0_init_us: f64,
    l_scale_init: f64,
    energy_scale_flight_path_m: f64,
    resolution: Option<PyTabulatedResolution>,
    flight_path_m: Option<f64>,
    delta_t_us: Option<f64>,
    delta_l_m: Option<f64>,
    groups: Option<Vec<PyIsotopeGroup>>,
    initial_densities: Option<Vec<f64>>,
    fix_densities: bool,
    density_free: Option<Vec<bool>>,
    tzero_jacobian: Option<&str>,
    fit_energy_range: Option<(f64, f64)>,
    fit_anorm: bool,
    baseline: bool,
    fit_b0: bool,
    fit_b1: bool,
    fit_b2: bool,
    b0_init: f64,
    b1_init: f64,
    b2_init: f64,
    b0_bounds: Option<(f64, f64)>,
    b1_bounds: Option<(f64, f64)>,
    b2_bounds: Option<(f64, f64)>,
    scale_by_chi2: bool,
) -> PyResult<PyFitResult> {
    use nereids_pipeline::pipeline::{InputData, fit_spectrum_typed};

    // Validate mutual exclusivity
    let has_isotopes = isotopes.is_some();
    let has_groups = groups.is_some();
    if has_isotopes && has_groups {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Provide either 'isotopes' or 'groups', not both.",
        ));
    }
    if !has_isotopes && !has_groups {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Must provide either 'isotopes' or 'groups'.",
        ));
    }

    let t_slice = transmission.as_slice()?;
    let u_slice = uncertainty.as_slice()?;
    let e_slice = energies.as_slice()?;

    if t_slice.len() != u_slice.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "transmission length ({}) must match uncertainty length ({})",
            t_slice.len(),
            u_slice.len(),
        )));
    }
    if t_slice.len() != e_slice.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "transmission length ({}) must match energies length ({})",
            t_slice.len(),
            e_slice.len(),
        )));
    }
    require_non_empty_energy_grid(e_slice)?;

    // Issue #458 V2: reject non-finite / negative initial densities.
    if let Some(ref init_d) = initial_densities {
        for (i, &d) in init_d.iter().enumerate() {
            if !d.is_finite() || d < 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "initial_densities[{i}] must be finite and non-negative, got {d}",
                )));
            }
        }
    }
    // Also validate embedded densities passed through `isotopes: list[(ResonanceData, float)]`.
    if let Some(ref iso) = isotopes {
        for (i, (_, d)) in iso.iter().enumerate() {
            if !d.is_finite() || *d < 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "isotopes[{i}] initial density must be finite and non-negative, got {d}",
                )));
            }
        }
    }

    let energies_vec = e_slice.to_vec();

    // Build resolution
    let res_fn = build_resolution(flight_path_m, delta_t_us, delta_l_m, resolution, None)?;

    // Build config based on isotopes or groups
    let mut config = if let Some(isotopes) = isotopes {
        let iso_names: Vec<String> = isotopes
            .iter()
            .map(|(d, _)| {
                let sym =
                    nereids_core::elements::element_symbol(d.inner.isotope.z()).unwrap_or("?");
                format!("{}-{}", sym, d.inner.isotope.a())
            })
            .collect();
        let init_densities: Vec<f64> = isotopes.iter().map(|(_, d)| *d).collect();
        let resonance_data: Vec<ResonanceData> = isotopes
            .into_iter()
            .map(|(d, _)| Arc::unwrap_or_clone(d.inner))
            .collect();

        UnifiedFitConfig::new(
            energies_vec,
            resonance_data,
            iso_names,
            temperature_k,
            res_fn,
            init_densities,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    } else {
        let groups = groups.unwrap();
        if groups.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "groups list must not be empty",
            ));
        }
        build_config_from_groups(
            &groups,
            energies_vec,
            temperature_k,
            res_fn,
            initial_densities,
        )?
    };

    // Solver — is_counts=false because this function only accepts transmission+uncertainty,
    // so "auto" always resolves to LM (the signature default is "lm" to match).
    let solver_config = parse_solver_config(solver, false, max_iter)?;
    config = config.with_solver(solver_config);

    // Issue #638: this function takes transmission input and therefore always
    // uses LM. LM already scales its covariance by reduced chi-squared, so this
    // compatibility flag is a no-op here.
    config = config.with_scale_by_chi2(scale_by_chi2);

    // Temperature fitting
    if fit_temperature {
        config = config.with_fit_temperature(true);
    }

    // Background
    if background {
        let mut bg = nereids_pipeline::pipeline::BackgroundConfig::default();
        bg.fit_anorm = fit_anorm;
        bg.fit_back_d = fit_back_d;
        bg.fit_back_f = fit_back_f;
        bg.back_d_init = back_d_init;
        bg.back_f_init = back_f_init;
        config = config.with_transmission_background(bg);
    } else if !fit_anorm {
        // Anorm only exists on the background model — reject the silent
        // no-op rather than letting the kwarg vanish.
        return Err(pyo3::exceptions::PyValueError::new_err(
            "fit_anorm=False requires background=True: Anorm is a parameter \
             of the SAMMY background model.",
        ));
    }

    // Bounded multiplicative baseline (issue #635).
    config = apply_baseline(
        config, baseline, fit_b0, fit_b1, fit_b2, b0_init, b1_init, b2_init, b0_bounds, b1_bounds,
        b2_bounds, true, background, fit_anorm,
    )?;

    // Energy-scale calibration (SAMMY TZERO equivalent)
    if fit_energy_scale {
        validate_energy_scale_params(t0_init_us, l_scale_init, energy_scale_flight_path_m)?;
        config = config.with_energy_scale(t0_init_us, l_scale_init, energy_scale_flight_path_m);
    }
    let tzero_method = parse_tzero_jacobian(tzero_jacobian)?;
    if tzero_method.is_some() {
        config = config.with_tzero_jacobian_method(tzero_method);
    }

    // SAMMY EMIN/EMAX-equivalent fit-energy-range (#514): when set, the
    // LM cost-function masks residuals to bins inside [E_min, E_max].
    // The Python caller is expected to pass full grid + per-bin data
    // and let the solver-side mask handle the restriction; the model
    // is still evaluated on the full grid so resolution broadening at
    // the boundaries remains correct (no caller-side margin slicing
    // needed).
    config = config
        .with_fit_energy_range(fit_energy_range)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    // Issue #633: freeze known densities (calibration-foil thermometry).
    config = apply_density_freeze(config, fix_densities, density_free)?;

    // Build 1D InputData
    let input = InputData::Transmission {
        transmission: t_slice.to_vec(),
        uncertainty: u_slice.to_vec(),
    };

    // Release the GIL for the fit computation.
    let result = py.detach(move || fit_spectrum_typed(&input, &config));

    // Config-class errors (InvalidParameter) surface as ValueError —
    // matching py_spatial_map_typed, which established the convention for
    // PipelineError at this boundary (review R2: identical invalid input,
    // e.g. a baseline init outside its bounds, previously raised ValueError
    // from the spatial API but RuntimeError here because the error variant
    // was stringified inside py.detach).  Other variants stay RuntimeError —
    // they signal solver / numeric failures, not bad input.
    let result = result.map_err(|e| {
        let msg = e.to_string();
        if matches!(
            e,
            nereids_pipeline::error::PipelineError::InvalidParameter(_)
        ) {
            pyo3::exceptions::PyValueError::new_err(msg)
        } else {
            pyo3::exceptions::PyRuntimeError::new_err(msg)
        }
    })?;

    Ok(PyFitResult {
        densities: result.densities,
        uncertainties: result.uncertainties,
        reduced_chi_squared: result.reduced_chi_squared,
        converged: result.converged,
        iterations: result.iterations,
        temperature_k: result.temperature_k,
        temperature_k_unc: result.temperature_k_unc,
        anorm: result.anorm,
        background: result.background,
        // BackD / BackF are only meaningful when the polynomial
        // background model was actually attached.  `result.back_d` /
        // `result.back_f` are `Option<f64>` (None when the inner
        // Rust fit didn't fit the exponential tail).  The outer gate
        // on `background && fit_back_d` is defensive — it ensures
        // PyFitResult.back_d is None whenever the caller didn't
        // request the tail, regardless of any future change to the
        // inner Rust contract.
        back_d: if background && fit_back_d {
            result.back_d
        } else {
            None
        },
        back_f: if background && fit_back_f {
            result.back_f
        } else {
            None
        },
        t0_us: result.t0_us,
        l_scale: result.l_scale,
        energy_scale_flight_path_m: result.energy_scale_flight_path_m,
        deviance_per_dof: result.deviance_per_dof,
        baseline: result.baseline,
        baseline_e_ref_ev: result.baseline_e_ref_ev,
        warnings: result.warnings,
    })
}
