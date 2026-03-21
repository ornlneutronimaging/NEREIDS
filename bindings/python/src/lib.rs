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
//! Use `from_counts()` or `from_transmission()` to create typed input data,
//! then pass to `spatial_map_typed()` for per-pixel fitting:
//!
//! - **Counts** → Poisson KL (statistically optimal for raw detector counts)
//! - **Transmission** → LM (default) or KL (opt-in via `solver="kl"`)
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
use nereids_io::normalization::{self as norm, NormalizationParams};
use nereids_io::tof::BeamlineParams;
use nereids_physics::doppler::{self, DopplerParams};
use nereids_physics::resolution::{
    self, ResolutionFunction, ResolutionParams, TabulatedResolution,
};
use nereids_physics::transmission::{self, InstrumentParams, SampleParams};
use nereids_pipeline::detectability;

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
        let n_res: usize = self
            .inner
            .ranges
            .iter()
            .flat_map(|r| &r.l_groups)
            .map(|lg| lg.resonances.len())
            .sum();
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

    /// Number of resonances.
    #[getter]
    fn n_resonances(&self) -> usize {
        self.inner
            .ranges
            .iter()
            .flat_map(|r| &r.l_groups)
            .map(|lg| lg.resonances.len())
            .sum()
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
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown library '{}'. Use one of: endf8.0, endf8.1, jeff3.3, jendl5",
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
                let mat_num = mat_number(iso).ok_or_else(|| {
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

        // Map results back into self.resonance_data.
        for (i, result) in results.into_iter().enumerate() {
            let data = result.map_err(|(is_parse, msg)| {
                if is_parse {
                    pyo3::exceptions::PyValueError::new_err(msg)
                } else {
                    pyo3::exceptions::PyRuntimeError::new_err(msg)
                }
            })?;
            self.resonance_data[i] = Some(Arc::new(data));
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
    /// Fitted normalization factor (1.0 when background not enabled).
    anorm: f64,
    /// Fitted background parameters [BackA, BackB, BackC] (all 0.0 when not enabled).
    background: [f64; 3],
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
    #[getter]
    fn temperature_k_unc(&self) -> Option<f64> {
        self.temperature_k_unc
    }

    /// Fitted normalization factor (1.0 if background not enabled).
    #[getter]
    fn anorm(&self) -> f64 {
        self.anorm
    }

    /// Fitted background parameters (BackA, BackB, BackC).
    #[getter]
    fn background(&self) -> [f64; 3] {
        self.background
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

/// Result of spatial (per-pixel) mapping.
///
/// Numpy arrays are constructed once and cached; property access returns
/// cheap references (refcount bump) rather than copying data.
#[pyclass(name = "SpatialResult")]
struct PySpatialResult {
    density_maps: Vec<Py<PyArray2<f64>>>,
    uncertainty_maps: Vec<Py<PyArray2<f64>>>,
    chi_squared_map: Py<PyArray2<f64>>,
    converged_map: Py<PyArray2<bool>>,
    n_converged: usize,
    n_total: usize,
    isotope_names: Vec<String>,
    shape: (usize, usize),
    /// Per-pixel fitted temperature (None when fit_temperature=False).
    temperature_map: Option<Py<PyArray2<f64>>>,
    /// Per-pixel normalization factor (None when background=False).
    anorm_map: Option<Py<PyArray2<f64>>>,
    /// Per-pixel background parameters [BackA, BackB, BackC] (None when background=False).
    background_maps: Option<[Py<PyArray2<f64>>; 3]>,
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

    /// Per-pixel normalization factor Anorm (None when background fitting was disabled).
    #[getter]
    fn anorm_map<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.anorm_map.as_ref().map(|m| m.bind(py).clone())
    }

    /// Per-pixel background parameter maps [BackA, BackB, BackC]
    /// (None when background fitting was disabled).
    #[getter]
    fn background_maps<'py>(&self, py: Python<'py>) -> Option<Vec<Bound<'py, PyArray2<f64>>>> {
        self.background_maps
            .as_ref()
            .map(|maps| maps.iter().map(|m| m.bind(py).clone()).collect())
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
///     energies: Energy grid in eV (1D numpy array).
///     data: ResonanceData for the isotope.
///
/// Returns:
///     Dictionary with 'total', 'elastic', 'capture', 'fission' arrays.
#[pyfunction]
fn cross_sections<'py>(
    py: Python<'py>,
    energies: PyReadonlyArray1<f64>,
    data: &PyResonanceData,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let e_owned = energies.as_slice()?.to_vec();
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

    let e_owned = energies.as_slice()?.to_vec();

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

/// Convert time-of-flight (μs) to energy (eV).
///
/// Args:
///     tof_us: Time-of-flight in microseconds.
///     flight_path_m: Flight path in meters.
///
/// Returns:
///     Energy in eV.
#[pyfunction]
fn tof_to_energy(tof_us: f64, flight_path_m: f64) -> f64 {
    nereids_core::constants::tof_to_energy(tof_us, flight_path_m)
}

/// Convert energy (eV) to time-of-flight (μs).
///
/// Args:
///     energy_ev: Energy in eV.
///     flight_path_m: Flight path in meters.
///
/// Returns:
///     Time-of-flight in microseconds.
#[pyfunction]
fn energy_to_tof(energy_ev: f64, flight_path_m: f64) -> f64 {
    nereids_core::constants::energy_to_tof(energy_ev, flight_path_m)
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
///              "jeff3.3", "jendl5".
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
        None => mat_number(&isotope).ok_or_else(|| {
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

/// Load a multi-frame TIFF file into a 3D numpy array.
///
/// Each TIFF frame becomes one slice along the first axis.
/// Data is converted to float64 regardless of the source pixel type.
///
/// Args:
///     path: Path to the multi-frame TIFF file.
///
/// Returns:
///     3D numpy array with shape (n_frames, height, width).
#[pyfunction]
fn load_tiff_stack<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let arr = nereids_io::tiff_stack::load_tiff_stack(std::path::Path::new(path))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
    Ok(PyArray3::from_owned_array(py, arr))
}

/// Load a folder of single-frame TIFFs into a 3D numpy array.
///
/// Files are sorted lexicographically by name, so they should be named with
/// zero-padded indices (e.g., ``frame_0001.tif``, ``frame_0002.tif``, ...).
///
/// Args:
///     folder: Path to the directory containing TIFF files.
///     pattern: Optional glob pattern matched against each filename (not the
///              full path).  Supports ``*`` and ``?`` wildcards
///              (case-insensitive).  Only files with ``.tif`` or ``.tiff``
///              extensions are ever loaded; the pattern adds an additional
///              filename filter on top of that.
///
/// Returns:
///     3D numpy array with shape (n_frames, height, width), dtype float64.
///
/// Raises:
///     FileNotFoundError: If the folder does not exist or no files match.
///     NotADirectoryError: If the provided path is not a directory.
///     ValueError: If matched frames have inconsistent dimensions.
///     IOError: For TIFF decoding errors or other I/O failures.
#[pyfunction]
#[pyo3(signature = (folder, pattern=None))]
fn load_tiff_folder<'py>(
    py: Python<'py>,
    folder: &str,
    pattern: Option<&str>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let arr = nereids_io::tiff_stack::load_tiff_folder(std::path::Path::new(folder), pattern)
        .map_err(|e| match &e {
            nereids_io::error::IoError::NoMatchingFiles { .. } => {
                pyo3::exceptions::PyFileNotFoundError::new_err(format!("{}", e))
            }
            nereids_io::error::IoError::NotADirectory(_) => {
                pyo3::exceptions::PyNotADirectoryError::new_err(format!("{}", e))
            }
            nereids_io::error::IoError::FileNotFound(_, source)
                if source.kind() == std::io::ErrorKind::NotFound =>
            {
                pyo3::exceptions::PyFileNotFoundError::new_err(format!("{}", e))
            }
            nereids_io::error::IoError::DimensionMismatch { .. } => {
                pyo3::exceptions::PyValueError::new_err(format!("{}", e))
            }
            _ => pyo3::exceptions::PyIOError::new_err(format!("{}", e)),
        })?;
    Ok(PyArray3::from_owned_array(py, arr))
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

    let res_data: Vec<ResonanceData> = isotopes
        .into_iter()
        .map(|d| Arc::unwrap_or_clone(d.inner))
        .collect();
    let res_fn = build_resolution(flight_path_m, delta_t_us, delta_l_m, resolution, delta_e_us)?;
    let instrument = res_fn.map(|r| InstrumentParams { resolution: r });

    // Copy numpy slice to owned Vec so we can release the GIL.
    let e_owned = e.to_vec();

    // Release the GIL for the heavy Doppler + resolution broadening.
    let xs = py.detach(move || {
        transmission::broadened_cross_sections(
            &e_owned,
            &res_data,
            temperature_k,
            instrument.as_ref(),
            None,
        )
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
/// ``spatial_map(dead_pixels=...)``.
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
    let e = energies_nominal.as_slice()?;
    let t = transmission.as_slice()?;
    let s = uncertainty.as_slice()?;

    let res_data: Vec<nereids_endf::resonance::ResonanceData> = isotopes
        .into_iter()
        .map(|d| Arc::unwrap_or_clone(d.inner))
        .collect();

    let instrument = resolution.map(|r| nereids_physics::transmission::InstrumentParams {
        resolution: nereids_physics::resolution::ResolutionFunction::Tabulated(r.inner.clone()),
    });

    let result = py.detach(move || {
        nereids_pipeline::calibration::calibrate_energy(
            e,
            t,
            s,
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

/// NEREIDS Python module.
#[pymodule]
fn nereids(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyResonanceData>()?;
    m.add_class::<PyFitResult>()?;
    m.add_class::<PyTabulatedResolution>()?;
    m.add_class::<PySpatialResult>()?;
    m.add_class::<PyTraceDetectabilityReport>()?;
    m.add_function(wrap_pyfunction!(cross_sections, m)?)?;
    m.add_function(wrap_pyfunction!(forward_model, m)?)?;
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
    m.add_function(wrap_pyfunction!(load_tiff_stack, m)?)?;
    m.add_function(wrap_pyfunction!(load_tiff_folder, m)?)?;
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
    m.add_function(wrap_pyfunction!(py_calibrate_energy, m)?)?;
    m.add_class::<PyCalibrationResult>()?;
    // Phase 5: Typed API
    m.add_class::<PyInputData>()?;
    m.add_class::<PyIsotopeGroup>()?;
    m.add_function(wrap_pyfunction!(py_from_counts, m)?)?;
    m.add_function(wrap_pyfunction!(py_from_transmission, m)?)?;
    m.add_function(wrap_pyfunction!(py_spatial_map_typed, m)?)?;
    m.add_function(wrap_pyfunction!(py_fit_spectrum_typed, m)?)?;
    Ok(())
}

// ── Phase 5: Typed Python API ────────────────────────────────────────────

use nereids_pipeline::pipeline::UnifiedFitConfig;
use nereids_pipeline::spatial::{InputData3D, spatial_map_typed};

/// Opaque wrapper around InputData3D for Python.
///
/// Created via `from_counts()` or `from_transmission()`.
/// Passed to `spatial_map_typed()`.
#[pyclass(name = "InputData")]
struct PyInputData {
    /// We store owned 3D arrays (ndarray::Array3) so the data lives
    /// as long as the Python object.
    kind: String, // "counts" or "transmission"
    data_a: ndarray::Array3<f64>,
    data_b: ndarray::Array3<f64>,
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
    })
}

/// Create InputData from normalized transmission and uncertainty.
///
/// The fitting engine will use LM by default. Pass solver="kl" to use
/// Poisson KL instead (for low-count transmission data).
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
        "lm" => Ok(
            nereids_pipeline::pipeline::SolverConfig::LevenbergMarquardt(
                nereids_fitting::lm::LmConfig {
                    max_iter,
                    ..Default::default()
                },
            ),
        ),
        "kl" | "poisson" => Ok(nereids_pipeline::pipeline::SolverConfig::PoissonKL(
            nereids_fitting::poisson::PoissonConfig {
                max_iter,
                ..Default::default()
            },
        )),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown solver: '{other}'. Use 'auto', 'lm', or 'kl'."
        ))),
    }
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
    let temperature_map = result
        .temperature_map
        .as_ref()
        .map(|m| PyArray2::from_array(py, m).into());

    PySpatialResult {
        density_maps,
        uncertainty_maps,
        chi_squared_map: PyArray2::from_array(py, &result.chi_squared_map).into(),
        converged_map: PyArray2::from_array(py, &result.converged_map).into(),
        n_converged: result.n_converged,
        n_total: result.n_total,
        isotope_names: result.isotope_labels.clone(),
        shape,
        temperature_map,
        anorm_map,
        background_maps,
    }
}

/// Spatial mapping using the typed input data API.
///
/// Dispatches per-pixel fitting based on the InputData type:
///   - from_counts → Poisson KL on raw counts (statistically optimal)
///   - from_transmission → LM by default, KL opt-in via solver="kl"
///
/// Either `isotopes` or `groups` must be provided, but not both.
/// When `groups` is provided, each group maps to one fitted density parameter.
///
/// Always returns SpatialResult.
///
/// Args:
///     data: InputData from from_counts() or from_transmission().
///     energies: 1D energy grid in eV (ascending).
///     isotopes: list of ResonanceData objects (mutually exclusive with groups).
///     temperature_k: Sample temperature in Kelvin (default 293.6).
///     fit_temperature: Whether to fit temperature per pixel (default False).
///     initial_densities: Initial density guesses (default 0.001 each).
///     dead_pixels: Optional 2D boolean dead pixel mask.
///     max_iter: Maximum iterations per pixel (default 200).
///     solver: "auto" (default), "lm", or "kl".
///     background: Enable SAMMY transmission background (only for transmission data).
///     resolution: Optional resolution function.
///     groups: list of IsotopeGroup objects (mutually exclusive with isotopes).
///
/// Returns:
///     SpatialResult with density_maps, chi_squared_map, converged_map, etc.
#[pyfunction]
#[pyo3(name = "spatial_map_typed", signature = (
    data, energies, isotopes=None, *,
    temperature_k = 293.6,
    fit_temperature = false,
    initial_densities = None,
    dead_pixels = None,
    max_iter = 200,
    solver = "auto",
    background = false,
    resolution = None,
    flight_path_m = None,
    delta_t_us = None,
    delta_l_m = None,
    groups = None,
))]
fn py_spatial_map_typed<'py>(
    py: Python<'py>,
    data: &PyInputData,
    energies: PyReadonlyArray1<'py, f64>,
    isotopes: Option<Vec<PyResonanceData>>,
    temperature_k: f64,
    fit_temperature: bool,
    initial_densities: Option<Vec<f64>>,
    dead_pixels: Option<PyReadonlyArray2<'py, bool>>,
    max_iter: usize,
    solver: &str,
    background: bool,
    resolution: Option<PyTabulatedResolution>,
    flight_path_m: Option<f64>,
    delta_t_us: Option<f64>,
    delta_l_m: Option<f64>,
    groups: Option<Vec<PyIsotopeGroup>>,
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

    let energies_vec = energies.as_slice()?.to_vec();

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
    let solver_config = parse_solver_config(solver, data.kind == "counts", max_iter)?;
    config = config.with_solver(solver_config);

    // Temperature fitting
    if fit_temperature {
        config = config.with_fit_temperature(true);
    }

    // Background (transmission only)
    if background {
        if data.kind == "counts" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "background=True is only supported for transmission data. \
                 Counts data uses nuisance estimation from the open beam.",
            ));
        }
        config = config
            .with_transmission_background(nereids_pipeline::pipeline::BackgroundConfig::default());
    }

    // Build InputData3D from the PyInputData
    let input = if data.kind == "counts" {
        InputData3D::Counts {
            sample_counts: data.data_a.view(),
            open_beam_counts: data.data_b.view(),
        }
    } else {
        InputData3D::Transmission {
            transmission: data.data_a.view(),
            uncertainty: data.data_b.view(),
        }
    };

    // Dead pixels
    let dead_arr = dead_pixels.map(|dp| dp.as_array().to_owned());

    // GIL held during computation.  InputData3D borrows PyInputData arrays
    // which are not Send, so we cannot use py.allow_threads().  The existing
    // py_spatial_map has the same limitation.  Rayon still parallelizes the
    // per-pixel fitting within the GIL.
    let result = spatial_map_typed(&input, &config, dead_arr.as_ref(), None, None)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(spatial_result_to_py(py, &result))
}

/// Fit a single spectrum using the typed input data API.
///
/// Dispatches per-pixel fitting based on the InputData type:
///   - from_counts → Poisson KL on raw counts (statistically optimal)
///   - from_transmission → LM by default, KL opt-in via solver="kl"
///
/// Either `isotopes` or `groups` must be provided, but not both.
///
/// Args:
///     transmission: 1D transmission spectrum.
///     uncertainty: 1D uncertainty (same length as transmission).
///     energies: 1D energy grid in eV (ascending).
///     isotopes: list of (ResonanceData, initial_density) tuples (mutually exclusive with groups).
///     temperature_k: Sample temperature in Kelvin (default 293.6).
///     fit_temperature: Whether to fit temperature (default False).
///     max_iter: Maximum iterations (default 200).
///     solver: "auto" (default), "lm", or "kl".
///     background: Enable SAMMY transmission background.
///     resolution: Optional resolution function.
///     groups: list of IsotopeGroup objects (mutually exclusive with isotopes).
///     initial_densities: Initial density guesses when using groups (default 0.001 each).
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
    resolution = None,
    flight_path_m = None,
    delta_t_us = None,
    delta_l_m = None,
    groups = None,
    initial_densities = None,
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
    resolution: Option<PyTabulatedResolution>,
    flight_path_m: Option<f64>,
    delta_t_us: Option<f64>,
    delta_l_m: Option<f64>,
    groups: Option<Vec<PyIsotopeGroup>>,
    initial_densities: Option<Vec<f64>>,
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

    // Solver
    let solver_config = parse_solver_config(solver, false, max_iter)?;
    config = config.with_solver(solver_config);

    // Temperature fitting
    if fit_temperature {
        config = config.with_fit_temperature(true);
    }

    // Background
    if background {
        config = config
            .with_transmission_background(nereids_pipeline::pipeline::BackgroundConfig::default());
    }

    // Build 1D InputData
    let input = InputData::Transmission {
        transmission: t_slice.to_vec(),
        uncertainty: u_slice.to_vec(),
    };

    // Release the GIL for the fit computation.
    let result = py.detach(move || fit_spectrum_typed(&input, &config).map_err(|e| e.to_string()));

    let result = result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

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
    })
}
