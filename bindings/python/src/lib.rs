//! # nereids-python
//!
//! PyO3 Python bindings for the NEREIDS neutron resonance imaging library.
//!
//! Provides a Pythonic API for:
//! - Computing theoretical transmission spectra
//! - Fitting measured transmission to recover isotopic compositions
//! - Spatial mapping across imaging data
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
//! transmission = nereids.forward_model(energies, [(isotope, 0.001)], temperature_k=300.0)
//!
//! # Fit a measured spectrum
//! result = nereids.fit_spectrum(measured_t, sigma, energies, [isotope])
//! ```

use std::sync::Arc;

use numpy::{
    PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;

use nereids_core::elements;
use nereids_core::types::Isotope;
use nereids_endf::parser::parse_endf_file2;
use nereids_endf::resonance::{
    LGroup, Resonance, ResonanceData, ResonanceFormalism, ResonanceRange,
};
use nereids_endf::retrieval::{EndfLibrary, EndfRetriever, mat_number};
use nereids_fitting::lm::{self, LmConfig};
use nereids_fitting::parameters::{FitParameter, ParameterSet};
use nereids_fitting::poisson::PoissonConfig;
use nereids_fitting::transmission_model::TransmissionFitModel;
use nereids_io::normalization::{self as norm, NormalizationParams};
use nereids_io::tof::BeamlineParams;
use nereids_physics::doppler::{self, DopplerParams};
use nereids_physics::resolution::{
    self, ResolutionFunction, ResolutionParams, TabulatedResolution,
};
use nereids_physics::transmission::{self, InstrumentParams, SampleParams};
use nereids_pipeline::pipeline::FitConfig;
use nereids_pipeline::sparse::{SparseConfig, estimate_nuisance, sparse_reconstruct};

/// Python wrapper for ENDF resonance data.
#[pyclass(name = "ResonanceData", from_py_object)]
#[derive(Clone)]
struct PyResonanceData {
    inner: ResonanceData,
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
            self.inner.isotope.z, self.inner.isotope.a, self.inner.awr, n_res
        )
    }

    /// Atomic number.
    #[getter]
    fn z(&self) -> u32 {
        self.inner.isotope.z
    }

    /// Mass number.
    #[getter]
    fn a(&self) -> u32 {
        self.inner.isotope.a
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

/// Result of fitting a spectrum.
#[pyclass(name = "FitResult")]
struct PyFitResult {
    densities: Vec<f64>,
    uncertainties: Vec<f64>,
    reduced_chi_squared: f64,
    converged: bool,
    iterations: usize,
}

#[pymethods]
impl PyFitResult {
    /// Fitted areal densities (atoms/barn).
    #[getter]
    fn densities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.densities.clone())
    }

    /// Uncertainties on fitted densities.
    #[getter]
    fn uncertainties<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.uncertainties.clone())
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

    fn __repr__(&self) -> String {
        format!(
            "FitResult(converged={}, chi2_red={:.4}, densities={:?})",
            self.converged, self.reduced_chi_squared, self.densities
        )
    }
}

/// Python wrapper for tabulated resolution function.
#[pyclass(name = "TabulatedResolution", from_py_object)]
#[derive(Clone)]
struct PyTabulatedResolution {
    inner: TabulatedResolution,
}

#[pymethods]
impl PyTabulatedResolution {
    /// Number of reference energies.
    #[getter]
    fn n_energies(&self) -> usize {
        self.inner.ref_energies.len()
    }

    /// Energy range (min, max) of the reference kernels in eV.
    #[getter]
    fn energy_range(&self) -> (f64, f64) {
        let e = &self.inner.ref_energies;
        if e.is_empty() {
            (0.0, 0.0)
        } else {
            (e[0], e[e.len() - 1])
        }
    }

    /// Flight path length in meters.
    #[getter]
    fn flight_path_m(&self) -> f64 {
        self.inner.flight_path_m
    }

    /// Number of points per kernel.
    #[getter]
    fn points_per_kernel(&self) -> usize {
        self.inner
            .kernels
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
            self.inner.flight_path_m,
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

/// Python wrapper for sparse (Poisson/KL) reconstruction results.
///
/// Returned by `spatial_map(..., fitter='poisson')`.
/// Uses raw count data and open-beam for statistically optimal fitting
/// at all count levels including very low statistics (< ~10 counts/bin).
#[pyclass(name = "SparseResult")]
struct PySparseResult {
    density_maps: Vec<Py<PyArray2<f64>>>,
    nll_map: Py<PyArray2<f64>>,
    converged_map: Py<PyArray2<bool>>,
    n_converged: usize,
    n_total: usize,
    isotope_names: Vec<String>,
    shape: (usize, usize),
}

#[pymethods]
impl PySparseResult {
    /// Density maps as a list of 2D numpy arrays, one per isotope.
    #[getter]
    fn density_maps<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f64>>> {
        self.density_maps
            .iter()
            .map(|m| m.bind(py).clone())
            .collect()
    }

    /// Poisson negative log-likelihood map (analogous to chi_squared_map for LM).
    #[getter]
    fn nll_map<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.nll_map.bind(py).clone()
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

    fn __repr__(&self) -> String {
        format!(
            "SparseResult(shape={}x{}, isotopes={}, converged={}/{})",
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
    let e = energies.as_slice()?;
    let mut total = Vec::with_capacity(e.len());
    let mut elastic = Vec::with_capacity(e.len());
    let mut capture = Vec::with_capacity(e.len());
    let mut fission = Vec::with_capacity(e.len());

    // The Rust dispatcher (`cross_sections_at_energy`) handles all supported
    // resonance formalisms per range (Reich-Moore, SLBW, MLBW via an SLBW
    // approximation, RML, URR), so no additional Python-side formalism
    // dispatch is needed.
    for &energy in e {
        let xs = nereids_physics::reich_moore::cross_sections_at_energy(&data.inner, energy);
        total.push(xs.total);
        elastic.push(xs.elastic);
        capture.push(xs.capture);
        fission.push(xs.fission);
    }

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
/// Args:
///     energies: Energy grid in eV (1D numpy array).
///     isotopes: List of (ResonanceData, areal_density) tuples.
///     temperature_k: Sample temperature in Kelvin (default 0.0).
///     flight_path_m: Flight path in meters for Gaussian resolution (optional).
///     delta_t_us: Timing uncertainty in microseconds (optional).
///     delta_l_m: Path length uncertainty in meters (optional).
///     resolution: TabulatedResolution from ``load_resolution()`` (optional).
///
/// Returns:
///     1D numpy array of transmission values.
#[pyfunction]
#[pyo3(signature = (energies, isotopes, temperature_k=0.0, flight_path_m=None, delta_t_us=None, delta_l_m=None, resolution=None))]
fn forward_model<'py>(
    py: Python<'py>,
    energies: PyReadonlyArray1<f64>,
    isotopes: Vec<(PyResonanceData, f64)>,
    temperature_k: f64,
    flight_path_m: Option<f64>,
    delta_t_us: Option<f64>,
    delta_l_m: Option<f64>,
    resolution: Option<PyTabulatedResolution>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let e = energies.as_slice()?;

    let sample_isotopes: Vec<(ResonanceData, f64)> = isotopes
        .into_iter()
        .map(|(d, thick)| (d.inner, thick))
        .collect();

    let sample = SampleParams {
        temperature_k,
        isotopes: sample_isotopes,
    };

    let res_fn = build_resolution(flight_path_m, delta_t_us, delta_l_m, resolution)?;
    let instrument = res_fn.map(|r| InstrumentParams { resolution: r });

    let t = transmission::forward_model(e, &sample, instrument.as_ref());
    Ok(PyArray1::from_vec(py, t))
}

/// Fit a measured transmission spectrum to recover isotopic areal densities.
///
/// Resolution broadening can be applied via either Gaussian parameters
/// (``flight_path_m``, ``delta_t_us``, ``delta_l_m``) or a tabulated
/// resolution function (``resolution``). Providing both is an error.
///
/// Args:
///     measured_t: Measured transmission (1D numpy array).
///     sigma: Measurement uncertainties (1D numpy array).
///     energies: Energy grid in eV (1D numpy array).
///     isotopes: List of ResonanceData objects.
///     temperature_k: Sample temperature in Kelvin (default 0.0).
///     initial_densities: Initial guess for areal densities (optional).
///     max_iter: Maximum LM iterations (default 100).
///     flight_path_m: Flight path in meters for Gaussian resolution (optional).
///     delta_t_us: Timing uncertainty in microseconds (optional).
///     delta_l_m: Path length uncertainty in meters (optional).
///     resolution: TabulatedResolution from ``load_resolution()`` (optional).
///
/// Returns:
///     FitResult with densities, uncertainties, and fit quality.
#[pyfunction]
#[pyo3(signature = (measured_t, sigma, energies, isotopes, temperature_k=0.0, initial_densities=None, max_iter=100, flight_path_m=None, delta_t_us=None, delta_l_m=None, resolution=None))]
fn fit_spectrum(
    measured_t: PyReadonlyArray1<f64>,
    sigma: PyReadonlyArray1<f64>,
    energies: PyReadonlyArray1<f64>,
    isotopes: Vec<PyResonanceData>,
    temperature_k: f64,
    initial_densities: Option<Vec<f64>>,
    max_iter: usize,
    flight_path_m: Option<f64>,
    delta_t_us: Option<f64>,
    delta_l_m: Option<f64>,
    resolution: Option<PyTabulatedResolution>,
) -> PyResult<PyFitResult> {
    let e = energies.as_slice()?;
    let t = measured_t.as_slice()?;
    let s = sigma.as_slice()?;

    if t.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "measured_t must not be empty",
        ));
    }
    if t.len() != s.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "measured_t length ({}) must match sigma length ({})",
            t.len(),
            s.len(),
        )));
    }
    if t.len() != e.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "measured_t length ({}) must match energies length ({})",
            t.len(),
            e.len(),
        )));
    }
    if isotopes.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "isotopes list must not be empty",
        ));
    }

    let n_isotopes = isotopes.len();
    let res_data: Vec<ResonanceData> = isotopes.into_iter().map(|d| d.inner).collect();

    let init = initial_densities.unwrap_or_else(|| vec![0.001; n_isotopes]);

    if init.len() != n_isotopes {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "initial_densities length ({}) must match isotopes length ({})",
            init.len(),
            n_isotopes,
        )));
    }

    let res_fn = build_resolution(flight_path_m, delta_t_us, delta_l_m, resolution)?;
    let instrument = res_fn.map(|r| Arc::new(InstrumentParams { resolution: r }));

    let model = TransmissionFitModel {
        energies: e.to_vec(),
        resonance_data: res_data,
        temperature_k,
        instrument,
        density_indices: (0..n_isotopes).collect(),
    };

    let mut params = ParameterSet::new(
        init.iter()
            .enumerate()
            .map(|(i, &d)| FitParameter::non_negative(format!("isotope_{}", i), d))
            .collect(),
    );

    let config = LmConfig {
        max_iter,
        ..LmConfig::default()
    };

    let result = lm::levenberg_marquardt(&model, t, s, &mut params, &config);

    let densities: Vec<f64> = (0..n_isotopes).map(|i| result.params[i]).collect();
    let uncertainties = result
        .uncertainties
        .unwrap_or_else(|| vec![f64::NAN; n_isotopes]);

    Ok(PyFitResult {
        densities,
        uncertainties,
        reduced_chi_squared: result.reduced_chi_squared,
        converged: result.converged,
        iterations: result.iterations,
    })
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
fn load_endf(z: u32, a: u32, library: &str, mat: Option<u32>) -> PyResult<PyResonanceData> {
    let lib = match library {
        "endf8.0" | "endf/b-viii.0" => EndfLibrary::EndfB8_0,
        "endf8.1" | "endf/b-viii.1" => EndfLibrary::EndfB8_1,
        "jeff3.3" => EndfLibrary::Jeff3_3,
        "jendl5" => EndfLibrary::Jendl5,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown library '{}'. Use one of: endf8.0, endf8.1, jeff3.3, jendl5",
                library
            )));
        }
    };

    let isotope = Isotope::new(z, a);

    let mat_num = match mat {
        Some(m) => m,
        None => mat_number(&isotope).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "MAT number not found for Z={} A={}; provide mat= explicitly",
                z, a
            ))
        })?,
    };

    let retriever = EndfRetriever::new();
    let (_path, contents) = retriever
        .get_endf_file(&isotope, lib, mat_num)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;

    let data = parse_endf_file2(&contents)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("ENDF parse error: {}", e)))?;

    // Validate that the parsed ENDF data matches the requested isotope.
    if data.isotope.z != z || data.isotope.a != a {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "ENDF data mismatch: requested Z={} A={} but file contains Z={} A={}",
            z, a, data.isotope.z, data.isotope.a
        )));
    }

    Ok(PyResonanceData { inner: data })
}

/// Load ENDF resonance data from a local file.
///
/// Args:
///     path: Path to an ENDF-format file on disk.
///
/// Returns:
///     ResonanceData parsed from the file.
#[pyfunction]
fn load_endf_file(path: &str) -> PyResult<PyResonanceData> {
    let contents = std::fs::read_to_string(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Cannot read '{}': {}", path, e))
    })?;

    let data = parse_endf_file2(&contents)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("ENDF parse error: {}", e)))?;

    Ok(PyResonanceData { inner: data })
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
                resonances: res,
            }]
        }
    };

    Ok(PyResonanceData {
        inner: ResonanceData {
            isotope: Isotope::new(z, a),
            za: z * 1000 + a,
            awr,
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high: 1e6,
                resolved: true,
                formalism: res_formalism,
                target_spin,
                scattering_radius,
                l_groups: groups,
                rml: None,
                urr: None,
                ap_table: None,
            }],
        },
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

/// Validate Gaussian resolution parameters: finite, positive flight path,
/// non-negative timing and path length uncertainties.
fn validate_gaussian_params(fp: f64, dt: f64, dl: f64) -> PyResult<()> {
    if !fp.is_finite() || fp <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "flight_path_m must be finite and positive",
        ));
    }
    if !dt.is_finite() || dt < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "delta_t_us must be finite and non-negative",
        ));
    }
    if !dl.is_finite() || dl < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "delta_l_m must be finite and non-negative",
        ));
    }
    Ok(())
}

/// Validate that an energy grid is finite, positive, and sorted ascending.
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

/// Build a `ResolutionFunction` from Python arguments.
///
/// Validates mutual exclusivity (Gaussian vs. tabulated) and completeness
/// of Gaussian parameters. Returns `None` when no resolution is requested.
fn build_resolution(
    flight_path_m: Option<f64>,
    delta_t_us: Option<f64>,
    delta_l_m: Option<f64>,
    resolution: Option<PyTabulatedResolution>,
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
        validate_gaussian_params(fp, dt, dl)?;
        Ok(Some(ResolutionFunction::Gaussian(ResolutionParams {
            flight_path_m: fp,
            delta_t_us: dt,
            delta_l_m: dl,
        })))
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
    if !awr.is_finite() || awr <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "awr must be finite and positive",
        ));
    }
    if !temperature_k.is_finite() || temperature_k < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "temperature_k must be finite and non-negative",
        ));
    }

    if temperature_k == 0.0 {
        return Ok(PyArray1::from_vec(py, xs.to_vec()));
    }

    let params = DopplerParams { temperature_k, awr };
    let result = doppler::doppler_broaden(e, xs, &params);
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
#[pyo3(signature = (energies, cross_sections, flight_path_m, delta_t_us, delta_l_m))]
fn resolution_broaden<'py>(
    py: Python<'py>,
    energies: PyReadonlyArray1<f64>,
    cross_sections: PyReadonlyArray1<f64>,
    flight_path_m: f64,
    delta_t_us: f64,
    delta_l_m: f64,
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
    if !flight_path_m.is_finite() || flight_path_m <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "flight_path_m must be finite and positive",
        ));
    }
    if !delta_t_us.is_finite() || delta_t_us < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "delta_t_us must be finite and non-negative",
        ));
    }
    if !delta_l_m.is_finite() || delta_l_m < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "delta_l_m must be finite and non-negative",
        ));
    }

    if delta_t_us == 0.0 && delta_l_m == 0.0 {
        return Ok(PyArray1::from_vec(py, xs.to_vec()));
    }

    let params = ResolutionParams {
        flight_path_m,
        delta_t_us,
        delta_l_m,
    };
    let result = resolution::resolution_broaden(e, xs, &params);
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

    Ok(PyTabulatedResolution { inner: tab })
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
    resolution: &PyTabulatedResolution,
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

    let res_fn = ResolutionFunction::Tabulated(resolution.inner.clone());
    let result = resolution::apply_resolution(e, s, &res_fn);
    Ok(PyArray1::from_vec(py, result))
}

/// Run per-pixel fitting across a transmission image stack.
///
/// Routes to either the Levenberg-Marquardt (LM/χ²) fitter or the
/// Poisson/KL sparse fitter depending on the `fitter` argument.
///
/// Args:
///     transmission: 3D array (n_energies, height, width).
///         - fitter='lm':      normalised transmission T ∈ [0,1].
///         - fitter='poisson': raw sample counts (integer-valued floats).
///     uncertainty: 3D array (n_energies, height, width).
///         - fitter='lm':      per-bin transmission uncertainty σ.
///         - fitter='poisson': raw open-beam counts (same shape as sample).
///     energies: 1D numpy array of energy values in eV.
///     isotopes: List of ResonanceData objects.
///     temperature_k: Sample temperature in Kelvin (default 300.0).
///     initial_densities: Initial guesses for areal densities (optional).
///     dead_pixels: 2D bool numpy array marking dead pixels (optional).
///     flight_path_m: Flight path for Gaussian resolution (optional).
///     delta_t_us: Timing uncertainty for Gaussian resolution (optional).
///     delta_l_m: Path length uncertainty for Gaussian resolution (optional).
///     resolution: TabulatedResolution for tabulated broadening (optional).
///     max_iter: Maximum LM iterations per pixel (default 100).
///     fitter: 'lm' (default) for Gaussian χ² or 'poisson' for Poisson NLL.
///     roi: [y0, y1, x0, x1] region for Stage-1 nuisance estimation
///         (Poisson path only, default uses full image).
///
/// Returns:
///     SpatialResult (fitter='lm') or SparseResult (fitter='poisson').
#[pyfunction]
#[pyo3(name = "spatial_map", signature = (transmission, uncertainty, energies, isotopes, temperature_k=300.0, initial_densities=None, dead_pixels=None, flight_path_m=None, delta_t_us=None, delta_l_m=None, resolution=None, max_iter=100, fitter="lm", roi=None))]
fn py_spatial_map(
    py: Python<'_>,
    transmission: PyReadonlyArray3<f64>,
    uncertainty: PyReadonlyArray3<f64>,
    energies: PyReadonlyArray1<f64>,
    isotopes: Vec<PyResonanceData>,
    temperature_k: f64,
    initial_densities: Option<Vec<f64>>,
    dead_pixels: Option<PyReadonlyArray2<bool>>,
    flight_path_m: Option<f64>,
    delta_t_us: Option<f64>,
    delta_l_m: Option<f64>,
    resolution: Option<PyTabulatedResolution>,
    max_iter: usize,
    fitter: &str,
    roi: Option<[usize; 4]>,
) -> PyResult<Py<PyAny>> {
    let e = energies.as_slice()?;

    if e.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "energies must not be empty",
        ));
    }

    // Validate shapes using cheap PyReadonly views before cloning
    let t_shape = transmission.shape();
    let u_shape = uncertainty.shape();
    if t_shape != u_shape {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "transmission shape {:?} must match uncertainty shape {:?}",
            t_shape, u_shape,
        )));
    }
    if e.len() != t_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "energies length ({}) must match spectral axis ({}) of transmission",
            e.len(),
            t_shape[0],
        )));
    }
    if let Some(ref dead) = dead_pixels {
        let d_shape = dead.shape();
        if d_shape[0] != t_shape[1] || d_shape[1] != t_shape[2] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dead_pixels shape ({}, {}) must match spatial dimensions ({}, {}) of transmission",
                d_shape[0], d_shape[1], t_shape[1], t_shape[2],
            )));
        }
    }

    if isotopes.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "isotopes list must not be empty",
        ));
    }

    let n_isotopes = isotopes.len();
    let isotope_names: Vec<String> = isotopes
        .iter()
        .map(|d| {
            let sym = elements::element_symbol(d.inner.isotope.z).unwrap_or("?");
            format!("{}-{}", sym, d.inner.isotope.a)
        })
        .collect();
    let res_data: Vec<ResonanceData> = isotopes.into_iter().map(|d| d.inner).collect();

    let init = initial_densities.unwrap_or_else(|| vec![0.001; n_isotopes]);
    if init.len() != n_isotopes {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "initial_densities length ({}) must match isotopes length ({})",
            init.len(),
            n_isotopes,
        )));
    }

    let res_fn = build_resolution(flight_path_m, delta_t_us, delta_l_m, resolution)?;

    match fitter {
        "lm" => {
            let config = FitConfig {
                energies: e.to_vec(),
                resonance_data: res_data,
                isotope_names: isotope_names.clone(),
                temperature_k,
                resolution: res_fn,
                initial_densities: init,
                lm_config: LmConfig {
                    max_iter,
                    ..LmConfig::default()
                },
                precomputed_cross_sections: None,
            };

            // Clone arrays only after all validation passes
            let trans = transmission.as_array().to_owned();
            let unc = uncertainty.as_array().to_owned();
            let dead = dead_pixels.map(|d| d.as_array().to_owned());

            let result =
                nereids_pipeline::spatial::spatial_map(&trans, &unc, &config, dead.as_ref(), None);

            let shape = (
                result.converged_map.shape()[0],
                result.converged_map.shape()[1],
            );
            let py_result = PySpatialResult {
                density_maps: result
                    .density_maps
                    .into_iter()
                    .map(|m| PyArray2::from_owned_array(py, m).unbind())
                    .collect(),
                uncertainty_maps: result
                    .uncertainty_maps
                    .into_iter()
                    .map(|m| PyArray2::from_owned_array(py, m).unbind())
                    .collect(),
                chi_squared_map: PyArray2::from_owned_array(py, result.chi_squared_map).unbind(),
                converged_map: PyArray2::from_owned_array(py, result.converged_map).unbind(),
                n_converged: result.n_converged,
                n_total: result.n_total,
                isotope_names,
                shape,
            };
            Py::new(py, py_result).map(|p| p.into_any())
        }

        "poisson" => {
            // Stage 1: estimate flux + background from open-beam (second arg).
            let sample = transmission.as_array().to_owned();
            let open_beam = uncertainty.as_array().to_owned();
            let dead = dead_pixels.map(|d| d.as_array().to_owned());

            let roi_ranges = roi.map(|r| (r[0]..r[1], r[2]..r[3]));
            let nuisance = estimate_nuisance(&sample, &open_beam, roi_ranges);

            let sparse_config = SparseConfig {
                energies: e.to_vec(),
                resonance_data: res_data,
                isotope_names: isotope_names.clone(),
                temperature_k,
                resolution: res_fn,
                initial_densities: init,
                poisson_config: PoissonConfig::default(),
            };

            // Stage 2: per-pixel Poisson NLL fitting.
            let result =
                sparse_reconstruct(&sample, &nuisance, &sparse_config, dead.as_ref(), None);

            let shape = (result.nll_map.shape()[0], result.nll_map.shape()[1]);
            let py_result = PySparseResult {
                density_maps: result
                    .density_maps
                    .into_iter()
                    .map(|m| PyArray2::from_owned_array(py, m).unbind())
                    .collect(),
                nll_map: PyArray2::from_owned_array(py, result.nll_map).unbind(),
                converged_map: PyArray2::from_owned_array(py, result.converged_map).unbind(),
                n_converged: result.n_converged,
                n_total: result.n_total,
                isotope_names,
                shape,
            };
            Py::new(py, py_result).map(|p| p.into_any())
        }

        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "fitter must be 'lm' or 'poisson', got '{}'",
            other,
        ))),
    }
}

/// Fit a single spectrum averaged over a region of interest.
///
/// Averages the transmission and uncertainty across the specified rectangular
/// region, then fits the resulting high-statistics spectrum. Useful for
/// getting reference densities before per-pixel mapping.
///
/// Args:
///     transmission: 3D numpy array (n_energies, height, width).
///     uncertainty: 3D numpy array (n_energies, height, width).
///     y_range: (start, end) row range for the ROI.
///     x_range: (start, end) column range for the ROI.
///     energies: 1D numpy array of energy values in eV.
///     isotopes: List of ResonanceData objects.
///     temperature_k: Sample temperature in Kelvin (default 300.0).
///     initial_densities: Initial guesses for areal densities (optional).
///     flight_path_m: Flight path for Gaussian resolution (optional).
///     delta_t_us: Timing uncertainty for Gaussian resolution (optional).
///     delta_l_m: Path length uncertainty for Gaussian resolution (optional).
///     resolution: TabulatedResolution for tabulated broadening (optional).
///     max_iter: Maximum LM iterations (default 100).
///
/// Returns:
///     FitResult with densities, uncertainties, and fit quality.
#[pyfunction]
#[pyo3(name = "fit_roi", signature = (transmission, uncertainty, y_range, x_range, energies, isotopes, temperature_k=300.0, initial_densities=None, flight_path_m=None, delta_t_us=None, delta_l_m=None, resolution=None, max_iter=100))]
fn py_fit_roi(
    transmission: PyReadonlyArray3<f64>,
    uncertainty: PyReadonlyArray3<f64>,
    y_range: (usize, usize),
    x_range: (usize, usize),
    energies: PyReadonlyArray1<f64>,
    isotopes: Vec<PyResonanceData>,
    temperature_k: f64,
    initial_densities: Option<Vec<f64>>,
    flight_path_m: Option<f64>,
    delta_t_us: Option<f64>,
    delta_l_m: Option<f64>,
    resolution: Option<PyTabulatedResolution>,
    max_iter: usize,
) -> PyResult<PyFitResult> {
    let e = energies.as_slice()?;

    if e.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "energies must not be empty",
        ));
    }

    // Validate shapes using cheap PyReadonly views before cloning
    let t_shape = transmission.shape();
    let u_shape = uncertainty.shape();
    if t_shape != u_shape {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "transmission shape {:?} must match uncertainty shape {:?}",
            t_shape, u_shape,
        )));
    }
    if e.len() != t_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "energies length ({}) must match spectral axis ({}) of transmission",
            e.len(),
            t_shape[0],
        )));
    }

    if isotopes.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "isotopes list must not be empty",
        ));
    }
    if y_range.0 >= y_range.1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "y_range must be non-empty: start ({}) >= end ({})",
            y_range.0, y_range.1,
        )));
    }
    if x_range.0 >= x_range.1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "x_range must be non-empty: start ({}) >= end ({})",
            x_range.0, x_range.1,
        )));
    }
    if y_range.1 > t_shape[1] || x_range.1 > t_shape[2] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "ROI exceeds image dimensions: y_end={} (max {}), x_end={} (max {})",
            y_range.1, t_shape[1], x_range.1, t_shape[2],
        )));
    }

    let n_isotopes = isotopes.len();
    let isotope_names: Vec<String> = isotopes
        .iter()
        .map(|d| {
            let sym = elements::element_symbol(d.inner.isotope.z).unwrap_or("?");
            format!("{}-{}", sym, d.inner.isotope.a)
        })
        .collect();
    let res_data: Vec<ResonanceData> = isotopes.into_iter().map(|d| d.inner).collect();

    let init = initial_densities.unwrap_or_else(|| vec![0.001; n_isotopes]);
    if init.len() != n_isotopes {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "initial_densities length ({}) must match isotopes length ({})",
            init.len(),
            n_isotopes,
        )));
    }

    let res_fn = build_resolution(flight_path_m, delta_t_us, delta_l_m, resolution)?;

    let config = FitConfig {
        energies: e.to_vec(),
        resonance_data: res_data,
        isotope_names,
        temperature_k,
        resolution: res_fn,
        initial_densities: init,
        lm_config: LmConfig {
            max_iter,
            ..LmConfig::default()
        },
        precomputed_cross_sections: None,
    };

    // Clone arrays only after all validation passes
    let trans = transmission.as_array().to_owned();
    let unc = uncertainty.as_array().to_owned();

    let result = nereids_pipeline::spatial::fit_roi(
        &trans,
        &unc,
        y_range.0..y_range.1,
        x_range.0..x_range.1,
        &config,
    );

    Ok(PyFitResult {
        densities: result.densities,
        uncertainties: result.uncertainties,
        reduced_chi_squared: result.reduced_chi_squared,
        converged: result.converged,
        iterations: result.iterations,
    })
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

    let result = norm::normalize(&s, &ob, &params, dc.as_ref())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;

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
    elements::parse_isotope_str(s).map(|iso| (iso.z, iso.a))
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
    elements::natural_abundance(&Isotope::new(z, a))
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
        .map(|(iso, frac)| ((iso.z, iso.a), frac))
        .collect()
}

/// NEREIDS Python module.
#[pymodule]
fn nereids(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyResonanceData>()?;
    m.add_class::<PyFitResult>()?;
    m.add_class::<PyTabulatedResolution>()?;
    m.add_class::<PySpatialResult>()?;
    m.add_class::<PySparseResult>()?;
    m.add_function(wrap_pyfunction!(cross_sections, m)?)?;
    m.add_function(wrap_pyfunction!(forward_model, m)?)?;
    m.add_function(wrap_pyfunction!(fit_spectrum, m)?)?;
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
    m.add_function(wrap_pyfunction!(py_spatial_map, m)?)?;
    m.add_function(wrap_pyfunction!(py_fit_roi, m)?)?;
    m.add_function(wrap_pyfunction!(load_tiff_stack, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(tof_to_energy_centers, m)?)?;
    m.add_function(wrap_pyfunction!(py_element_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(py_element_name, m)?)?;
    m.add_function(wrap_pyfunction!(py_parse_isotope_str, m)?)?;
    m.add_function(wrap_pyfunction!(py_natural_abundance, m)?)?;
    m.add_function(wrap_pyfunction!(py_natural_isotopes, m)?)?;
    Ok(())
}
