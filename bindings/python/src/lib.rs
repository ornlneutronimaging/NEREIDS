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

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use nereids_core::types::Isotope;
use nereids_endf::parser::parse_endf_file2;
use nereids_endf::resonance::{
    LGroup, Resonance, ResonanceData, ResonanceFormalism, ResonanceRange,
};
use nereids_endf::retrieval::{EndfLibrary, EndfRetriever, mat_number};
use nereids_fitting::lm::{self, LmConfig};
use nereids_fitting::parameters::{FitParameter, ParameterSet};
use nereids_fitting::transmission_model::TransmissionFitModel;
use nereids_physics::doppler::{self, DopplerParams};
use nereids_physics::resolution::{self, ResolutionParams};
use nereids_physics::transmission::{self, InstrumentParams, SampleParams};

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

    /// Scattering radius (fm) of the first resonance range.
    #[getter]
    fn scattering_radius(&self) -> f64 {
        self.inner
            .ranges
            .first()
            .map(|r| r.scattering_radius)
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
/// Args:
///     energies: Energy grid in eV (1D numpy array).
///     isotopes: List of (ResonanceData, areal_density) tuples.
///     temperature_k: Sample temperature in Kelvin (default 0.0).
///     flight_path_m: Flight path in meters for resolution (optional).
///     delta_t_us: Timing uncertainty in microseconds (optional).
///     delta_l_m: Path length uncertainty in meters (optional).
///
/// Returns:
///     1D numpy array of transmission values.
#[pyfunction]
#[pyo3(signature = (energies, isotopes, temperature_k=0.0, flight_path_m=None, delta_t_us=None, delta_l_m=None))]
fn forward_model<'py>(
    py: Python<'py>,
    energies: PyReadonlyArray1<f64>,
    isotopes: Vec<(PyResonanceData, f64)>,
    temperature_k: f64,
    flight_path_m: Option<f64>,
    delta_t_us: Option<f64>,
    delta_l_m: Option<f64>,
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

    let instrument = match (flight_path_m, delta_t_us, delta_l_m) {
        (Some(fp), Some(dt), Some(dl)) => Some(InstrumentParams {
            resolution: ResolutionParams {
                flight_path_m: fp,
                delta_t_us: dt,
                delta_l_m: dl,
            },
        }),
        _ => None,
    };

    let t = transmission::forward_model(e, &sample, instrument.as_ref());
    Ok(PyArray1::from_vec(py, t))
}

/// Fit a measured transmission spectrum to recover isotopic areal densities.
///
/// Args:
///     measured_t: Measured transmission (1D numpy array).
///     sigma: Measurement uncertainties (1D numpy array).
///     energies: Energy grid in eV (1D numpy array).
///     isotopes: List of ResonanceData objects.
///     temperature_k: Sample temperature in Kelvin (default 0.0).
///     initial_densities: Initial guess for areal densities (optional).
///     max_iter: Maximum LM iterations (default 100).
///
/// Returns:
///     FitResult with densities, uncertainties, and fit quality.
#[pyfunction]
#[pyo3(signature = (measured_t, sigma, energies, isotopes, temperature_k=0.0, initial_densities=None, max_iter=100))]
fn fit_spectrum(
    measured_t: PyReadonlyArray1<f64>,
    sigma: PyReadonlyArray1<f64>,
    energies: PyReadonlyArray1<f64>,
    isotopes: Vec<PyResonanceData>,
    temperature_k: f64,
    initial_densities: Option<Vec<f64>>,
    max_iter: usize,
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

    let model = TransmissionFitModel {
        energies: e.to_vec(),
        resonance_data: res_data,
        temperature_k,
        instrument: None,
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
///     library: ENDF library name. One of "endf8.0" (default), "endf8.1",
///              "jeff3.3", "jendl5".
///     mat: ENDF MAT (material) number. If None, looks up from built-in table
///          (~40 common isotopes). Provide explicitly for uncommon isotopes.
///
/// Returns:
///     ResonanceData parsed from the ENDF file.
#[pyfunction]
#[pyo3(signature = (z, a, library="endf8.0", mat=None))]
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
///
/// Returns:
///     ResonanceData object.
#[pyfunction]
#[pyo3(signature = (z, a, awr, scattering_radius, resonances, target_spin=0.0, l_groups=None))]
fn create_resonance_data(
    z: u32,
    a: u32,
    awr: f64,
    scattering_radius: f64,
    resonances: Vec<(f64, f64, f64, f64)>,
    target_spin: f64,
    l_groups: Option<Vec<(u32, Vec<(f64, f64, f64, f64)>)>>,
) -> PyResonanceData {
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

    PyResonanceData {
        inner: ResonanceData {
            isotope: Isotope::new(z, a),
            za: z * 1000 + a,
            awr,
            ranges: vec![ResonanceRange {
                energy_low: 1e-5,
                energy_high: 1e6,
                resolved: true,
                formalism: ResonanceFormalism::ReichMoore,
                target_spin,
                scattering_radius,
                l_groups: groups,
            }],
        },
    }
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

/// NEREIDS Python module.
#[pymodule]
fn nereids(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyResonanceData>()?;
    m.add_class::<PyFitResult>()?;
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
    Ok(())
}
