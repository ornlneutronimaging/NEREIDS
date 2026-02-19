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
use nereids_endf::resonance::{LGroup, Resonance, ResonanceData, ResonanceFormalism, ResonanceRange};
use nereids_fitting::lm::{self, LmConfig};
use nereids_fitting::parameters::{FitParameter, ParameterSet};
use nereids_fitting::transmission_model::TransmissionFitModel;
use nereids_physics::resolution::ResolutionParams;
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

    let n_isotopes = isotopes.len();
    let res_data: Vec<ResonanceData> = isotopes.into_iter().map(|d| d.inner).collect();

    let init = initial_densities.unwrap_or_else(|| vec![0.001; n_isotopes]);

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

/// Create ResonanceData from parameters (for testing/custom isotopes).
///
/// Args:
///     z: Atomic number.
///     a: Mass number.
///     awr: Atomic weight ratio.
///     scattering_radius: Scattering radius in fm.
///     resonances: List of (energy_eV, j, gn, gg) tuples.
///
/// Returns:
///     ResonanceData object.
#[pyfunction]
fn create_resonance_data(
    z: u32,
    a: u32,
    awr: f64,
    scattering_radius: f64,
    resonances: Vec<(f64, f64, f64, f64)>,
) -> PyResonanceData {
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
                target_spin: 0.0,
                scattering_radius,
                l_groups: vec![LGroup {
                    l: 0,
                    awr,
                    apl: 0.0,
                    resonances: res,
                }],
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
    m.add_function(wrap_pyfunction!(create_resonance_data, m)?)?;
    m.add_function(wrap_pyfunction!(beer_lambert, m)?)?;
    Ok(())
}
