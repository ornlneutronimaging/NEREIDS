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
use nereids_endf::retrieval::{EndfLibrary, EndfRetrievalError, EndfRetriever, mat_number};
use nereids_fitting::lm::FlatMatrix;
use nereids_io::tof::BeamlineParams;
use nereids_physics::doppler::{self, DopplerParams};
use nereids_physics::ikeda_carpenter::{
    EnergyLaw, IkedaCarpenter, IkedaCarpenterParams, SynthesisGrid,
};
use nereids_physics::resolution::{
    self, ResolutionFunction, ResolutionParams, TabulatedResolution,
};
use nereids_physics::transmission::{self, InstrumentParams, SampleParams};
use nereids_pipeline::counts_fit::{
    CountsFit, Material, Measurement, NO_TIME_BACKGROUND, Region, RegionFit, Value,
};
use nereids_pipeline::counts_map::{CountsMap, MapMeasurement, Patch};
use nereids_pipeline::error::PipelineError;
use nereids_pipeline::open_beam::{Calibration, Pulse};

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
        // Discrete-resonance count over evaluable LRF=1/2/3 ranges; skipped
        // LRF=7 / LRU=2 placeholder ranges contribute 0.
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

    /// Number of discrete resonances (across all evaluable ranges).
    ///
    /// Delegates to [`ResonanceData::total_resonance_count`], which counts the
    /// L-grouped resonances of evaluable LRF=1/2/3 ranges. LRF=7 (and LRU=2)
    /// ranges are parsed-and-skipped — NEREIDS does not evaluate them — so they
    /// contribute 0.
    #[getter]
    fn n_resonances(&self) -> usize {
        self.inner.total_resonance_count()
    }

    /// Total resonance count across all ranges.
    ///
    /// Explicit alias for [`Self::n_resonances`]; both delegate to the Rust
    /// `ResonanceData::total_resonance_count()`, which counts LRF=1/2/3
    /// L-group resonances only (skipped LRF=7 / LRU=2 ranges contribute 0).
    #[getter]
    fn total_resonance_count(&self) -> usize {
        self.inner.total_resonance_count()
    }

    /// Whether the evaluation carries parsed-but-not-evaluated ranges.
    ///
    /// `True` when any range is a parse-and-skip placeholder (LRF=7
    /// R-Matrix Limited, LRU=2 URR, or LRU=0 scattering-radius-only). Those
    /// spans contribute zero
    /// cross-section — a `UserWarning` listing them is emitted at load time
    /// by `load_endf` / `load_endf_file`. Files with NO evaluable range at
    /// all fail to load instead (ValueError).
    #[getter]
    fn has_unevaluated_ranges(&self) -> bool {
        self.inner.has_unevaluated_ranges()
    }

    /// One-line descriptions of the parse-and-skip placeholder ranges.
    ///
    /// One string per non-evaluable range (LRF=7 R-Matrix Limited, LRU=2 URR,
    /// or LRU=0 scattering-radius-only), formatted by the shared Rust
    /// `ResonanceRange::skip_description` — the same text used in the load-time
    /// `UserWarning` and the parser's rejection message. Empty when
    /// [`Self::has_unevaluated_ranges`] is `False`.
    #[getter]
    fn skipped_ranges(&self) -> Vec<String> {
        self.inner
            .unevaluated_ranges()
            .iter()
            .map(|r| r.skip_description())
            .collect()
    }

    /// Target spin (I) of the first evaluable resonance range.
    ///
    /// Parse-and-skip placeholders (LRU=0 / LRF=7 / URR) may precede the
    /// evaluable ranges on mixed tapes; their SPI describes a span the
    /// cross-section code never evaluates, so the getter skips them (falling
    /// back to the first range only if nothing is evaluable).
    #[getter]
    fn target_spin(&self) -> f64 {
        self.inner
            .ranges
            .iter()
            .find(|r| r.is_evaluable())
            .or_else(|| self.inner.ranges.first())
            .map(|r| r.target_spin)
            .unwrap_or(0.0)
    }

    /// Effective scattering radius (fm) of the first evaluable range.
    ///
    /// Returns the global AP from the first evaluable range (skipping
    /// parse-and-skip placeholders, as for `target_spin`). If AP=0 (common in
    /// ENDF Reich-Moore data that uses energy-dependent radii), falls back
    /// to the first L-group's channel radius APL.
    #[getter]
    fn scattering_radius(&self) -> f64 {
        self.inner
            .ranges
            .iter()
            .find(|r| r.is_evaluable())
            .or_else(|| self.inner.ranges.first())
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

fn load_and_parse_endf(
    isotope: &Isotope,
    lib: EndfLibrary,
    mat_num: u32,
    prefix: &str,
) -> PyResult<ResonanceData> {
    let retriever = EndfRetriever::new();
    let (_path, contents) =
        retriever
            .get_endf_file(isotope, lib, mat_num)
            .map_err(|e| match e {
                EndfRetrievalError::Io(io) => {
                    let path = retriever.cache_file_path(isotope, lib);
                    let message = format!("{prefix}{}: {io}", path.display());
                    PyErr::from(std::io::Error::new(io.kind(), message))
                }
                e if e.is_unavailable() => {
                    pyo3::exceptions::PyConnectionError::new_err(format!("{prefix}{e}"))
                }
                e => pyo3::exceptions::PyRuntimeError::new_err(format!("{prefix}{e}")),
            })?;
    let data = parse_endf_file2(&contents).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("{prefix}ENDF parse error: {e}"))
    })?;
    if data.isotope != *isotope {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{prefix}ENDF data mismatch: requested Z={} A={} but file contains Z={} A={}",
            isotope.z(),
            isotope.a(),
            data.isotope.z(),
            data.isotope.a()
        )));
    }
    Ok(data)
}

/// Emit a Python ``UserWarning`` when parsed ENDF data contains
/// parse-and-skip placeholder ranges (LRF=7 R-Matrix Limited, LRU=2 URR, or
/// LRU=0 scattering-radius-only).
///
/// Those spans contribute exactly zero cross-section, so any physics computed
/// over them reflects only the evaluation's other ranges. The parser already
/// hard-errors when NO range is evaluable; this warning covers the mixed case
/// (e.g. Ta-181, U-238), where the file loads but part of its declared energy
/// coverage is silently inert without this signal. Callers can catch, filter,
/// or escalate it with the stdlib ``warnings`` module.
///
/// ``stacklevel=1`` attributes the warning to the frame that invoked the
/// pyo3 function — extension functions execute inside the caller's Python
/// frame, so level 1 is already the Python call site (same rationale as
/// `emit_tiff_load_warnings`).
fn warn_unevaluated_ranges(py: Python<'_>, data: &ResonanceData, origin: &str) -> PyResult<()> {
    const WARN_STACKLEVEL: i32 = 1;
    let skipped = data.unevaluated_ranges();
    if skipped.is_empty() {
        return Ok(());
    }
    let spans: Vec<String> = skipped.iter().map(|r| r.skip_description()).collect();
    let msg = format!(
        "{origin}: {} of {} resonance range(s) are parsed but NOT evaluated: {}. \
         These spans contribute zero cross-section (transmission = 1); NEREIDS \
         evaluates resolved LRF=1/2/3 ranges only.",
        skipped.len(),
        data.ranges.len(),
        spans.join("; ")
    );
    let c_msg = std::ffi::CString::new(msg).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "internal ENDF-load warning message contained a NUL byte: {e}"
        ))
    })?;
    let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
    PyErr::warn(py, category.as_any(), &c_msg, WARN_STACKLEVEL)
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
    ///
    /// Raises:
    ///     ConnectionError, OSError, RuntimeError, ValueError: as ``load_endf``;
    ///         an error loading a member starts with that member's Z and A.
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
        let results: Vec<PyResult<ResonanceData>> = py.detach(move || {
            members
                .iter()
                .map(|(iso, mat)| {
                    let prefix = format!("Z={} A={}: ", iso.z(), iso.a());
                    load_and_parse_endf(iso, lib, *mat, &prefix)
                })
                .collect()
        });

        // Stage all results first — if any failed, return error without
        // modifying self.resonance_data (atomic update).
        let staged: Vec<Arc<ResonanceData>> = results
            .into_iter()
            .map(|result| result.map(Arc::new))
            .collect::<PyResult<Vec<_>>>()?;
        // Warn about skipped ranges BEFORE mutating self: a caller that
        // escalates warnings to exceptions (warnings.simplefilter("error"))
        // must not observe a partially-updated group.
        for (i, data) in staged.iter().enumerate() {
            let member = &self.inner.members()[i];
            warn_unevaluated_ranges(
                py,
                data,
                &format!("ENDF Z={} A={}", member.0.z(), member.0.a()),
            )?;
        }
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
/// [`IkedaCarpenter.as_tabulated`] anywhere a loaded resolution file is
/// accepted (`forward_model`). Note that
/// `precompute_cross_sections` does NOT take a resolution — broadening is applied
/// after Beer–Lambert, not on the cross-sections. The synthesized kernel rides
/// the *same* broadening path as a Monte-Carlo file, so the IC-vs-tabulated
/// comparison differs only in kernel source.
///
/// Parameters: `alpha`/`r` are [`EnergyLaw`]s (fixed-or-fit general case).
/// `beta` keeps the original constant-rate API; the optional trailing
/// `beta_law` overrides it with an energy-dependent rate. Optional
/// `burst_sigma_us` (Gaussian) and `channel_fwhm_us` (triangle) fold in the
/// proton-burst and chopper terms; detector-time bin probabilities refuse a
/// burst.
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
    /// ascending, 0 = pulse start, weights peak-normalized.
    ///
    /// Raises ``ValueError`` when the τ-grid cannot resolve the prompt core
    /// and requested folds within the sample cap at this energy (construction
    /// validates the reference energies; a probe energy outside their range
    /// can still be unresolvable), or when a parameter law is invalid at this
    /// probe energy (non-positive or singular rate, storage fraction outside
    /// [0, 1]).
    fn kernel_at(&self, energy_ev: f64) -> PyResult<(Vec<f64>, Vec<f64>)> {
        self.inner
            .kernel_at(energy_ev)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// `(moderator_delay_us, density)` for the physical source pulse at one
    /// true energy. Unlike `kernel_at`, this keeps the pulse's time origin.
    /// The density is a peak-normalized shape (maximum 1), not a probability
    /// density. Offsets are from the moderator origin; a symmetric burst or
    /// channel fold reaches before it, so leading delays can be negative.
    /// `kernel_at` returns the same synthesis on the same origin.
    fn source_pulse_at(&self, true_energy_ev: f64) -> PyResult<(Vec<f64>, Vec<f64>)> {
        self.inner
            .source_pulse_at(true_energy_ev)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Probability that a neutron at one true energy is recorded in each
    /// adjacent detector-time bin. The result is not renormalized when the
    /// supplied time window omits part of the pulse. A Gaussian burst is
    /// refused.
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
    // the SLBW / Reich-Moore leaf paths.  Sibling PyO3 entries
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
///     delta_t_us: Timing width in microseconds, W-parameter (optional).
///     delta_l_m: Path length width in meters, W-parameter (optional).
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
    // added to the SLBW / Reich-Moore leaf paths.  Empty grids are accepted
    // (the downstream `transmission::forward_model` returns an empty vector
    // for an empty grid).
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

/// Load ENDF resonance data for an isotope, downloading the evaluation on
/// first use and caching it in the user's cache directory.
///
/// Args:
///     z: Atomic number (e.g. 92 for uranium).
///     a: Mass number (e.g. 238).
///     library: ENDF library name. One of "endf8.0", "endf8.1" (default),
///              "jeff3.3", "jendl5", "tendl2023", "cendl3.2".
///     mat: ENDF MAT (material) number. If None, looked up from the built-in
///          table; required for an isotope the table lacks.
///
/// Returns:
///     ResonanceData parsed from the ENDF file.
///
/// Raises:
///     ConnectionError: no download site could be reached, or one answered
///         with an HTTP error other than 404 or refused the download.
///     OSError: the cache file named in the message could not be read or
///         written.
///     RuntimeError: the isotope is absent from the library (HTTP 404), or the
///         download is not a valid archive.
///     ValueError: an unknown library, an invalid isotope or MAT, a file for
///         another isotope, or ENDF text that does not parse or has no
///         evaluable resolved range.
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
    let data = py.detach(move || load_and_parse_endf(&isotope, lib, mat_num, ""))?;

    warn_unevaluated_ranges(py, &data, &format!("ENDF Z={z} A={a}"))?;

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

    warn_unevaluated_ranges(py, &data, path)?;

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

    // A resolved range with no resonances at all evaluates to exactly zero
    // cross-section everywhere (transmission = 1) — the same silently-inert
    // shape the ENDF load path rejects. Reject it here too.
    if groups.iter().all(|g| g.resonances.is_empty()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "create_resonance_data: no resonances given (empty resonance list \
             in every L-group); cross-sections would be identically zero \
             everywhere (transmission = 1)",
        ));
    }

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
/// the instrument's timing and flight-path widths. This is the same broadening
/// applied internally by `forward_model()`, but exposed here for independent
/// use on arbitrary arrays.
///
/// Width convention: the kernel is ``exp(-x^2 / W^2)``, so ``delta_t_us`` and
/// ``delta_l_m`` are W-parameters — ``sigma = W / sqrt(2)`` and
/// ``FWHM = 2*sqrt(ln 2)*W``. This is SAMMY's convention and matches the
/// Doppler width. Passing a measured 1-sigma value directly gives a kernel
/// sqrt(2) too narrow; multiply it by ``sqrt(2)`` first.
///
/// Args:
///     energies: Energy grid in eV (1D numpy array, sorted ascending).
///     cross_sections: Values to broaden (1D numpy array, same length).
///     flight_path_m: Flight path length in meters (source to detector).
///     delta_t_us: Total timing width in microseconds, as a W-parameter
///         (sigma = W/sqrt(2)), NOT a standard deviation.
///     delta_l_m: Flight path width in meters, as a W-parameter
///         (sigma = W/sqrt(2)), NOT a standard deviation.
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

/// Convert a standard deviation to the W-parameter this API expects.
///
/// ``delta_t_us`` and ``delta_l_m`` are W-parameters — the width in
/// ``exp(-x^2/W^2)`` — while instrument metrology is usually quoted as a
/// 1-sigma jitter. This applies ``W = sigma * sqrt(2)``.
///
/// Args:
///     sigma: A standard deviation, in any unit.
///
/// Returns:
///     The same width expressed as a W-parameter, in the same unit.
///
/// Example:
///     >>> nereids.width_from_sigma(0.5)  # a measured 0.5 us 1-sigma jitter
///     0.7071067811865476
#[pyfunction]
fn width_from_sigma(sigma: f64) -> f64 {
    sigma * std::f64::consts::SQRT_2
}

/// Convert a full width at half maximum to the W-parameter this API expects.
///
/// Applies ``W = FWHM / (2*sqrt(ln 2))``. This is the measure SAMMY's
/// ``Deltag`` uses, so a timing width taken from a SAMMY ``.inp`` can be passed
/// through this function directly.
///
/// Args:
///     fwhm: A full width at half maximum, in any unit.
///
/// Returns:
///     The same width expressed as a W-parameter, in the same unit.
#[pyfunction]
fn width_from_fwhm(fwhm: f64) -> f64 {
    fwhm / (2.0 * 2.0_f64.ln().sqrt())
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
///     delta_t_us: Timing width in microseconds, W-parameter (optional).
///     delta_l_m: Path length width in meters, W-parameter (optional).
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

    // resolution-broadened cross-sections are not physically
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
/// axis are exactly zero.  The returned mask can be passed as the
/// ``excluded`` mask of a ``MapMeasurement``.
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
    let mask = py.detach(move || nereids_io::pixels::detect_dead_pixels(&arr));
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
#[pyo3(signature = (data, k_mad = nereids_io::pixels::HOT_PIXEL_K_MAD))]
fn detect_hot_pixels<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<f64>,
    k_mad: f64,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let arr = data.as_array().to_owned();
    let mask = py.detach(move || nereids_io::pixels::detect_hot_pixels(&arr, k_mad));
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
    let mask = py.detach(move || nereids_io::pixels::detect_dead_pixels_chunked(&owned));
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
#[pyo3(signature = (sample, open_beam = None, hot_k_mad = Some(nereids_io::pixels::HOT_PIXEL_K_MAD)))]
fn detect_bad_pixels<'py>(
    py: Python<'py>,
    sample: PyReadonlyArray3<f64>,
    open_beam: Option<PyReadonlyArray3<f64>>,
    hot_k_mad: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let s = sample.as_array().to_owned();
    let ob = open_beam.map(|o| o.as_array().to_owned());
    let mask = py.detach(move || nereids_io::pixels::detect_bad_pixels(&s, ob.as_ref(), hot_k_mad));
    let mask = mask.map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    Ok(PyArray2::from_owned_array(py, mask))
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
/// By default this function refuses multi-angle files
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

/// Read a DASlogs PV from a NeXus file as a transition log.
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
/// `min_value <= value <= max_value`.
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

/// Intersect two interval lists.
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
/// optionally keeping only pulses inside wall-clock intervals.
///
/// Reads `/entry/<bank>/{event_time_offset, event_index, event_time_zero}`
/// (the facility NeXus convention).  The `units` attributes on
/// `event_time_offset` AND `event_time_zero` are required — NXevent_data
/// specifies no defaults, and refusing to guess closes the
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

// ── Counts-domain fitting (Engine C) ─────────────────────────────────────

/// Map a pipeline error to a Python ValueError.
fn counts_error(error: PipelineError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(error.to_string())
}

/// A [C, A, tau] triple of PyValues.
fn value_trio(values: Vec<PyValue>) -> PyResult<[Value; 3]> {
    let [a, b, c] = values.as_slice() else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "expected exactly three values",
        ));
    };
    Ok([a.inner, b.inner, c.inner])
}

/// A covariance matrix as a 2D numpy array.
fn matrix_to_py<'py>(py: Python<'py>, matrix: &FlatMatrix) -> Bound<'py, PyArray2<f64>> {
    PyArray2::from_owned_array(
        py,
        ndarray::Array2::from_shape_fn((matrix.nrows, matrix.ncols), |(row, column)| {
            matrix.get(row, column)
        }),
    )
}

/// A quantity the fit holds at a known value, fits from a starting value over
/// the quantity's range, fits from start within lower..=upper, or fits under a
/// Gaussian prior N(value, sd^2).
#[pyclass(name = "Value", from_py_object)]
#[derive(Clone, Copy)]
struct PyValue {
    inner: Value,
}

#[pymethods]
impl PyValue {
    /// Hold the quantity at value.
    #[staticmethod]
    fn known(value: f64) -> Self {
        Self {
            inner: Value::Known(value),
        }
    }

    /// Fit the quantity from start over its full range.
    #[staticmethod]
    fn fitted(start: f64) -> Self {
        Self {
            inner: Value::Fitted(start),
        }
    }

    /// Fit the quantity from start within lower..=upper.
    #[staticmethod]
    fn within(start: f64, lower: f64, upper: f64) -> Self {
        Self {
            inner: Value::Within {
                start,
                lower,
                upper,
            },
        }
    }

    /// Fit the quantity with the Gaussian prior N(value, sd^2).
    #[staticmethod]
    fn measured(value: f64, sd: f64) -> Self {
        Self {
            inner: Value::Measured { value, sd },
        }
    }

    /// "known" | "fitted" | "within" | "measured".
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            Value::Known(_) => "known",
            Value::Fitted(_) => "fitted",
            Value::Within { .. } => "within",
            Value::Measured { .. } => "measured",
        }
    }

    /// The held, starting or measured value; None for within.
    #[getter]
    fn value(&self) -> Option<f64> {
        match self.inner {
            Value::Known(v) | Value::Fitted(v) | Value::Measured { value: v, .. } => Some(v),
            Value::Within { .. } => None,
        }
    }

    /// The starting value.
    #[getter]
    fn start(&self) -> Option<f64> {
        match self.inner {
            Value::Within { start, .. } => Some(start),
            _ => self.value(),
        }
    }

    /// The lower bound of a within value.
    #[getter]
    fn lower(&self) -> Option<f64> {
        match self.inner {
            Value::Within { lower, .. } => Some(lower),
            _ => None,
        }
    }

    /// The upper bound of a within value.
    #[getter]
    fn upper(&self) -> Option<f64> {
        match self.inner {
            Value::Within { upper, .. } => Some(upper),
            _ => None,
        }
    }

    /// The standard deviation of a measured value.
    #[getter]
    fn sd(&self) -> Option<f64> {
        match self.inner {
            Value::Measured { sd, .. } => Some(sd),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!("Value({:?})", self.inner)
    }
}

/// A material's isotopes, each with an areal density in atoms/barn, and its
/// temperature in K.
#[pyclass(name = "Material", from_py_object)]
#[derive(Clone)]
struct PyMaterial {
    inner: Material,
}

#[pymethods]
impl PyMaterial {
    /// library names the ENDF library the isotopes' resonance data were read
    /// from, e.g. "endf8.1".
    #[new]
    #[pyo3(signature = (isotopes, temperature_k, library="endf8.1"))]
    fn new(
        isotopes: Vec<(PyResonanceData, PyValue)>,
        temperature_k: PyValue,
        library: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Material {
                isotopes: isotopes
                    .into_iter()
                    .map(|(data, density)| ((*data.inner).clone(), density.inner))
                    .collect(),
                temperature_k: temperature_k.inner,
                library: parse_library_name(library)?,
                covariances: Vec::new(),
            },
        })
    }

    /// Each isotope with its areal density in atoms/barn.
    #[getter]
    fn isotopes(&self) -> Vec<(PyResonanceData, PyValue)> {
        self.inner
            .isotopes
            .iter()
            .map(|(data, density)| {
                (
                    PyResonanceData {
                        inner: Arc::new(data.clone()),
                    },
                    PyValue { inner: *density },
                )
            })
            .collect()
    }

    /// The temperature in K.
    #[getter]
    fn temperature_k(&self) -> PyValue {
        PyValue {
            inner: self.inner.temperature_k,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Material(isotopes={}, temperature_k={:?})",
            self.inner.isotopes.len(),
            self.inner.temperature_k
        )
    }
}

/// The counts of a set of pixels in the open-beam and sample runs, its
/// background, and the material the beam crosses there.
#[pyclass(name = "Region", from_py_object)]
#[derive(Clone)]
struct PyRegion {
    inner: Region,
}

#[pymethods]
impl PyRegion {
    /// background is [b0, b1, b2] of b(E) = b0 + b1/sqrt(E) + b2*sqrt(E);
    /// pixels is the region's area in pixels, finite and positive.
    #[new]
    #[pyo3(signature = (open_counts, sample_counts, pixels, background, gamma_transmission, material=None, open_live=None, sample_live=None))]
    fn new(
        open_counts: Vec<f64>,
        sample_counts: Vec<f64>,
        pixels: f64,
        background: Vec<PyValue>,
        gamma_transmission: PyValue,
        material: Option<PyMaterial>,
        open_live: Option<Vec<f64>>,
        sample_live: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Region {
                open_counts,
                sample_counts,
                open_live,
                sample_live,
                background: value_trio(background)?,
                pixels,
                gamma_transmission: gamma_transmission.inner,
                material: material.map(|material| material.inner),
            },
        })
    }

    /// Raw counts of the open-beam run, one per bin.
    #[getter]
    fn open_counts(&self) -> Vec<f64> {
        self.inner.open_counts.clone()
    }

    /// Raw counts of the sample run, one per bin.
    #[getter]
    fn sample_counts(&self) -> Vec<f64> {
        self.inner.sample_counts.clone()
    }

    /// The open-beam run's live fractions, in (0, 1], or None.
    #[getter]
    fn open_live(&self) -> Option<Vec<f64>> {
        self.inner.open_live.clone()
    }

    /// The sample run's live fractions, in (0, 1], or None.
    #[getter]
    fn sample_live(&self) -> Option<Vec<f64>> {
        self.inner.sample_live.clone()
    }

    /// The region's background terms [b0, b1, b2].
    #[getter]
    fn background(&self) -> Vec<PyValue> {
        self.inner
            .background
            .iter()
            .map(|value| PyValue { inner: *value })
            .collect()
    }

    /// The region's area in pixels.
    #[getter]
    fn pixels(&self) -> f64 {
        self.inner.pixels
    }

    /// The fraction of the time background that reaches the region's pixels in
    /// the sample run, within 0-1.
    #[getter]
    fn gamma_transmission(&self) -> PyValue {
        PyValue {
            inner: self.inner.gamma_transmission,
        }
    }

    /// The material the beam crosses, or None.
    #[getter]
    fn material(&self) -> Option<PyMaterial> {
        self.inner
            .material
            .clone()
            .map(|inner| PyMaterial { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "Region(pixels={}, bins={}, material={})",
            self.inner.pixels,
            self.inner.open_counts.len(),
            self.inner.material.is_some()
        )
    }
}

/// An open-beam run and a sample run recorded in the same time bins, read in
/// one or more regions of the detector.
#[pyclass(name = "Measurement", from_py_object)]
#[derive(Clone)]
struct PyMeasurement {
    inner: Measurement,
}

#[pymethods]
impl PyMeasurement {
    /// time_edges_us are the time-bin edges in microseconds; charge_ratio is
    /// the sample run's proton charge over the open-beam run's.
    #[new]
    #[pyo3(signature = (time_edges_us, charge_ratio, normalization, regions, time_background=None))]
    fn new(
        time_edges_us: Vec<f64>,
        charge_ratio: f64,
        normalization: PyValue,
        regions: Vec<PyRegion>,
        time_background: Option<Vec<PyValue>>,
    ) -> PyResult<Self> {
        let time_background = match time_background {
            Some(values) => value_trio(values)?,
            None => NO_TIME_BACKGROUND,
        };
        Ok(Self {
            inner: Measurement {
                time_edges_us,
                charge_ratio,
                normalization: normalization.inner,
                time_background,
                regions: regions.into_iter().map(|region| region.inner).collect(),
            },
        })
    }

    /// Time-bin edges in microseconds.
    #[getter]
    fn time_edges_us(&self) -> Vec<f64> {
        self.inner.time_edges_us.clone()
    }

    /// The sample run's proton charge over the open-beam run's.
    #[getter]
    fn charge_ratio(&self) -> f64 {
        self.inner.charge_ratio
    }

    /// The sample run's normalization a.
    #[getter]
    fn normalization(&self) -> PyValue {
        PyValue {
            inner: self.inner.normalization,
        }
    }

    /// The time background's [C, A, tau].
    #[getter]
    fn time_background(&self) -> Vec<PyValue> {
        self.inner
            .time_background
            .iter()
            .map(|value| PyValue { inner: *value })
            .collect()
    }

    /// The regions, in the order given.
    #[getter]
    fn regions(&self) -> Vec<PyRegion> {
        self.inner
            .regions
            .iter()
            .map(|inner| PyRegion {
                inner: inner.clone(),
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Measurement(bins={}, regions={})",
            self.inner.time_edges_us.len().saturating_sub(1),
            self.inner.regions.len()
        )
    }
}

/// The Ikeda-Carpenter pulse with alpha = alpha0*sqrt(E) + alpha1 and
/// beta = beta0*sqrt(E) + beta1 in 1/microsecond, E in eV, a storage fraction
/// R constant over energy_span_ev, and the proton pulse's triangle of FWHM h
/// in microseconds, given by h^2.
#[pyclass(name = "Pulse", from_py_object)]
#[derive(Clone)]
struct PyPulse {
    inner: Pulse,
}

#[pymethods]
impl PyPulse {
    /// energy_span_ev is (low, high) in eV; n_tau is the number of samples
    /// across the prompt core (at least 8).  line_span_ev gives the (low, high)
    /// energies in eV of a calibration foil's resonances, or None to check
    /// nothing.
    #[new]
    #[pyo3(signature = (alpha0, alpha1, beta0, beta1, r, fwhm_squared_us2, energy_span_ev, n_tau=600, line_span_ev=None))]
    fn new(
        alpha0: PyValue,
        alpha1: PyValue,
        beta0: PyValue,
        beta1: PyValue,
        r: PyValue,
        fwhm_squared_us2: PyValue,
        energy_span_ev: (f64, f64),
        n_tau: usize,
        line_span_ev: Option<(f64, f64)>,
    ) -> Self {
        Self {
            inner: Pulse {
                alpha: [alpha0.inner, alpha1.inner],
                beta: [beta0.inner, beta1.inner],
                r: r.inner,
                fwhm_squared_us2: fwhm_squared_us2.inner,
                energy_span_ev,
                n_tau,
                line_span_ev,
                prior: None,
            },
        }
    }

    /// [alpha0, alpha1], in 1/(microsecond*sqrt(eV)) and 1/microsecond.
    #[getter]
    fn alpha(&self) -> Vec<PyValue> {
        self.inner
            .alpha
            .iter()
            .map(|value| PyValue { inner: *value })
            .collect()
    }

    /// [beta0, beta1], in 1/(microsecond*sqrt(eV)) and 1/microsecond.
    #[getter]
    fn beta(&self) -> Vec<PyValue> {
        self.inner
            .beta
            .iter()
            .map(|value| PyValue { inner: *value })
            .collect()
    }

    /// The storage fraction R.
    #[getter]
    fn r(&self) -> PyValue {
        PyValue {
            inner: self.inner.r,
        }
    }

    /// h^2 in microseconds squared.
    #[getter]
    fn fwhm_squared_us2(&self) -> PyValue {
        PyValue {
            inner: self.inner.fwhm_squared_us2,
        }
    }

    /// The (low, high) energies in eV the laws hold over.
    #[getter]
    fn energy_span_ev(&self) -> (f64, f64) {
        self.inner.energy_span_ev
    }

    /// Samples across the prompt core.
    #[getter]
    fn n_tau(&self) -> usize {
        self.inner.n_tau
    }

    fn __repr__(&self) -> String {
        format!(
            "Pulse(energy_span_ev={:?}, n_tau={})",
            self.inner.energy_span_ev, self.inner.n_tau
        )
    }
}

/// A neutron of flight time u over flight_path_m (m) arrives at t0_us + u
/// (microseconds) plus a delay drawn from pulse.
#[pyclass(name = "Calibration", from_py_object)]
#[derive(Clone)]
struct PyCalibration {
    inner: Calibration,
}

#[pymethods]
impl PyCalibration {
    #[new]
    fn new(t0_us: PyValue, flight_path_m: PyValue, pulse: PyPulse) -> Self {
        Self {
            inner: Calibration {
                t0_us: t0_us.inner,
                flight_path_m: flight_path_m.inner,
                pulse: pulse.inner,
            },
        }
    }

    /// The timing offset t0 in microseconds.
    #[getter]
    fn t0_us(&self) -> PyValue {
        PyValue {
            inner: self.inner.t0_us,
        }
    }

    /// The flight path in m.
    #[getter]
    fn flight_path_m(&self) -> PyValue {
        PyValue {
            inner: self.inner.flight_path_m,
        }
    }

    /// The pulse.
    #[getter]
    fn pulse(&self) -> PyPulse {
        PyPulse {
            inner: self.inner.pulse.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Calibration(t0_us={:?}, flight_path_m={:?})",
            self.inner.t0_us, self.inner.flight_path_m
        )
    }
}

/// One region's fitted quantities and counts.
#[pyclass(name = "RegionFit", from_py_object)]
#[derive(Clone)]
struct PyRegionFit {
    densities: Vec<f64>,
    temperature_k: Option<f64>,
    background: [f64; 3],
    gamma_transmission: f64,
    beam_at_limit: bool,
    predicted_open: Vec<f64>,
    predicted_sample: Vec<f64>,
    overdispersion: [Option<f64>; 2],
    excluded_bins: Vec<usize>,
}

impl PyRegionFit {
    fn wrap(region: &RegionFit) -> Self {
        Self {
            densities: region.densities.clone(),
            temperature_k: region.temperature_k,
            background: region.background,
            gamma_transmission: region.gamma_transmission,
            beam_at_limit: region.beam_at_limit,
            predicted_open: region.predicted[0].clone(),
            predicted_sample: region.predicted[1].clone(),
            overdispersion: region.overdispersion,
            excluded_bins: region.excluded_bins.clone(),
        }
    }
}

#[pymethods]
impl PyRegionFit {
    /// Areal density of each isotope of the region's material in atoms/barn.
    #[getter]
    fn densities(&self) -> Vec<f64> {
        self.densities.clone()
    }

    /// The material's temperature in K, or None without a material.
    #[getter]
    fn temperature_k(&self) -> Option<f64> {
        self.temperature_k
    }

    /// [b0, b1, b2].
    #[getter]
    fn background(&self) -> [f64; 3] {
        self.background
    }

    /// The gamma transmission.
    #[getter]
    fn gamma_transmission(&self) -> f64 {
        self.gamma_transmission
    }

    /// Whether the region's open-beam fit chose its richest beam.
    #[getter]
    fn beam_at_limit(&self) -> bool {
        self.beam_at_limit
    }

    /// The counts the fit predicts in the open-beam run, one per bin.
    #[getter]
    fn predicted_open<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.predicted_open.clone())
    }

    /// The counts the fit predicts in the sample run, one per bin.
    #[getter]
    fn predicted_sample<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.predicted_sample.clone())
    }

    /// [open-beam, sample] overdispersion, each None when unmeasured.
    #[getter]
    fn overdispersion(&self) -> Vec<Option<f64>> {
        self.overdispersion.to_vec()
    }

    /// Indices of the sample run's bins the material's library leaves out.
    #[getter]
    fn excluded_bins(&self) -> Vec<usize> {
        self.excluded_bins.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RegionFit(densities={:?}, temperature_k={:?})",
            self.densities, self.temperature_k
        )
    }
}

/// The fitted quantities of every region, and the normalization, time
/// background, timing offset, flight path and pulse numbers the regions share.
#[pyclass(name = "CountsFit", from_py_object)]
#[derive(Clone)]
struct PyCountsFit {
    inner: CountsFit,
}

#[pymethods]
impl PyCountsFit {
    /// Each region's fitted quantities and counts, in the order given.
    #[getter]
    fn regions(&self) -> Vec<PyRegionFit> {
        self.inner.regions.iter().map(PyRegionFit::wrap).collect()
    }

    /// The normalization a.
    #[getter]
    fn normalization(&self) -> f64 {
        self.inner.normalization
    }

    /// The timing offset t0 in microseconds.
    #[getter]
    fn t0_us(&self) -> f64 {
        self.inner.t0_us
    }

    /// The flight path in m.
    #[getter]
    fn flight_path_m(&self) -> f64 {
        self.inner.flight_path_m
    }

    /// The pulse's [alpha0, alpha1].
    #[getter]
    fn alpha(&self) -> [f64; 2] {
        self.inner.alpha
    }

    /// The pulse's [beta0, beta1].
    #[getter]
    fn beta(&self) -> [f64; 2] {
        self.inner.beta
    }

    /// The pulse's R.
    #[getter]
    fn r(&self) -> f64 {
        self.inner.r
    }

    /// The square of the pulse's triangle FWHM, h^2 in microseconds squared.
    #[getter]
    fn fwhm_squared_us2(&self) -> f64 {
        self.inner.fwhm_squared_us2
    }

    /// The time background's [C, A, tau].
    #[getter]
    fn time_background(&self) -> [f64; 3] {
        self.inner.time_background
    }

    /// Covariance of the fitted quantities, or None when the fit did not
    /// converge.
    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner
            .covariance
            .as_ref()
            .map(|covariance| matrix_to_py(py, covariance))
    }

    /// The mean of the fitted quantities' Gaussian without their bounds.
    #[getter]
    fn unbounded_mean(&self) -> Option<Vec<f64>> {
        self.inner
            .unbounded
            .as_ref()
            .map(|unbounded| unbounded.mean.clone())
    }

    /// The covariance of the fitted quantities without their bounds.
    #[getter]
    fn unbounded_covariance<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner
            .unbounded
            .as_ref()
            .map(|unbounded| matrix_to_py(py, &unbounded.covariance))
    }

    /// Whether each fitted quantity, in the covariance's order, ended on one
    /// of its bounds.
    #[getter]
    fn on_bound(&self) -> Vec<bool> {
        self.inner.on_bound.clone()
    }

    /// Each run's half Poisson deviance over its overdispersion, summed over
    /// both runs of every region.
    #[getter]
    fn deviance(&self) -> f64 {
        self.inner.deviance
    }

    /// Whether the fit converged.
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    /// For each measured quantity, its pull; None when the fitter reports none.
    #[getter]
    fn measured_pulls(&self) -> Option<Vec<f64>> {
        self.inner.measured_pulls.clone()
    }

    /// The pulse calibration's consistency test as (q, dof, p), or None.
    #[getter]
    fn pulse_consistency(&self) -> Option<(f64, usize, f64)> {
        self.inner
            .pulse_consistency
            .map(|consistency| (consistency.q, consistency.dof, consistency.p))
    }

    /// Step, in microseconds, of the fit's grid.
    #[getter]
    fn step_us(&self) -> f64 {
        self.inner.step_us
    }

    /// Number of points of that grid.
    #[getter]
    fn points(&self) -> usize {
        self.inner.points
    }

    /// How many times the grid was halved to reach its final step.
    #[getter]
    fn halvings(&self) -> usize {
        self.inner.halvings
    }

    fn __repr__(&self) -> String {
        format!(
            "CountsFit(regions={}, deviance={:.6}, converged={})",
            self.inner.regions.len(),
            self.inner.deviance,
            self.inner.converged
        )
    }
}

/// What the pixels of a patch that are not excluded lie behind.
#[pyclass(name = "Patch", from_py_object)]
#[derive(Clone, Copy)]
struct PyPatch {
    kind: Patch,
}

#[pymethods]
impl PyPatch {
    /// "sample" | "mixed" | "outside".
    #[getter]
    fn kind(&self) -> &'static str {
        match self.kind {
            Patch::Sample => "sample",
            Patch::Mixed => "mixed",
            Patch::Outside => "outside",
        }
    }

    fn __repr__(&self) -> String {
        match self.kind {
            Patch::Sample => "Patch(sample)".to_string(),
            Patch::Mixed => "Patch(mixed)".to_string(),
            Patch::Outside => "Patch(outside)".to_string(),
        }
    }
}

#[derive(Clone)]
struct MapData {
    time_edges_us: Vec<f64>,
    charge_ratio: f64,
    normalization: Value,
    time_background: [Value; 3],
    open_counts: ndarray::Array3<f64>,
    sample_counts: ndarray::Array3<f64>,
    open_live: Option<Vec<f64>>,
    sample_live: Option<Vec<f64>>,
    excluded: ndarray::Array2<bool>,
    sample: ndarray::Array2<bool>,
    empty: ndarray::Array2<bool>,
    binning: usize,
    material: Material,
    background: [Value; 3],
    gamma_transmission: Value,
    empty_background: [Value; 3],
}

impl MapData {
    fn measurement(&self) -> MapMeasurement<'_> {
        MapMeasurement {
            time_edges_us: self.time_edges_us.clone(),
            charge_ratio: self.charge_ratio,
            normalization: self.normalization,
            time_background: self.time_background,
            open_counts: self.open_counts.view(),
            sample_counts: self.sample_counts.view(),
            open_live: self.open_live.clone(),
            sample_live: self.sample_live.clone(),
            excluded: self.excluded.view(),
            sample: self.sample.view(),
            empty: self.empty.view(),
            binning: self.binning,
            material: self.material.clone(),
            background: self.background,
            gamma_transmission: self.gamma_transmission,
            empty_background: self.empty_background,
        }
    }
}

/// The counts of every pixel of an open-beam run and a sample run recorded in
/// the same time bins, which pixels to fit, and the material behind the sample.
#[pyclass(name = "MapMeasurement", from_py_object)]
#[derive(Clone)]
struct PyMapMeasurement {
    inner: MapData,
}

#[pymethods]
impl PyMapMeasurement {
    /// open_counts and sample_counts have shape (time bin, row, column);
    /// excluded, sample and empty have shape (row, column).
    #[new]
    #[pyo3(signature = (time_edges_us, open_counts, sample_counts, excluded, sample, empty, material, charge_ratio, normalization, background, binning=1, empty_background=None, open_live=None, sample_live=None, time_background=None, gamma_transmission=None))]
    fn new(
        time_edges_us: Vec<f64>,
        open_counts: PyReadonlyArray3<f64>,
        sample_counts: PyReadonlyArray3<f64>,
        excluded: PyReadonlyArray2<bool>,
        sample: PyReadonlyArray2<bool>,
        empty: PyReadonlyArray2<bool>,
        material: PyMaterial,
        charge_ratio: f64,
        normalization: PyValue,
        background: Vec<PyValue>,
        binning: usize,
        empty_background: Option<Vec<PyValue>>,
        open_live: Option<Vec<f64>>,
        sample_live: Option<Vec<f64>>,
        time_background: Option<Vec<PyValue>>,
        gamma_transmission: Option<PyValue>,
    ) -> PyResult<Self> {
        let empty_background = match empty_background {
            Some(values) => value_trio(values)?,
            None => [Value::Known(0.0); 3],
        };
        let time_background = match time_background {
            Some(values) => value_trio(values)?,
            None => NO_TIME_BACKGROUND,
        };
        Ok(Self {
            inner: MapData {
                time_edges_us,
                charge_ratio,
                normalization: normalization.inner,
                time_background,
                open_counts: open_counts.as_array().to_owned(),
                sample_counts: sample_counts.as_array().to_owned(),
                open_live,
                sample_live,
                excluded: excluded.as_array().to_owned(),
                sample: sample.as_array().to_owned(),
                empty: empty.as_array().to_owned(),
                binning,
                material: material.inner,
                background: value_trio(background)?,
                gamma_transmission: gamma_transmission
                    .map_or(Value::Known(1.0), |value| value.inner),
                empty_background,
            },
        })
    }

    /// Each patch's kind, in (patch row, patch column).
    fn patches(&self) -> PyResult<Vec<Vec<PyPatch>>> {
        let measurement = self.inner.measurement();
        let patches = measurement.patches().map_err(counts_error)?;
        Ok(patch_rows(&patches))
    }

    /// Time-bin edges in microseconds.
    #[getter]
    fn time_edges_us(&self) -> Vec<f64> {
        self.inner.time_edges_us.clone()
    }

    /// The sample run's proton charge over the open-beam run's.
    #[getter]
    fn charge_ratio(&self) -> f64 {
        self.inner.charge_ratio
    }

    /// The side of a square patch, in pixels.
    #[getter]
    fn binning(&self) -> usize {
        self.inner.binning
    }

    /// The time background's [C, A, tau] per pixel.
    #[getter]
    fn time_background(&self) -> Vec<PyValue> {
        self.inner
            .time_background
            .iter()
            .map(|&inner| PyValue { inner })
            .collect()
    }

    /// The gamma transmission of every sample patch.
    #[getter]
    fn gamma_transmission(&self) -> PyValue {
        PyValue {
            inner: self.inner.gamma_transmission,
        }
    }

    fn __repr__(&self) -> String {
        let (bins, height, width) = self.inner.open_counts.dim();
        format!(
            "MapMeasurement(bins={bins}, pixels={height}x{width}, binning={})",
            self.inner.binning
        )
    }
}

fn patch_rows(patches: &ndarray::Array2<Patch>) -> Vec<Vec<PyPatch>> {
    patches
        .outer_iter()
        .map(|row| row.iter().map(|&kind| PyPatch { kind }).collect())
        .collect()
}

/// The fitted map, one value per patch.
#[pyclass(name = "CountsMap", from_py_object)]
#[derive(Clone)]
struct PyCountsMap {
    patches: ndarray::Array2<Patch>,
    fits: ndarray::Array2<Option<RegionFit>>,
    failed: ndarray::Array2<Option<String>>,
    empty: Option<RegionFit>,
    densities: Vec<ndarray::Array2<f64>>,
    density_sd: Vec<ndarray::Array2<f64>>,
    temperature_k: ndarray::Array2<f64>,
    temperature_sd_k: ndarray::Array2<f64>,
    overdispersion: [ndarray::Array2<f64>; 2],
    trusted: ndarray::Array2<bool>,
    normalization: f64,
    time_background: [f64; 3],
    t0_us: f64,
    flight_path_m: f64,
    alpha: [f64; 2],
    beta: [f64; 2],
    r: f64,
    fwhm_squared_us2: f64,
    shared: Vec<String>,
    shared_covariance: Option<FlatMatrix>,
    measured_pulls: Option<Vec<f64>>,
    pulse_consistency: Option<(f64, usize, f64)>,
    deviance: f64,
    converged: bool,
    unconverged: Option<String>,
    steps: usize,
}

impl PyCountsMap {
    fn wrap(map: &CountsMap<'_>) -> Self {
        Self {
            patches: map.patches.clone(),
            fits: map.fits.clone(),
            failed: map.failed.clone(),
            empty: map.empty.clone(),
            densities: map.densities.clone(),
            density_sd: map.density_sd.clone(),
            temperature_k: map.temperature_k.clone(),
            temperature_sd_k: map.temperature_sd_k.clone(),
            overdispersion: map.overdispersion.clone(),
            trusted: map.trusted.clone(),
            normalization: map.normalization,
            time_background: map.time_background,
            t0_us: map.t0_us,
            flight_path_m: map.flight_path_m,
            alpha: map.alpha,
            beta: map.beta,
            r: map.r,
            fwhm_squared_us2: map.fwhm_squared_us2,
            shared: map.shared.iter().map(|name| name.to_string()).collect(),
            shared_covariance: map.shared_covariance.clone(),
            measured_pulls: map.measured_pulls.clone(),
            pulse_consistency: map
                .pulse_consistency
                .map(|consistency| (consistency.q, consistency.dof, consistency.p)),
            deviance: map.deviance,
            converged: map.converged,
            unconverged: map.unconverged.clone(),
            steps: map.steps,
        }
    }
}

#[pymethods]
impl PyCountsMap {
    /// Each patch's kind, in (patch row, patch column).
    #[getter]
    fn patches(&self) -> Vec<Vec<PyPatch>> {
        patch_rows(&self.patches)
    }

    /// Each fitted patch's fit, in (patch row, patch column).
    #[getter]
    fn fits(&self) -> Vec<Vec<Option<PyRegionFit>>> {
        self.fits
            .outer_iter()
            .map(|row| {
                row.iter()
                    .map(|fit| fit.as_ref().map(PyRegionFit::wrap))
                    .collect()
            })
            .collect()
    }

    /// Why a patch left out of the shared quantities failed, or None.
    #[getter]
    fn failed(&self) -> Vec<Vec<Option<String>>> {
        self.failed
            .outer_iter()
            .map(|row| row.iter().cloned().collect())
            .collect()
    }

    /// The empty pixels' fit, or None.
    #[getter]
    fn empty(&self) -> Option<PyRegionFit> {
        self.empty.as_ref().map(PyRegionFit::wrap)
    }

    /// Each isotope's areal density in atoms/barn, one map per isotope.
    #[getter]
    fn densities<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f64>>> {
        self.densities
            .iter()
            .map(|density| PyArray2::from_owned_array(py, density.clone()))
            .collect()
    }

    /// Standard deviation of each density, one map per isotope.
    #[getter]
    fn density_sd<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f64>>> {
        self.density_sd
            .iter()
            .map(|sd| PyArray2::from_owned_array(py, sd.clone()))
            .collect()
    }

    /// The temperature in K.
    #[getter]
    fn temperature_k<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_owned_array(py, self.temperature_k.clone())
    }

    /// Standard deviation of the temperature in K.
    #[getter]
    fn temperature_sd_k<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_owned_array(py, self.temperature_sd_k.clone())
    }

    /// [open-beam, sample] overdispersion maps.
    #[getter]
    fn overdispersion<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f64>>> {
        self.overdispersion
            .iter()
            .map(|map| PyArray2::from_owned_array(py, map.clone()))
            .collect()
    }

    /// Numerical convergence and uncertainty status, not physical validation.
    #[getter]
    fn trusted<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<bool>> {
        PyArray2::from_owned_array(py, self.trusted.clone())
    }

    /// The normalization a.
    #[getter]
    fn normalization(&self) -> f64 {
        self.normalization
    }

    /// The time background's [C, A, tau].
    #[getter]
    fn time_background(&self) -> [f64; 3] {
        self.time_background
    }

    /// The timing offset t0 in microseconds.
    #[getter]
    fn t0_us(&self) -> f64 {
        self.t0_us
    }

    /// The flight path in m.
    #[getter]
    fn flight_path_m(&self) -> f64 {
        self.flight_path_m
    }

    /// The pulse's [alpha0, alpha1].
    #[getter]
    fn alpha(&self) -> [f64; 2] {
        self.alpha
    }

    /// The pulse's [beta0, beta1].
    #[getter]
    fn beta(&self) -> [f64; 2] {
        self.beta
    }

    /// The pulse's R.
    #[getter]
    fn r(&self) -> f64 {
        self.r
    }

    /// The square of the pulse's triangle FWHM, h^2 in microseconds squared.
    #[getter]
    fn fwhm_squared_us2(&self) -> f64 {
        self.fwhm_squared_us2
    }

    /// The fitted shared quantities, in order.
    #[getter]
    fn shared(&self) -> Vec<String> {
        self.shared.clone()
    }

    /// Covariance of the shared quantities, or None.
    #[getter]
    fn shared_covariance<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.shared_covariance
            .as_ref()
            .map(|covariance| matrix_to_py(py, covariance))
    }

    /// For each measured shared quantity, its pull.
    #[getter]
    fn measured_pulls(&self) -> Option<Vec<f64>> {
        self.measured_pulls.clone()
    }

    /// The pulse calibration's consistency test as (q, dof, p), or None.
    #[getter]
    fn pulse_consistency(&self) -> Option<(f64, usize, f64)> {
        self.pulse_consistency
    }

    /// Each fit's half Poisson deviance over its overdispersion.
    #[getter]
    fn deviance(&self) -> f64 {
        self.deviance
    }

    /// Whether the map converged.
    #[getter]
    fn converged(&self) -> bool {
        self.converged
    }

    /// Why the map did not converge, or None.
    #[getter]
    fn unconverged(&self) -> Option<String> {
        self.unconverged.clone()
    }

    /// Newton steps tried on the shared quantities.
    #[getter]
    fn steps(&self) -> usize {
        self.steps
    }

    /// The number of fitted patches.
    #[getter]
    fn n_fitted(&self) -> usize {
        self.fits.iter().filter(|fit| fit.is_some()).count()
    }

    fn __repr__(&self) -> String {
        format!(
            "CountsMap(patches={}x{}, fitted={}, converged={})",
            self.patches.nrows(),
            self.patches.ncols(),
            self.n_fitted(),
            self.converged
        )
    }
}

/// Fit the areal densities and temperatures of the measurement's regions to
/// the raw counts of both runs.
#[pyfunction]
fn fit_counts(
    py: Python<'_>,
    measurement: PyRef<'_, PyMeasurement>,
    calibration: PyRef<'_, PyCalibration>,
) -> PyResult<PyCountsFit> {
    let measurement = measurement.inner.clone();
    let calibration = calibration.inner.clone();
    let fit = py
        .detach(move || nereids_pipeline::counts_fit::fit_counts(&measurement, &calibration))
        .map_err(counts_error)?;
    Ok(PyCountsFit { inner: fit })
}

/// Fit a material over patches of the map's pixels to the raw counts of both
/// runs.
#[pyfunction]
fn fit_map(
    py: Python<'_>,
    map: PyRef<'_, PyMapMeasurement>,
    calibration: PyRef<'_, PyCalibration>,
) -> PyResult<PyCountsMap> {
    let data = map.inner.clone();
    let calibration = calibration.inner.clone();
    py.detach(move || {
        let measurement = data.measurement();
        nereids_pipeline::counts_map::fit_map(&measurement, &calibration)
            .map(|fit| PyCountsMap::wrap(&fit))
    })
    .map_err(counts_error)
}

/// NEREIDS Python module.
#[pymodule]
fn nereids(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyResonanceData>()?;
    m.add_class::<PyIsotopeGroup>()?;
    m.add_class::<PyTabulatedResolution>()?;
    m.add_class::<PyEnergyLaw>()?;
    m.add_class::<PyIkedaCarpenter>()?;
    m.add_class::<PyNexusMetadata>()?;
    m.add_class::<PyNexusData>()?;
    m.add_class::<PyRunHealth>()?;
    m.add_class::<PyRunLog>()?;
    m.add_class::<PyBankSpectrum>()?;
    m.add_class::<PyValue>()?;
    m.add_class::<PyMaterial>()?;
    m.add_class::<PyRegion>()?;
    m.add_class::<PyMeasurement>()?;
    m.add_class::<PyPulse>()?;
    m.add_class::<PyCalibration>()?;
    m.add_class::<PyRegionFit>()?;
    m.add_class::<PyCountsFit>()?;
    m.add_class::<PyPatch>()?;
    m.add_class::<PyMapMeasurement>()?;
    m.add_class::<PyCountsMap>()?;
    m.add(
        "NO_TIME_BACKGROUND",
        (
            PyValue {
                inner: NO_TIME_BACKGROUND[0],
            },
            PyValue {
                inner: NO_TIME_BACKGROUND[1],
            },
            PyValue {
                inner: NO_TIME_BACKGROUND[2],
            },
        ),
    )?;
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
    m.add_function(wrap_pyfunction!(width_from_sigma, m)?)?;
    m.add_function(wrap_pyfunction!(width_from_fwhm, m)?)?;
    m.add_function(wrap_pyfunction!(load_resolution, m)?)?;
    m.add_function(wrap_pyfunction!(py_apply_resolution, m)?)?;
    m.add_function(wrap_pyfunction!(load_tiff_stack, m)?)?;
    m.add_function(wrap_pyfunction!(load_tiff_folder, m)?)?;
    m.add_function(wrap_pyfunction!(read_tof_sidecar, m)?)?;
    m.add_function(wrap_pyfunction!(probe_nexus, m)?)?;
    m.add_function(wrap_pyfunction!(run_health, m)?)?;
    m.add_function(wrap_pyfunction!(load_nexus_histogram, m)?)?;
    m.add_function(wrap_pyfunction!(load_nexus_events, m)?)?;
    m.add_function(wrap_pyfunction!(read_run_log, m)?)?;
    m.add_function(wrap_pyfunction!(intervals_where, m)?)?;
    m.add_function(wrap_pyfunction!(intervals_intersect, m)?)?;
    m.add_function(wrap_pyfunction!(load_nexus_bank_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(tof_to_energy_centers, m)?)?;
    m.add_function(wrap_pyfunction!(py_element_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(py_element_name, m)?)?;
    m.add_function(wrap_pyfunction!(py_parse_isotope_str, m)?)?;
    m.add_function(wrap_pyfunction!(py_natural_abundance, m)?)?;
    m.add_function(wrap_pyfunction!(py_natural_isotopes, m)?)?;
    m.add_function(wrap_pyfunction!(precompute_cross_sections, m)?)?;
    m.add_function(wrap_pyfunction!(detect_dead_pixels, m)?)?;
    m.add_function(wrap_pyfunction!(detect_hot_pixels, m)?)?;
    m.add_function(wrap_pyfunction!(detect_dead_pixels_chunked, m)?)?;
    m.add_function(wrap_pyfunction!(detect_bad_pixels, m)?)?;
    m.add_function(wrap_pyfunction!(fit_counts, m)?)?;
    m.add_function(wrap_pyfunction!(fit_map, m)?)?;
    Ok(())
}
