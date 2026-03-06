"""Type stubs for the NEREIDS Python bindings (PEP 561)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

class ResonanceData:
    """ENDF resonance data for an isotope."""

    @property
    def z(self) -> int:
        """Atomic number."""
        ...

    @property
    def a(self) -> int:
        """Mass number."""
        ...

    @property
    def awr(self) -> float:
        """Atomic weight ratio (target mass / neutron mass)."""
        ...

    @property
    def n_resonances(self) -> int:
        """Total number of resonances across all L-groups and ranges."""
        ...

    @property
    def target_spin(self) -> float:
        """Target nuclear spin (I) of the first resonance range."""
        ...

    @property
    def scattering_radius(self) -> float:
        """Effective scattering radius in fm."""
        ...

    @property
    def l_values(self) -> list[int]:
        """Orbital angular momentum values (L) present in the data."""
        ...

class FitResult:
    """Result of fitting a spectrum."""

    @property
    def densities(self) -> NDArray[np.float64]:
        """Fitted areal densities (atoms/barn)."""
        ...

    @property
    def uncertainties(self) -> NDArray[np.float64]:
        """Uncertainties on fitted densities."""
        ...

    @property
    def reduced_chi_squared(self) -> float:
        """Reduced chi-squared of the fit."""
        ...

    @property
    def converged(self) -> bool:
        """Whether the fit converged."""
        ...

    @property
    def iterations(self) -> int:
        """Number of iterations."""
        ...

    @property
    def temperature_k(self) -> float | None:
        """Fitted sample temperature in Kelvin (None when fit_temperature=False)."""
        ...

    @property
    def temperature_k_unc(self) -> float | None:
        """1-sigma uncertainty on fitted temperature (None when fit_temperature=False)."""
        ...

class TabulatedResolution:
    """Tabulated instrument resolution function."""

    @property
    def n_energies(self) -> int:
        """Number of reference energies."""
        ...

    @property
    def energy_range(self) -> tuple[float, float]:
        """Energy range (min, max) of the reference kernels in eV."""
        ...

    @property
    def flight_path_m(self) -> float:
        """Flight path length in meters."""
        ...

    @property
    def points_per_kernel(self) -> int:
        """Number of points per kernel."""
        ...

class SpatialResult:
    """Result of per-pixel spatial mapping (LM fitter)."""

    @property
    def density_maps(self) -> list[NDArray[np.float64]]:
        """Density maps as a list of 2D arrays, one per isotope."""
        ...

    @property
    def uncertainty_maps(self) -> list[NDArray[np.float64]]:
        """Uncertainty maps as a list of 2D arrays."""
        ...

    @property
    def chi_squared_map(self) -> NDArray[np.float64]:
        """Reduced chi-squared map."""
        ...

    @property
    def converged_map(self) -> NDArray[np.bool_]:
        """Convergence map (True = converged)."""
        ...

    @property
    def n_converged(self) -> int:
        """Number of converged pixels."""
        ...

    @property
    def n_total(self) -> int:
        """Total number of fitted pixels."""
        ...

    @property
    def isotope_names(self) -> list[str]:
        """Isotope names."""
        ...

class SparseResult:
    """Result of per-pixel sparse/Poisson fitting."""

    @property
    def density_maps(self) -> list[NDArray[np.float64]]:
        """Density maps as a list of 2D arrays, one per isotope."""
        ...

    @property
    def nll_map(self) -> NDArray[np.float64]:
        """Poisson negative log-likelihood map."""
        ...

    @property
    def converged_map(self) -> NDArray[np.bool_]:
        """Convergence map (True = converged)."""
        ...

    @property
    def n_converged(self) -> int:
        """Number of converged pixels."""
        ...

    @property
    def n_total(self) -> int:
        """Total number of fitted pixels."""
        ...

    @property
    def isotope_names(self) -> list[str]:
        """Isotope names."""
        ...

class TraceDetectabilityReport:
    """Result of a trace-detectability analysis."""

    @property
    def peak_delta_t_per_ppm(self) -> float:
        """Peak |DeltaT| per ppm concentration at the most sensitive energy."""
        ...

    @property
    def peak_energy_ev(self) -> float:
        """Energy at which peak contrast occurs (eV)."""
        ...

    @property
    def peak_snr(self) -> float:
        """Estimated peak SNR at the given concentration and I0."""
        ...

    @property
    def detectable(self) -> bool:
        """Whether the combination is detectable (SNR > threshold)."""
        ...

    @property
    def delta_t_spectrum(self) -> NDArray[np.float64]:
        """Energy-resolved |DeltaT| spectrum for the given concentration."""
        ...

    @property
    def energies(self) -> NDArray[np.float64]:
        """Energies used (eV)."""
        ...

    @property
    def opaque_fraction(self) -> float:
        """Fraction of energy bins where matrix baseline is opaque (T < 1e-15)."""
        ...

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def cross_sections(
    energies: NDArray[np.float64],
    data: ResonanceData,
) -> dict[str, NDArray[np.float64]]:
    """Compute cross-sections at given energies for an isotope.

    Returns a dict with keys 'total', 'elastic', 'capture', 'fission'.

    Note: MLBW (Multi-Level Breit-Wigner) ranges use SLBW approximation
    (resonance-resonance interference is ignored).
    """
    ...

def forward_model(
    energies: NDArray[np.float64],
    isotopes: list[tuple[ResonanceData, float]],
    temperature_k: float = 0.0,
    flight_path_m: float | None = None,
    delta_t_us: float | None = None,
    delta_l_m: float | None = None,
    resolution: TabulatedResolution | None = None,
) -> NDArray[np.float64]:
    """Compute theoretical transmission spectrum."""
    ...

def fit_spectrum(
    measured_t: NDArray[np.float64],
    sigma: NDArray[np.float64],
    energies: NDArray[np.float64],
    isotopes: list[ResonanceData],
    temperature_k: float = 0.0,
    initial_densities: list[float] | None = None,
    max_iter: int = 100,
    flight_path_m: float | None = None,
    delta_t_us: float | None = None,
    delta_l_m: float | None = None,
    resolution: TabulatedResolution | None = None,
    fit_temperature: bool = False,
    fitter: str = "lm",
) -> FitResult:
    """Fit a measured transmission spectrum to recover isotopic areal densities."""
    ...

def tof_to_energy(tof_us: float, flight_path_m: float) -> float:
    """Convert time-of-flight (us) to energy (eV)."""
    ...

def energy_to_tof(energy_ev: float, flight_path_m: float) -> float:
    """Convert energy (eV) to time-of-flight (us)."""
    ...

def load_endf(
    z: int,
    a: int,
    library: str = "endf8.1",
    mat: int | None = None,
) -> ResonanceData:
    """Load ENDF resonance data for an isotope from the IAEA database."""
    ...

def load_endf_file(path: str) -> ResonanceData:
    """Load ENDF resonance data from a local file."""
    ...

def create_resonance_data(
    z: int,
    a: int,
    awr: float,
    scattering_radius: float,
    resonances: list[tuple[float, float, float, float]],
    target_spin: float = 0.0,
    l_groups: list[tuple[int, list[tuple[float, float, float, float]]]] | None = None,
    formalism: str | None = None,
) -> ResonanceData:
    """Create ResonanceData from parameters (for testing/custom isotopes)."""
    ...

def beer_lambert(
    cross_sections: NDArray[np.float64],
    thickness: float,
) -> NDArray[np.float64]:
    """Beer-Lambert transmission: T = exp(-thickness * sigma)."""
    ...

def doppler_broaden(
    energies: NDArray[np.float64],
    cross_sections: NDArray[np.float64],
    awr: float,
    temperature_k: float,
) -> NDArray[np.float64]:
    """Apply Free Gas Model (FGM) Doppler broadening to a cross-section array."""
    ...

def resolution_broaden(
    energies: NDArray[np.float64],
    cross_sections: NDArray[np.float64],
    flight_path_m: float,
    delta_t_us: float,
    delta_l_m: float,
) -> NDArray[np.float64]:
    """Apply Gaussian resolution broadening to a cross-section or spectrum array."""
    ...

def load_resolution(
    path: str,
    flight_path_m: float,
) -> TabulatedResolution:
    """Load a tabulated resolution function from a VENUS/FTS-format file."""
    ...

def apply_resolution(
    energies: NDArray[np.float64],
    spectrum: NDArray[np.float64],
    resolution: TabulatedResolution,
) -> NDArray[np.float64]:
    """Apply tabulated resolution broadening to a spectrum."""
    ...

def spatial_map(
    transmission: NDArray[np.float64],
    uncertainty: NDArray[np.float64],
    energies: NDArray[np.float64],
    isotopes: list[ResonanceData],
    temperature_k: float = 300.0,
    initial_densities: list[float] | None = None,
    dead_pixels: NDArray[np.bool_] | None = None,
    flight_path_m: float | None = None,
    delta_t_us: float | None = None,
    delta_l_m: float | None = None,
    resolution: TabulatedResolution | None = None,
    max_iter: int = 100,
    fitter: str = "lm",
    roi: list[int] | None = None,
) -> SpatialResult | SparseResult:
    """Run per-pixel fitting across a transmission image stack."""
    ...

def fit_roi(
    transmission: NDArray[np.float64],
    uncertainty: NDArray[np.float64],
    y_range: tuple[int, int],
    x_range: tuple[int, int],
    energies: NDArray[np.float64],
    isotopes: list[ResonanceData],
    temperature_k: float = 300.0,
    initial_densities: list[float] | None = None,
    flight_path_m: float | None = None,
    delta_t_us: float | None = None,
    delta_l_m: float | None = None,
    resolution: TabulatedResolution | None = None,
    max_iter: int = 100,
) -> FitResult:
    """Fit a single spectrum averaged over a region of interest."""
    ...

def load_tiff_stack(path: str) -> NDArray[np.float64]:
    """Load a multi-frame TIFF file into a 3D numpy array."""
    ...

def load_tiff_folder(
    folder: str,
    pattern: str | None = None,
) -> NDArray[np.float64]:
    """Load a folder of single-frame TIFFs into a 3D numpy array."""
    ...

def normalize(
    sample: NDArray[np.float64],
    open_beam: NDArray[np.float64],
    pc_sample: float,
    pc_ob: float,
    dark_current: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Normalize raw sample and open-beam data to transmission."""
    ...

def tof_to_energy_centers(
    tof_edges: NDArray[np.float64],
    flight_path_m: float,
    delay_us: float = 0.0,
) -> NDArray[np.float64]:
    """Convert TOF bin edges to energy bin centers."""
    ...

def element_symbol(z: int) -> str | None:
    """Get the element symbol for a given atomic number Z."""
    ...

def element_name(z: int) -> str | None:
    """Get the element name for a given atomic number Z."""
    ...

def parse_isotope_str(s: str) -> tuple[int, int] | None:
    """Parse an isotope string like 'U-238' into (Z, A)."""
    ...

def natural_abundance(z: int, a: int) -> float | None:
    """Get the natural isotopic abundance for a specific isotope."""
    ...

def natural_isotopes(z: int) -> list[tuple[tuple[int, int], float]]:
    """Get all naturally occurring isotopes for an element."""
    ...

def trace_detectability(
    matrix: ResonanceData,
    matrix_density: float,
    trace: ResonanceData,
    trace_ppm: float,
    energies: NDArray[np.float64],
    i0: float,
    temperature_k: float = 293.6,
    flight_path_m: float | None = None,
    delta_t_us: float | None = None,
    delta_l_m: float | None = None,
    resolution: TabulatedResolution | None = None,
    snr_threshold: float = 3.0,
) -> TraceDetectabilityReport:
    """Compute trace-detectability for a matrix + trace isotope pair."""
    ...

def trace_detectability_survey(
    matrix: ResonanceData,
    matrix_density: float,
    trace_candidates: list[ResonanceData],
    trace_ppm: float,
    energies: NDArray[np.float64],
    i0: float,
    temperature_k: float = 293.6,
    flight_path_m: float | None = None,
    delta_t_us: float | None = None,
    delta_l_m: float | None = None,
    resolution: TabulatedResolution | None = None,
    snr_threshold: float = 3.0,
) -> list[tuple[str, TraceDetectabilityReport]]:
    """Survey multiple trace candidates against a single matrix."""
    ...

def precompute_cross_sections(
    energies: NDArray[np.float64],
    isotopes: list[ResonanceData],
    temperature_k: float = 0.0,
    flight_path_m: float | None = None,
    delta_t_us: float | None = None,
    delta_l_m: float | None = None,
    resolution: TabulatedResolution | None = None,
) -> list[NDArray[np.float64]]:
    """Precompute Doppler- and resolution-broadened total cross-sections.

    Returns one broadened total cross-section array per isotope. This is the
    expensive physics step; caching the result avoids redundant computation.
    """
    ...

def detect_dead_pixels(
    data: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Detect dead pixels (all-zero across the spectral axis).

    Parameters
    ----------
    data :
        3D NumPy array with shape ``(n_frames, height, width)``. The spectral
        axis corresponds to the first dimension (``n_frames``).

    Returns
    -------
    NDArray[np.bool_]
        2D boolean mask with shape ``(height, width)``, where ``True`` marks
        a dead pixel (all-zero across the spectral axis).
    """
    ...
