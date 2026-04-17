"""Type stubs for the NEREIDS Python bindings (PEP 561)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

class NexusMetadata:
    """Metadata from probing a NeXus/HDF5 file."""

    @property
    def has_histogram(self) -> bool: ...
    @property
    def has_events(self) -> bool: ...
    @property
    def histogram_shape(self) -> tuple[int, int, int, int] | None: ...
    @property
    def n_events(self) -> int | None: ...
    @property
    def flight_path_m(self) -> float | None: ...
    @property
    def tof_offset_ns(self) -> float | None: ...

class NexusData:
    """Result of loading NeXus histogram or event data."""

    @property
    def counts(self) -> NDArray[np.float64]:
        """3D counts array (n_tof, height, width)."""
        ...
    @property
    def tof_edges_us(self) -> NDArray[np.float64]:
        """TOF bin edges in microseconds (length = n_tof + 1)."""
        ...
    @property
    def flight_path_m(self) -> float | None: ...
    @property
    def dead_pixels(self) -> NDArray[np.bool_] | None:
        """Dead pixel mask (height, width). True = dead."""
        ...
    @property
    def n_rotation_angles(self) -> int: ...
    @property
    def event_total(self) -> int | None:
        """Total events before filtering (event data only)."""
        ...
    @property
    def event_kept(self) -> int | None:
        """Events kept after filtering (event data only)."""
        ...

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

    @property
    def anorm(self) -> float:
        """Fitted normalization factor (1.0 if background not enabled)."""
        ...

    @property
    def background(self) -> tuple[float, float, float]:
        """Fitted background parameters (BackA, BackB, BackC)."""
        ...

    @property
    def back_d(self) -> float:
        """Fitted exponential background amplitude (SAMMY BackD). Zero when not fitted."""
        ...

    @property
    def back_f(self) -> float:
        """Fitted exponential background decay constant (SAMMY BackF). Zero when not fitted."""
        ...

    @property
    def t0_us(self) -> float | None:
        """Fitted TOF offset (SAMMY TZERO t0) in microseconds, or None."""
        ...

    @property
    def l_scale(self) -> float | None:
        """Fitted flight-path scale factor (SAMMY TZERO L0), or None."""
        ...

    @property
    def deviance_per_dof(self) -> float | None:
        """Conditional binomial deviance / (n - k) from the counts-KL
        dispatch (joint-Poisson profile-deviance fitter).

        Primary goodness-of-fit for ``solver='kl'`` (or the
        ``'poisson'`` / ``'joint_poisson'`` aliases) on counts data,
        per memo 35 §P1.2 — replaces the fixed-flux Pearson chi-squared
        that scaled with ``c``.  ``None`` for LM fits and for
        transmission + PoissonKL (those populate
        ``reduced_chi_squared`` with Pearson chi-squared / (n - k)).
        """
        ...

class CalibrationResult:
    """Result of energy axis calibration."""

    @property
    def flight_path_m(self) -> float:
        """Fitted flight path length in metres."""
        ...

    @property
    def t0_us(self) -> float:
        """Fitted TOF delay in microseconds."""
        ...

    @property
    def total_density(self) -> float:
        """Fitted total areal density in atoms/barn."""
        ...

    @property
    def reduced_chi_squared(self) -> float:
        """Reduced chi-squared at the best parameters."""
        ...

    @property
    def energies_corrected(self) -> NDArray[np.float64]:
        """Corrected energy grid (ascending, eV)."""
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
        """Reduced chi-squared map.  For the counts-KL dispatch this mirrors
        ``deviance_per_dof_map`` (back-compat)."""
        ...

    @property
    def deviance_per_dof_map(self) -> NDArray[np.float64] | None:
        """Counts-KL conditional binomial deviance / (n − k) per pixel
        (memo 35 §P1.2).  ``None`` for LM-only runs and for transmission +
        PoissonKL (those populate ``chi_squared_map`` with Pearson χ²/dof)."""
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
    def n_failed(self) -> int:
        """Number of pixels where the fitter returned a hard error (NaN density)."""
        ...

    @property
    def isotope_names(self) -> list[str]:
        """Isotope names."""
        ...

    @property
    def temperature_map(self) -> NDArray[np.float64] | None:
        """Per-pixel fitted temperature map (None when fit_temperature=False)."""
        ...

    @property
    def temperature_uncertainty_map(self) -> NDArray[np.float64] | None:
        """Per-pixel temperature uncertainty map (None when fit_temperature=False).
        Entries are NaN where uncertainty was unavailable for that pixel."""
        ...

    @property
    def anorm_map(self) -> NDArray[np.float64] | None:
        """Per-pixel normalization factor Anorm (None when background=False)."""
        ...

    @property
    def background_maps(self) -> list[NDArray[np.float64]] | None:
        """Per-pixel background parameter maps [BackA, BackB, BackC] (None when background=False)."""
        ...

class IsotopeGroup:
    """A group of isotopes sharing one fitted density parameter.

    Members have fixed fractional ratios summing to 1.0. During fitting,
    the effective cross-section sigma_eff(E) = sum(f_i * sigma_i(E)) reduces
    the group to a single virtual isotope with one free density parameter.
    """

    @staticmethod
    def natural(z: int) -> IsotopeGroup:
        """Create a group from all natural isotopes of element Z at IUPAC abundances."""
        ...

    @staticmethod
    def subset(z: int, mass_numbers: list[int]) -> IsotopeGroup:
        """Create a group from a subset of natural isotopes, re-normalized."""
        ...

    @staticmethod
    def custom(name: str, members: list[tuple[int, int, float]]) -> IsotopeGroup:
        """Create a group with arbitrary isotope/ratio pairs.

        Args:
            name: Display name for the group.
            members: List of (z, a, ratio) tuples. Ratios must sum to 1.0.
        """
        ...

    def load_endf(self, library: str | None = None) -> None:
        """Fetch ENDF data for all members.

        Args:
            library: ENDF library name (default "endf8.1").
        """
        ...

    @property
    def name(self) -> str:
        """Group display name (e.g., 'W (nat)', 'Eu-151/153')."""
        ...

    @property
    def n_members(self) -> int:
        """Number of member isotopes."""
        ...

    @property
    def members(self) -> list[tuple[tuple[int, int], float]]:
        """Member isotopes with their fractional ratios as ((z, a), ratio) tuples."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Whether ENDF data has been loaded for all members."""
        ...

    @property
    def resonance_data(self) -> list[ResonanceData]:
        """Get loaded resonance data for all members.

        Raises:
            ValueError: If not all members have loaded ENDF data.
        """
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
    isotopes: list[tuple[ResonanceData, float]] | None = None,
    temperature_k: float = 293.6,
    flight_path_m: float | None = None,
    delta_t_us: float | None = None,
    delta_l_m: float | None = None,
    resolution: TabulatedResolution | None = None,
    delta_e_us: float | None = None,
    groups: list[tuple[IsotopeGroup, float]] | None = None,
) -> NDArray[np.float64]:
    """Compute theoretical transmission spectrum.

    Either ``isotopes`` or ``groups`` must be provided, but not both.
    When ``groups`` is provided, each group is expanded into its members
    with effective densities = group_density * member_ratio.
    """
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
    delta_e_us: float = 0.0,
) -> NDArray[np.float64]:
    """Apply resolution broadening (Gaussian, or Gaussian+exponential tail) to a cross-section or spectrum array."""
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

def load_tiff_stack(path: str) -> NDArray[np.float64]:
    """Load a multi-frame TIFF file into a 3D numpy array."""
    ...

def load_tiff_folder(
    folder: str,
    pattern: str | None = None,
) -> NDArray[np.float64]:
    """Load a folder of single-frame TIFFs into a 3D numpy array."""
    ...

def probe_nexus(path: str) -> NexusMetadata:
    """Probe a NeXus/HDF5 file for available data without loading it."""
    ...

def load_nexus_histogram(path: str) -> NexusData:
    """Load pre-histogrammed counts from a NeXus/HDF5 file.

    Reads ``/entry/histogram/counts``, sums over rotation angles,
    and returns a ``NexusData`` with shape ``(n_tof, height, width)``.
    """
    ...

def load_nexus_events(
    path: str,
    n_bins: int,
    tof_min_us: float,
    tof_max_us: float,
    height: int,
    width: int,
) -> NexusData:
    """Load event data from a NeXus/HDF5 file, histogramming into TOF bins.

    Reads ``/entry/neutrons/event_time_offset``, ``/x``, ``/y`` and bins
    events into a linear TOF grid.
    """
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
    delta_e_us: float | None = None,
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
    delta_e_us: float | None = None,
    snr_threshold: float = 3.0,
) -> list[tuple[str, TraceDetectabilityReport]]:
    """Survey multiple trace candidates against a single matrix."""
    ...

def precompute_cross_sections(
    energies: NDArray[np.float64],
    isotopes: list[ResonanceData],
    temperature_k: float = 293.6,
    flight_path_m: float | None = None,
    delta_t_us: float | None = None,
    delta_l_m: float | None = None,
    resolution: TabulatedResolution | None = None,
    delta_e_us: float | None = None,
) -> list[NDArray[np.float64]]:
    """Precompute Doppler-broadened total cross-sections.

    Returns one Doppler-broadened total cross-section array per isotope.
    This is the expensive physics step; caching the result avoids redundant
    computation.

    Raises ``ValueError`` if any resolution parameters are passed.
    Resolution broadening cannot be precomputed as broadened cross-sections
    because it must be applied after Beer-Lambert on the total transmission,
    which depends on per-pixel densities.  Use ``forward_model()`` instead.
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

def calibrate_energy(
    energies_nominal: NDArray[np.float64],
    transmission: NDArray[np.float64],
    uncertainty: NDArray[np.float64],
    isotopes: list[ResonanceData],
    abundances: list[float],
    assumed_flight_path_m: float,
    temperature_k: float = 293.6,
) -> CalibrationResult:
    """Calibrate the energy axis by fitting flight path and TOF delay.

    Finds the (L, t0, n_total) that best align the ENDF resonance model
    with measured transmission data from a known-composition reference.
    """
    ...


# ---------------------------------------------------------------------------
# Typed Input Data API (Phase 5)
# ---------------------------------------------------------------------------

class InputData:
    """Opaque typed input data for spatial mapping.

    Created via ``from_counts()`` or ``from_transmission()``.
    Passed to ``spatial_map_typed()``.
    """

    @property
    def kind(self) -> str:
        """'counts' or 'transmission'."""
        ...

    @property
    def shape(self) -> tuple[int, int, int]:
        """(n_energies, height, width)."""
        ...


def from_counts(
    sample_counts: NDArray[np.float64],
    open_beam_counts: NDArray[np.float64],
) -> InputData:
    """Create InputData from raw detector counts and open beam.

    The fitting engine uses Poisson KL by default (statistically
    optimal for count data).
    """
    ...


def from_transmission(
    transmission: NDArray[np.float64],
    uncertainty: NDArray[np.float64],
) -> InputData:
    """Create InputData from normalized transmission and uncertainty.

    The fitting engine uses LM by default. Pass solver="kl" to
    spatial_map_typed() for low-count transmission data.
    """
    ...


def spatial_map_typed(
    data: InputData,
    energies: NDArray[np.float64],
    isotopes: list[ResonanceData] | None = None,
    *,
    temperature_k: float = 293.6,
    fit_temperature: bool = False,
    initial_densities: list[float] | None = None,
    dead_pixels: NDArray[np.bool_] | None = None,
    max_iter: int = 200,
    solver: str = "auto",
    background: bool = False,
    fit_alpha_1: bool = False,
    fit_alpha_2: bool = False,
    alpha_1_init: float = 1.0,
    alpha_2_init: float = 1.0,
    c: float = 1.0,
    enable_polish: bool | None = None,
    resolution: TabulatedResolution | None = None,
    flight_path_m: float | None = None,
    delta_t_us: float | None = None,
    delta_l_m: float | None = None,
    groups: list[IsotopeGroup] | None = None,
) -> SpatialResult:
    """Spatial mapping using the typed input data API.

    Either ``isotopes`` or ``groups`` must be provided, but not both.
    When ``groups`` is provided, each group maps to one fitted density parameter.

    Dispatches per-pixel fitting based on InputData type:
      - from_counts / from_counts_with_nuisance + solver="kl" / "auto"
        -> counts-KL (joint-Poisson deviance) — the counts-path solver
        validated in memo 35 §P1/§P2 and memo 38.
      - from_transmission + solver="lm" (default for transmission) -> LM.
      - from_transmission + solver="kl" -> Poisson NLL on transmission values
        (legacy niche).

    Args:
        data: InputData from `from_counts()`, `from_counts_with_nuisance()`,
            or `from_transmission()`.
        c: Proton-charge ratio ``Q_s / Q_ob`` for the counts-KL dispatch
            (memo 35 §P1.3).  Default 1.0 (assumes caller PC-normalized
            the flux already).  Ignored for LM / transmission-KL paths.
        enable_polish: Override the Nelder-Mead polish flag.  ``None``
            (default) = the dispatcher auto-disables polish when
            ``n_pixels > 1`` (memo 38 §6 — polish costs ~1000 s per pixel
            on realistic data).  ``True`` forces polish on, ``False`` off.

    Always returns SpatialResult.  For counts-KL runs,
    ``SpatialResult.deviance_per_dof_map`` is populated as the primary GOF.
    """
    ...


def fit_spectrum_typed(
    transmission: NDArray[np.float64],
    uncertainty: NDArray[np.float64],
    energies: NDArray[np.float64],
    isotopes: list[tuple[ResonanceData, float]] | None = None,
    *,
    temperature_k: float = 293.6,
    fit_temperature: bool = False,
    max_iter: int = 200,
    solver: str = "lm",
    background: bool = False,
    fit_back_d: bool = False,
    fit_back_f: bool = False,
    back_d_init: float = 0.01,
    back_f_init: float = 1.0,
    fit_energy_scale: bool = False,
    t0_init_us: float = 0.0,
    l_scale_init: float = 1.0,
    energy_scale_flight_path_m: float = 25.0,
    resolution: TabulatedResolution | None = None,
    flight_path_m: float | None = None,
    delta_t_us: float | None = None,
    delta_l_m: float | None = None,
    groups: list[IsotopeGroup] | None = None,
    initial_densities: list[float] | None = None,
) -> FitResult:
    """Fit a single pre-normalized transmission spectrum.

    This function accepts **transmission** data only (T = sample/open-beam).
    For raw-count fitting, use ``fit_counts_spectrum_typed(...)``.

    Either ``isotopes`` or ``groups`` must be provided, but not both.
    When ``groups`` is provided, each group maps to one fitted density parameter.

    Args:
        transmission: 1D pre-normalized transmission spectrum.
        uncertainty: 1D uncertainty (same length as transmission).
        energies: 1D energy grid in eV (ascending).
        isotopes: List of (ResonanceData, initial_density) tuples.
        temperature_k: Sample temperature in Kelvin (default 293.6).
        fit_temperature: Whether to fit temperature (default False).
        max_iter: Maximum iterations (default 200).
        solver: 'lm' (default), 'kl', or 'auto'.
        background: Enable SAMMY transmission background.
        resolution: Optional resolution function.
        groups: List of IsotopeGroup objects (mutually exclusive with isotopes).
        initial_densities: Initial density guesses when using groups.
    """
    ...


def fit_counts_spectrum_typed(
    sample_counts: NDArray[np.float64],
    open_beam_counts: NDArray[np.float64],
    energies: NDArray[np.float64],
    isotopes: list[tuple[ResonanceData, float]] | None = None,
    *,
    temperature_k: float = 293.6,
    fit_temperature: bool = False,
    max_iter: int = 200,
    solver: str = "auto",
    background: bool = False,
    fit_back_d: bool = False,
    fit_back_f: bool = False,
    back_d_init: float = 0.01,
    back_f_init: float = 1.0,
    fit_energy_scale: bool = False,
    t0_init_us: float = 0.0,
    l_scale_init: float = 1.0,
    energy_scale_flight_path_m: float = 25.0,
    detector_background: NDArray[np.float64] | None = None,
    fit_alpha_1: bool = False,
    fit_alpha_2: bool = False,
    alpha_1_init: float = 1.0,
    alpha_2_init: float = 1.0,
    c: float = 1.0,
    resolution: TabulatedResolution | None = None,
    flight_path_m: float | None = None,
    delta_t_us: float | None = None,
    delta_l_m: float | None = None,
    groups: list[IsotopeGroup] | None = None,
    initial_densities: list[float] | None = None,
) -> FitResult:
    """Fit a single raw-count spectrum (sample + open-beam counts).

    Dispatches to a counts-domain solver based on ``solver``:

    - ``'auto'`` (default), ``'kl'``, ``'poisson'``, and ``'joint_poisson'``
      all route to the **counts-KL dispatch**: the joint-Poisson profile
      binomial-deviance fitter (memo 35 §P1/§P2; collapsed to a single
      path in this PR).  Uses the explicit proton-charge ratio
      ``c = Q_s / Q_ob`` from the ``c`` kwarg and populates
      ``FitResult.deviance_per_dof`` as the primary GOF.
      ``'joint_poisson'`` is kept as a compatibility alias; prefer ``'kl'``
      for new code.
    - ``'lm'`` converts counts to transmission internally and runs
      Levenberg-Marquardt on the resulting ratio (information-lossy
      fallback).

    For pre-normalized transmission data, use ``fit_spectrum_typed(...)``.

    Either ``isotopes`` or ``groups`` must be provided, but not both.
    When ``groups`` is provided, each group maps to one fitted density parameter.

    Args:
        sample_counts: 1D sample counts spectrum.
        open_beam_counts: 1D open-beam counts reference.
        energies: 1D energy grid in eV (ascending).
        isotopes: List of (ResonanceData, initial_density) tuples.
        temperature_k: Sample temperature in Kelvin (default 293.6).
        fit_temperature: Whether to fit temperature (default False).
        max_iter: Maximum iterations (default 200).
        solver: ``'auto'`` (default), ``'kl'`` / ``'poisson'`` /
            ``'joint_poisson'`` (all equivalent — counts-KL dispatch),
            or ``'lm'``.
        background: Enable the SAMMY-style transmission-background
            wrapper inside the counts-KL fit (A_n + B_A + B_B/√E + B_C√E).
        detector_background: Optional detector/counts background reference
            (for LM-converted path only; counts-KL rejects non-zero values,
            deferred to memo 35 §P3.2).
        fit_alpha_1: Research-only; rejected by the counts-KL dispatch
            because the profile λ̂ absorbs the global flux scale.
        fit_alpha_2: Research-only; rejected by the counts-KL dispatch
            (B_det / alpha_2 wiring deferred to memo 35 §P3.2).
        alpha_1_init: Initial value for alpha_1 (default 1.0); only
            consumed by the research Fisher helper.
        alpha_2_init: Initial value for alpha_2 (default 1.0); same.
        c: Proton-charge ratio ``Q_s / Q_ob`` (memo 35 §P1.3).  Default
            1.0 assumes the caller has already PC-normalized the flux.
            For raw VENUS-style counts, set this to the actual ratio
            (typically ~5–6).  Used by the counts-KL dispatch; ignored
            by the LM path.
        resolution: Optional resolution function.
        groups: List of IsotopeGroup objects (mutually exclusive with isotopes).
        initial_densities: Initial density guesses when using groups.
    """
    ...

class ModelJacobianResult:
    """Result of exact Jacobian/Fisher evaluation from the Rust engine."""

    @property
    def jacobian(self) -> NDArray[np.float64]:
        """Analytical Jacobian (n_energy × n_free_params), row-major."""
        ...

    @property
    def fisher(self) -> NDArray[np.float64]:
        """Expected Poisson Fisher F = J^T diag(1/μ) J (n_free × n_free)."""
        ...

    @property
    def model_prediction(self) -> NDArray[np.float64]:
        """Model prediction μ(E) at the evaluation point."""
        ...

    @property
    def param_names(self) -> list[str]:
        """Names of free parameters in Jacobian column order."""
        ...

def compute_model_jacobian(
    open_beam_counts: NDArray[np.float64],
    energies: NDArray[np.float64],
    isotopes: list[tuple[ResonanceData, float]] | None = None,
    *,
    temperature_k: float = 293.6,
    fit_temperature: bool = False,
    flight_path_m: float | None = None,
    delta_t_us: float | None = None,
    delta_l_m: float | None = None,
    resolution: TabulatedResolution | None = None,
    detector_background: NDArray[np.float64] | None = None,
    fit_alpha_1: bool = False,
    fit_alpha_2: bool = False,
    alpha_1: float = 1.0,
    alpha_2: float = 1.0,
    groups: list[IsotopeGroup] | None = None,
    initial_densities: list[float] | None = None,
) -> ModelJacobianResult:
    """Compute exact resolved analytical Jacobian and expected Fisher.

    Uses the same model construction as ``fit_counts_spectrum_typed()`` but
    evaluates at the given parameter values without optimising.

    Either ``isotopes`` or ``groups`` must be provided, but not both.
    When ``groups`` is provided, each group maps to one density parameter.

    Research-oriented function for Fisher-based regularisation studies.
    """
    ...
