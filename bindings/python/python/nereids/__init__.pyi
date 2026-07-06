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

class RunHealth:
    """DASlogs-based run-health summary (``None`` = PV absent)."""

    @property
    def pause_fraction(self) -> float | None:
        """Time-weighted fraction of the run spent paused."""
        ...
    @property
    def beam_dip_fraction(self) -> float | None:
        """Time-weighted fraction with power below the dip threshold."""
        ...
    @property
    def median_power(self) -> float | None:
        """Sample median of the power PV entries (not time-weighted)."""
        ...
    @property
    def duration_s(self) -> float | None:
        """Run duration in seconds (file value, or last-timestamp lower bound)."""
        ...
    @property
    def n_pause_entries(self) -> int: ...
    @property
    def n_power_entries(self) -> int: ...

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
    def back_d(self) -> float | None:
        """Fitted exponential background amplitude (SAMMY BackD), or None when ``background=False`` or ``fit_back_d=False``."""
        ...

    @property
    def back_f(self) -> float | None:
        """Fitted exponential background decay constant (SAMMY BackF), or None when ``background=False`` or ``fit_back_f=False``."""
        ...

    @property
    def t0_us(self) -> float | None:
        """Fitted TOF offset (SAMMY TZERO t0) in microseconds, or None."""
        ...

    @property
    def l_scale(self) -> float | None:
        """Fitted flight-path scale factor (SAMMY TZERO L0), or None."""
        ...

    def corrected_energies(
        self, nominal_energies: NDArray[np.float64]
    ) -> NDArray[np.float64] | None:
        """Map a nominal energy grid through the fitted ``(t0_us, l_scale)``
        energy scale to the corrected (calibrated) energies the fit used
        (issue #634), using the exact SAMMY −t0 convention and the SAME
        flight path the fit was configured with (stored on the result).

        Returns ``None`` when energy-scale fitting was not enabled; raises
        ``ValueError`` on an invalid nominal grid (non-finite, non-positive,
        or non-ascending) or a degenerate calibration (t0 past the shortest
        flight time).
        """
        ...

    @property
    def deviance_per_dof(self) -> float | None:
        """Conditional binomial deviance / (n - k) from the counts-KL
        dispatch (joint-Poisson profile-deviance fitter).

        Primary goodness-of-fit for ``solver='kl'`` (or the
        ``'poisson'`` / ``'joint_poisson'`` aliases) on counts data —
        replaces the fixed-flux Pearson chi-squared
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

class EnergyLaw:
    """Energy-dependence law for an Ikeda-Carpenter parameter.

    Build via the static constructors; pass to :class:`IkedaCarpenter`.
    """

    @staticmethod
    def const(value: float) -> EnergyLaw:
        """Energy-independent constant value."""
        ...

    @staticmethod
    def sqrt_e(a0: float, a1: float) -> EnergyLaw:
        """``a0*sqrt(E[eV]) + a1`` — leading epithermal scaling of alpha(E)."""
        ...

    @staticmethod
    def inverse_lambda(a0: float, a1: float) -> EnergyLaw:
        """Mantid IC form ``1/(a0 + a1*lambda)`` (alpha ~ sqrt(E) low-E)."""
        ...

    @staticmethod
    def exp_mev(kappa: float) -> EnergyLaw:
        """``exp(-E[meV]/kappa)`` — storage fraction R(E), ->0 in the eV regime."""
        ...

    def eval(self, energy_ev: float) -> float:
        """Evaluate the law at ``energy_ev`` (eV)."""
        ...

class IkedaCarpenter:
    """Analytical Ikeda-Carpenter instrument-resolution model.

    Synthesizes a dense tabulated kernel at construction; pass
    :meth:`as_tabulated` anywhere a loaded resolution file is accepted
    (``forward_model``, ``fit_spectrum_typed``, ``calibrate_resolution``).
    Note ``precompute_cross_sections`` does NOT take a resolution -- broadening
    is applied after Beer-Lambert, not on the cross-sections.
    """

    def __init__(
        self,
        flight_path_m: float,
        e_min_ev: float,
        e_max_ev: float,
        alpha: EnergyLaw,
        beta: float,
        r: EnergyLaw,
        n_energies: int = 64,
        n_tau: int = 600,
        burst_sigma_us: float | None = None,
        channel_fwhm_us: float | None = None,
    ) -> None: ...
    def as_tabulated(self) -> TabulatedResolution:
        """The synthesized tabulated kernel (usable as a resolution file)."""
        ...

    def kernel_at(self, energy_ev: float) -> tuple[list[float], list[float]]:
        """``(tof_offsets_us, weights)`` at one energy; mode at offset 0.

        Raises ``ValueError`` when the tau-grid cannot resolve the prompt
        core and requested folds within the sample cap at this energy.
        """
        ...

    @property
    def flight_path_m(self) -> float:
        """Flight path length in meters."""
        ...

    @property
    def n_energies(self) -> int:
        """Number of synthesized reference energies."""
        ...

class ResolutionCalibration:
    """Result of :func:`calibrate_resolution`."""

    @property
    def family(self) -> str:
        """Resolution-model family (``"gaussian"`` | ``"udr_corr"`` | ``"ic"``)."""
        ...

    @property
    def theta(self) -> list[float]:
        """Raw fitted parameter vector (optimizer space)."""
        ...

    @property
    def chi2(self) -> float:
        """chi-squared per degree of freedom of the calibration fit."""
        ...

    @property
    def converged(self) -> bool:
        """Whether the optimizer self-converged."""
        ...

    @property
    def iterations(self) -> int:
        """Optimizer iterations."""
        ...

    @property
    def position_t0_us(self) -> float:
        """Fitted (or pinned) SAMMY energy-scale TOF zero t0 (us). Equals
        t0_center_us when fit_t0=False (the default; position pinned). When fit, a
        SHARED energy-scale parameter under a metrology prior, not a per-family
        nuisance (the asymmetric-kernel lag is confounded with flight-path
        L_scale)."""
        ...

    @property
    def position_l_scale(self) -> float:
        """Fitted (or pinned) flight-path scale L_scale. Equals l_scale_center when
        fit_l_scale=False (the default)."""
        ...

    @property
    def prior_penalty(self) -> float:
        """Gaussian-prior penalty on the fitted (t0, L_scale) at the solution
        (0 when position is pinned or has no prior). objective = chi2_data +
        prior_penalty; a large value flags a family that needed a big position move
        to fit."""
        ...

    @property
    def n_free_params(self) -> int:
        """Number of outer-loop free parameters: resolution theta plus any
        fitted position coordinates (4-5 for family="ic", 2 for the other
        families)."""
        ...

    @property
    def bounds_hit(self) -> list[str]:
        """Coordinates pinned at a box bound at the solution, as
        "name:lower" / "name:upper" strings (empty list = interior solution).
        E.g. "r:lower" flags the beta-R ridge: the calibrant shows no storage
        tail, so the reported beta carries no information."""
        ...

    def params(self) -> dict[str, float]:
        """Decoded, human-readable fitted parameters.

        For family="ic" the keys are a0/a1 (alpha(E) = a0*sqrt(E) + a1,
        positive by construction), beta, r and psr_fwhm_us — decoded from the
        calibrated resolution itself (the raw theta is ln/box-encoded
        optimizer space)."""
        ...

    def as_tabulated(self) -> TabulatedResolution | None:
        """The calibrated resolution as a kernel to pin into a fit
        (``udr_corr`` / ``ic``); ``None`` for the Gaussian family."""
        ...

    def gaussian_params(self) -> tuple[float, float] | None:
        """``(delta_t_us, delta_l_m)`` for the Gaussian family; ``None`` otherwise."""
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
        """Counts-KL conditional binomial deviance / (n − k) per pixel.
        ``None`` for LM-only runs and for transmission +
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

    @property
    def back_d_map(self) -> NDArray[np.float64] | None:
        """Per-pixel SAMMY exponential background amplitude (BackD) map.

        ``None`` whenever the polynomial transmission background was
        not active (``background=False``) OR the exponential tail was
        not fit (``fit_back_d=False``).  For counts inputs the map is
        always ``None`` — the joint-Poisson dispatch (counts-KL) never
        fits BackD/BackF, and the spatial pipeline rejects counts +
        ``fit_back_d=True`` up-front."""
        ...

    @property
    def back_f_map(self) -> NDArray[np.float64] | None:
        """Per-pixel SAMMY exponential background decay constant (BackF) map.

        ``None`` under the same conditions as :py:attr:`back_d_map`."""
        ...

    @property
    def t0_us_map(self) -> NDArray[np.float64] | None:
        """Per-pixel SAMMY TZERO offset t0 (µs) map.
        ``None`` when the run did not use ``fit_energy_scale=True``."""
        ...

    @property
    def l_scale_map(self) -> NDArray[np.float64] | None:
        """Per-pixel SAMMY TZERO flight-path scale factor map.
        ``None`` when the run did not use ``fit_energy_scale=True``."""
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

def calibrate_resolution(
    energies: NDArray[np.float64],
    data: NDArray[np.float64],
    uncertainty: NDArray[np.float64],
    family: str,
    isotopes: list[tuple[ResonanceData, float]] | None = None,
    groups: list[tuple[IsotopeGroup, float]] | None = None,
    temperature_k: float = 293.6,
    base_udr: TabulatedResolution | None = None,
    flight_path_m: float = 25.0,
    fit_background: bool = False,
    restarts: int = 1,
    ic_n_energies: int = 64,
    ic_n_tau: int = 500,
    fit_t0: bool = False,
    fit_l_scale: bool = False,
    t0_center_us: float = 0.0,
    l_scale_center: float = 1.0,
    t0_prior_us: float | None = None,
    l_scale_prior: float | None = None,
    psr_fwhm_ns: float = 350.0,
    fit_psr: bool = False,
) -> ResolutionCalibration:
    """Calibrate instrument-resolution parameters against a known-(rho,T) calibrant.

    Fits the resolution parameters of ``family`` (``"gaussian"`` | ``"udr_corr"``
    | ``"ic"``) while holding the calibrant density (in ``isotopes``/``groups``)
    and ``temperature_k`` fixed. ``base_udr`` is required for ``"udr_corr"``.
    Pin the returned resolution into a sample fit via ``.as_tabulated()`` (or
    ``.gaussian_params()`` for the Gaussian family).

    The ``"ic"`` family fits the full bounded moderator shape:
    ``alpha(E) = a0*sqrt(E) + a1`` (positive by construction), free bounded
    ``beta`` and storage fraction ``r``, folded with the SNS PSR channel
    triangle of FWHM ``psr_fwhm_ns`` (default 350 ns, the VENUS FTS header
    value; 0 disables; applies to "ic" only — tabulated/UDR kernels already
    carry the fold). ``psr_fwhm_ns`` is NANOSECONDS: nonzero values above
    10_000 ns (10 us) raise ``ValueError`` as a us-as-ns unit slip (kernel
    synthesis cost is quadratic in the fold width, so e.g. 350 meaning us
    would hang for hours). ``fit_psr=True`` ("ic" only) also fits the PSR FWHM as a
    5th parameter (box-bounded 0.05-1 us), started at ``psr_fwhm_ns`` clamped
    into that box: an out-of-box start (legal as a pin up to 10 us) starts at
    the nearer box edge with a stderr warning, and a fit that stays there
    reports ``psr_fwhm_us:lower`` / ``:upper`` in ``bounds_hit``.
    ``psr_fwhm_ns`` must then be > 0 (a zero start contradicts "0 disables"
    and raises ``ValueError``). ``psr_fwhm_ns`` / ``fit_psr`` sit at the END of the
    signature so pre-existing positional calls keep their meaning. Degenerate
    directions are reported via ``bounds_hit``.

    By default position is PINNED (``fit_t0=fit_l_scale=False``): a pure
    shape/width fit on the already energy-calibrated grid. Set ``fit_t0`` /
    ``fit_l_scale`` to also fit the SHARED SAMMY energy-scale ``(t0, L_scale)``
    under a Gaussian metrology prior (``t0_prior_us`` / ``l_scale_prior``, centered
    at ``t0_center_us`` / ``l_scale_center``) — for joint energy-scale or cross-
    family identifiability work. Do NOT fit position with a flat prior in
    production: the asymmetric-kernel lag is the same ``1/sqrt(E)`` basis as
    ``L_scale``, so a free ``L_scale`` absorbs the lag and corrupts the width.
    """
    ...

def tof_to_energy(tof_us: float, flight_path_m: float) -> float:
    """Convert time-of-flight (us) to energy (eV).

    Raises ``ValueError`` if ``tof_us`` or ``flight_path_m`` is non-positive
    or non-finite.
    """
    ...

def energy_to_tof(energy_ev: float, flight_path_m: float) -> float:
    """Convert energy (eV) to time-of-flight (us).

    Raises ``ValueError`` if ``energy_ev`` or ``flight_path_m`` is
    non-positive or non-finite.
    """
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
    """Apply Free Gas Model (FGM) Doppler broadening to a cross-section array.

    Exact FGM kernel (SAMMY manual Eq. III B1.7, w²-weighted integrand —
    the same weighting as SAMMY's Dopfgm; the numerical quadrature differs).
    Near the grid edges sigma is 1/v-extrapolated beyond the supplied grid;
    edge points whose Doppler window is both grid-truncated and
    under-resolved (fewer than 3 nodes) are returned unbroadened, matching
    SAMMY.
    """
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

def load_tiff_stack(
    path: str,
    pixel_policy: str = "reject",
) -> NDArray[np.float64]:
    """Load a multi-frame TIFF file into a 3D numpy array.

    ``pixel_policy``: ``"reject"`` (default) errors on negative or
    non-finite pixels; ``"clip"`` clamps negatives to 0.0 (NaN still
    errors); ``"allow"`` passes values through verbatim (pre-normalized
    transmission stacks).
    """
    ...

def load_tiff_folder(
    folder: str,
    pattern: str | None = None,
    sum_chunks: bool = True,
    pixel_policy: str = "reject",
) -> NDArray[np.float64]:
    """Load a folder of single-frame TIFFs into a 3D numpy array.

    Chunked VENUS folders (``<prefix>_<chunk>_<frame>.tif``) are detected
    automatically and summed element-wise across chunks by default
    (``sum_chunks=True``); ragged chunks are an error.  ``pixel_policy``
    is ``"reject"`` | ``"clip"`` | ``"allow"`` as in ``load_tiff_stack``.
    """
    ...

def read_tof_sidecar(
    path: str,
    n_frames: int | None = None,
) -> NDArray[np.float64]:
    """Read a VENUS ``*_Spectra.txt`` sidecar into TOF bin edges (µs).

    Column 0 holds each frame's start time in seconds; the result is the
    N+1 ascending microsecond edges expected by ``tof_to_energy_centers``
    (the closing edge extrapolates the last frame width).  When
    ``n_frames`` is given, the edge count is validated against it.
    """
    ...

def probe_nexus(path: str) -> NexusMetadata:
    """Probe a NeXus/HDF5 file for available data without loading it."""
    ...

def run_health(
    path: str,
    pause_pv: str = "pause",
    power_pv: str = "proton_charge",
    power_dip_fraction: float = 0.5,
) -> RunHealth:
    """Compute a run-health summary from ``/entry/DASlogs``.

    DASlogs PVs log transitions, so statistics use last-value-held
    time-weighted integration over the run window, never entry means.
    The PV-name defaults are the SNS ones; other facilities pass their
    own.  Absent PVs yield ``None`` fields, not errors.
    """
    ...

def load_nexus_histogram(
    path: str,
    multi_angle_mode: str = "error",
    angle_index: int = 0,
) -> NexusData:
    """Load pre-histogrammed counts from a NeXus/HDF5 file.

    Reads ``/entry/histogram/counts`` (4D: ``rot × y × x × tof``) and
    returns a ``NexusData`` with shape ``(n_tof, height, width)``.

    **Issue #430**: by default this function refuses multi-angle files
    (``n_rot > 1``).  Silently summing over rotation angles at load
    time destroys projection-resolved information; callers must
    choose explicitly via ``multi_angle_mode``:

    - ``"error"`` (default): reject multi-angle files with a clear
      ``ValueError``.
    - ``"sum"``: sum over all rotation angles into a single
      ``(tof, y, x)`` volume (legacy behaviour, now opt-in).
    - ``"select"``: extract a single projection at ``angle_index``.
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

    Pixel masks are a pipeline-integrity screen only — they exclude pixels
    whose data stream is broken (dead, hot/railed), never low-count or
    poorly covered pixels.  Low-count pixels are alive and must be kept.
    Prefer ``detect_bad_pixels()`` (validating, unions sample and open-beam,
    optional hot screen); see also ``detect_hot_pixels()`` and
    ``detect_dead_pixels_chunked()``.

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

def detect_hot_pixels(
    data: NDArray[np.float64],
    k_mad: float = 6.0,
) -> NDArray[np.bool_]:
    """Detect hot (railed / runaway) pixels — two-stage screen.

    Stage 1 (global): robust one-sided cut on per-pixel total counts — a
    pixel is a candidate when ``ln(total) > median + k_mad * sigma``, with
    median and MAD taken over the live (``total > 0``) pixels only and
    ``sigma`` floored by the Poisson counting noise of the median total.
    Stage 2 (local), iterated to a fixpoint: a candidate is flagged only if
    its total also exceeds 10x the median of its 8-neighborhood reference
    sample — live unflagged neighbors contribute their totals,
    already-flagged neighbors contribute 0 (a known defect cannot vouch for
    its neighbors), dead neighbors are omitted; edge pixels use whatever
    neighbors exist, and a candidate with no live neighbor keeps the global
    verdict.  Passes repeat until no new flag is added (bounded by
    ``height * width`` passes; in practice ~the defect-cluster radius),
    eroding railed CLUSTERS from the boundary inward — a single pass would
    miss the interior of clusters >= 2 px wide, whose neighbors are railed
    too.  Clusters up to 3 px wide are fully consumed PROVIDED they expose
    at least one end cap or convex corner to normal-scene neighbors
    (erosion must seed somewhere): an EDGE-TO-EDGE railed band >= 2 px
    wide, spanning the full detector width or height with both ends
    off-detector, has no seed and is NOT caught — deliberately, because a
    slit-aperture open beam produces a genuine full-width bright scene
    band pixel-for-pixel indistinguishable from it, and a full-span screen
    would mask that scene (the bimodal failure).  Declare such full-span
    detector pathologies in a file mask.  A full-span width-1 railed line
    IS caught (each pixel keeps >= 4 normal neighbors).  The local
    confirmation keeps bimodal scenes honest: with a dark majority (a
    sample covering >50% of the field of view, or an aperture-limited open
    beam), the global statistics describe only the dark population and the
    entire bright region would otherwise be masked — a contiguous bright
    region is scene, not a defect.  Upper tail only — stuck-low pixels are
    indistinguishable from low-count-alive pixels and are kept (masks are
    pipeline-integrity only, never a low-count screen).

    Bright SCENE regions never erode: a boundary pixel of a contiguous
    bright region >= 2 px wide keeps >= 4 same-side neighbors for any
    straight or diagonal edge, so its reference median stays bright and
    scene gradients (<= 2-3x across real edges) never reach the 10x factor
    — the erosion has no seed.  Documented width-1 limitation (accepted
    trade-off): a 1-px-wide bright scene line at >= 10x local contrast is
    spatially indistinguishable from a railed line and IS masked; real
    scene features on VENUS are PSF-blurred over >= 2 px, so >= 10x
    single-pixel scene contrast is physically rare, and contiguous bright
    regions of width >= 2 are safe from the local stage.

    ``data`` must be RAW detected counts (unscaled): the Poisson floor
    assumes ``Var[N] = N``.  Scaled inputs silently distort the floor —
    down-scaling (proton-charge-normalized rates << 1, gain division)
    inflates it and can suppress real flags; up-scaling (event weights > 1)
    deflates it below true counting noise.  Detect on raw counts and
    normalize afterwards.

    Parameters
    ----------
    data :
        3D NumPy array of raw counts with shape ``(n_frames, height, width)``.
    k_mad :
        Robust-sigma multiplier for the stage-1 upper-tail cut. The default
        6.0 corresponds to a one-sided Gaussian tail of ~1e-9 — on a
        unimodal image it essentially never flags a statistically plausible
        pixel.

    Returns
    -------
    NDArray[np.bool_]
        2D boolean mask with shape ``(height, width)``, where ``True`` marks
        a hot pixel.

    Raises
    ------
    ValueError
        If ``data`` contains non-finite or negative values or has zero
        frames (``shape[0] == 0``), or ``k_mad`` is not finite and positive.
    """
    ...

def detect_dead_pixels_chunked(
    chunks: list[NDArray[np.float64]],
) -> NDArray[np.bool_]:
    """Detect dead pixels across acquisition chunks (dead in ANY chunk).

    Catches intermittent deadness invisible to ``detect_dead_pixels()`` on
    the summed stack: a pixel dead for one acquisition chunk but alive in
    another has nonzero summed counts, yet its dead-chunk data corrupts the
    combined spectrum.  Chunk the acquisition so each live pixel has an
    expected >= 20 total counts per chunk (misflag probability per live
    pixel is ``m * exp(-lambda)`` over ``m`` chunks).

    Parameters
    ----------
    chunks :
        List of 3D NumPy arrays, one per acquisition chunk, each with shape
        ``(n_frames_i, height, width)``.  Frame counts may differ between
        chunks (ragged event re-histogramming is fine); spatial dimensions
        must agree.

    Returns
    -------
    NDArray[np.bool_]
        2D boolean mask with shape ``(height, width)``, where ``True`` marks
        a pixel that is all-zero in at least one chunk.

    Raises
    ------
    ValueError
        If ``chunks`` is empty, any chunk has zero frames
        (``shape[0] == 0`` — its all-zero test would vacuously mark every
        pixel dead), any chunk contains non-finite or negative values, or
        the spatial dimensions differ between chunks.
    """
    ...

def detect_bad_pixels(
    sample: NDArray[np.float64],
    open_beam: NDArray[np.float64] | None = None,
    hot_k_mad: float | None = 6.0,
) -> NDArray[np.bool_]:
    """Detect all pipeline-corrupting pixels: dead + hot, sample and OB.

    The validating entry point.  Deadness/hotness is per-acquisition — a
    pixel dead only in the open-beam run still corrupts every transmission
    ratio computed from it — so the masks of both stacks are unioned:
    ``dead(sample) | hot(sample) [| dead(open_beam) | hot(open_beam)]``.
    Frame counts may differ between the stacks; spatial dimensions must
    agree.  The result can be passed to ``spatial_map_typed(dead_pixels=...)``.

    Both stacks must be RAW detected counts (unscaled) — see
    ``detect_hot_pixels()``: scaling distorts the Poisson floor of the hot
    screen.  Detect on raw counts, before any normalization.

    Parameters
    ----------
    sample :
        3D NumPy array of raw counts with shape ``(n_frames, height, width)``.
    open_beam :
        Optional 3D NumPy array of raw counts with shape
        ``(n_frames2, height, width)``.
    hot_k_mad :
        Robust-sigma multiplier for the hot-pixel screen (default 6.0), or
        ``None`` to disable it (dead-only detection).

    Returns
    -------
    NDArray[np.bool_]
        2D boolean mask with shape ``(height, width)``, where ``True`` marks
        a pixel to exclude from fitting.

    Raises
    ------
    ValueError
        If either stack contains non-finite or negative values or has zero
        frames (``shape[0] == 0``), the spatial dimensions differ, or
        ``hot_k_mad`` is not finite and positive.
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
    resolution: TabulatedResolution | None = None,
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


def from_counts_with_nuisance(
    sample_counts: NDArray[np.float64],
    flux: NDArray[np.float64],
    background: NDArray[np.float64],
) -> InputData:
    """Create InputData from raw detector counts plus explicit nuisance spectra.

    Use this when the detector/counts background spectrum has been
    estimated outside NEREIDS and should be supplied explicitly
    alongside the open-beam flux.  Routes through the counts-KL
    (joint-Poisson) dispatch when passed to ``spatial_map_typed``
    with ``solver="auto"`` / ``"kl"``.  (Per-spectrum counts fitting
    uses ``fit_counts_spectrum_typed``, which takes the raw 1D
    ``sample_counts`` / ``open_beam_counts`` / ``detector_background``
    arrays directly rather than an ``InputData`` wrapper.)

    Args:
        sample_counts: 3D float64 array (n_energies, height, width).
        flux: 3D float64 array (n_energies, height, width) of open-beam flux.
        background: 3D float64 array (n_energies, height, width) of
            detector/counts background.

    Returns:
        InputData object to pass to ``spatial_map_typed()``.
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
    fix_densities: bool = False,
    density_free: list[bool] | None = None,
    dead_pixels: NDArray[np.bool_] | None = None,
    max_iter: int = 200,
    solver: str = "auto",
    background: bool = False,
    fit_back_d: bool = False,
    fit_back_f: bool = False,
    back_d_init: float = 0.01,
    back_f_init: float = 1.0,
    fit_alpha_1: bool = False,
    fit_alpha_2: bool = False,
    alpha_1_init: float = 1.0,
    alpha_2_init: float = 1.0,
    c: float = 1.0,
    enable_polish: bool | None = None,
    fit_energy_scale: bool = False,
    t0_init_us: float = 0.0,
    l_scale_init: float = 1.0,
    energy_scale_flight_path_m: float = 25.0,
    resolution: TabulatedResolution | None = None,
    flight_path_m: float | None = None,
    delta_t_us: float | None = None,
    delta_l_m: float | None = None,
    groups: list[IsotopeGroup] | None = None,
    tzero_jacobian: str | None = None,
    fit_energy_range: tuple[float, float] | None = None,
) -> SpatialResult:
    """Spatial mapping using the typed input data API.

    Either ``isotopes`` or ``groups`` must be provided, but not both.
    When ``groups`` is provided, each group maps to one fitted density parameter.

    Dispatches per-pixel fitting based on InputData type:
      - from_counts / from_counts_with_nuisance + solver="kl" / "auto"
        -> counts-KL (joint-Poisson deviance) — the counts-path solver,
        validated against synthetic counts benchmarks and locked by a
        real-VENUS counts regression test on the committed aggregated-Hf
        fixture.
      - from_transmission + solver="lm" (default for transmission) -> LM.
      - from_transmission + solver="kl" -> Poisson NLL on transmission values
        (legacy niche).

    Args:
        data: InputData from `from_counts()`, `from_counts_with_nuisance()`,
            or `from_transmission()`.
        fix_densities: Freeze all densities at their initial values across
            every pixel (per-pixel temperature-only fits with a known
            calibration-foil density — the fastest thermometry path).
        density_free: Per-density free/fixed mask applied to every pixel
            (``free[i] == False`` freezes density ``i``); length must equal
            the number of density parameters. Mutually exclusive with
            ``fix_densities``.
        fit_back_d, fit_back_f: When ``background=True``, fit the
            optional SAMMY exponential background tail
            ``BackD * exp(-BackF / √E)``.  SAMMY pairs the two — fitting
            only one is rejected.  Materialises
            :py:attr:`SpatialResult.back_d_map` /
            :py:attr:`SpatialResult.back_f_map` per-pixel.
        back_d_init, back_f_init: Initial values for the exponential
            tail.  Both must be strictly positive when their fit flags
            are set (BackF's Jacobian column zeros out at BackD ≈ 0,
            and BackD becomes a constant duplicate of BackA at
            BackF ≈ 0).
        c: Proton-charge ratio ``Q_s / Q_ob`` for the counts-KL dispatch.
            Default 1.0 (assumes caller PC-normalized
            the flux already).  Ignored for LM / transmission-KL paths.
        enable_polish: Override the Nelder-Mead polish flag.  ``None``
            (default) = the dispatcher auto-disables polish when
            ``n_pixels > 1`` (polish costs ~1000 s per pixel
            on realistic data).  ``True`` forces polish on, ``False`` off.
        fit_energy_scale: Fit per-pixel SAMMY TZERO calibration
            (t0 and L_scale).  Required for real VENUS data to match SAMMY
            chi-squared performance — without it, sharp resonances are
            offset in TOF and per-pixel chi-squared explodes.
        t0_init_us: Initial TOF offset in microseconds (default 0.0).
        l_scale_init: Initial flight-path scale factor (default 1.0).
        energy_scale_flight_path_m: Nominal flight path (m) for the
            energy-scale model (default 25.0).

    Always returns SpatialResult.  For counts-KL runs,
    ``SpatialResult.deviance_per_dof_map`` is populated as the primary GOF.
    When ``fit_energy_scale=True``, per-pixel ``t0_us_map`` and
    ``l_scale_map`` are populated on the returned SpatialResult.  When
    ``background=True`` with ``fit_back_d=True`` / ``fit_back_f=True``,
    per-pixel ``back_d_map`` / ``back_f_map`` are populated.
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
    fix_densities: bool = False,
    density_free: list[bool] | None = None,
    tzero_jacobian: str | None = None,
    fit_energy_range: tuple[float, float] | None = None,
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
        fix_densities: Freeze all densities at their initial values and fit
            only the remaining free parameters (temperature / energy scale).
            The standard resonance-thermometry workflow when the areal density
            is known from a calibration foil.
        density_free: Per-density free/fixed mask (``free[i] == False`` freezes
            density ``i``); length must equal the number of density parameters.
            Mutually exclusive with ``fix_densities``.
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
    fix_densities: bool = False,
    density_free: list[bool] | None = None,
    enable_polish: bool | None = None,
    tzero_jacobian: str | None = None,
    fit_energy_range: tuple[float, float] | None = None,
) -> FitResult:
    """Fit a single raw-count spectrum (sample + open-beam counts).

    Dispatches to a counts-domain solver based on ``solver``:

    - ``'auto'`` (default), ``'kl'``, ``'poisson'``, and ``'joint_poisson'``
      all route to the **counts-KL dispatch**: the joint-Poisson profile
      binomial-deviance fitter.  Uses the explicit proton-charge ratio
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
            (for LM-converted path only; counts-KL rejects non-zero values).
        fit_alpha_1: Research-only; rejected by the counts-KL dispatch
            because the profile λ̂ absorbs the global flux scale.
        fit_alpha_2: Research-only; rejected by the counts-KL dispatch
            (non-zero detector background not currently wired).
        alpha_1_init: Initial value for alpha_1 (default 1.0); only
            consumed by the research Fisher helper.
        alpha_2_init: Initial value for alpha_2 (default 1.0); same.
        c: Proton-charge ratio ``Q_s / Q_ob``.  Default
            1.0 assumes the caller has already PC-normalized the flux.
            For raw VENUS-style counts, set this to the actual ratio
            (typically ~5–6).  Used by the counts-KL dispatch; ignored
            by the LM path.
        resolution: Optional resolution function.
        groups: List of IsotopeGroup objects (mutually exclusive with isotopes).
        initial_densities: Initial density guesses when using groups.
        fix_densities: Freeze all densities at their initial values and fit
            only the remaining free parameters (temperature / energy scale).
            The standard resonance-thermometry workflow when the areal density
            is known from a calibration foil.
        density_free: Per-density free/fixed mask (``free[i] == False`` freezes
            density ``i``); length must equal the number of density parameters.
            Mutually exclusive with ``fix_densities``.
        enable_polish: Override the Nelder-Mead polish flag for the
            counts-KL solver.  ``None`` (default) falls through to the
            library default — currently ``False`` (#486) because the
            polish ``fatol = 1e-10`` is sub-ULP on real counts data
            where ``D`` saturates at ``10⁴``-``10⁵`` and burns
            ``max_iter = 5000`` for ≤ 0.35 Fisher σ parameter
            movement.  Pass ``True`` to opt in for synthetic / clean
            (``D ≈ 1``) regimes where the absolute tolerance is
            physically meaningful; ``False`` to force off explicitly.
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
