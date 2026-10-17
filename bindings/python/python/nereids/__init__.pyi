"""Type stubs for the NEREIDS Python bindings (PEP 561)."""

from __future__ import annotations

from typing import Any, Literal, overload

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
    """DASlogs-based run-health summary (``None`` = cannot compute:
    PV absent or empty, or the dip threshold is undefined)."""

    @property
    def pause_fraction(self) -> float | None:
        """Time-weighted fraction of the run spent paused."""
        ...
    @property
    def beam_dip_fraction(self) -> float | None:
        """Time-weighted fraction with power below the dip threshold.

        ``None`` when the power PV is absent or empty, or when the dip
        threshold is undefined because the sample median of the power
        entries is non-positive (e.g. beam off for at least half the
        entries) — check ``median_power``, which is co-reported.
        """
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
        """Total number of discrete resonances across all evaluable ranges.

        Counts LRF=1/2/3 L-group resonances only. LRF=7 (R-Matrix Limited)
        and LRU=2 (URR) ranges are parsed-and-skipped — not evaluated — so
        they contribute 0. A file with NO evaluable range at all fails to
        load with a ValueError instead of reporting 0 here.
        """
        ...

    @property
    def total_resonance_count(self) -> int:
        """Total resonance count across all ranges.

        Explicit alias for :attr:`n_resonances`; both delegate to the Rust
        ``ResonanceData::total_resonance_count()`` (skipped LRF=7 / LRU=2
        ranges contribute 0).
        """
        ...

    @property
    def has_unevaluated_ranges(self) -> bool:
        """Whether the evaluation carries parsed-but-not-evaluated ranges.

        True when any range is a parse-and-skip placeholder (LRF=7 R-Matrix
        Limited, LRU=2 URR, or LRU=0 scattering-radius-only). Those spans
        contribute zero cross-section; a ``UserWarning`` listing them is
        emitted at load time by :func:`load_endf` / :func:`load_endf_file`.
        """
        ...

    @property
    def skipped_ranges(self) -> list[str]:
        """One-line descriptions of the parse-and-skip placeholder ranges.

        One string per non-evaluable range (LRF=7, LRU=2, or LRU=0), using the
        same text as the load-time ``UserWarning`` and the parser's rejection
        message. Empty when :attr:`has_unevaluated_ranges` is ``False``.
        """
        ...

    @property
    def target_spin(self) -> float:
        """Target nuclear spin (I) of the first evaluable resonance range.

        Parse-and-skip placeholder ranges (LRU=0 / LRF=7 / URR) are skipped.
        """
        ...

    @property
    def scattering_radius(self) -> float:
        """Effective scattering radius in fm, from the first evaluable range.

        Parse-and-skip placeholder ranges (LRU=0 / LRF=7 / URR) are skipped.
        """
        ...

    @property
    def l_values(self) -> list[int]:
        """Orbital angular momentum values (L) present in the data."""
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

    def detector_bin_probabilities(
        self,
        true_energy_ev: float,
        detector_time_edges_us: list[float],
        timing_offset_us: float,
    ) -> list[float]:
        """Probability in each adjacent detector-time bin.

        The loaded tabulated response is evaluated at ``true_energy_ev``.
        Probability outside the supplied time window is not renormalized into
        the window.
        """
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
    (``forward_model``, ``fit_counts``, ``fit_counts``).
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
        beta_law: EnergyLaw | None = None,
    ) -> None:
        """When ``beta_law`` is supplied it overrides the scalar ``beta``
        (which is still required positionally but otherwise ignored).
        ``detector_bin_probabilities`` refuses a nonzero ``burst_sigma_us``."""
        ...
    def as_tabulated(self) -> TabulatedResolution:
        """The synthesized tabulated kernel (usable as a resolution file)."""
        ...

    def kernel_at(self, energy_ev: float) -> tuple[list[float], list[float]]:
        """``(tof_offsets_us, weights)`` at one energy, from the moderator
        origin. A symmetric fold reaches before it, so offsets can be negative.

        Raises ``ValueError`` when the tau-grid cannot resolve the prompt
        core and requested folds within the sample cap at this energy, or
        when a parameter law is invalid at this probe energy (non-positive
        or singular rate, storage fraction outside [0, 1]).
        """
        ...

    def source_pulse_at(self, true_energy_ev: float) -> tuple[list[float], list[float]]:
        """Physical ``(moderator_delay_us, density)`` at one true energy.

        The same synthesis and the same origin as :meth:`kernel_at`. The
        density is a peak-normalized shape (maximum 1), not a probability
        density. Raises ``ValueError`` for a nonphysical probe energy or a
        parameter law that is invalid there.
        """
        ...

    def detector_bin_probabilities(
        self,
        true_energy_ev: float,
        detector_time_edges_us: list[float],
        timing_offset_us: float,
    ) -> list[float]:
        """Probability in each detector-time bin for one true energy.

        The probabilities are not renormalized when the supplied time window
        omits part of the pulse. A Gaussian burst is refused with ValueError.
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

        Raises:
            ConnectionError, OSError, RuntimeError, ValueError: as
                ``load_endf``; an error loading a member starts with that
                member's Z and A.
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
    """Load ENDF resonance data for an isotope, downloading the evaluation on
    first use and caching it in the user's cache directory.

    Mixed evaluations load with a ``UserWarning`` naming the
    parsed-but-not-evaluated spans.

    Raises:
        ConnectionError: no download site could be reached, or one answered
            with an HTTP error other than 404 or refused the download.
        OSError: the cache file named in the message could not be read or
            written.
        RuntimeError: the isotope is absent from the library (HTTP 404), or
            the download is not a valid archive.
        ValueError: an unknown library, an invalid isotope or MAT, a file for
            another isotope, or ENDF text that does not parse or has no
            evaluable resolved (LRF=1/2/3) range.
    """
    ...

def load_endf_file(path: str) -> ResonanceData:
    """Load ENDF resonance data from a local file.

    Raises ``ValueError`` when the file has no evaluable resolved
    (LRF=1/2/3) range — loading it would yield zero cross-section
    everywhere. Mixed evaluations load with a ``UserWarning`` naming the
    parsed-but-not-evaluated spans.
    """
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
    """Apply resolution broadening (Gaussian, or Gaussian+exponential tail) to a cross-section or spectrum array.

    ``delta_t_us`` and ``delta_l_m`` are W-parameters, the width in
    ``exp(-x^2/W^2)``: ``sigma = W/sqrt(2)`` and ``FWHM = 1.6651*W``. Convert a
    measured width with :func:`width_from_sigma` or :func:`width_from_fwhm`
    rather than passing it unconverted.
    """
    ...

def width_from_sigma(sigma: float) -> float:
    """Convert a standard deviation to the W-parameter this API expects (``W = sigma*sqrt(2)``)."""
    ...

def width_from_fwhm(fwhm: float) -> float:
    """Convert a full width at half maximum to the W-parameter this API expects (``W = FWHM/(2*sqrt(ln 2))``)."""
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

    Raises ``FileNotFoundError`` when ``path`` does not exist,
    ``ValueError`` for bad pixel values or an invalid policy, and
    ``OSError`` for other I/O failures (e.g. permission denied).
    """
    ...

@overload
def load_tiff_folder(
    folder: str,
    pattern: str | None = None,
    sum_chunks: bool = True,
    pixel_policy: str = "reject",
    *,
    return_info: Literal[False] = False,
) -> NDArray[np.float64]:
    """Load a folder of single-frame TIFFs into a 3D numpy array.

    Chunked VENUS folders (``<prefix>_<chunk>_<frame>.tif``) are detected
    automatically and summed element-wise across chunks by default
    (``sum_chunks=True``, with a ``UserWarning`` naming the chunks); ragged
    or duplicated chunks are an error on the summing path, but with
    ``sum_chunks=False`` they load as the lexicographic concatenation and
    are flagged via ``chunk_inconsistent`` (with a ``UserWarning``) instead
    of raising.  ``sum_chunks`` otherwise only affects folders with two
    or more chunks — a *consistent* single-chunk (or non-chunk-patterned)
    folder loads identically either way; an internally inconsistent one
    (e.g. an accidental ``.tif``/``.tiff`` duplicate) still raises under
    ``sum_chunks=True`` but concatenates under ``sum_chunks=False``.  A
    *mixed* folder (chunk-patterned names
    alongside non-conforming files) loads lexicographically with a
    ``UserWarning`` counting the files that disabled chunk detection.  The
    heuristic assumes one acquisition per folder; use ``pattern`` (or
    ``sum_chunks=False``) when a folder may hold multiple same-prefix
    runs.  ``pixel_policy`` is ``"reject"`` | ``"clip"`` | ``"allow"`` as
    in ``load_tiff_stack``.  With ``return_info=True`` (keyword-only),
    returns ``(array, info)`` where ``info`` has keys ``n_files``,
    ``n_chunks``, ``chunk_ids``, ``chunks_summed``, ``n_clipped_pixels``,
    ``n_unrecognized_files`` (files that broke chunk detection in a mixed
    folder; 0 otherwise), ``unrecognized_examples`` (up to three of their
    names), and ``chunk_inconsistent`` (``True`` when inconsistent chunks
    were concatenated under ``sum_chunks=False`` instead of raising).
    """
    ...

@overload
def load_tiff_folder(
    folder: str,
    pattern: str | None = None,
    sum_chunks: bool = True,
    pixel_policy: str = "reject",
    *,
    return_info: Literal[True],
) -> tuple[NDArray[np.float64], dict[str, Any]]: ...

def read_tof_sidecar(
    path: str,
    n_frames: int | None = None,
) -> NDArray[np.float64]:
    """Read a VENUS ``*_Spectra.txt`` sidecar into TOF bin edges (µs).

    Column 0 holds each frame's start time in seconds — the frame START
    (left bin edge; verified on measured VENUS autoreduce output, where
    every value is an exact integer multiple of the bin width).  The
    result is the N+1 ascending microsecond edges expected by
    ``tof_to_energy_centers`` (the closing edge extrapolates the last
    frame width).  When ``n_frames`` is given, the edge count is
    validated against it.  If the first edge is exactly 0, crop the
    first frame from both stack and edges (``stack[1:]``, ``edges[1:]``)
    before energy conversion.

    Raises ``FileNotFoundError`` only when the file genuinely does not
    exist; other open failures (e.g. permission denied) raise
    ``OSError``, and malformed content raises ``ValueError``.
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
    own.  Absent PVs and present-but-empty PVs (zero entries logged)
    yield ``None`` fields, not errors; ``beam_dip_fraction`` is also
    ``None`` when the median power is non-positive (dip threshold
    undefined — see ``RunHealth.beam_dip_fraction``).
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

class RunLog:
    """A slow-control PV from ``/entry/DASlogs/<pv>`` as a transition log.

    ``values[i]`` takes effect at ``times[i]`` and persists until the next
    entry (the last value persists to ``duration_s``).  Do NOT average
    ``values`` directly — use :func:`intervals_where` (issue #637).
    """

    @property
    def times(self) -> NDArray[np.float64]: ...
    @property
    def values(self) -> NDArray[np.float64]: ...
    @property
    def duration_s(self) -> float: ...
    @property
    def offset_iso(self) -> str | None: ...
    @property
    def n_dropped_corrupt(self) -> int: ...

class BankSpectrum:
    """1-D TOF spectrum from one NXevent_data bank plus retention stats."""

    @property
    def tof_edges_us(self) -> NDArray[np.float64]: ...
    @property
    def counts(self) -> NDArray[np.uint64]: ...
    @property
    def pulses_total(self) -> int: ...
    @property
    def pulses_kept(self) -> int: ...
    @property
    def events_total(self) -> int: ...
    @property
    def events_kept(self) -> int: ...
    @property
    def dropped_tof_range(self) -> int: ...
    @property
    def dropped_non_finite(self) -> int: ...
    @property
    def pulse_time_offset_iso(self) -> str | None: ...

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

def read_run_log(path: str, pv: str) -> RunLog:
    """Read a DASlogs PV from a NeXus file as a transition log (issue #637).

    SNS DASlogs record TRANSITIONS, not uniform samples: on a real VENUS
    run the entry-mean of the ``pause`` log read 0.43 while the
    time-weighted truth was 0.90.  Derive intervals with
    :func:`intervals_where` instead of averaging ``values``.
    """
    ...

def intervals_where(
    times: NDArray[np.float64] | list[float],
    values: NDArray[np.float64] | list[float],
    duration_s: float,
    min_value: float | None = None,
    max_value: float | None = None,
) -> list[tuple[float, float]]:
    """Intervals where ``min_value <= value <= max_value`` under correct
    step-function semantics (issue #637).  Time before the first log entry
    never matches; NaN never matches; adjacent segments merge.
    """
    ...

def intervals_intersect(
    a: list[tuple[float, float]],
    b: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Intersect two interval lists, e.g. pause==0 AND power>1.5 MW.

    Inputs may be unsorted/overlapping (normalised by sort+merge first);
    every pair must be finite with t_end > t_start.
    """
    ...

def load_nexus_bank_spectrum(
    path: str,
    bank: str,
    n_bins: int,
    tof_min_us: float,
    tof_max_us: float,
    keep_intervals: list[tuple[float, float]] | None = None,
) -> BankSpectrum:
    """Load one NXevent_data bank (e.g. ``"monitor1"``) as a 1-D TOF
    spectrum, keeping only pulses inside ``keep_intervals`` (seconds on the
    pulse clock = DASlogs clock at SNS).  Empty banks load gracefully to a
    zero spectrum; ``units`` attrs are required on both event datasets
    (issue #637).
    """
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
    agree.  The result can be passed to ``fit_map(dead_pixels=...)``.

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


# ---------------------------------------------------------------------------
# Counts-domain fitting
# ---------------------------------------------------------------------------

class Value:
    """A quantity the fit holds known, fits from a start, fits within bounds,
    or fits under a Gaussian prior."""

    @staticmethod
    def known(value: float) -> Value:
        """Hold the quantity at the given value."""
        ...
    @staticmethod
    def fitted(start: float) -> Value:
        """Fit the quantity from the given start over its full range."""
        ...
    @staticmethod
    def within(start: float, lower: float, upper: float) -> Value:
        """Fit the quantity from start within lower..=upper."""
        ...
    @staticmethod
    def measured(value: float, sd: float) -> Value:
        """Fit the quantity under the Gaussian prior N(value, sd^2)."""
        ...
    @property
    def kind(self) -> Literal["known", "fitted", "within", "measured"]: ...
    @property
    def value(self) -> float | None: ...
    @property
    def start(self) -> float | None: ...
    @property
    def lower(self) -> float | None: ...
    @property
    def upper(self) -> float | None: ...
    @property
    def sd(self) -> float | None: ...

class PhononSpectrum:
    """A crystal's phonon density of states in eV: values every step_ev from
    0, the first 0, quadratic in the energy up to the first point and linear
    between the others."""

    def __init__(self, step_ev: float, density: list[float]) -> None: ...
    @staticmethod
    def debye(theta_k: float, points: int = 1000) -> PhononSpectrum: ...
    @property
    def max_ev(self) -> float: ...
    def effective_temperature_k(self, temperature_k: float) -> float: ...

class Material:
    """A material isotopes, each with an areal density in atoms/barn, and its
    temperature in K.  The library names the ENDF library the isotopes were
    read from; thickness_variance is the relative variance of the areal
    density over the region's pixels, known 0 by default; lattice is the
    PhononSpectrum the nuclei vibrate with, which makes temperature_k the
    thermodynamic temperature, or None for the free gas."""

    def __init__(
        self,
        isotopes: list[tuple[ResonanceData, Value]],
        temperature_k: Value,
        library: str = "endf8.1",
        thickness_variance: Value | None = None,
        lattice: PhononSpectrum | None = None,
    ) -> None: ...
    @property
    def isotopes(self) -> list[tuple[ResonanceData, Value]]: ...
    @property
    def temperature_k(self) -> Value: ...
    @property
    def thickness_variance(self) -> Value: ...
    @property
    def lattice(self) -> PhononSpectrum | None: ...

class Region:
    """The counts of a set of pixels in both runs, its background and the
    material the beam crosses there.

    The background is [b0, b1, b2] or [b0, b1, b2, d, f] of
    b(E) = b0 + b1/sqrt(E) + b2*sqrt(E) + d*exp(-f/sqrt(E));
    pixels is the region area in pixels; capture_gammas is the share of the
    sample's captures whose gammas the region records, in_scatter the share of
    its forward single scatters the region records, and timing_offset_us how
    much later than t0's clock the region records every count, in
    microseconds, each known 0 by default.
    """

    def __init__(
        self,
        open_counts: NDArray[np.float64],
        sample_counts: NDArray[np.float64],
        pixels: float,
        background: list[Value],
        gamma_transmission: Value,
        material: Material | None = None,
        open_live: NDArray[np.float64] | None = None,
        sample_live: NDArray[np.float64] | None = None,
        capture_gammas: Value | None = None,
        in_scatter: Value | None = None,
        timing_offset_us: Value | None = None,
    ) -> None: ...
    @property
    def capture_gammas(self) -> Value: ...
    @property
    def in_scatter(self) -> Value: ...
    @property
    def timing_offset_us(self) -> Value: ...
    @property
    def open_counts(self) -> NDArray[np.float64]: ...
    @property
    def sample_counts(self) -> NDArray[np.float64]: ...
    @property
    def open_live(self) -> NDArray[np.float64] | None: ...
    @property
    def sample_live(self) -> NDArray[np.float64] | None: ...
    @property
    def background(self) -> list[Value]: ...
    @property
    def pixels(self) -> float: ...
    @property
    def gamma_transmission(self) -> Value: ...
    @property
    def material(self) -> Material | None: ...

class RoomBackground:
    """Counts that follow each run's duration, not its proton charge.

    rate is per microsecond per pixel per frame; frames are those of the
    open-beam run and the sample run.
    """

    def __init__(self, rate: Value, frames: tuple[float, float]) -> None: ...
    @property
    def rate(self) -> Value: ...
    @property
    def frames(self) -> tuple[float, float]: ...

class Spread:
    """Counts the detector records in another region than the one they arrive in.

    share is h; kernel holds (weight, length in cells) terms of the radial
    kernel sum(w*exp(-r/s)/(2*pi*s**2)); grid is (rows, columns); cells
    holds each region's cells as (row, column), in the order of the regions.
    """

    def __init__(
        self,
        share: Value,
        kernel: list[tuple[float, float]],
        grid: tuple[int, int],
        cells: list[list[tuple[int, int]]],
    ) -> None: ...
    @property
    def share(self) -> Value: ...
    @property
    def kernel(self) -> list[tuple[float, float]]: ...

class Measurement:
    """An open-beam run and a sample run recorded in the same time bins.

    time_edges_us are the time-bin edges in microseconds and charge_ratio is
    the sample run proton charge over the open-beam run.  normalization is the
    sample run normalization a.  time_background is [C, A, tau] or
    NO_TIME_BACKGROUND; room_background is a RoomBackground or None; spread
    is a Spread or None.
    """

    def __init__(
        self,
        time_edges_us: NDArray[np.float64],
        charge_ratio: float,
        normalization: Value,
        regions: list[Region],
        time_background: list[Value] | None = None,
        room_background: RoomBackground | None = None,
        spread: Spread | None = None,
    ) -> None: ...
    @property
    def time_edges_us(self) -> NDArray[np.float64]: ...
    @property
    def charge_ratio(self) -> float: ...
    @property
    def normalization(self) -> Value: ...
    @property
    def time_background(self) -> list[Value]: ...
    @property
    def room_background(self) -> RoomBackground | None: ...
    @property
    def spread(self) -> Spread | None: ...
    @property
    def regions(self) -> list[Region]: ...

class Pulse:
    """The Ikeda-Carpenter pulse: alpha = alpha0*sqrt(E) + alpha1 and
    beta = beta0*sqrt(E) + beta1 in 1/microsecond, a storage fraction
    R over energy_span_ev (eV), or linear in sqrt(E) from R at its low end to
    r_high at its high end, triangle FWHM h with fwhm_squared_us2 equal to h^2
    in microseconds squared, and an optional detector tail (f, tau_D): the detector records a
    fraction f of the counts late by an exponential delay of mean tau_D
    microseconds."""

    def __init__(
        self,
        alpha0: Value,
        alpha1: Value,
        beta0: Value,
        beta1: Value,
        r: Value,
        fwhm_squared_us2: Value,
        energy_span_ev: tuple[float, float],
        n_tau: int = 600,
        line_span_ev: tuple[float, float] | None = None,
        r_high: Value | None = None,
        detector_tail: tuple[Value, Value] | None = None,
    ) -> None: ...
    @property
    def alpha(self) -> list[Value]: ...
    @property
    def beta(self) -> list[Value]: ...
    @property
    def r(self) -> Value: ...
    @property
    def fwhm_squared_us2(self) -> Value: ...
    @property
    def r_high(self) -> Value | None: ...
    @property
    def detector_tail(self) -> tuple[Value, Value] | None: ...
    @property
    def energy_span_ev(self) -> tuple[float, float]: ...
    @property
    def n_tau(self) -> int: ...

class Calibration:
    """A neutron of flight time u over flight_path_m (m) arrives at
    t0_us + u (microseconds) plus a delay drawn from pulse."""

    def __init__(self, t0_us: Value, flight_path_m: Value, pulse: Pulse) -> None: ...
    @property
    def t0_us(self) -> Value: ...
    @property
    def flight_path_m(self) -> Value: ...
    @property
    def pulse(self) -> Pulse: ...

class RegionFit:
    """One region fitted quantities and counts."""

    @property
    def densities(self) -> list[float]:
        """Areal density of each isotope in atoms/barn."""
        ...
    @property
    def temperature_k(self) -> float | None: ...
    @property
    def thickness_variance(self) -> float | None: ...
    @property
    def capture_gammas(self) -> float: ...
    @property
    def in_scatter(self) -> float: ...
    @property
    def timing_offset_us(self) -> float: ...
    @property
    def background(self) -> list[float]: ...
    @property
    def gamma_transmission(self) -> float: ...
    @property
    def beam_at_limit(self) -> bool: ...
    @property
    def predicted_open(self) -> NDArray[np.float64]: ...
    @property
    def predicted_sample(self) -> NDArray[np.float64]: ...
    @property
    def overdispersion(self) -> list[float | None]: ...
    @property
    def excluded_bins(self) -> list[int]: ...

class CountsFit:
    """The fitted quantities of every region and the quantities they share."""

    @property
    def regions(self) -> list[RegionFit]: ...
    @property
    def normalization(self) -> float: ...
    @property
    def t0_us(self) -> float: ...
    @property
    def flight_path_m(self) -> float: ...
    @property
    def alpha(self) -> list[float]: ...
    @property
    def beta(self) -> list[float]: ...
    @property
    def r(self) -> float: ...
    @property
    def fwhm_squared_us2(self) -> float: ...
    @property
    def r_high(self) -> float | None: ...
    @property
    def detector_tail(self) -> tuple[float, float] | None: ...
    @property
    def time_background(self) -> list[float]: ...
    @property
    def room_background(self) -> float: ...
    @property
    def spread_share(self) -> float: ...
    @property
    def covariance(self) -> NDArray[np.float64] | None: ...
    @property
    def unbounded_mean(self) -> list[float] | None: ...
    @property
    def unbounded_covariance(self) -> NDArray[np.float64] | None: ...
    @property
    def on_bound(self) -> list[bool]: ...
    @property
    def deviance(self) -> float:
        """Each run half Poisson deviance over its overdispersion."""
        ...
    @property
    def converged(self) -> bool: ...
    @property
    def measured_pulls(self) -> list[float] | None: ...
    @property
    def pulse_consistency(self) -> tuple[float, int, float] | None: ...
    @property
    def step_us(self) -> float: ...
    @property
    def points(self) -> int: ...
    @property
    def halvings(self) -> int: ...

class Patch:
    """What the pixels of a patch that are not excluded lie behind."""

    @property
    def kind(self) -> Literal["sample", "mixed", "outside"]: ...

class MapMeasurement:
    """The counts of every pixel of both runs, the masks, and the material.

    open_counts and sample_counts have shape (time bin, row, column);
    excluded, sample and empty have shape (row, column).
    time_background is [C, A, tau] per pixel, shared by every patch, or None
    for none; gamma_transmission is that of every sample patch, or None for
    known 1.
    """

    def __init__(
        self,
        time_edges_us: NDArray[np.float64],
        open_counts: NDArray[np.float64],
        sample_counts: NDArray[np.float64],
        excluded: NDArray[np.bool_],
        sample: NDArray[np.bool_],
        empty: NDArray[np.bool_],
        material: Material,
        charge_ratio: float,
        normalization: Value,
        background: list[Value],
        binning: int = 1,
        empty_background: list[Value] | None = None,
        open_live: NDArray[np.float64] | None = None,
        sample_live: NDArray[np.float64] | None = None,
        time_background: list[Value] | None = None,
        gamma_transmission: Value | None = None,
        room_background: RoomBackground | None = None,
        capture_gammas: Value | None = None,
        in_scatter: Value | None = None,
    ) -> None: ...
    def patches(self) -> list[list[Patch]]:
        """Each patch kind, in (patch row, patch column)."""
        ...
    @property
    def time_edges_us(self) -> list[float]: ...
    @property
    def charge_ratio(self) -> float: ...
    @property
    def binning(self) -> int: ...
    @property
    def time_background(self) -> list[Value]: ...
    @property
    def gamma_transmission(self) -> Value: ...
    @property
    def room_background(self) -> RoomBackground | None: ...
    @property
    def capture_gammas(self) -> Value: ...
    @property
    def in_scatter(self) -> Value: ...

class CountsMap:
    """The fitted map, one value per patch."""

    @property
    def patches(self) -> list[list[Patch]]: ...
    @property
    def fits(self) -> list[list[RegionFit | None]]: ...
    @property
    def failed(self) -> list[list[str | None]]: ...
    @property
    def empty(self) -> RegionFit | None: ...
    @property
    def densities(self) -> list[NDArray[np.float64]]: ...
    @property
    def density_sd(self) -> list[NDArray[np.float64]]: ...
    @property
    def temperature_k(self) -> NDArray[np.float64]: ...
    @property
    def temperature_sd_k(self) -> NDArray[np.float64]: ...
    @property
    def overdispersion(self) -> list[NDArray[np.float64]]: ...
    @property
    def identified(self) -> NDArray[np.bool_]:
        """Whether the map converged, the patch did not fail, and each fitted
        density and the fitted temperature has a finite standard deviation or,
        for a density, is 0."""
        ...
    @property
    def normalization(self) -> float: ...
    @property
    def time_background(self) -> list[float]: ...
    @property
    def room_background(self) -> float: ...
    @property
    def t0_us(self) -> float: ...
    @property
    def flight_path_m(self) -> float: ...
    @property
    def alpha(self) -> list[float]: ...
    @property
    def beta(self) -> list[float]: ...
    @property
    def r(self) -> float: ...
    @property
    def fwhm_squared_us2(self) -> float: ...
    @property
    def r_high(self) -> float | None: ...
    @property
    def detector_tail(self) -> tuple[float, float] | None: ...
    @property
    def shared(self) -> list[str]: ...
    @property
    def shared_covariance(self) -> NDArray[np.float64] | None: ...
    @property
    def measured_pulls(self) -> list[float] | None: ...
    @property
    def pulse_consistency(self) -> tuple[float, int, float] | None: ...
    @property
    def deviance(self) -> float: ...
    @property
    def converged(self) -> bool: ...
    @property
    def unconverged(self) -> str | None: ...
    @property
    def steps(self) -> int: ...
    @property
    def n_fitted(self) -> int: ...

NO_TIME_BACKGROUND: tuple[Value, Value, Value]
"""No time background: Value.known(0.0), Value.known(0.0), Value.known(1.0)."""

def fit_counts(measurement: Measurement, calibration: Calibration) -> CountsFit:
    """Fit the regions densities and temperatures to the raw counts of both
    runs, together with the normalization, time background, timing offset,
    flight path and pulse numbers that are not known."""
    ...

def fit_map(map: MapMeasurement, calibration: Calibration) -> CountsMap:
    """Fit a material over patches of the map pixels to the raw counts of
    both runs."""
    ...
