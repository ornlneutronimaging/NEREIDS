"""Aggregate 1D room calibration followed by a frozen hot-spectrum fit.

This module is the supported version of the workflow developed for VENUS.  It
keeps open-beam and sample counts as separate measurements, integrates the
instrument response over the actual detector-time bins, infers the incident
source from the open beam only, and returns the complete fitted curve.  The hot
fit reuses the room response and source without recalibrating either one.

The functions never edit measured counts or evaluated nuclear data.  Energy
selection only chooses which unchanged bins contribute to the objective; the
response is still evaluated on the full supplied acquisition interval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.optimize import least_squares, lsq_linear, minimize

from .nereids import (
    DetectorResponseMatrix,
    EnergyLaw,
    IkedaCarpenter,
    energy_to_tof,
    precompute_cross_sections,
    tof_to_energy,
    tof_to_energy_centers,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class IcShapeProfile:
    """Fixed energy law for an analytical Ikeda-Carpenter response.

    ``alpha`` and ``beta`` are inverse-microsecond rates.  Their coefficients
    use ``rate(E) = sqrt_coefficient * sqrt(E/eV) + offset``.  ``slow_fraction``
    is the fraction assigned to the slow storage component.
    """

    alpha_sqrt_coefficient: float
    alpha_offset: float
    beta_sqrt_coefficient: float
    beta_offset: float
    slow_fraction: float
    channel_fwhm_us: float
    pivot_energy_ev: float = 20.0
    synthesis_energies: int = 48
    synthesis_times: int = 400


# Analytical approximation inferred from the archived VENUS UDR shapes.  It is
# a compact instrument model, not a replacement for the high-fidelity UDR and
# not a change to any nuclear evaluation.
VENUS_UDR_MATCHED_IC_PROFILE = IcShapeProfile(
    alpha_sqrt_coefficient=0.6450616623046456,
    alpha_offset=2.223255165220424,
    beta_sqrt_coefficient=0.08670235276707496,
    beta_offset=0.12226218901463583,
    slow_fraction=0.23645799735769557,
    channel_fwhm_us=0.350,
)


@dataclass(frozen=True)
class SourceInferenceResult:
    """Open-beam-only incident-source estimate and numerical diagnostics."""

    weights: FloatArray
    expected_open_counts: FloatArray
    observed_open_counts: FloatArray
    converged: bool
    message: str
    iterations: int
    reduced_poisson_deviance: float
    max_abs_scaled_projected_gradient: float


@dataclass(frozen=True)
class Aggregated1DFitResult:
    """Complete aggregate-spectrum fit result in increasing-energy order."""

    energy_ev: FloatArray
    measured_transmission: FloatArray
    poisson_uncertainty: FloatArray
    uncertainty: FloatArray
    signal_prediction: FloatArray
    background_prediction: FloatArray
    model_prediction: FloatArray
    poisson_residual: FloatArray
    residual: FloatArray
    objective_residual: FloatArray
    fit_mask: NDArray[np.bool_]
    fit_energy_range_ev: tuple[float, float]
    open_variance_factor: float
    sample_variance_factor: float
    parameters: dict[str, float]
    converged: bool
    message: str
    function_evaluations: int
    bound_hits: tuple[str, ...]
    max_abs_residual: float
    rms_residual: float
    bins_above_five: int


@dataclass(frozen=True)
class Aggregated1DCalibration:
    """Room result plus the frozen objects required by a later hot fit."""

    fit: Aggregated1DFitResult
    instrument_parameters: dict[str, float]
    source: SourceInferenceResult
    detector_time_edges_us: FloatArray
    reference_energy_ev: FloatArray
    energy_cell_edges_ev: FloatArray
    quadrature_points_ev: FloatArray
    quadrature_weights: FloatArray
    response_matrix: Any = field(repr=False)
    isotopes: tuple[Any, ...] = field(repr=False)
    atomic_fractions: FloatArray = field(repr=False)
    ic_profile: IcShapeProfile = field(repr=False)
    room_temperature_k: float
    debye_temperature_k: float | None
    open_variance_factor: float
    room_variance_factor: float


def solid_debye_effective_temperature(
    thermodynamic_temperature_k: float, debye_temperature_k: float
) -> float:
    """Return the effective free-gas temperature for a Debye solid.

    This maps a solid's thermodynamic temperature onto the mean kinetic energy
    used by the free-gas Doppler model.  It changes only Doppler broadening; it
    does not alter resonance parameters.
    """

    temperature = float(thermodynamic_temperature_k)
    debye = float(debye_temperature_k)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("thermodynamic_temperature_k must be finite and positive")
    if not math.isfinite(debye) or debye <= 0.0:
        raise ValueError("debye_temperature_k must be finite and positive")
    upper = debye / temperature

    def integrand(value: float) -> float:
        return 0.0 if value == 0.0 else value**3 / math.expm1(value)

    integral, _ = quad(integrand, 0.0, upper, epsabs=1.0e-12, epsrel=1.0e-12)
    ratio = 3.0 * debye / (8.0 * temperature) + 3.0 * (
        temperature / debye
    ) ** 3 * integral
    return temperature * ratio


def profiled_two_arm_residual(
    open_counts: Sequence[float] | FloatArray,
    sample_counts: Sequence[float] | FloatArray,
    sample_over_open_exposure: float,
    apparent_transmission: Sequence[float] | FloatArray,
) -> FloatArray:
    """Signed square root of the two-arm Poisson deviance.

    The unknown incident intensity in each measured bin is removed exactly.
    No transmission ratio or Gaussian error approximation is used in this
    objective.
    """

    open_array = _one_dimensional("open_counts", open_counts)
    sample_array = _one_dimensional("sample_counts", sample_counts)
    apparent = _one_dimensional("apparent_transmission", apparent_transmission)
    if open_array.shape != sample_array.shape or open_array.shape != apparent.shape:
        raise ValueError("open, sample, and apparent-transmission arrays must match")
    if np.any(open_array < 0.0) or np.any(sample_array < 0.0):
        raise ValueError("counts must be nonnegative")
    exposure = float(sample_over_open_exposure)
    if not math.isfinite(exposure) or exposure <= 0.0:
        raise ValueError("sample_over_open_exposure must be finite and positive")
    if np.any(apparent <= 0.0):
        raise ValueError("apparent_transmission must be positive")

    total = open_array + sample_array
    odds = exposure * apparent
    probability = odds / (1.0 + odds)
    expected_sample = total * probability
    expected_open = total - expected_sample
    contribution = np.zeros_like(total)
    sample_positive = sample_array > 0.0
    open_positive = open_array > 0.0
    contribution[sample_positive] += sample_array[sample_positive] * np.log(
        sample_array[sample_positive] / expected_sample[sample_positive]
    )
    contribution[open_positive] += open_array[open_positive] * np.log(
        open_array[open_positive] / expected_open[open_positive]
    )
    sign = np.sign(sample_array - expected_sample)
    return sign * np.sqrt(np.maximum(2.0 * contribution, 0.0))


def select_energy_ordered_detector_bins(
    detector_time_edges_us: Sequence[float] | FloatArray,
    reference_energy_ev: Sequence[float] | FloatArray,
    arrays: Sequence[Sequence[float] | FloatArray],
    energy_range_ev: tuple[float, float],
) -> tuple[FloatArray, FloatArray, tuple[FloatArray, ...]]:
    """Select one contiguous energy interval and its matching TOF edges.

    Input spectra and ``reference_energy_ev`` are in increasing-energy order;
    detector-time edges are in increasing-time order.  The returned spectra
    remain in increasing-energy order.  This helper prevents the common
    off-by-one and accidental-axis-reversal errors in notebook preprocessing.
    """

    edges = _one_dimensional("detector_time_edges_us", detector_time_edges_us)
    energy = _one_dimensional("reference_energy_ev", reference_energy_ev)
    if edges.size != energy.size + 1 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("detector edges must bracket every reference-energy bin")
    if np.any(np.diff(energy) <= 0.0):
        raise ValueError("reference_energy_ev must be strictly increasing")
    mask = _energy_mask(energy, energy_range_ev)
    indices = np.flatnonzero(mask)
    if np.any(np.diff(indices) != 1):
        raise ValueError("energy_range_ev did not select one contiguous interval")
    n_bins = energy.size
    first_time_bin = n_bins - 1 - int(indices[-1])
    last_time_bin = n_bins - 1 - int(indices[0])
    selected_edges = np.ascontiguousarray(edges[first_time_bin : last_time_bin + 2])
    selected_arrays: list[FloatArray] = []
    for index, values in enumerate(arrays):
        array = _one_dimensional(f"arrays[{index}]", values)
        if array.shape != energy.shape:
            raise ValueError(f"arrays[{index}] must match reference_energy_ev")
        selected_arrays.append(np.ascontiguousarray(array[mask]))
    return selected_edges, np.ascontiguousarray(energy[mask]), tuple(selected_arrays)


def calibrate_aggregated_1d(
    *,
    detector_time_edges_us: Sequence[float] | FloatArray,
    open_counts: Sequence[float] | FloatArray,
    room_counts: Sequence[float] | FloatArray,
    sample_over_open_exposure: float,
    isotopes: Sequence[Any],
    room_temperature_k: float,
    reference_flight_path_m: float,
    reference_timing_offset_us: float,
    initial_physical_parameters: Sequence[float],
    fit_energy_range_ev: tuple[float, float],
    ic_profile: IcShapeProfile,
    physical_lower_bounds: Sequence[float],
    physical_upper_bounds: Sequence[float],
    initial_background_parameters: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    atomic_fractions: Sequence[float] | None = None,
    debye_temperature_k: float | None = None,
    background_lower_bounds: Sequence[float] = (0.5, -0.5, -0.5, -0.5),
    background_upper_bounds: Sequence[float] = (2.0, 0.5, 0.5, 0.5),
    physical_scale: Sequence[float] | None = None,
    outer_max_evaluations: int = 40,
    inner_max_evaluations: int = 200,
    source_max_iterations: int = 10_000,
    open_variance_factor: float = 1.0,
    room_variance_factor: float = 1.0,
    progress: Callable[[str, dict[str, float]], None] | None = None,
) -> Aggregated1DCalibration:
    """Calibrate the analytical IC instrument using room data only.

    Counts must be in increasing-energy order.  ``detector_time_edges_us``
    must be in increasing-time order and therefore describes those same bins
    in reverse.  The four physical parameters are timing offset (microseconds),
    effective centroid path (metres), response width scale, and areal density
    (atoms/barn).  Background parameters are normalization, BackA, BackB, and
    BackC from the SAMMY apparent-transmission equation.

    The IC profile, fit range, and physical bounds are explicit because they
    are instrument- and experiment-specific. The hot spectrum is not an
    argument and cannot influence this calibration.
    """

    detector_edges, open_array, room_array = _validate_aggregate_inputs(
        detector_time_edges_us, open_counts, room_counts
    )
    exposure = _positive_scalar("sample_over_open_exposure", sample_over_open_exposure)
    room_temperature = _positive_scalar("room_temperature_k", room_temperature_k)
    reference_path = _positive_scalar("reference_flight_path_m", reference_flight_path_m)
    reference_t0 = _finite_scalar("reference_timing_offset_us", reference_timing_offset_us)
    open_factor = _positive_scalar("open_variance_factor", open_variance_factor)
    room_factor = _positive_scalar("room_variance_factor", room_variance_factor)
    source_iterations = _positive_integer("source_max_iterations", source_max_iterations)
    physical_start = _fixed_vector("initial_physical_parameters", initial_physical_parameters, 4)
    background_start = _fixed_vector(
        "initial_background_parameters", initial_background_parameters, 4
    )
    physical_lower = _fixed_vector("physical_lower_bounds", physical_lower_bounds, 4)
    physical_upper = _fixed_vector("physical_upper_bounds", physical_upper_bounds, 4)
    background_lower = _fixed_vector(
        "background_lower_bounds", background_lower_bounds, 4
    )
    background_upper = _fixed_vector(
        "background_upper_bounds", background_upper_bounds, 4
    )
    _validate_bounds(physical_start, physical_lower, physical_upper, "physical")
    _validate_bounds(background_start, background_lower, background_upper, "background")
    if physical_scale is None:
        scale = np.array(
            [1.0, 0.05, 0.5, max(abs(physical_start[3]), 1.0e-6)], dtype=float
        )
    else:
        scale = _fixed_vector("physical_scale", physical_scale, 4)
        if np.any(scale <= 0.0):
            raise ValueError("physical_scale values must be positive")

    reference_energy = np.asarray(
        tof_to_energy_centers(detector_edges, reference_path, reference_t0),
        dtype=float,
    )
    energy_edges = np.array(
        [
            tof_to_energy(float(time - reference_t0), reference_path)
            for time in detector_edges[::-1]
        ],
        dtype=float,
    )
    if np.any(np.diff(reference_energy) <= 0.0) or np.any(np.diff(energy_edges) <= 0.0):
        raise ValueError("reference conversion did not produce increasing-energy bins")
    fit_mask = _energy_mask(reference_energy, fit_energy_range_ev)
    points, weights = _kronrod_15_cells(energy_edges)
    isotope_tuple, fractions = _validate_material(isotopes, atomic_fractions)
    doppler_temperature = (
        room_temperature
        if debye_temperature_k is None
        else solid_debye_effective_temperature(room_temperature, debye_temperature_k)
    )
    room_cross_section = _total_cross_section(
        points.ravel(), isotope_tuple, fractions, doppler_temperature
    ).reshape(points.shape)
    measured = room_array / open_array / exposure
    uncertainty = measured * np.sqrt(
        room_factor / room_array + open_factor / open_array
    )
    poisson_uncertainty = measured * np.sqrt(1.0 / room_array + 1.0 / open_array)
    response_cache: dict[bytes, dict[str, Any]] = {}
    profile_cache: dict[bytes, dict[str, Any]] = {}

    def response_record(parameters: FloatArray) -> dict[str, Any]:
        key = np.ascontiguousarray(parameters[:3], dtype=float).tobytes()
        if key in response_cache:
            return response_cache[key]
        resolution, physical = _build_ic(
            ic_profile, float(energy_edges[0]), float(energy_edges[-1]), parameters
        )
        operator = _CellResponse(points, weights, detector_edges, resolution, parameters[0])
        source = _infer_source(operator, open_array, source_iterations)
        record = {
            "operator": operator,
            "source": source,
            "physical": physical,
        }
        response_cache[key] = record
        if progress is not None:
            progress(
                "room_response",
                {
                    "response_builds": float(len(response_cache)),
                    "source_iterations": float(source.iterations),
                    "source_reduced_poisson_deviance": source.reduced_poisson_deviance,
                },
            )
        return record

    def profile(parameters: FloatArray) -> dict[str, Any]:
        key = np.ascontiguousarray(parameters, dtype=float).tobytes()
        if key in profile_cache:
            return profile_cache[key]
        response = response_record(parameters)
        operator: _CellResponse = response["operator"]
        source: SourceInferenceResult = response["source"]
        transmission = np.exp(-float(parameters[3]) * room_cross_section)
        sample_prediction = operator.project_points(
            np.repeat(source.weights, points.shape[1]) * transmission.ravel()
        )
        ratio = sample_prediction / np.maximum(
            source.expected_open_counts, np.finfo(float).tiny
        )
        detector_energy = _detector_energy_axis(
            detector_edges, float(parameters[0]), float(response["physical"]["physical_path_m"])
        )
        basis = np.column_stack(
            (ratio, np.ones_like(ratio), 1.0 / np.sqrt(detector_energy), np.sqrt(detector_energy))
        )

        def inner_residual(coefficients: FloatArray) -> FloatArray:
            model = basis @ coefficients
            return profiled_two_arm_residual(open_array, room_array, exposure, model)[fit_mask]

        inner = least_squares(
            inner_residual,
            background_start,
            jac="3-point",
            diff_step=np.array([1.0e-4, 1.0e-4, 1.0e-4, 1.0e-5])
            / np.maximum(np.abs(background_start), np.finfo(float).tiny),
            bounds=(background_lower, background_upper),
            x_scale=np.array([0.2, 0.1, 0.1, 0.01]),
            max_nfev=int(inner_max_evaluations),
            xtol=1.0e-8,
            ftol=1.0e-8,
            gtol=1.0e-8,
        )
        signal = inner.x[0] * ratio
        background = basis[:, 1:] @ inner.x[1:]
        model = signal + background
        record = {
            "response": response,
            "ratio": ratio,
            "detector_energy": detector_energy,
            "inner": inner,
            "signal": signal,
            "background": background,
            "model": model,
            "objective_residual": profiled_two_arm_residual(
                open_array, room_array, exposure, model
            ),
        }
        profile_cache[key] = record
        return record

    absolute_steps = np.array(
        [0.01, 0.01, 0.01 * physical_start[2], 0.01 * physical_start[3]], dtype=float
    )

    def outer_residual(parameters: FloatArray) -> FloatArray:
        selected = np.asarray(
            profile(parameters)["objective_residual"], dtype=float
        )[fit_mask]
        if progress is not None:
            progress(
                "room_fit",
                {
                    "profile_evaluations": float(len(profile_cache)),
                    "rms_objective_residual": float(np.sqrt(np.mean(selected * selected))),
                    "max_abs_objective_residual": float(np.max(np.abs(selected))),
                },
            )
        return selected

    def outer_jacobian(parameters: FloatArray) -> FloatArray:
        columns: list[FloatArray] = []
        for index, step in enumerate(absolute_steps):
            plus = parameters.copy()
            minus = parameters.copy()
            plus[index] += step
            minus[index] -= step
            if minus[index] <= physical_lower[index] or plus[index] >= physical_upper[index]:
                raise ValueError("fixed room derivative step crossed a parameter bound")
            columns.append((outer_residual(plus) - outer_residual(minus)) / (2.0 * step))
        return np.column_stack(columns)

    outer = least_squares(
        outer_residual,
        physical_start,
        jac=outer_jacobian,
        bounds=(physical_lower, physical_upper),
        x_scale=scale,
        max_nfev=int(outer_max_evaluations),
        xtol=1.0e-8,
        ftol=1.0e-8,
        gtol=1.0e-8,
    )
    final = profile(np.asarray(outer.x, dtype=float))
    inner = final["inner"]
    parameters = np.concatenate((outer.x, inner.x))
    names = (
        "timing_offset_us",
        "effective_centroid_path_m",
        "width_scale",
        "density_atoms_per_barn",
        "anorm",
        "back_a",
        "back_b",
        "back_c",
    )
    lower = np.concatenate((physical_lower, background_lower))
    upper = np.concatenate((physical_upper, background_upper))
    bound_hits = _bound_hits(names, parameters, lower, upper)
    standardized = (measured - final["model"]) / uncertainty
    response = final["response"]
    result = _fit_result(
        energy=np.asarray(final["detector_energy"], dtype=float),
        measured=measured,
        poisson_uncertainty=poisson_uncertainty,
        uncertainty=uncertainty,
        signal=np.asarray(final["signal"], dtype=float),
        background=np.asarray(final["background"], dtype=float),
        objective=np.asarray(final["objective_residual"], dtype=float),
        standardized=standardized,
        fit_mask=fit_mask,
        names=names,
        values=parameters,
        converged=bool(outer.success and inner.success and response["source"].converged),
        message=(
            f"outer: {outer.message}; background: {inner.message}; "
            f"source: {response['source'].message}"
        ),
        evaluations=int(outer.nfev),
        bound_hits=bound_hits,
        fit_energy_range_ev=fit_energy_range_ev,
        open_variance_factor=open_factor,
        sample_variance_factor=room_factor,
    )
    physical = dict(response["physical"])
    instrument_parameters = {
        name: float(value) for name, value in zip(names, parameters, strict=True)
    }
    return Aggregated1DCalibration(
        fit=result,
        instrument_parameters={**instrument_parameters, **physical},
        source=response["source"],
        detector_time_edges_us=detector_edges,
        reference_energy_ev=reference_energy,
        energy_cell_edges_ev=energy_edges,
        quadrature_points_ev=points,
        quadrature_weights=weights,
        response_matrix=response["operator"].matrix,
        isotopes=isotope_tuple,
        atomic_fractions=fractions,
        ic_profile=ic_profile,
        room_temperature_k=room_temperature,
        debye_temperature_k=debye_temperature_k,
        open_variance_factor=open_factor,
        room_variance_factor=room_factor,
    )


def fit_frozen_aggregated_1d(
    calibration: Aggregated1DCalibration,
    *,
    hot_counts: Sequence[float] | FloatArray,
    sample_over_open_exposure: float,
    initial_temperature_k: float,
    initial_density_atoms_per_barn: float,
    hot_variance_factor: float = 1.0,
    temperature_bounds_k: tuple[float, float] = (1.0, 5000.0),
    density_bounds_atoms_per_barn: tuple[float, float] = (0.0, 0.01),
    background_lower_bounds: Sequence[float] = (0.5, -0.5, -0.5, -0.5),
    background_upper_bounds: Sequence[float] = (2.0, 0.5, 0.5, 0.5),
    temperature_derivative_step_k: float = 5.0,
    density_derivative_step_atoms_per_barn: float | None = None,
    max_evaluations: int = 40,
    progress: Callable[[str, dict[str, float]], None] | None = None,
) -> Aggregated1DFitResult:
    """Fit temperature and amount with the room instrument fully frozen."""

    hot = _one_dimensional("hot_counts", hot_counts)
    if not calibration.source.converged:
        raise RuntimeError(
            "the frozen room calibration has a non-converged open-beam source; "
            "the hot fit cannot use it"
        )
    # The observed open counts, rather than their fitted expectation, define
    # both the reported transmission and its propagated uncertainty.  Recover
    # them exactly from the room result and room exposure-independent ratio.
    # They are stored on the source result below by the calibration constructor.
    open_array = np.asarray(calibration.source.observed_open_counts, dtype=float)
    if hot.shape != open_array.shape:
        raise ValueError("hot_counts must match the calibrated detector bins")
    if np.any(hot <= 0.0):
        raise ValueError("hot_counts must be positive on the aggregate fit interval")
    exposure = _positive_scalar("sample_over_open_exposure", sample_over_open_exposure)
    hot_factor = _positive_scalar("hot_variance_factor", hot_variance_factor)
    temperature_start = _positive_scalar("initial_temperature_k", initial_temperature_k)
    density_start = _positive_scalar(
        "initial_density_atoms_per_barn", initial_density_atoms_per_barn
    )
    background_lower = _fixed_vector(
        "background_lower_bounds", background_lower_bounds, 4
    )
    background_upper = _fixed_vector(
        "background_upper_bounds", background_upper_bounds, 4
    )
    measured = hot / open_array / exposure
    uncertainty = measured * np.sqrt(
        hot_factor / hot + calibration.open_variance_factor / open_array
    )
    poisson_uncertainty = measured * np.sqrt(1.0 / hot + 1.0 / open_array)
    points = calibration.quadrature_points_ev
    weights = calibration.quadrature_weights
    operator = _CellResponse.from_existing(
        points, weights, calibration.response_matrix
    )
    source = calibration.source.weights
    predicted_open = calibration.source.expected_open_counts
    detector_energy = calibration.fit.energy_ev
    fit_mask = calibration.fit.fit_mask
    sigma_cache: dict[bytes, FloatArray] = {}
    profile_cache: dict[bytes, dict[str, Any]] = {}

    def cross_section(temperature_k: float) -> FloatArray:
        key = np.asarray([temperature_k], dtype=np.float64).tobytes()
        if key not in sigma_cache:
            effective = (
                temperature_k
                if calibration.debye_temperature_k is None
                else solid_debye_effective_temperature(
                    temperature_k, calibration.debye_temperature_k
                )
            )
            sigma_cache[key] = _total_cross_section(
                points.ravel(), calibration.isotopes, calibration.atomic_fractions, effective
            ).reshape(points.shape)
        return sigma_cache[key]

    def profile(parameters: FloatArray) -> dict[str, Any]:
        key = np.ascontiguousarray(parameters, dtype=float).tobytes()
        if key in profile_cache:
            return profile_cache[key]
        temperature, density = map(float, parameters)
        transmission = np.exp(-density * cross_section(temperature))
        sample_prediction = operator.project_points(
            np.repeat(source, points.shape[1]) * transmission.ravel()
        )
        ratio = sample_prediction / np.maximum(predicted_open, np.finfo(float).tiny)
        basis = np.column_stack(
            (
                ratio[fit_mask],
                np.ones(int(fit_mask.sum())),
                1.0 / np.sqrt(detector_energy[fit_mask]),
                np.sqrt(detector_energy[fit_mask]),
            )
        )
        background_fit = lsq_linear(
            basis / uncertainty[fit_mask, None],
            measured[fit_mask] / uncertainty[fit_mask],
            bounds=(background_lower, background_upper),
            method="trf",
            tol=1.0e-12,
            max_iter=500,
        )
        if not background_fit.success:
            raise RuntimeError(f"hot background fit failed: {background_fit.message}")
        signal = background_fit.x[0] * ratio
        background = (
            background_fit.x[1]
            + background_fit.x[2] / np.sqrt(detector_energy)
            + background_fit.x[3] * np.sqrt(detector_energy)
        )
        model = signal + background
        record = {
            "background_fit": background_fit,
            "signal": signal,
            "background": background,
            "model": model,
            "residual": (measured - model) / uncertainty,
        }
        profile_cache[key] = record
        return record

    start = np.array([temperature_start, density_start], dtype=float)
    lower = np.array([temperature_bounds_k[0], density_bounds_atoms_per_barn[0]], dtype=float)
    upper = np.array([temperature_bounds_k[1], density_bounds_atoms_per_barn[1]], dtype=float)
    _validate_bounds(start, lower, upper, "hot")
    density_step = (
        0.01 * density_start
        if density_derivative_step_atoms_per_barn is None
        else _positive_scalar(
            "density_derivative_step_atoms_per_barn",
            density_derivative_step_atoms_per_barn,
        )
    )
    steps = np.array(
        [
            _positive_scalar(
                "temperature_derivative_step_k", temperature_derivative_step_k
            ),
            density_step,
        ]
    )

    def residual(parameters: FloatArray) -> FloatArray:
        selected = np.asarray(profile(parameters)["residual"], dtype=float)[fit_mask]
        if progress is not None:
            progress(
                "hot_fit",
                {
                    "profile_evaluations": float(len(profile_cache)),
                    "rms_residual": float(np.sqrt(np.mean(selected * selected))),
                    "max_abs_residual": float(np.max(np.abs(selected))),
                },
            )
        return selected

    def jacobian(parameters: FloatArray) -> FloatArray:
        columns: list[FloatArray] = []
        for index, step in enumerate(steps):
            plus = parameters.copy()
            minus = parameters.copy()
            plus[index] += step
            minus[index] -= step
            if minus[index] <= lower[index] or plus[index] >= upper[index]:
                raise ValueError("fixed hot derivative step crossed a parameter bound")
            columns.append((residual(plus) - residual(minus)) / (2.0 * step))
        return np.column_stack(columns)

    outer = least_squares(
        residual,
        start,
        jac=jacobian,
        bounds=(lower, upper),
        x_scale=np.array([500.0, max(density_start, 1.0e-6)]),
        max_nfev=int(max_evaluations),
        xtol=1.0e-8,
        ftol=1.0e-8,
        gtol=1.0e-8,
    )
    final = profile(np.asarray(outer.x, dtype=float))
    background_fit = final["background_fit"]
    values = np.concatenate((outer.x, background_fit.x))
    names = (
        "temperature_k",
        "density_atoms_per_barn",
        "anorm",
        "back_a",
        "back_b",
        "back_c",
    )
    all_lower = np.concatenate((lower, background_lower))
    all_upper = np.concatenate((upper, background_upper))
    return _fit_result(
        energy=detector_energy,
        measured=measured,
        poisson_uncertainty=poisson_uncertainty,
        uncertainty=uncertainty,
        signal=np.asarray(final["signal"], dtype=float),
        background=np.asarray(final["background"], dtype=float),
        objective=np.asarray(final["residual"], dtype=float),
        standardized=np.asarray(final["residual"], dtype=float),
        fit_mask=fit_mask,
        names=names,
        values=values,
        converged=bool(outer.success and background_fit.success),
        message=f"outer: {outer.message}; background: {background_fit.message}",
        evaluations=int(outer.nfev),
        bound_hits=_bound_hits(names, values, all_lower, all_upper),
        fit_energy_range_ev=calibration.fit.fit_energy_range_ev,
        open_variance_factor=calibration.open_variance_factor,
        sample_variance_factor=hot_factor,
    )


class _CellResponse:
    def __init__(
        self,
        points: FloatArray,
        weights: FloatArray,
        detector_edges: FloatArray,
        resolution: Any,
        timing_offset_us: float,
    ) -> None:
        self.points = points
        self.weights = weights
        self.matrix = DetectorResponseMatrix(
            np.ascontiguousarray(points.ravel()),
            np.ascontiguousarray(detector_edges),
            resolution,
            float(timing_offset_us),
        )
        self.cell_matrix = self.matrix.collapse_true_energy_groups(
            np.ascontiguousarray(weights.ravel()), points.shape[1]
        )

    @classmethod
    def from_existing(
        cls, points: FloatArray, weights: FloatArray, matrix: Any
    ) -> _CellResponse:
        value = cls.__new__(cls)
        value.points = points
        value.weights = weights
        value.matrix = matrix
        value.cell_matrix = None
        return value

    def project_points(self, point_values: FloatArray) -> FloatArray:
        weighted = np.asarray(point_values, dtype=float) * self.weights.ravel()
        return np.asarray(self.matrix.project(np.ascontiguousarray(weighted)), dtype=float)[::-1]

    def transpose_cells(self, energy_ordered_detector_values: FloatArray) -> FloatArray:
        if self.cell_matrix is None:
            raise RuntimeError("cell response is unavailable on a sample-only operator")
        time_ordered = np.ascontiguousarray(energy_ordered_detector_values[::-1])
        return np.asarray(self.cell_matrix.transpose_project(time_ordered), dtype=float)

    def project_cells(self, cell_values: FloatArray) -> FloatArray:
        if self.cell_matrix is None:
            raise RuntimeError("cell response is unavailable on a sample-only operator")
        return np.asarray(
            self.cell_matrix.project(np.ascontiguousarray(cell_values)), dtype=float
        )[::-1]


def _infer_source(
    operator: _CellResponse, observed_open: FloatArray, max_iterations: int
) -> SourceInferenceResult:
    column_mass = operator.transpose_cells(np.ones_like(observed_open))
    if np.any(column_mass <= 0.0):
        raise RuntimeError("instrument response contains an unobservable source cell")
    scale = float(np.median(observed_open))
    if scale <= 0.0:
        raise ValueError("median open count must be positive")
    start = np.maximum(observed_open / column_mass / scale, np.finfo(float).tiny)
    n_bins = observed_open.size

    def objective_and_gradient(value: FloatArray) -> tuple[float, FloatArray]:
        expected = operator.project_cells(value * scale)
        safe = np.maximum(expected, np.finfo(float).tiny)
        objective = _poisson_deviance(observed_open, safe) / n_bins
        gradient = (
            2.0
            * scale
            * operator.transpose_cells(1.0 - observed_open / safe)
            / n_bins
        )
        return objective, gradient

    initial = minimize(
        objective_and_gradient,
        start,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * n_bins,
        options={
            "maxiter": min(2_000, int(max_iterations)),
            "ftol": 1.0e-12,
            "gtol": 1.0e-8,
            "maxls": 40,
        },
    )
    # Continue the same convex objective with a larger correction history.
    # The two stages reproduce the validated VENUS workflow while keeping the
    # source entirely open-beam-only.
    result = minimize(
        objective_and_gradient,
        np.asarray(initial.x, dtype=float),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * n_bins,
        options={
            "maxiter": int(max_iterations),
            "ftol": 1.0e-12,
            "gtol": 1.0e-8,
            "maxls": 40,
            "maxcor": 50,
        },
    )
    weights = np.asarray(result.x, dtype=float) * scale
    expected = operator.project_cells(weights)
    _, gradient = objective_and_gradient(np.asarray(result.x, dtype=float))
    projected = gradient.copy()
    at_lower = np.asarray(result.x) <= 1.0e-14
    projected[at_lower] = np.minimum(projected[at_lower], 0.0)
    return SourceInferenceResult(
        weights=weights,
        expected_open_counts=expected,
        observed_open_counts=np.asarray(observed_open, dtype=float),
        converged=bool(result.success),
        message=str(result.message),
        iterations=int(initial.nit + result.nit),
        reduced_poisson_deviance=float(result.fun),
        max_abs_scaled_projected_gradient=float(np.max(np.abs(projected))),
    )


def _build_ic(
    profile: IcShapeProfile,
    minimum_energy_ev: float,
    maximum_energy_ev: float,
    physical_parameters: FloatArray,
) -> tuple[Any, dict[str, float]]:
    timing_offset, effective_path, width_scale, _density = map(float, physical_parameters)
    if width_scale <= 0.0:
        raise ValueError("IC width scale must be positive")
    alpha_sqrt = profile.alpha_sqrt_coefficient / width_scale
    alpha_offset = profile.alpha_offset / width_scale
    beta_sqrt = profile.beta_sqrt_coefficient / width_scale
    beta_offset = profile.beta_offset / width_scale
    root_pivot = math.sqrt(profile.pivot_energy_ev)
    alpha_pivot = alpha_sqrt * root_pivot + alpha_offset
    beta_pivot = beta_sqrt * root_pivot + beta_offset
    mean_delay = 3.0 / alpha_pivot + profile.slow_fraction / beta_pivot
    # energy_to_tof is proportional to path; this converts the mean delay at
    # the pivot into the equivalent extra flight path without a copied constant.
    one_metre_tof = energy_to_tof(profile.pivot_energy_ev, 1.0)
    equivalent_path = mean_delay / one_metre_tof
    physical_path = effective_path - equivalent_path
    if physical_path <= 0.0:
        raise ValueError("IC physical flight path became non-positive")
    resolution = IkedaCarpenter(
        flight_path_m=physical_path,
        e_min_ev=minimum_energy_ev * (1.0 - 1.0e-12),
        e_max_ev=maximum_energy_ev * (1.0 + 1.0e-12),
        alpha=EnergyLaw.sqrt_e(alpha_sqrt, alpha_offset),
        beta=1.0,
        beta_law=EnergyLaw.sqrt_e(beta_sqrt, beta_offset),
        r=EnergyLaw.const(profile.slow_fraction),
        n_energies=int(profile.synthesis_energies),
        n_tau=int(profile.synthesis_times),
        channel_fwhm_us=profile.channel_fwhm_us,
    )
    return resolution, {
        "timing_offset_us": timing_offset,
        "effective_centroid_path_m": effective_path,
        "physical_path_m": physical_path,
        "width_scale": width_scale,
        "mean_delay_at_pivot_us": mean_delay,
        "mean_equivalent_path_at_pivot_m": equivalent_path,
        "alpha_sqrt_coefficient": alpha_sqrt,
        "alpha_offset": alpha_offset,
        "beta_sqrt_coefficient": beta_sqrt,
        "beta_offset": beta_offset,
        "slow_fraction": profile.slow_fraction,
    }


def _kronrod_15_cells(energy_edges: FloatArray) -> tuple[FloatArray, FloatArray]:
    positive_nodes = np.array(
        [
            0.9914553711208126,
            0.9491079123427585,
            0.8648644233597691,
            0.7415311855993945,
            0.5860872354676911,
            0.4058451513773972,
            0.2077849550078985,
        ]
    )
    positive_weights = np.array(
        [
            0.02293532201052922,
            0.06309209262997855,
            0.1047900103222502,
            0.1406532597155259,
            0.1690047266392679,
            0.1903505780647854,
            0.2044329400752989,
        ]
    )
    nodes = np.concatenate((-positive_nodes, [0.0], positive_nodes[::-1]))
    weights = np.concatenate((positive_weights, [0.2094821410847278], positive_weights[::-1]))
    order = np.argsort(nodes)
    nodes = nodes[order]
    weights = weights[order]
    lower = energy_edges[:-1, None]
    upper = energy_edges[1:, None]
    points = 0.5 * ((upper - lower) * nodes[None, :] + upper + lower)
    normalized_weights = np.broadcast_to(0.5 * weights[None, :], points.shape).copy()
    return np.ascontiguousarray(points), np.ascontiguousarray(normalized_weights)


def _total_cross_section(
    energies: FloatArray,
    isotopes: tuple[Any, ...],
    fractions: FloatArray,
    temperature_k: float,
) -> FloatArray:
    components = precompute_cross_sections(
        np.ascontiguousarray(energies), list(isotopes), temperature_k=float(temperature_k)
    )
    arrays = np.vstack([np.asarray(component, dtype=float) for component in components])
    if np.any(~np.isfinite(arrays)) or np.any(arrays < 0.0):
        raise RuntimeError("cross-section calculation returned an invalid value")
    return np.asarray(fractions @ arrays, dtype=float)


def _detector_energy_axis(
    detector_edges_us: FloatArray, timing_offset_us: float, flight_path_m: float
) -> FloatArray:
    centers = 0.5 * (detector_edges_us[:-1] + detector_edges_us[1:])
    corrected = centers[::-1] - timing_offset_us
    if np.any(corrected <= 0.0):
        raise ValueError("timing offset makes a detector-bin time non-positive")
    return np.array([tof_to_energy(float(value), flight_path_m) for value in corrected])


def _fit_result(
    *,
    energy: FloatArray,
    measured: FloatArray,
    poisson_uncertainty: FloatArray,
    uncertainty: FloatArray,
    signal: FloatArray,
    background: FloatArray,
    objective: FloatArray,
    standardized: FloatArray,
    fit_mask: NDArray[np.bool_],
    names: Sequence[str],
    values: FloatArray,
    converged: bool,
    message: str,
    evaluations: int,
    bound_hits: tuple[str, ...],
    fit_energy_range_ev: tuple[float, float],
    open_variance_factor: float,
    sample_variance_factor: float,
) -> Aggregated1DFitResult:
    selected = np.asarray(standardized, dtype=float)[fit_mask]
    model = np.asarray(signal + background, dtype=float)
    return Aggregated1DFitResult(
        energy_ev=np.asarray(energy, dtype=float),
        measured_transmission=np.asarray(measured, dtype=float),
        poisson_uncertainty=np.asarray(poisson_uncertainty, dtype=float),
        uncertainty=np.asarray(uncertainty, dtype=float),
        signal_prediction=np.asarray(signal, dtype=float),
        background_prediction=np.asarray(background, dtype=float),
        model_prediction=model,
        poisson_residual=np.asarray((measured - model) / poisson_uncertainty, dtype=float),
        residual=np.asarray(standardized, dtype=float),
        objective_residual=np.asarray(objective, dtype=float),
        fit_mask=np.asarray(fit_mask, dtype=bool),
        fit_energy_range_ev=tuple(map(float, fit_energy_range_ev)),
        open_variance_factor=float(open_variance_factor),
        sample_variance_factor=float(sample_variance_factor),
        parameters={name: float(value) for name, value in zip(names, values, strict=True)},
        converged=converged,
        message=message,
        function_evaluations=evaluations,
        bound_hits=bound_hits,
        max_abs_residual=float(np.max(np.abs(selected))),
        rms_residual=float(np.sqrt(np.mean(selected * selected))),
        bins_above_five=int(np.count_nonzero(np.abs(selected) > 5.0)),
    )


def _poisson_deviance(observed: FloatArray, expected: FloatArray) -> float:
    safe = np.maximum(expected, np.finfo(float).tiny)
    terms = safe - observed
    positive = observed > 0.0
    terms[positive] += observed[positive] * np.log(observed[positive] / safe[positive])
    return float(2.0 * np.sum(terms))


def _validate_aggregate_inputs(
    detector_edges: Sequence[float] | FloatArray,
    open_counts: Sequence[float] | FloatArray,
    sample_counts: Sequence[float] | FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    edges = _one_dimensional("detector_time_edges_us", detector_edges)
    open_array = _one_dimensional("open_counts", open_counts)
    sample_array = _one_dimensional("sample_counts", sample_counts)
    if edges.size != open_array.size + 1 or sample_array.shape != open_array.shape:
        raise ValueError("detector edges must bracket every matching open/sample count bin")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("detector_time_edges_us must be strictly increasing")
    if np.any(open_array <= 0.0) or np.any(sample_array <= 0.0):
        raise ValueError("aggregate open and sample counts must be positive")
    return edges, open_array, sample_array


def _validate_material(
    isotopes: Sequence[Any], atomic_fractions: Sequence[float] | None
) -> tuple[tuple[Any, ...], FloatArray]:
    values = tuple(isotopes)
    if not values:
        raise ValueError("isotopes must not be empty")
    if atomic_fractions is None:
        if len(values) != 1:
            raise ValueError("atomic_fractions are required for multiple isotopes")
        fractions = np.ones(1, dtype=float)
    else:
        fractions = _fixed_vector("atomic_fractions", atomic_fractions, len(values))
        if np.any(fractions < 0.0) or not np.isclose(np.sum(fractions), 1.0, atol=1.0e-12):
            raise ValueError("atomic_fractions must be nonnegative and sum to one")
    return values, fractions


def _one_dimensional(name: str, values: Sequence[float] | FloatArray) -> FloatArray:
    # Always copy so a calibration remains frozen if the caller later mutates
    # an input array that was already contiguous and float64.
    array = np.array(values, dtype=float, order="C", copy=True)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _fixed_vector(name: str, values: Sequence[float], length: int) -> FloatArray:
    array = _one_dimensional(name, values)
    if array.size != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return array


def _finite_scalar(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_scalar(name: str, value: float) -> float:
    result = _finite_scalar(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_integer(name: str, value: int) -> int:
    result = int(value)
    if result != value or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _energy_mask(energy: FloatArray, limits: tuple[float, float]) -> NDArray[np.bool_]:
    lower, upper = map(float, limits)
    if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 < lower < upper):
        raise ValueError("fit_energy_range_ev must be finite, positive, and increasing")
    mask = (energy >= lower) & (energy <= upper)
    if not np.any(mask):
        raise ValueError("fit_energy_range_ev selects no detector bins")
    return mask


def _validate_bounds(
    start: FloatArray, lower: FloatArray, upper: FloatArray, label: str
) -> None:
    if np.any(lower >= upper):
        raise ValueError(f"{label} lower bounds must be below upper bounds")
    if np.any(start <= lower) or np.any(start >= upper):
        raise ValueError(f"{label} initial values must be strictly inside their bounds")


def _bound_hits(
    names: Sequence[str], values: FloatArray, lower: FloatArray, upper: FloatArray
) -> tuple[str, ...]:
    return tuple(
        name
        for name, value, low, high in zip(names, values, lower, upper, strict=True)
        if np.isclose(value, low, rtol=0.0, atol=1.0e-8 * max(1.0, abs(low)))
        or np.isclose(value, high, rtol=0.0, atol=1.0e-8 * max(1.0, abs(high)))
    )


__all__ = [
    "Aggregated1DCalibration",
    "Aggregated1DFitResult",
    "IcShapeProfile",
    "SourceInferenceResult",
    "VENUS_UDR_MATCHED_IC_PROFILE",
    "calibrate_aggregated_1d",
    "fit_frozen_aggregated_1d",
    "profiled_two_arm_residual",
    "select_energy_ordered_detector_bins",
    "solid_debye_effective_temperature",
]
