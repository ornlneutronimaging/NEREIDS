"""Public aggregate 1D calibration-to-hot-fit workflow regression."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

import nereids


def _kronrod_cells(energy_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    weights = np.concatenate(
        (positive_weights, [0.2094821410847278], positive_weights[::-1])
    )
    order = np.argsort(nodes)
    lower = energy_edges[:-1, None]
    upper = energy_edges[1:, None]
    points = 0.5 * (
        (upper - lower) * nodes[order][None, :] + upper + lower
    )
    normalized = np.broadcast_to(0.5 * weights[order][None, :], points.shape)
    return np.ascontiguousarray(points), np.ascontiguousarray(normalized)


def test_room_calibration_then_frozen_hot_fit_uses_public_api_only():
    flight_path_m = 25.0
    timing_offset_us = 0.7
    n_bins = 18
    detector_edges = np.linspace(
        nereids.energy_to_tof(9.0, flight_path_m) + timing_offset_us,
        nereids.energy_to_tof(5.0, flight_path_m) + timing_offset_us,
        n_bins + 1,
    )
    energy_edges = np.array(
        [
            nereids.tof_to_energy(time - timing_offset_us, flight_path_m)
            for time in detector_edges[::-1]
        ]
    )
    points, weights = _kronrod_cells(energy_edges)
    profile = nereids.IcShapeProfile(
        alpha_sqrt_coefficient=0.0,
        alpha_offset=4.0,
        beta_sqrt_coefficient=0.0,
        beta_offset=0.5,
        slow_fraction=0.0,
        channel_fwhm_us=0.0,
        synthesis_energies=16,
        synthesis_times=200,
    )
    mean_delay_us = 3.0 / profile.alpha_offset
    effective_path_m = flight_path_m + mean_delay_us / nereids.energy_to_tof(
        profile.pivot_energy_ev, 1.0
    )
    physical_parameters = np.array(
        [timing_offset_us, effective_path_m, 1.0, 7.0e-4]
    )
    response = nereids.IkedaCarpenter(
        flight_path_m=flight_path_m,
        e_min_ev=float(energy_edges[0] * (1.0 - 1.0e-12)),
        e_max_ev=float(energy_edges[-1] * (1.0 + 1.0e-12)),
        alpha=nereids.EnergyLaw.const(4.0),
        beta=0.5,
        r=nereids.EnergyLaw.const(0.0),
        n_energies=profile.synthesis_energies,
        n_tau=profile.synthesis_times,
        channel_fwhm_us=0.0,
    )
    isotope = nereids.create_resonance_data(
        73,
        181,
        179.0,
        7.0,
        [(6.2, 0.08, 0.02, 0.04), (7.7, 0.12, 0.03, 0.05)],
    )
    room_sigma = np.asarray(
        nereids.precompute_cross_sections(
            points.ravel(), [isotope], temperature_k=300.0
        )[0]
    ).reshape(points.shape)
    matrix = nereids.DetectorResponseMatrix(
        points.ravel(), detector_edges, response, timing_offset_us
    )
    source = 1.0e6 * (1.0 + 0.1 * np.linspace(-1.0, 1.0, n_bins))
    integrated_source = np.repeat(source, points.shape[1]) * weights.ravel()
    open_counts = np.asarray(matrix.project(integrated_source))[::-1]
    room_counts = np.asarray(
        matrix.project(
            integrated_source
            * np.exp(-physical_parameters[3] * room_sigma.ravel())
        )
    )[::-1]
    original_room = room_counts.copy()
    original_open = open_counts.copy()

    calibration = nereids.calibrate_aggregated_1d(
        detector_time_edges_us=detector_edges,
        open_counts=open_counts,
        room_counts=room_counts,
        sample_over_open_exposure=1.0,
        isotopes=[isotope],
        room_temperature_k=300.0,
        reference_flight_path_m=flight_path_m,
        reference_timing_offset_us=timing_offset_us,
        initial_physical_parameters=physical_parameters,
        fit_energy_range_ev=(5.1, 8.9),
        ic_profile=profile,
        physical_lower_bounds=(0.4, effective_path_m - 0.06, 0.8, 5.0e-4),
        physical_upper_bounds=(1.0, effective_path_m + 0.06, 1.2, 9.0e-4),
        background_lower_bounds=(0.9, -1.0e-6, -1.0e-6, -1.0e-6),
        background_upper_bounds=(1.1, 1.0e-6, 1.0e-6, 1.0e-6),
        outer_max_evaluations=8,
        inner_max_evaluations=30,
        source_max_iterations=500,
    )

    assert calibration.source.converged
    assert calibration.fit.converged
    np.testing.assert_array_equal(room_counts, original_room)
    assert not np.shares_memory(calibration.source.observed_open_counts, open_counts)
    open_counts[:] = 1.0
    np.testing.assert_array_equal(
        calibration.source.observed_open_counts, original_open
    )
    np.testing.assert_allclose(
        calibration.fit.signal_prediction
        + calibration.fit.background_prediction,
        calibration.fit.model_prediction,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(calibration.fit.poisson_uncertainty, calibration.fit.uncertainty)
    np.testing.assert_allclose(calibration.fit.poisson_residual, calibration.fit.residual)
    assert calibration.fit.fit_energy_range_ev == (5.1, 8.9)
    assert calibration.fit.open_variance_factor == 1.0
    assert calibration.fit.sample_variance_factor == 1.0
    assert calibration.fit.rms_residual < 2.0e-4
    assert calibration.fit.parameters["density_atoms_per_barn"] == pytest.approx(
        physical_parameters[3], abs=2.0e-7
    )

    hot_temperature_k = 700.0
    hot_density = 7.5e-4
    hot_sigma = np.asarray(
        nereids.precompute_cross_sections(
            points.ravel(), [isotope], temperature_k=hot_temperature_k
        )[0]
    ).reshape(points.shape)
    hot_counts = np.asarray(
        matrix.project(integrated_source * np.exp(-hot_density * hot_sigma.ravel()))
    )[::-1]
    original_hot = hot_counts.copy()
    hot = nereids.fit_frozen_aggregated_1d(
        calibration,
        hot_counts=hot_counts,
        sample_over_open_exposure=1.0,
        initial_temperature_k=650.0,
        initial_density_atoms_per_barn=7.0e-4,
        temperature_bounds_k=(100.0, 1200.0),
        density_bounds_atoms_per_barn=(1.0e-4, 2.0e-3),
        background_lower_bounds=(0.9, -1.0e-6, -1.0e-6, -1.0e-6),
        background_upper_bounds=(1.1, 1.0e-6, 1.0e-6, 1.0e-6),
        max_evaluations=12,
    )

    np.testing.assert_array_equal(hot_counts, original_hot)
    np.testing.assert_allclose(
        hot.signal_prediction + hot.background_prediction,
        hot.model_prediction,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(hot.poisson_uncertainty, hot.uncertainty)
    np.testing.assert_allclose(hot.poisson_residual, hot.residual)
    assert hot.fit_energy_range_ev == calibration.fit.fit_energy_range_ev
    assert hot.parameters["temperature_k"] == pytest.approx(hot_temperature_k, abs=0.2)
    assert hot.parameters["density_atoms_per_barn"] == pytest.approx(
        hot_density, abs=2.0e-7
    )
    assert hot.rms_residual < 2.0e-3

    failed_source = replace(calibration.source, converged=False)
    failed_calibration = replace(calibration, source=failed_source)
    with pytest.raises(RuntimeError, match="non-converged open-beam source"):
        nereids.fit_frozen_aggregated_1d(
            failed_calibration,
            hot_counts=hot_counts,
            sample_over_open_exposure=1.0,
            initial_temperature_k=650.0,
            initial_density_atoms_per_barn=7.0e-4,
        )


def test_source_iteration_limit_must_be_positive():
    with pytest.raises(ValueError, match="source_max_iterations must be a positive integer"):
        nereids.calibrate_aggregated_1d(
            detector_time_edges_us=np.array([1.0, 2.0]),
            open_counts=np.array([10.0]),
            room_counts=np.array([9.0]),
            sample_over_open_exposure=1.0,
            isotopes=[object()],
            room_temperature_k=300.0,
            reference_flight_path_m=25.0,
            reference_timing_offset_us=0.0,
            initial_physical_parameters=(0.0, 25.0, 1.0, 1.0e-3),
            fit_energy_range_ev=(1.0, 2.0),
            ic_profile=nereids.VENUS_UDR_MATCHED_IC_PROFILE,
            physical_lower_bounds=(-1.0, 24.0, 0.5, 0.0),
            physical_upper_bounds=(1.0, 26.0, 2.0, 0.01),
            source_max_iterations=0,
        )


def test_debye_effective_temperature_matches_solid_tantalum_reference():
    observed = nereids.solid_debye_effective_temperature(300.82681290235263, 217.0)
    assert observed == pytest.approx(308.60538943915935, rel=2.0e-14)


def test_profiled_two_arm_residual_is_zero_at_observed_ratio():
    open_counts = np.array([100.0, 150.0, 200.0])
    sample_counts = np.array([40.0, 90.0, 160.0])
    exposure = 0.8
    ratio = sample_counts / open_counts / exposure
    residual = nereids.profiled_two_arm_residual(
        open_counts, sample_counts, exposure, ratio
    )
    np.testing.assert_allclose(residual, 0.0, atol=2.0e-7)
