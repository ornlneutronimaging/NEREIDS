"""Comprehensive pytest suite for the NEREIDS Python bindings.

All tests use synthetic data built with ``nereids.create_resonance_data()``
so no network access or ENDF downloads are required.
"""

import os
import tempfile

import numpy as np
import pytest

import nereids

from _fixtures import _make_single_resonance


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def u238_data():
    """Single-resonance U-238-like isotope (6.674 eV resonance)."""
    return _make_single_resonance()


@pytest.fixture
def energy_grid():
    """Fine energy grid around the 6.67 eV resonance."""
    return np.linspace(1.0, 30.0, 2000)


# ===========================================================================
# Element utilities
# ===========================================================================


class TestElementUtilities:
    """Tests for element_symbol, element_name, parse_isotope_str, etc."""

    def test_element_symbol(self):
        assert nereids.element_symbol(92) == "U"
        assert nereids.element_symbol(1) == "H"
        assert nereids.element_symbol(26) == "Fe"

    def test_element_symbol_edge_cases(self):
        # Z=0 is the neutron in ENDF convention
        assert nereids.element_symbol(0) == "n"
        assert nereids.element_symbol(999) is None

    def test_element_name(self):
        assert nereids.element_name(92) == "Uranium"
        assert nereids.element_name(1) == "Hydrogen"
        assert nereids.element_name(26) == "Iron"

    def test_element_name_edge_cases(self):
        # Z=0 is the neutron in ENDF convention
        assert nereids.element_name(0) == "neutron"
        assert nereids.element_name(999) is None

    def test_parse_isotope_str(self):
        result = nereids.parse_isotope_str("U-238")
        assert result == (92, 238)

    def test_parse_isotope_str_various(self):
        assert nereids.parse_isotope_str("Fe-56") == (26, 56)
        assert nereids.parse_isotope_str("H-1") == (1, 1)

    def test_parse_isotope_str_invalid(self):
        assert nereids.parse_isotope_str("invalid") is None
        assert nereids.parse_isotope_str("Xx-999") is None

    def test_natural_abundance(self):
        abundance = nereids.natural_abundance(92, 238)
        assert abundance is not None
        # U-238 is ~99.27% abundant
        assert 0.99 < abundance < 1.0

    def test_natural_abundance_u235(self):
        abundance = nereids.natural_abundance(92, 235)
        assert abundance is not None
        # U-235 is ~0.72%
        assert 0.005 < abundance < 0.01

    def test_natural_abundance_synthetic(self):
        # Tc-99 is synthetic -- may return None
        result = nereids.natural_abundance(43, 99)
        # either None or 0.0 is acceptable
        assert result is None or result == 0.0

    def test_natural_isotopes(self):
        isotopes = nereids.natural_isotopes(92)
        assert len(isotopes) > 0
        # U should have at least U-234, U-235, U-238
        mass_numbers = [a for ((_z, a), _frac) in isotopes]
        assert 238 in mass_numbers
        assert 235 in mass_numbers
        # Abundances should sum to ~1.0
        total = sum(frac for (_, frac) in isotopes)
        assert abs(total - 1.0) < 0.01

    def test_natural_isotopes_iron(self):
        isotopes = nereids.natural_isotopes(26)
        assert len(isotopes) >= 4  # Fe has 4 stable isotopes
        mass_numbers = [a for ((_z, a), _frac) in isotopes]
        assert 56 in mass_numbers  # Fe-56 is most abundant


# ===========================================================================
# TOF / energy conversion
# ===========================================================================


class TestTofConversion:
    """Tests for tof_to_energy, energy_to_tof, tof_to_energy_centers."""

    def test_tof_energy_roundtrip(self):
        """Energy -> TOF -> energy should roundtrip to machine precision."""
        energy = 6.67  # eV
        flight_path = 20.0  # meters
        tof = nereids.energy_to_tof(energy, flight_path)
        assert tof > 0.0
        recovered = nereids.tof_to_energy(tof, flight_path)
        assert abs(recovered - energy) / energy < 1e-10

    def test_tof_energy_roundtrip_multiple(self):
        """Roundtrip at several energies."""
        flight_path = 15.0
        for energy in [0.025, 1.0, 6.67, 100.0, 1000.0]:
            tof = nereids.energy_to_tof(energy, flight_path)
            recovered = nereids.tof_to_energy(tof, flight_path)
            assert abs(recovered - energy) / energy < 1e-10

    def test_higher_energy_shorter_tof(self):
        """Higher energy neutrons should have shorter time-of-flight."""
        fp = 20.0
        tof_low = nereids.energy_to_tof(1.0, fp)
        tof_high = nereids.energy_to_tof(100.0, fp)
        assert tof_high < tof_low

    def test_tof_to_energy_centers(self):
        """TOF bin edges to energy centers."""
        flight_path = 20.0
        # Create TOF edges in ascending order
        tof_edges = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        centers = nereids.tof_to_energy_centers(tof_edges, flight_path)
        assert len(centers) == len(tof_edges) - 1
        # The function returns energy centers in ascending order
        for i in range(len(centers) - 1):
            assert centers[i] < centers[i + 1], f"Expected ascending energy: centers[{i}]={centers[i]} >= centers[{i+1}]={centers[i+1]}"

    def test_tof_to_energy_rejects_non_positive_and_non_finite(self):
        """Non-positive / non-finite TOF or flight path must raise ValueError.

        Without the guard the conversion silently returned a positive energy
        for negative TOF (v**2 hides the sign) or +inf at zero TOF, masking a
        bad TOF axis downstream.
        """
        fp = 20.0
        for bad in (-100.0, 0.0, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                nereids.tof_to_energy(bad, fp)
        for bad_fp in (-1.0, 0.0, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                nereids.tof_to_energy(100.0, bad_fp)

    def test_energy_to_tof_rejects_non_positive_and_non_finite(self):
        """Non-positive / non-finite energy or flight path must raise ValueError."""
        fp = 20.0
        for bad in (-1.0, 0.0, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                nereids.energy_to_tof(bad, fp)
        for bad_fp in (-1.0, 0.0, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                nereids.energy_to_tof(10.0, bad_fp)


# ===========================================================================
# ResonanceData creation
# ===========================================================================


class TestResonanceData:
    """Tests for create_resonance_data and ResonanceData properties."""

    def test_basic_creation(self, u238_data):
        assert u238_data.z == 92
        assert u238_data.a == 238
        assert abs(u238_data.awr - 236.006) < 0.01
        assert u238_data.n_resonances == 1

    def test_scattering_radius(self, u238_data):
        assert u238_data.scattering_radius > 0.0

    def test_target_spin(self, u238_data):
        assert u238_data.target_spin == 0.0

    def test_l_values(self, u238_data):
        assert 0 in u238_data.l_values

    def test_repr(self, u238_data):
        r = repr(u238_data)
        assert "ResonanceData" in r
        assert "Z=92" in r
        assert "A=238" in r

    def test_multi_resonance(self):
        """Create with multiple resonances."""
        data = nereids.create_resonance_data(
            z=92,
            a=238,
            awr=236.006,
            scattering_radius=9.48,
            resonances=[
                (6.67, 0.5, 0.0015, 0.023),
                (20.87, 0.5, 0.010, 0.023),
                (36.68, 0.5, 0.034, 0.023),
            ],
            target_spin=0.0,
        )
        assert data.n_resonances == 3

    def test_l_groups(self):
        """Create with explicit L-groups."""
        data = nereids.create_resonance_data(
            z=26,
            a=56,
            awr=55.347,
            scattering_radius=6.0,
            resonances=[],  # ignored when l_groups is provided
            l_groups=[
                (0, [(7.8, 0.5, 0.001, 0.01)]),
                (1, [(27.4, 1.5, 0.002, 0.015)]),
            ],
        )
        assert data.n_resonances == 2
        l_vals = data.l_values
        assert 0 in l_vals
        assert 1 in l_vals

    def test_slbw_formalism(self):
        """Create SLBW formalism data."""
        data = nereids.create_resonance_data(
            z=92,
            a=238,
            awr=236.006,
            scattering_radius=9.48,
            resonances=[(6.67, 0.5, 0.0015, 0.023)],
            formalism="slbw",
        )
        assert data.n_resonances == 1

    def test_invalid_formalism(self):
        with pytest.raises(ValueError, match="Unknown formalism"):
            nereids.create_resonance_data(
                z=92,
                a=238,
                awr=236.006,
                scattering_radius=9.48,
                resonances=[(6.67, 0.5, 0.0015, 0.023)],
                formalism="bogus",
            )


# ===========================================================================
# Cross-sections
# ===========================================================================


class TestCrossSections:
    """Tests for cross_sections()."""

    def test_basic(self, u238_data, energy_grid):
        xs = nereids.cross_sections(energy_grid, u238_data)
        assert isinstance(xs, dict)
        for key in ("total", "elastic", "capture", "fission"):
            assert key in xs
            arr = np.asarray(xs[key])
            assert arr.shape == energy_grid.shape

    def test_non_negative(self, u238_data, energy_grid):
        xs = nereids.cross_sections(energy_grid, u238_data)
        for key in ("total", "elastic", "capture", "fission"):
            arr = np.asarray(xs[key])
            assert np.all(arr >= 0.0), f"{key} has negative values"

    def test_peak_near_resonance(self, u238_data):
        """Total cross-section should peak near 6.67 eV."""
        energies = np.linspace(1.0, 30.0, 5000)
        xs = nereids.cross_sections(energies, u238_data)
        total = np.asarray(xs["total"])
        peak_idx = np.argmax(total)
        peak_energy = energies[peak_idx]
        assert abs(peak_energy - 6.67) < 0.5, (
            f"Peak at {peak_energy} eV, expected near 6.67 eV"
        )

    def test_capture_dominates_at_resonance(self, u238_data):
        """For U-238 at low energies, capture cross-section should be significant."""
        energies = np.array([6.67])
        xs = nereids.cross_sections(energies, u238_data)
        capture = float(np.asarray(xs["capture"])[0])
        assert capture > 0.0


# ===========================================================================
# Forward model (transmission)
# ===========================================================================


class TestForwardModel:
    """Tests for forward_model()."""

    def test_basic(self, u238_data, energy_grid):
        t = nereids.forward_model(energy_grid, [(u238_data, 0.001)])
        t = np.asarray(t)
        assert t.shape == energy_grid.shape

    def test_bounded_0_1(self, u238_data, energy_grid):
        """Transmission should be between 0 and 1."""
        t = np.asarray(nereids.forward_model(energy_grid, [(u238_data, 0.001)]))
        assert np.all(t >= 0.0)
        assert np.all(t <= 1.0)

    def test_dip_near_resonance(self, u238_data):
        """Transmission should dip near the resonance energy."""
        energies = np.linspace(1.0, 30.0, 5000)
        t = np.asarray(nereids.forward_model(energies, [(u238_data, 0.001)]))
        min_idx = np.argmin(t)
        min_energy = energies[min_idx]
        assert abs(min_energy - 6.67) < 0.5

    def test_zero_density_is_unity(self, u238_data, energy_grid):
        """Zero density -> transmission = 1."""
        t = np.asarray(nereids.forward_model(energy_grid, [(u238_data, 0.0)]))
        np.testing.assert_allclose(t, 1.0, atol=1e-12)

    def test_higher_density_lower_transmission(self, u238_data, energy_grid):
        """Higher density should give lower (or equal) transmission everywhere."""
        t_low = np.asarray(
            nereids.forward_model(energy_grid, [(u238_data, 0.0005)])
        )
        t_high = np.asarray(
            nereids.forward_model(energy_grid, [(u238_data, 0.005)])
        )
        assert np.all(t_high <= t_low + 1e-12)

    def test_temperature_kwarg(self, u238_data, energy_grid):
        """forward_model with temperature_k should not raise."""
        t = nereids.forward_model(
            energy_grid, [(u238_data, 0.001)], temperature_k=300.0
        )
        assert len(t) == len(energy_grid)


# ===========================================================================
# Beer-Lambert
# ===========================================================================


class TestBeerLambert:
    """Tests for the standalone beer_lambert() function."""

    def test_basic(self):
        xs = np.array([10.0, 20.0, 30.0])
        t = np.asarray(nereids.beer_lambert(xs, 0.001))
        expected = np.exp(-0.001 * xs)
        np.testing.assert_allclose(t, expected, rtol=1e-12)

    def test_zero_thickness(self):
        xs = np.array([10.0, 20.0])
        t = np.asarray(nereids.beer_lambert(xs, 0.0))
        np.testing.assert_allclose(t, 1.0, atol=1e-15)


# ===========================================================================
# Doppler broadening
# ===========================================================================


class TestDopplerBroadening:
    """Tests for doppler_broaden()."""

    def test_zero_temperature_passthrough(self, u238_data, energy_grid):
        """At T=0, doppler_broaden should return the input unchanged."""
        xs_dict = nereids.cross_sections(energy_grid, u238_data)
        xs_total = np.asarray(xs_dict["total"])
        broadened = np.asarray(
            nereids.doppler_broaden(energy_grid, xs_total, 236.006, 0.0)
        )
        np.testing.assert_allclose(broadened, xs_total, rtol=1e-12)

    def test_broadened_peak_lower(self, u238_data):
        """Doppler broadening at 300K should reduce the peak height."""
        energies = np.linspace(1.0, 30.0, 5000)
        xs_dict = nereids.cross_sections(energies, u238_data)
        xs_total = np.asarray(xs_dict["total"])
        broadened = np.asarray(
            nereids.doppler_broaden(energies, xs_total, 236.006, 300.0)
        )
        assert np.max(broadened) < np.max(xs_total)

    def test_broadened_preserves_length(self, energy_grid, u238_data):
        xs_dict = nereids.cross_sections(energy_grid, u238_data)
        xs_total = np.asarray(xs_dict["total"])
        broadened = np.asarray(
            nereids.doppler_broaden(energy_grid, xs_total, 236.006, 300.0)
        )
        assert broadened.shape == xs_total.shape

    def test_shape_mismatch_raises(self):
        e = np.linspace(1.0, 10.0, 100)
        xs = np.ones(50)  # wrong length
        with pytest.raises(ValueError):
            nereids.doppler_broaden(e, xs, 236.0, 300.0)


# ===========================================================================
# Resolution broadening
# ===========================================================================


class TestResolutionBroadening:
    """Tests for resolution_broaden()."""

    def test_zero_resolution_passthrough(self, u238_data):
        """Zero timing and path uncertainty -> no change."""
        energies = np.linspace(1.0, 30.0, 2000)
        xs_dict = nereids.cross_sections(energies, u238_data)
        xs_total = np.asarray(xs_dict["total"])
        result = np.asarray(
            nereids.resolution_broaden(energies, xs_total, 20.0, 0.0, 0.0)
        )
        np.testing.assert_allclose(result, xs_total, rtol=1e-12)

    def test_broadening_reduces_peak(self, u238_data):
        """Resolution broadening should reduce peak height."""
        energies = np.linspace(1.0, 30.0, 5000)
        xs_dict = nereids.cross_sections(energies, u238_data)
        xs_total = np.asarray(xs_dict["total"])
        broadened = np.asarray(
            nereids.resolution_broaden(energies, xs_total, 20.0, 0.5, 0.001)
        )
        assert np.max(broadened) < np.max(xs_total)

    def test_shape_preserved(self, u238_data):
        energies = np.linspace(1.0, 30.0, 1000)
        xs_dict = nereids.cross_sections(energies, u238_data)
        xs_total = np.asarray(xs_dict["total"])
        broadened = np.asarray(
            nereids.resolution_broaden(energies, xs_total, 20.0, 0.3, 0.001)
        )
        assert broadened.shape == xs_total.shape

    def test_shape_mismatch_raises(self):
        e = np.linspace(1.0, 10.0, 100)
        xs = np.ones(50)
        with pytest.raises(ValueError):
            nereids.resolution_broaden(e, xs, 20.0, 0.5, 0.001)

    def test_invalid_flight_path_raises(self):
        e = np.linspace(1.0, 10.0, 100)
        xs = np.ones(100)
        with pytest.raises(ValueError, match="flight_path_m"):
            nereids.resolution_broaden(e, xs, -1.0, 0.5, 0.001)


class TestTabulatedKernelOrientation:
    """Regression for issue #631: tabulated kernels must be applied as a
    convolution (theory gathered at t - dt), not time-mirrored.

    Kernel convention: mode at TOF offset 0, delayed-emission tail at
    POSITIVE offsets. A delayed neutron measured at TOF t really flew
    t - dt (it is faster than nominal), so a broadened symmetric dip
    must acquire its tail toward later TOF = LOWER apparent energy:
    centroid shift and skew both strictly negative. The pre-fix code
    produced +0.297 eV / +1.53 on this exact setup.

    Also the first coverage of the load_resolution / apply_resolution
    bindings.
    """

    @staticmethod
    def _write_kernel(path):
        dt = np.linspace(-1.0, 15.0, 499)  # us
        amp = np.where(dt >= 0, np.exp(-dt / 3.0), 0.0)
        amp[np.argmin(np.abs(dt))] = 1.0  # mode at offset 0
        lines = [
            "synthetic asymmetric kernel, tail at positive TOF offsets",
            "-----",
        ]
        for eref in (10.0, 100.0):
            lines.append(f"   {eref:.5e}   0.00000e+000")
            lines += [f"{d:.15f} {a:.15e}" for d, a in zip(dt, amp)]
            lines.append("")
        path.write_text("\n".join(lines))

    def test_delayed_tail_shifts_dip_to_lower_energy(self, tmp_path):
        kernel_path = tmp_path / "synthetic_kernel.txt"
        self._write_kernel(kernel_path)
        tab = nereids.load_resolution(str(kernel_path), 25.0)

        e0 = 20.0
        e = np.linspace(e0 - 3, e0 + 3, 24001)
        spec = 1.0 - 0.6 * np.exp(-0.5 * ((e - e0) / 0.02) ** 2)
        b = np.asarray(nereids.apply_resolution(e, spec, tab))

        # Non-vacuity: the kernel must visibly reshape the dip.
        assert np.max(np.abs(b - spec)) > 0.01 * np.max(np.abs(spec))

        a = 1.0 - b
        mu = np.sum(e * a) / np.sum(a)
        sd = np.sqrt(np.sum((e - mu) ** 2 * a) / np.sum(a))
        skew = np.sum(((e - mu) / sd) ** 3 * a) / np.sum(a)

        assert mu - e0 < 0, (
            f"centroid must shift to LOWER energy (delayed arrival): "
            f"{mu - e0:+.4f} eV — positive means the kernel was applied "
            f"time-mirrored"
        )
        assert skew < 0, (
            f"broadened dip must be skewed toward LOWER energy: {skew:+.3f}"
        )


class TestTabulatedKernelWidthInterpolation:
    """Regression for issue #632: kernel width between reference energies
    must follow the physical power law, not the arithmetic blend chord.

    Symmetric Gaussian kernels (mode 0, immune to the #631 orientation)
    at 10 and 50 eV with sigma_t ~ E^-1/2 widths: a width-correct
    interpolation at 20 eV gives sigma_t = 2/sqrt(2) = 1.4142 us. The
    pre-fix element-wise blend measured 1.524 us (+7.8%); the issue's
    acceptance bar is <3% at the midpoint.
    """

    TOF_FACTOR = 72.298254398292800  # us*sqrt(eV)/m (resolution.rs)
    L = 25.0
    SIGMA_10EV = 2.0

    @classmethod
    def _write_kernel(cls, path):
        lines = ["synthetic Gaussian kernels, sigma_t ~ E^-1/2", "-----"]
        for eref in (10.0, 50.0):
            sigma = cls.SIGMA_10EV * (eref / 10.0) ** -0.5
            d = np.linspace(-6.0 * sigma, 6.0 * sigma, 499)
            a = np.exp(-0.5 * (d / sigma) ** 2)
            lines.append(f"   {eref:.5e}   0.00000e+000")
            lines += [f"{x:.15f} {y:.15e}" for x, y in zip(d, a)]
            lines.append("")
        path.write_text("\n".join(lines))

    def _measured_sigma_t(self, tab, e0):
        e = np.linspace(e0 * 0.7, e0 * 1.3, 30001)
        spec = 1.0 - 0.8 * np.exp(-0.5 * ((e - e0) / (e0 * 1e-4)) ** 2)
        b = np.asarray(nereids.apply_resolution(e, spec, tab))
        # Non-vacuity: the kernel must visibly reshape the dip.
        assert np.max(np.abs(b - spec)) > 0.01
        a = 1.0 - b
        mu = np.sum(e * a) / np.sum(a)
        sd_e = np.sqrt(np.sum((e - mu) ** 2 * a) / np.sum(a))
        tof = self.TOF_FACTOR * self.L / np.sqrt(e0)
        return sd_e * tof / (2.0 * e0)

    def test_width_follows_power_law_between_references(self, tmp_path):
        kernel_path = tmp_path / "synthetic_two_ref.txt"
        self._write_kernel(kernel_path)
        tab = nereids.load_resolution(str(kernel_path), self.L)

        for e0, expected in [
            (10.0, self.SIGMA_10EV),
            (50.0, self.SIGMA_10EV / np.sqrt(5.0)),
            (20.0, self.SIGMA_10EV / np.sqrt(2.0)),
        ]:
            measured = self._measured_sigma_t(tab, e0)
            surplus = measured / expected - 1.0
            assert abs(surplus) < 0.03, (
                f"kernel width at {e0} eV is {surplus * 100:+.1f}% off the "
                f"power law (measured {measured:.4f} us, expected "
                f"{expected:.4f} us; the pre-fix blend gave +7.8% at 20 eV)"
            )


# ===========================================================================
# LM fitting
# ===========================================================================


# (TestFitSpectrumLM and TestFitSpectrumPoisson removed — old fit_spectrum API deleted)


# ===========================================================================
# Spatial mapping (LM)
# ===========================================================================


class TestSpatialMapTransmission:
    """Tests for spatial_map_typed() with from_transmission (LM solver)."""

    def test_basic_spatial_map(self, u238_data):
        """3x3 spatial map should return correct shapes."""
        energies = np.linspace(1.0, 30.0, 200)
        true_density = 0.002
        ny, nx = 3, 3

        # Build a (n_e, ny, nx) transmission cube at 293.6 K (default)
        t_1d = np.asarray(
            nereids.forward_model(energies, [(u238_data, true_density)])
        )
        trans = np.tile(t_1d[:, None, None], (1, ny, nx))
        unc = np.full_like(trans, 0.005)

        data = nereids.from_transmission(trans, unc)
        result = nereids.spatial_map_typed(
            data, energies, [u238_data], max_iter=50
        )
        # Should return SpatialResult
        assert hasattr(result, "density_maps")
        assert hasattr(result, "uncertainty_maps")
        assert hasattr(result, "chi_squared_map")
        assert hasattr(result, "converged_map")
        assert hasattr(result, "n_converged")
        assert hasattr(result, "n_total")
        assert hasattr(result, "isotope_names")

        density_maps = result.density_maps
        assert len(density_maps) == 1
        dmap = np.asarray(density_maps[0])
        assert dmap.shape == (ny, nx)
        assert trans.shape[0] == len(energies)

        converged = np.asarray(result.converged_map)
        assert converged.shape == (ny, nx)
        assert result.n_total == ny * nx

        # Density recovery: fitted values should be close to ground truth
        np.testing.assert_allclose(dmap, true_density, rtol=0.15)

    def test_spatial_map_repr(self, u238_data):
        energies = np.linspace(1.0, 30.0, 100)
        t_1d = np.asarray(
            nereids.forward_model(energies, [(u238_data, 0.001)])
        )
        trans = np.tile(t_1d[:, None, None], (1, 2, 2))
        unc = np.full_like(trans, 0.01)

        data = nereids.from_transmission(trans, unc)
        # Use default temperature_k (293.6) to match forward_model default
        result = nereids.spatial_map_typed(
            data, energies, [u238_data], max_iter=20
        )
        r = repr(result)
        assert "SpatialResult" in r

    def test_spatial_map_back_d_f_maps_none_when_disabled(self, u238_data):
        """Issue #538: ``back_d_map`` / ``back_f_map`` must be ``None``
        whenever ``fit_back_d`` / ``fit_back_f`` are left at their
        defaults — even when ``background=True`` attaches the polynomial
        terms.  This is the "exponential tail never engaged" gate."""
        energies = np.linspace(1.0, 30.0, 200)
        t_1d = np.asarray(nereids.forward_model(energies, [(u238_data, 0.001)]))
        trans = np.tile(t_1d[:, None, None], (1, 2, 2))
        unc = np.full_like(trans, 0.01)

        data = nereids.from_transmission(trans, unc)
        result = nereids.spatial_map_typed(
            data,
            energies,
            [u238_data],
            max_iter=30,
            background=True,
            # fit_back_d / fit_back_f left at False (default).
        )
        # Polynomial background maps materialise.
        assert result.anorm_map is not None
        assert result.background_maps is not None
        # Exponential tail maps stay None when not requested.
        assert result.back_d_map is None, (
            f"back_d_map must be None when fit_back_d=False, got "
            f"{result.back_d_map!r}"
        )
        assert result.back_f_map is None, (
            f"back_f_map must be None when fit_back_f=False, got "
            f"{result.back_f_map!r}"
        )

    def test_spatial_map_back_d_f_maps_some_when_enabled(self, u238_data):
        """Issue #538: ``back_d_map`` / ``back_f_map`` are 2-D float64
        arrays when ``background=True`` and both ``fit_back_d`` /
        ``fit_back_f`` are set.  Uses the *real* PyO3 binding (not a
        SimpleNamespace stub) per the project's PyO3-contract testing
        convention.

        The original fixture was resonance-only
        (no exponential tail injected), which made BackD/BackF
        unidentifiable — ``anorm`` absorbed them and LM stalled at
        ``back_d ≈ 0`` with `converged = false` per pixel, so the
        finite-value assertion was vacuous or flaky.  Mirror the Rust
        coverage in
        ``test_spatial_map_back_d_f_maps_some_when_fit_enabled`` by
        injecting a known ``BackD * exp(-BackF / √E)`` tail on top of
        the resonance-only transmission so the BackD/BackF Jacobian
        columns carry non-degenerate signal."""
        # 1.0 to 11.0 eV (101 bins) matches the Rust fixture and keeps
        # the 1/√E factor identifiable across the range.
        energies = np.linspace(1.0, 11.0, 101)
        t_1d = np.asarray(nereids.forward_model(energies, [(u238_data, 0.001)]))
        # Inject a known exponential tail so the BackD/BackF columns
        # are not degenerate.  Values mirror the Rust test.
        true_back_d = 0.03
        true_back_f = 2.0
        tail = true_back_d * np.exp(-true_back_f / np.sqrt(energies))
        t_1d = t_1d + tail
        ny, nx = 2, 2
        trans = np.tile(t_1d[:, None, None], (1, ny, nx))
        unc = np.full_like(trans, 0.01)

        data = nereids.from_transmission(trans, unc)
        result = nereids.spatial_map_typed(
            data,
            energies,
            [u238_data],
            max_iter=500,
            background=True,
            fit_back_d=True,
            fit_back_f=True,
            back_d_init=0.01,
            back_f_init=1.0,
        )
        assert result.back_d_map is not None, (
            "back_d_map must be populated when fit_back_d=True"
        )
        assert result.back_f_map is not None, (
            "back_f_map must be populated when fit_back_f=True"
        )
        bd = np.asarray(result.back_d_map)
        bf = np.asarray(result.back_f_map)
        assert bd.shape == (ny, nx)
        assert bf.shape == (ny, nx)
        assert bd.dtype == np.float64
        assert bf.dtype == np.float64
        # At least one converged pixel must populate a finite back_d/back_f
        # value, otherwise the gating is vacuous (matches the Rust
        # `n_converged > 0` precondition).
        converged = np.asarray(result.converged_map)
        assert converged.any(), (
            "test fixture produced zero converged pixels — the LM fit failed "
            "to recover BackD/BackF on the injected tail; the test fixture "
            "is no longer exercising the gating contract"
        )
        finite_bd = np.isfinite(bd[converged])
        finite_bf = np.isfinite(bf[converged])
        assert finite_bd.any() and finite_bf.any(), (
            f"at least one converged pixel must produce a finite back_d/back_f "
            f"(finite_bd={finite_bd.sum()}, finite_bf={finite_bf.sum()})"
        )

    def test_spatial_map_back_d_f_requires_background_kwarg(self, u238_data):
        """Issue #538: ``fit_back_d`` / ``fit_back_f`` with
        ``background=False`` is rejected at the binding boundary —
        the exponential tail of the SAMMY background only exists when
        the polynomial background is attached, so silently producing
        ``back_d_map=None`` would be misleading."""
        energies = np.linspace(1.0, 30.0, 100)
        t_1d = np.asarray(nereids.forward_model(energies, [(u238_data, 0.001)]))
        trans = np.tile(t_1d[:, None, None], (1, 2, 2))
        unc = np.full_like(trans, 0.01)
        data = nereids.from_transmission(trans, unc)
        with pytest.raises(ValueError, match="background=True"):
            nereids.spatial_map_typed(
                data,
                energies,
                [u238_data],
                max_iter=10,
                background=False,
                fit_back_d=True,
                fit_back_f=True,
            )


# ===========================================================================
# Spatial mapping (Poisson)
# ===========================================================================


class TestSpatialMapCounts:
    """Tests for spatial_map_typed() with from_counts (Poisson KL solver)."""

    def test_counts_spatial_map(self, u238_data):
        """Poisson spatial map with synthetic count data."""
        energies = np.linspace(1.0, 30.0, 150)
        true_density = 0.002
        flux = 5000.0
        n_e = len(energies)
        ny, nx = 2, 2

        # Use default temperature (293.6 K) for consistency
        t_1d = np.asarray(
            nereids.forward_model(energies, [(u238_data, true_density)])
        )

        rng = np.random.default_rng(999)
        open_beam = np.full((n_e, ny, nx), flux)
        sample = np.zeros((n_e, ny, nx))
        for y in range(ny):
            for x in range(nx):
                sample[:, y, x] = rng.poisson(flux * t_1d).astype(float)

        data = nereids.from_counts(sample, open_beam)
        result = nereids.spatial_map_typed(
            data,
            energies,
            [u238_data],
            max_iter=50,
        )
        # Should return SpatialResult (typed API always returns SpatialResult)
        assert hasattr(result, "density_maps")
        assert hasattr(result, "chi_squared_map")
        assert hasattr(result, "converged_map")

        density_maps = result.density_maps
        assert len(density_maps) == 1
        dmap = np.asarray(density_maps[0])
        assert dmap.shape == (ny, nx)

        # Density recovery: Poisson is noisier, so use wider tolerance
        np.testing.assert_allclose(dmap, true_density, rtol=0.5)

    def test_spatial_rejects_non_positive_c(self, u238_data):
        """Issue #458 V1: spatial_map_typed must reject c <= 0 or NaN at the
        binding boundary with a clear PyValueError, not let it crash deep in
        the solver as an opaque PyRuntimeError."""
        energies = np.linspace(1.0, 10.0, 20)
        n_e = len(energies)
        sample = np.full((n_e, 2, 2), 100.0)
        ob = np.full((n_e, 2, 2), 200.0)
        data = nereids.from_counts(sample, ob)

        for bad_c in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="c.*positive and finite"):
                nereids.spatial_map_typed(
                    data, energies, [u238_data], c=bad_c, max_iter=5
                )

    def test_spatial_c_validation_scoped_to_counts(self, u238_data):
        """Issue #458 V1: `c` is only consumed on counts
        inputs.  A transmission caller who passes `c=0.0` should NOT be
        rejected — the value is ignored on their path.  Rejecting it would
        produce a misleading error that doesn't apply to their input type.
        """
        energies = np.linspace(1.0, 10.0, 20)
        n_e = len(energies)
        t = np.full((n_e, 2, 2), 0.5)
        u = np.full((n_e, 2, 2), 0.01)
        data = nereids.from_transmission(t, u)

        # Should not raise — `c=0.0` is ignored on the transmission path.
        result = nereids.spatial_map_typed(
            data, energies, [u238_data], c=0.0, solver="lm", max_iter=5
        )
        assert result is not None  # call succeeded

    def test_spatial_rejects_bad_initial_densities(self, u238_data):
        """Issue #458 V2: spatial_map_typed must reject NaN / negative
        initial densities at the binding boundary."""
        energies = np.linspace(1.0, 10.0, 20)
        n_e = len(energies)
        sample = np.full((n_e, 2, 2), 100.0)
        ob = np.full((n_e, 2, 2), 200.0)
        data = nereids.from_counts(sample, ob)

        for bad in ([float("nan")], [-0.001], [0.001, float("inf")]):
            with pytest.raises(ValueError, match="initial_densities.*finite and non-negative"):
                nereids.spatial_map_typed(
                    data,
                    energies,
                    [u238_data] if len(bad) == 1 else [u238_data, u238_data],
                    initial_densities=bad,
                    max_iter=5,
                )

    def test_spatial_rejects_shape_mismatches(self, u238_data):
        """Issue #458 V3: spatial_map_typed must reject energies and
        dead_pixels shape mismatches upfront with a clear PyValueError,
        rather than letting them panic deep in the Rust pipeline."""
        energies = np.linspace(1.0, 10.0, 20)
        n_e = len(energies)
        sample = np.full((n_e, 3, 4), 100.0)
        ob = np.full((n_e, 3, 4), 200.0)
        data = nereids.from_counts(sample, ob)

        # energies length mismatch
        short_energies = energies[:10]
        with pytest.raises(ValueError, match="energies length.*data spectral axis length"):
            nereids.spatial_map_typed(data, short_energies, [u238_data], max_iter=5)

        # dead_pixels shape mismatch
        wrong_mask = np.zeros((5, 5), dtype=bool)
        with pytest.raises(ValueError, match="dead_pixels shape.*data spatial dims"):
            nereids.spatial_map_typed(
                data,
                energies,
                [u238_data],
                dead_pixels=wrong_mask,
                max_iter=5,
            )

    def test_spatial_rejects_bad_tzero_params(self, u238_data):
        """Issue #458: when `fit_energy_scale=True`,
        the TZERO kwargs `t0_init_us`, `l_scale_init`, and
        `energy_scale_flight_path_m` must be validated at the binding
        boundary.  Non-finite or non-positive values (for flight path)
        produced opaque PyRuntimeError from the solver rather than a clear
        PyValueError.
        """
        energies = np.linspace(1.0, 10.0, 20)
        n_e = len(energies)
        t = np.full((n_e, 2, 2), 0.5)
        u = np.full((n_e, 2, 2), 0.01)
        data = nereids.from_transmission(t, u)

        # t0_init_us non-finite
        for bad_t0 in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError, match="t0_init_us must be finite"):
                nereids.spatial_map_typed(
                    data, energies, [u238_data],
                    solver="lm",
                    fit_energy_scale=True,
                    t0_init_us=bad_t0,
                    max_iter=5,
                )

        # l_scale_init non-finite
        for bad_l in (float("nan"), float("inf")):
            with pytest.raises(ValueError, match="l_scale_init must be finite"):
                nereids.spatial_map_typed(
                    data, energies, [u238_data],
                    solver="lm",
                    fit_energy_scale=True,
                    l_scale_init=bad_l,
                    max_iter=5,
                )

        # flight_path_m non-positive or non-finite
        for bad_fp in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="energy_scale_flight_path_m"):
                nereids.spatial_map_typed(
                    data, energies, [u238_data],
                    solver="lm",
                    fit_energy_scale=True,
                    energy_scale_flight_path_m=bad_fp,
                    max_iter=5,
                )

    def test_spatial_all_dead_pixels_returns_nan_density(self, u238_data):
        """Issue #458: when every pixel is
        masked dead, the early-return path must honour the NaN-on-failure
        contract — density_maps must be NaN, not zeros.
        """
        energies = np.linspace(1.0, 10.0, 20)
        n_e = len(energies)
        h, w = 3, 3
        t = np.full((n_e, h, w), 0.5)
        u = np.full((n_e, h, w), 0.01)
        data = nereids.from_transmission(t, u)
        all_dead = np.ones((h, w), dtype=bool)

        result = nereids.spatial_map_typed(
            data, energies, [u238_data],
            solver="lm",
            dead_pixels=all_dead,
            max_iter=5,
        )
        # converged_map is the signal that no fits ran.
        assert result.n_converged == 0
        assert result.n_total == 0
        # density_map must be all NaN (not zero-filled).
        density = np.asarray(result.density_maps[0])
        assert density.shape == (h, w)
        assert np.all(np.isnan(density)), (
            "all-dead-pixels early-return must honour NaN-on-failure contract; "
            "got density map with non-NaN entries"
        )

    def test_counts_with_nuisance_auto_dispatches_to_kl(self, u238_data):
        """Regression for issue #458 B4.

        `spatial_map_typed(solver="auto")` with a `from_counts_with_nuisance`
        input must dispatch to the counts-KL (joint-Poisson) path — previously
        the ``data.kind == "counts"`` check in the Python binding missed the
        `"counts_with_nuisance"` variant and silently fell through to LM.

        Proxy observable: the counts-KL dispatch populates
        `deviance_per_dof_map`; LM on counts does not.  If we see
        `deviance_per_dof_map is not None`, the auto-dispatch is correct.
        """
        energies = np.linspace(1.0, 30.0, 80)
        true_density = 0.002
        flux = 4000.0
        n_e = len(energies)
        ny, nx = 2, 2

        t_1d = np.asarray(
            nereids.forward_model(energies, [(u238_data, true_density)])
        )

        rng = np.random.default_rng(1234)
        flux_3d = np.full((n_e, ny, nx), flux)
        # Explicit nuisance spectra: flux and zero detector background.
        background_3d = np.zeros((n_e, ny, nx))
        sample_3d = np.zeros((n_e, ny, nx))
        for y in range(ny):
            for x in range(nx):
                sample_3d[:, y, x] = rng.poisson(flux * t_1d).astype(float)

        data = nereids.from_counts_with_nuisance(sample_3d, flux_3d, background_3d)
        result = nereids.spatial_map_typed(
            data,
            energies,
            [u238_data],
            solver="auto",
            max_iter=50,
        )
        # Definitive signal that we routed through the counts-KL dispatch:
        assert result.deviance_per_dof_map is not None, (
            "from_counts_with_nuisance + solver='auto' should dispatch to "
            "counts-KL (joint-Poisson), which populates deviance_per_dof_map. "
            "None means the binding mis-dispatched to LM (issue #458 B4)."
        )


class TestSpatialMapBadValues:
    """spatial_map_typed must reject non-finite / out-of-domain detector-cube
    VALUES up front with a ``ValueError`` instead of silently clamping them via
    the per-pixel ``v.max(0.0)`` / ``sigma.max(1e-10)`` sanitation.  The Rust
    core raises ``InvalidParameter``, mapped to ``ValueError`` at the binding
    boundary.  Shapes are valid, so the only possible ``ValueError`` is the
    value check."""

    @staticmethod
    def _energies(n=20):
        return np.linspace(1.0, 10.0, n)

    def test_rejects_nan_sample(self, u238_data):
        energies = self._energies()
        n_e = len(energies)
        sample = np.full((n_e, 2, 2), 100.0)
        ob = np.full((n_e, 2, 2), 200.0)
        sample[5, 0, 1] = np.nan
        data = nereids.from_counts(sample, ob)
        with pytest.raises(ValueError, match="sample_counts"):
            nereids.spatial_map_typed(data, energies, [u238_data], max_iter=10)

    def test_rejects_negative_sample(self, u238_data):
        energies = self._energies()
        n_e = len(energies)
        sample = np.full((n_e, 2, 2), 100.0)
        ob = np.full((n_e, 2, 2), 200.0)
        sample[3, 1, 0] = -1.0
        data = nereids.from_counts(sample, ob)
        with pytest.raises(ValueError, match="sample_counts"):
            nereids.spatial_map_typed(data, energies, [u238_data], max_iter=10)

    def test_rejects_inf_open_beam(self, u238_data):
        # A single bad open-beam bin would poison the spatially-averaged flux
        # for every pixel; it must surface as a hard error.
        energies = self._energies()
        n_e = len(energies)
        sample = np.full((n_e, 2, 2), 100.0)
        ob = np.full((n_e, 2, 2), 200.0)
        ob[7, 0, 0] = np.inf
        data = nereids.from_counts(sample, ob)
        with pytest.raises(ValueError, match="open_beam_counts"):
            nereids.spatial_map_typed(data, energies, [u238_data], max_iter=10)

    def test_rejects_nan_transmission(self, u238_data):
        energies = self._energies()
        n_e = len(energies)
        trans = np.full((n_e, 2, 2), 0.8)
        unc = np.full((n_e, 2, 2), 0.01)
        trans[4, 1, 1] = np.nan
        data = nereids.from_transmission(trans, unc)
        with pytest.raises(ValueError, match="transmission"):
            nereids.spatial_map_typed(data, energies, [u238_data], max_iter=10)

    def test_rejects_zero_uncertainty(self, u238_data):
        energies = self._energies()
        n_e = len(energies)
        trans = np.full((n_e, 2, 2), 0.8)
        unc = np.full((n_e, 2, 2), 0.01)
        unc[2, 0, 0] = 0.0
        data = nereids.from_transmission(trans, unc)
        with pytest.raises(ValueError, match="uncertainty"):
            nereids.spatial_map_typed(data, energies, [u238_data], max_iter=10)

    def test_rejects_nan_flux(self, u238_data):
        energies = self._energies()
        n_e = len(energies)
        sample = np.full((n_e, 2, 2), 100.0)
        flux = np.full((n_e, 2, 2), 200.0)
        background = np.zeros((n_e, 2, 2))
        flux[6, 1, 0] = np.nan
        data = nereids.from_counts_with_nuisance(sample, flux, background)
        with pytest.raises(ValueError, match="flux"):
            nereids.spatial_map_typed(data, energies, [u238_data], max_iter=10)

    def test_accepts_negative_transmission(self, u238_data):
        # SAMMY does not reject negative transmission (noise / over-
        # subtraction); only finiteness is required, so this must NOT raise.
        energies = self._energies()
        n_e = len(energies)
        trans = np.full((n_e, 2, 2), 0.8)
        unc = np.full((n_e, 2, 2), 0.01)
        trans[4, 1, 1] = -0.05
        data = nereids.from_transmission(trans, unc)
        result = nereids.spatial_map_typed(
            data, energies, [u238_data], max_iter=10
        )
        assert result.n_total == 4


# ===========================================================================
# fit_counts_spectrum_typed — single-spectrum counts-KL bindings
# ===========================================================================


class TestFitCountsSpectrumTyped:
    """Tests for the single-spectrum counts-KL binding."""

    def test_enable_polish_kwarg_accepted_and_toggles_behavior(self, u238_data):
        """Issue #486: `enable_polish` kwarg must be accepted on the
        Python `fit_counts_spectrum_typed` binding, default `None` must
        fall through to the library default (now ``False``), and
        explicit ``True``/``False`` must produce different iteration
        counts on a synthetic case.

        This is the regression test guarding against a future binding
        refactor silently dropping the kwarg or stranding the override
        plumbing.
        """
        # Tiny synthetic counts setup — uses the U-238 single-resonance
        # fixture so the test runs without ENDF retrieval.
        energies = np.linspace(1.0, 30.0, 200)
        true_density = 0.0008
        flux = 1000.0  # open-beam mean cts/bin

        t_1d = np.asarray(
            nereids.forward_model(energies, [(u238_data, true_density)])
        )
        rng = np.random.default_rng(20260424)
        open_beam = rng.poisson(np.full_like(t_1d, flux)).astype(float)
        sample = rng.poisson(flux * t_1d).astype(float)
        # Floor open_beam at 1 to avoid divide-by-zero in the c-norm path.
        open_beam = np.maximum(open_beam, 1.0)

        kwargs = dict(
            sample_counts=sample,
            open_beam_counts=open_beam,
            energies=energies,
            isotopes=[(u238_data, true_density)],
            solver="kl",
            c=1.0,
            temperature_k=293.6,
            max_iter=200,
        )

        # Default kwarg → library default (False post-#486) → fast.
        r_default = nereids.fit_counts_spectrum_typed(**kwargs)
        # Explicit False → same path as default, gates against future drift.
        r_off = nereids.fit_counts_spectrum_typed(**kwargs, enable_polish=False)
        # Explicit True → opt-in polish → must run more iterations than
        # polish-off on a regime where polish has any room to move
        # (NM still runs even if it doesn't improve, so iter count
        # rises by polish_iters >= 1).
        r_on = nereids.fit_counts_spectrum_typed(**kwargs, enable_polish=True)

        # All three must converge / produce a valid result.
        for r in (r_default, r_off, r_on):
            assert r.densities[0] > 0.0
            assert r.iterations > 0

        # Default == explicit-False: the kwarg's `None` path must not
        # accidentally toggle polish on.
        assert r_default.iterations == r_off.iterations, (
            f"enable_polish=None must match enable_polish=False; "
            f"got iter None={r_default.iterations} False={r_off.iterations}"
        )

        # Explicit True must produce more iterations than False — polish
        # is the only stage that runs after Gauss-Newton convergence,
        # so any non-zero polish_iters bumps the summed iteration count.
        assert r_on.iterations > r_off.iterations, (
            f"enable_polish=True must run more iterations than =False; "
            f"got True={r_on.iterations} False={r_off.iterations}"
        )

        # All three should agree on the converged density to within
        # Fisher σ — polish should not move the answer beyond solver
        # noise on this synthetic case (matches issue #486 ablation
        # finding: shift ≤ 0.35 σ on every parameter / scenario).
        d_off = float(r_off.densities[0])
        d_on = float(r_on.densities[0])
        sigma = float(r_on.uncertainties[0])
        if np.isfinite(sigma) and sigma > 0.0:
            assert abs(d_on - d_off) / sigma < 1.0, (
                f"polish moved density beyond 1σ: "
                f"|Δn|/σ = {abs(d_on - d_off) / sigma:.3f}"
            )

    def test_tzero_jacobian_kwarg_accepted_and_alias_resolution(self, u238_data):
        """Issue #489: `tzero_jacobian` kwarg must be accepted on the
        Python `fit_counts_spectrum_typed` binding, default `None` must
        defer to the library default (PartialGal post-#489), explicit
        ``"fd2"`` / ``"partial_gal"`` must be accepted, and underscore +
        dash aliases must resolve identically.  The legacy ``"chain"`` /
        ``"frozen-r"`` FrozenResolutionChainRule method was removed in #608
        and must now be rejected.

        Guards against a future binding refactor silently dropping the
        kwarg or stranding the override plumbing — same defect class
        as the ``enable_polish`` kwarg test above.
        """
        energies = np.linspace(1.0, 30.0, 200)
        true_density = 0.0008
        flux = 1000.0

        t_1d = np.asarray(
            nereids.forward_model(energies, [(u238_data, true_density)])
        )
        rng = np.random.default_rng(20260426)
        open_beam = rng.poisson(np.full_like(t_1d, flux)).astype(float)
        sample = rng.poisson(flux * t_1d).astype(float)
        open_beam = np.maximum(open_beam, 1.0)

        kwargs = dict(
            sample_counts=sample,
            open_beam_counts=open_beam,
            energies=energies,
            isotopes=[(u238_data, true_density)],
            solver="kl",
            c=1.0,
            temperature_k=293.6,
            max_iter=200,
            fit_energy_scale=True,
            t0_init_us=0.0,
            l_scale_init=1.0,
        )

        # Default kwarg None == explicit "partial_gal" (post-#489 default).
        r_default = nereids.fit_counts_spectrum_typed(**kwargs)
        r_pg = nereids.fit_counts_spectrum_typed(
            **kwargs, tzero_jacobian="partial_gal"
        )
        # Underscore / dash aliases.
        r_pg_dash = nereids.fit_counts_spectrum_typed(
            **kwargs, tzero_jacobian="partial-gal"
        )
        # Explicit FD2 opt-out.
        r_fd2 = nereids.fit_counts_spectrum_typed(**kwargs, tzero_jacobian="fd2")
        r_fd2_dash = nereids.fit_counts_spectrum_typed(
            **kwargs, tzero_jacobian="finite-difference"
        )
        # All variants must produce a valid result.
        for r in (
            r_default,
            r_pg,
            r_pg_dash,
            r_fd2,
            r_fd2_dash,
        ):
            assert r.densities[0] > 0.0
            assert r.iterations > 0
            assert np.isfinite(r.t0_us)
            assert np.isfinite(r.l_scale)

        # Default == explicit "partial_gal": the None path must defer
        # to PartialGal, not FD2 or chain.
        assert r_default.densities[0] == r_pg.densities[0], (
            f"tzero_jacobian=None must match =\"partial_gal\"; got "
            f"None={r_default.densities[0]} pg={r_pg.densities[0]}"
        )
        assert r_default.t0_us == r_pg.t0_us
        assert r_default.l_scale == r_pg.l_scale

        # Underscore / dash aliases must resolve identically.
        assert r_pg.densities[0] == r_pg_dash.densities[0]
        assert r_fd2.densities[0] == r_fd2_dash.densities[0]

        # The FrozenResolutionChainRule method was removed in #608; its legacy
        # aliases must now be rejected like any other invalid value.
        for removed in ("chain", "frozen-r", "frozen_r"):
            with pytest.raises(ValueError, match="tzero_jacobian must be one of"):
                nereids.fit_counts_spectrum_typed(**kwargs, tzero_jacobian=removed)

        # Invalid value must raise ValueError listing the supported set.
        with pytest.raises(ValueError, match="tzero_jacobian must be one of"):
            nereids.fit_counts_spectrum_typed(
                **kwargs, tzero_jacobian="not-a-real-method"
            )


# ===========================================================================
# Normalization
# ===========================================================================


class TestNormalization:
    """Tests for normalize()."""

    def test_basic_normalization(self):
        """Normalize identical sample and open_beam -> T=1."""
        n_e, ny, nx = 10, 3, 3
        rng = np.random.default_rng(42)
        counts = rng.poisson(1000, size=(n_e, ny, nx)).astype(float)

        t, unc = nereids.normalize(counts, counts, 1.0, 1.0)
        t = np.asarray(t)
        unc = np.asarray(unc)
        assert t.shape == (n_e, ny, nx)
        assert unc.shape == (n_e, ny, nx)
        np.testing.assert_allclose(t, 1.0, atol=1e-12)

    def test_proton_charge_scaling(self):
        """Different proton charges should scale the transmission."""
        n_e, ny, nx = 5, 2, 2
        sample = np.full((n_e, ny, nx), 100.0)
        open_beam = np.full((n_e, ny, nx), 200.0)

        t1, _ = nereids.normalize(sample, open_beam, 1.0, 1.0)
        t2, _ = nereids.normalize(sample, open_beam, 1.0, 2.0)
        t1 = np.asarray(t1)
        t2 = np.asarray(t2)
        # T = (sample/ob) * (pc_ob/pc_sample)
        # t1 = 0.5 * 1.0 = 0.5; t2 = 0.5 * 2.0 = 1.0
        np.testing.assert_allclose(t1, 0.5, atol=1e-12)
        np.testing.assert_allclose(t2, 1.0, atol=1e-12)

    def test_shape_mismatch_raises(self):
        s = np.ones((5, 3, 3))
        ob = np.ones((5, 3, 4))  # different width
        with pytest.raises(ValueError, match="shape"):
            nereids.normalize(s, ob, 1.0, 1.0)


# ===========================================================================
# Pixel masks (pipeline-integrity screens, #643)
# ===========================================================================


class TestPixelMasks:
    """Tests for detect_dead_pixels, detect_hot_pixels,
    detect_dead_pixels_chunked, and detect_bad_pixels (#643)."""

    def test_detect_dead_pixels_backcompat(self):
        """The original all-zero-stack detector is unchanged."""
        data = np.full((3, 2, 2), 5.0)
        data[:, 0, 0] = 0.0
        mask = np.asarray(nereids.detect_dead_pixels(data))
        assert mask.dtype == np.bool_
        assert mask.shape == (2, 2)
        assert mask[0, 0]
        assert not mask[1, 1]

    def test_detect_bad_pixels_union(self):
        """dead(sample) | hot(sample) | dead(ob) | hot(ob); low-count kept."""
        sample = np.full((3, 3, 3), 100.0)
        sample[:, 0, 0] = 0.0  # dead in sample only
        sample[:, 1, 2] = 0.0
        sample[0, 1, 2] = 1.0  # low-count-ALIVE: 1 total count
        ob = np.full((3, 3, 3), 200.0)
        ob[:, 0, 1] = 0.0  # dead in OB only
        ob[:, 2, 2] = 65535.0  # hot (railed) in OB only

        mask = np.asarray(nereids.detect_bad_pixels(sample, ob))
        assert mask.dtype == np.bool_
        assert mask.shape == (3, 3)
        assert mask[0, 0], "dead-in-sample-only must be flagged"
        assert mask[0, 1], "dead-in-OB-only must be flagged"
        assert mask[2, 2], "hot-in-OB-only must be flagged"
        assert not mask[1, 2], "low-count-alive pixel must be kept"
        assert mask.sum() == 3

    def test_detect_bad_pixels_hot_k_mad_none_disables_hot(self):
        sample = np.full((3, 3, 3), 100.0)
        sample[:, 1, 1] = 65535.0  # railed
        dead_only = np.asarray(nereids.detect_bad_pixels(sample, hot_k_mad=None))
        assert not dead_only.any()
        with_hot = np.asarray(nereids.detect_bad_pixels(sample))
        assert with_hot[1, 1]

    def test_detect_hot_pixels_flags_railed_keeps_low_count(self):
        data = np.full((3, 3, 3), 100.0)
        data[:, 2, 0] = 65535.0  # railed
        data[:, 0, 2] = 0.0
        data[1, 0, 2] = 1.0  # low-count-alive
        mask = np.asarray(nereids.detect_hot_pixels(data))
        assert mask[2, 0], "railed pixel must be flagged"
        assert not mask[0, 2], "low-count-alive pixel must be kept"
        assert mask.sum() == 1

    def test_detect_hot_pixels_mad_scale_decides_stage1_threshold(self):
        """Mirror of the Rust deciding-branch test (#646 review R4, P1-2):
        the robust-MAD branch of the stage-1 scale
        sigma = max(MAD_TO_SIGMA*mad, exp(-med/2)) decides the outcome.

        8x8 single-bin grid; two all-dead 3x3 moats (rows/cols 1-3 and
        4-6) isolate a probe at (2,2) and a control at (5,5) so their
        stage-2 reference sample is empty and the flag outcome is decided
        purely by the stage-1 threshold.  Background: 24 px at A = 8000,
        22 px at B = 12500 (A*B = 1e8, B/A = 1.25**2).  Worked stage-1
        arithmetic over the 48 live pixels (ranks pin the statistics):

          med = (ln A + ln B)/2 = ln 1e4         = 9.2103404
          mad = ln 1.25                          = 0.2231436
          MAD term = 1.4826022 * 0.2231436       = 0.3308331
          floor = exp(-med/2) = 1e-2             = 0.01  (MAD wins)
          threshold = med + 6*sigma              = 11.1953391

        Probe ln 40000 = 10.5966347 < threshold -> kept; under a
        floor-only mutation the threshold collapses to 9.2703404 and the
        probe would be flagged.  Control ln 200000 = 12.2060726 >
        threshold -> flagged.
        """
        data = np.zeros((1, 8, 8))
        data[0, 2, 2] = 40000.0  # probe
        data[0, 5, 5] = 200000.0  # control
        filled = 0
        for y in range(8):
            for x in range(8):
                in_moat1 = 1 <= y <= 3 and 1 <= x <= 3
                in_moat2 = 4 <= y <= 6 and 4 <= x <= 6
                if in_moat1 or in_moat2:
                    continue  # dead moat cells stay 0.0
                data[0, y, x] = 8000.0 if filled < 24 else 12500.0
                filled += 1
        assert filled == 46

        mask = np.asarray(nereids.detect_hot_pixels(data))
        assert not mask[2, 2], "probe below the MAD-driven threshold must be kept"
        assert mask[5, 5], "control above the MAD-driven threshold must be flagged"
        assert mask.sum() == 1

    def test_detect_dead_pixels_chunked_catches_intermittent(self):
        chunk0 = np.full((3, 2, 2), 5.0)
        chunk0[:, 0, 1] = 0.0  # dead throughout chunk 0 only
        chunk1 = np.full((3, 2, 2), 5.0)
        mask = np.asarray(nereids.detect_dead_pixels_chunked([chunk0, chunk1]))
        assert mask[0, 1]
        assert mask.sum() == 1
        # The gap being closed: the summed stack cannot see it.
        summed_mask = np.asarray(nereids.detect_dead_pixels(chunk0 + chunk1))
        assert not summed_mask[0, 1]

    def test_shape_mismatch_raises(self):
        sample = np.ones((3, 2, 2))
        ob = np.ones((3, 2, 3))
        with pytest.raises(ValueError, match="[Ss]hape"):
            nereids.detect_bad_pixels(sample, ob)
        with pytest.raises(ValueError, match="[Ss]hape"):
            nereids.detect_dead_pixels_chunked([sample, ob])

    def test_nan_raises(self):
        data = np.ones((2, 2, 2))
        data[0, 0, 0] = np.nan
        with pytest.raises(ValueError):
            nereids.detect_bad_pixels(data)
        with pytest.raises(ValueError):
            nereids.detect_hot_pixels(data)
        with pytest.raises(ValueError):
            nereids.detect_dead_pixels_chunked([data])

    def test_bad_k_raises(self):
        data = np.ones((2, 2, 2))
        for bad_k in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="k_mad"):
                nereids.detect_hot_pixels(data, k_mad=bad_k)
            with pytest.raises(ValueError, match="k_mad"):
                nereids.detect_bad_pixels(data, hot_k_mad=bad_k)

    def test_empty_chunks_raises(self):
        with pytest.raises(ValueError):
            nereids.detect_dead_pixels_chunked([])

    def test_bimodal_bright_region_not_flagged(self):
        """PR #646 P0 regression guard: dark-majority bimodal scene (60%
        of the FOV at ~50 counts, contiguous 40% bright region at ~5000).
        Every bright pixel passes the global median+MAD cut (the dark
        population holds the median), but the local-neighborhood
        confirmation must veto them all — bright scene is not a defect."""
        data = np.full((1, 10, 10), 50.0)
        data[:, :, 6:] = 5000.0
        assert not np.asarray(nereids.detect_hot_pixels(data)).any()
        assert not np.asarray(nereids.detect_bad_pixels(data)).any()

    def test_bimodal_railed_inside_bright_region_caught(self):
        """A genuinely railed pixel INSIDE the bright region of a bimodal
        scene (~200x its bright neighbors) is still caught."""
        data = np.full((1, 10, 10), 50.0)
        data[:, :, 6:] = 5000.0
        data[0, 5, 8] = 1.0e6
        mask = np.asarray(nereids.detect_hot_pixels(data))
        assert mask[5, 8]
        assert mask.sum() == 1

    def test_railed_blob_fully_caught_fixpoint(self):
        """#646 review R2 (F1): the stage-2 fixpoint erodes a 3x3 railed
        blob from the boundary inward — a single local pass flags only the
        4 corners (5 background neighbors each) and misses the edge centers
        and the interior, whose neighbors are railed too."""
        data = np.full((4, 9, 9), 100.0)
        data[:, 3:6, 3:6] = 65535.0
        mask = np.asarray(nereids.detect_hot_pixels(data))
        assert mask[3:6, 3:6].all(), "3x3 blob must be fully caught"
        assert mask.sum() == 9, "only the blob may be flagged"

    def test_large_psf_bright_region_not_eroded(self):
        """#646 review R2 (F1 safety): a large bright scene region (20x20
        core at 100x background) with the >= 2-px PSF edge blur real VENUS
        features have (adjacent ratios <= 5x) passes the global cut but
        never seeds the erosion — zero flags."""
        data = np.full((1, 50, 50), 100.0)
        data[:, 13:37, 13:37] = 400.0
        data[:, 14:36, 14:36] = 2000.0
        data[:, 15:35, 15:35] = 10000.0
        assert not np.asarray(nereids.detect_hot_pixels(data)).any()

    def test_edge_to_edge_2px_band_not_flagged_by_design(self):
        """#646 review R3 (F1, pinned limitation): an EDGE-TO-EDGE railed
        band >= 2 px wide (both ends off-detector) exposes no end cap or
        convex corner, so the fixpoint erosion has no seed and the band is
        NOT caught — deliberately: a slit-aperture open beam produces a
        genuine full-width bright scene band indistinguishable from it,
        and a full-span screen would mask that scene (bimodal failure).
        Declare such full-span pathologies in a file mask.  The same band
        with one end cap inside the detector IS fully consumed."""
        edge_to_edge = np.full((4, 9, 9), 100.0)
        edge_to_edge[:, 3:5, :] = 65535.0
        assert not np.asarray(nereids.detect_hot_pixels(edge_to_edge)).any()

        one_end_inside = np.full((4, 9, 9), 100.0)
        one_end_inside[:, 3:5, :7] = 65535.0
        mask = np.asarray(nereids.detect_hot_pixels(one_end_inside))
        assert mask[3:5, :7].all(), "band with an end cap must be caught"
        assert mask.sum() == 14, "only the railed band may be flagged"

    def test_1px_bright_line_flagged_by_design(self):
        """#646 review R2 (F3, pinned): a 1-px-wide bright scene line at
        >= 10x local contrast is spatially indistinguishable from a railed
        line and IS masked — the documented, accepted trade-off (real VENUS
        scene features are PSF-blurred over >= 2 px)."""
        data = np.full((1, 9, 9), 100.0)
        data[:, :, 4] = 5000.0
        mask = np.asarray(nereids.detect_hot_pixels(data))
        assert mask[:, 4].all(), "width-1 line is flagged by design"
        assert mask.sum() == 9

    def test_empty_tof_axis_raises(self):
        """shape[0] == 0: the all-zero dead test would pass vacuously and
        mask the whole detector — the validating entry points reject it."""
        empty = np.empty((0, 2, 2))
        with pytest.raises(ValueError):
            nereids.detect_bad_pixels(empty)
        with pytest.raises(ValueError):
            nereids.detect_bad_pixels(np.ones((3, 2, 2)), empty)
        with pytest.raises(ValueError):
            nereids.detect_hot_pixels(empty)
        with pytest.raises(ValueError):
            nereids.detect_dead_pixels_chunked([empty])

    def test_mask_round_trips_into_spatial_map(self, u238_data):
        """A detect_bad_pixels mask feeds spatial_map_typed(dead_pixels=...):
        the masked pixel is hard-excluded (NaN in the density map)."""
        energies = np.linspace(1.0, 10.0, 20)
        n_e = len(energies)
        h, w = 3, 3
        sample = np.full((n_e, h, w), 100.0)
        sample[:, 1, 1] = 0.0  # dead pixel
        mask = np.asarray(nereids.detect_bad_pixels(sample))
        assert mask.dtype == np.bool_
        assert mask[1, 1] and mask.sum() == 1

        t = np.full((n_e, h, w), 0.5)
        u = np.full((n_e, h, w), 0.01)
        data = nereids.from_transmission(t, u)
        result = nereids.spatial_map_typed(
            data, energies, [u238_data],
            solver="lm",
            dead_pixels=mask,
            max_iter=5,
        )
        density = np.asarray(result.density_maps[0])
        assert density.shape == (h, w)
        assert np.isnan(density[1, 1]), (
            "masked pixel must be hard-excluded (NaN in the density map)"
        )


# ===========================================================================
# TIFF I/O
# ===========================================================================


class TestTiffIO:
    """Tests for load_tiff_stack and related I/O.

    Uses tifffile for writing synthetic test TIFFs.
    """

    def test_roundtrip_tiff_stack(self):
        """Write a multi-frame TIFF and load it back."""
        tifffile = pytest.importorskip("tifffile")
        n_frames, h, w = 5, 8, 10
        data = np.random.default_rng(42).random((n_frames, h, w)).astype(
            np.float32
        )

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = f.name
        try:
            tifffile.imwrite(path, data)
            loaded = np.asarray(nereids.load_tiff_stack(path))
            assert loaded.shape == (n_frames, h, w)
            np.testing.assert_allclose(loaded, data.astype(np.float64), atol=1e-5)
        finally:
            os.unlink(path)

    def test_load_tiff_folder(self):
        """Write single-frame TIFFs to a folder and load them."""
        tifffile = pytest.importorskip("tifffile")
        n_frames, h, w = 3, 4, 5
        data = np.random.default_rng(7).random((n_frames, h, w)).astype(
            np.float32
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(n_frames):
                tifffile.imwrite(
                    os.path.join(tmpdir, f"frame_{i:04d}.tif"), data[i]
                )
            loaded = np.asarray(nereids.load_tiff_folder(tmpdir))
            assert loaded.shape == (n_frames, h, w)
            np.testing.assert_allclose(
                loaded, data.astype(np.float64), atol=1e-5
            )

    def test_load_tiff_stack_missing_file(self):
        with pytest.raises(OSError):
            nereids.load_tiff_stack("/nonexistent/path.tif")


# ===========================================================================
# Error handling / validation
# ===========================================================================


class TestErrorHandling:
    """Tests for input validation and error messages."""

    def test_spatial_map_typed_shape_mismatch(self, u238_data):
        e = np.linspace(1.0, 10.0, 5)
        trans = np.ones((5, 3, 3))
        unc = np.ones((5, 3, 4))  # width mismatch
        with pytest.raises(ValueError, match="shape"):
            nereids.from_transmission(trans, unc)

    def test_spatial_map_typed_empty_spectral(self, u238_data):
        trans = np.ones((0, 2, 2))
        unc = np.ones((0, 2, 2))
        with pytest.raises(ValueError, match="spectral"):
            nereids.from_transmission(trans, unc)

    def test_doppler_broaden_invalid_awr(self):
        e = np.linspace(1.0, 10.0, 100)
        xs = np.ones(100)
        with pytest.raises(ValueError, match="AWR must be positive"):
            nereids.doppler_broaden(e, xs, -1.0, 300.0)

    def test_doppler_broaden_invalid_temperature(self):
        e = np.linspace(1.0, 10.0, 100)
        xs = np.ones(100)
        with pytest.raises(ValueError, match="temperature must be non-negative"):
            nereids.doppler_broaden(e, xs, 236.0, -1.0)

    def test_resolution_broaden_invalid_delta_t(self):
        e = np.linspace(1.0, 10.0, 100)
        xs = np.ones(100)
        with pytest.raises(ValueError, match="delta_t_us"):
            nereids.resolution_broaden(e, xs, 20.0, -0.5, 0.001)


# ===========================================================================
# Trace detectability
# ===========================================================================


class TestTraceDetectability:
    """Tests for trace_detectability() and trace_detectability_survey()."""

    def test_basic_detectability(self, u238_data):
        """Single trace analysis should return a report."""
        trace = _make_single_resonance(
            z=26,
            a=56,
            awr=55.347,
            scattering_radius=6.0,
            energy=7.8,
            j=0.5,
            gn=0.001,
            gg=0.01,
        )
        energies = np.linspace(1.0, 30.0, 1000)
        report = nereids.trace_detectability(
            matrix=u238_data,
            matrix_density=0.01,
            trace=trace,
            trace_ppm=100.0,
            energies=energies,
            i0=10000.0,
        )
        assert isinstance(report.peak_snr, float)
        assert isinstance(report.peak_energy_ev, float)
        assert isinstance(report.detectable, bool)
        assert len(np.asarray(report.delta_t_spectrum)) == len(energies)
        assert len(np.asarray(report.energies)) == len(energies)
        r = repr(report)
        assert "TraceDetectabilityReport" in r

    def test_detectability_survey(self, u238_data):
        """Survey with multiple trace candidates."""
        trace1 = _make_single_resonance(
            z=26, a=56, awr=55.347, scattering_radius=6.0,
            energy=7.8, j=0.5, gn=0.001, gg=0.01,
        )
        trace2 = _make_single_resonance(
            z=29, a=63, awr=62.442, scattering_radius=6.5,
            energy=12.0, j=0.5, gn=0.002, gg=0.015,
        )
        energies = np.linspace(1.0, 30.0, 500)
        results = nereids.trace_detectability_survey(
            matrix=u238_data,
            matrix_density=0.01,
            trace_candidates=[trace1, trace2],
            trace_ppm=100.0,
            energies=energies,
            i0=10000.0,
        )
        assert len(results) == 2
        for name, report in results:
            assert isinstance(name, str)
            assert isinstance(report.peak_snr, float)

    def test_detectability_empty_candidates_raises(self, u238_data):
        energies = np.linspace(1.0, 30.0, 100)
        with pytest.raises(ValueError, match="trace_candidates"):
            nereids.trace_detectability_survey(
                matrix=u238_data,
                matrix_density=0.01,
                trace_candidates=[],
                trace_ppm=100.0,
                energies=energies,
                i0=10000.0,
            )


# ===========================================================================
# NeXus I/O Tests
# ===========================================================================

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def _create_synthetic_nexus_histogram(path, n_tof=10, height=4, width=4):
    """Create a minimal VENUS-schema NeXus file with histogram data."""
    import h5py

    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        hist = entry.create_group("histogram")
        # Shape: (1 rotation, height, width, n_tof) — u64
        counts = np.random.default_rng(42).integers(
            0, 100, size=(1, height, width, n_tof), dtype=np.uint64
        )
        hist.create_dataset("counts", data=counts)
        # TOF edges in nanoseconds (n_tof + 1) — dataset name matches VENUS schema
        tof_ns = np.linspace(1e4, 5e4, n_tof + 1)
        hist.create_dataset("time_of_flight", data=tof_ns)
        # Flight path — attribute name matches VENUS schema expected by Rust reader
        entry.attrs["flight_path_m"] = 25.0


def _create_synthetic_nexus_histogram_multi_angle(
    path, n_rot=3, n_tof=10, height=4, width=4, seed=42
):
    """Create a NeXus histogram file with more than one rotation angle.

    Used to exercise issue #430: the default loader must refuse these
    files, and explicit multi-angle policies (``sum`` / ``select``)
    must produce predictable output.
    """
    import h5py

    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        hist = entry.create_group("histogram")
        counts = np.random.default_rng(seed).integers(
            0, 100, size=(n_rot, height, width, n_tof), dtype=np.uint64
        )
        hist.create_dataset("counts", data=counts)
        tof_ns = np.linspace(1e4, 5e4, n_tof + 1)
        hist.create_dataset("time_of_flight", data=tof_ns)
        entry.attrs["flight_path_m"] = 25.0


def _create_synthetic_nexus_events(path, n_events=1000, height=4, width=4):
    """Create a minimal VENUS-schema NeXus file with event data."""
    import h5py

    rng = np.random.default_rng(43)
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        neutrons = entry.create_group("neutrons")
        # Event time offsets in nanoseconds (u64)
        tof_ns = rng.integers(10_000, 50_000, size=n_events, dtype=np.uint64)
        neutrons.create_dataset("event_time_offset", data=tof_ns)
        # Pixel coordinates (f64)
        x = rng.uniform(0, width - 1, size=n_events)
        y = rng.uniform(0, height - 1, size=n_events)
        neutrons.create_dataset("x", data=x)
        neutrons.create_dataset("y", data=y)
        # Flight path — attribute name matches VENUS schema
        entry.attrs["flight_path_m"] = 25.0


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
class TestNexusIO:
    """Tests for NeXus loading Python bindings."""

    def test_probe_nexus_histogram(self, tmp_path):
        path = str(tmp_path / "hist.h5")
        _create_synthetic_nexus_histogram(path)
        meta = nereids.probe_nexus(path)
        assert isinstance(meta, nereids.NexusMetadata)
        assert meta.has_histogram is True
        assert meta.has_events is False
        assert meta.flight_path_m == pytest.approx(25.0)

    def test_probe_nexus_events(self, tmp_path):
        path = str(tmp_path / "events.h5")
        _create_synthetic_nexus_events(path)
        meta = nereids.probe_nexus(path)
        assert meta.has_events is True
        assert meta.n_events == 1000
        assert meta.flight_path_m == pytest.approx(25.0)

    def test_load_nexus_histogram(self, tmp_path):
        path = str(tmp_path / "hist.h5")
        _create_synthetic_nexus_histogram(path, n_tof=10, height=4, width=4)
        data = nereids.load_nexus_histogram(path)
        assert isinstance(data, nereids.NexusData)
        # Shape should be (n_tof, height, width)
        assert data.counts.shape == (10, 4, 4)
        assert data.tof_edges_us.shape == (11,)
        assert data.n_rotation_angles == 1
        # Counts should be non-negative
        assert np.all(data.counts >= 0)
        # Flight path from metadata
        assert data.flight_path_m == pytest.approx(25.0)

    def test_load_nexus_events(self, tmp_path):
        path = str(tmp_path / "events.h5")
        _create_synthetic_nexus_events(path, n_events=5000, height=4, width=4)
        data = nereids.load_nexus_events(
            path,
            n_bins=20,
            tof_min_us=10.0,
            tof_max_us=50.0,
            height=4,
            width=4,
        )
        assert isinstance(data, nereids.NexusData)
        assert data.counts.shape == (20, 4, 4)
        assert data.tof_edges_us.shape == (21,)
        # Event stats should be populated
        assert data.event_total is not None
        assert data.event_total == 5000
        assert data.event_kept is not None
        assert data.event_kept > 0
        assert data.event_kept <= 5000
        assert data.flight_path_m == pytest.approx(25.0)

    def test_load_nexus_histogram_bad_path(self):
        with pytest.raises(IOError):
            nereids.load_nexus_histogram("/nonexistent/file.h5")

    def test_probe_nexus_bad_path(self):
        with pytest.raises(IOError):
            nereids.probe_nexus("/nonexistent/file.h5")

    def test_load_nexus_histogram_rejects_multi_angle_by_default(self, tmp_path):
        """Issue #430: default `load_nexus_histogram(path)` must refuse
        multi-angle files — silent sum-over-angles is a data-loss bug
        the loader used to hide."""
        path = str(tmp_path / "multi_angle.h5")
        _create_synthetic_nexus_histogram_multi_angle(path, n_rot=3)
        with pytest.raises(ValueError, match="3 rotation angles"):
            nereids.load_nexus_histogram(path)

    def test_load_nexus_histogram_multi_angle_sum_mode(self, tmp_path):
        """Issue #430: opt-in `multi_angle_mode='sum'` recovers the
        legacy auto-sum behaviour."""
        path = str(tmp_path / "multi_angle_sum.h5")
        _create_synthetic_nexus_histogram_multi_angle(
            path, n_rot=3, n_tof=5, height=2, width=2
        )
        data = nereids.load_nexus_histogram(path, multi_angle_mode="sum")
        # Summed into single volume, per-angle info is lost.
        assert data.counts.shape == (5, 2, 2)
        assert data.n_rotation_angles == 3
        # Sum across angles must not exceed 3 × max(single-angle) = 3 × 99.
        assert np.all(data.counts <= 3 * 99)

    def test_load_nexus_histogram_multi_angle_select_mode(self, tmp_path):
        """Issue #430: `multi_angle_mode='select'` extracts a single
        projection by `angle_index`."""
        path = str(tmp_path / "multi_angle_select.h5")
        _create_synthetic_nexus_histogram_multi_angle(
            path, n_rot=3, n_tof=5, height=2, width=2
        )
        data0 = nereids.load_nexus_histogram(
            path, multi_angle_mode="select", angle_index=0
        )
        data1 = nereids.load_nexus_histogram(
            path, multi_angle_mode="select", angle_index=1
        )
        assert data0.counts.shape == (5, 2, 2)
        assert data1.counts.shape == (5, 2, 2)
        # Different angles yield different counts for random-seeded data.
        assert not np.array_equal(data0.counts, data1.counts)

        # Out-of-range angle index → ValueError
        with pytest.raises(ValueError, match="SelectAngle.*out of range"):
            nereids.load_nexus_histogram(
                path, multi_angle_mode="select", angle_index=99
            )

    def test_load_nexus_histogram_unknown_mode_rejected(self, tmp_path):
        """Invalid `multi_angle_mode` string must error with a clear
        message listing the allowed values."""
        path = str(tmp_path / "mode_str.h5")
        _create_synthetic_nexus_histogram(path, n_tof=5, height=2, width=2)
        with pytest.raises(ValueError, match="multi_angle_mode.*error.*sum.*select"):
            nereids.load_nexus_histogram(path, multi_angle_mode="average")

    def test_nexus_histogram_to_fitting_workflow(self, tmp_path):
        """End-to-end: load histogram → normalize → fit."""
        path = str(tmp_path / "hist.h5")
        n_tof, h, w = 50, 2, 2
        _create_synthetic_nexus_histogram(path, n_tof=n_tof, height=h, width=w)
        data = nereids.load_nexus_histogram(path)
        assert data.counts.shape == (n_tof, h, w)
        # Verify the loaded data can be used in from_counts
        sample = data.counts
        ob = np.full_like(sample, 100.0)  # synthetic OB
        input_data = nereids.from_counts(sample, ob)
        assert input_data is not None  # successfully created InputData


# ---------------------------------------------------------------------------
# Real VENUS regression gate (issue #465)
# ---------------------------------------------------------------------------
#
# Guards against silent changes to the MLBW dispatch path.  The fit below
# runs on a committed aggregated VENUS Hf 120 min spectrum + the committed
# Hf-177 ENDF fixture (LRF=2, MLBW) with a Gaussian resolution derived from
# the VENUS beamline parameters.  If any future change shifts a fit output
# on this real-data workload, the assertion fails and the PR cannot land
# without an explicit re-baseline + explanation.
#
# The fit deliberately uses a single MLBW isotope + no background so the
# failure mode is physics-clean (bit-exact bracket of the MLBW evaluator)
# rather than masked by incidental model complexity.  The expected values
# are those produced by the fixed code path introduced in issue #465 — a
# mismatch means either the MLBW evaluator drifted or the batch / per-point
# paths disagreed again.


class TestVenusMlbwRegression:
    """Real-data regression for the issue #465 MLBW correctness fix.

    Uses the committed aggregated VENUS Hf 120 min spectrum and the
    committed Hf-177 ENDF file to lock a single LM fit's outputs
    bit-exactly.  Any change to cross-section evaluation that shifts a
    real-world fit result fails this test.
    """

    @pytest.fixture
    def venus_data(self):
        """Aggregated VENUS Hf 120 min spectrum + Hf-177 MLBW isotope.

        **Fails** (not skips) if the committed fixtures are not present:
        they are checked into ``tests/data/`` so their absence means a
        broken checkout or a packaging step that stripped them, which
        would silently hide the regression gate.
        """
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fixture = os.path.join(root, "tests/data/venus/aggregated_hf_120min.npz")
        endf = os.path.join(root, "tests/data/endf/Hf-177.endf")
        if not os.path.exists(fixture) or not os.path.exists(endf):
            pytest.fail(
                f"VENUS / Hf-177 regression-gate fixtures missing: "
                f"{fixture}, {endf}. "
                f"These are committed under tests/data/; absence means a broken "
                f"checkout. Do NOT silently skip — refusing to run the gate is a "
                f"regression by itself."
            )
        with np.load(fixture) as f:
            E = np.ascontiguousarray(f["energies_ev"])
            S_agg = np.ascontiguousarray(f["sample_counts"])
            O_agg = np.ascontiguousarray(f["open_beam_counts"])
            c = float(f["pc_ratio"])
        hf177 = nereids.load_endf_file(endf)
        return E, S_agg, O_agg, c, hf177

    def test_mlbw_lm_fit_matches_baseline(self, venus_data):
        """LM fit on aggregated Hf-177 must match the committed baseline.

        Regression gate for #465.  The #465 bug produced up to 33 %
        relative error on this workload's σ; any serious dispatch /
        evaluator regression therefore shifts the converged density by
        orders of magnitude more than the ``rel=1e-6`` tolerance used
        here.

        Note: bit-exact (`==`) was used initially but relaxed to a tight
        ``pytest.approx`` because LM traverses BLAS and sum/dot
        ordering differs between Accelerate (macOS) and OpenBLAS
        (Linux CI), which flapped the gate without catching real bugs.
        ``rel=1e-6`` is three orders of magnitude tighter than the
        smallest regression this test is meant to catch, so the gate
        still fires on any real correctness drift.
        """
        E, S_agg, O_agg, c, hf177 = venus_data

        T_agg = S_agg / np.maximum(c * O_agg, 1.0)
        sigT_agg = T_agg * np.sqrt(
            1.0 / np.maximum(S_agg, 1.0) + 1.0 / np.maximum(O_agg, 1.0)
        )

        result = nereids.fit_spectrum_typed(
            transmission=T_agg,
            uncertainty=sigT_agg,
            energies=E,
            solver="lm",
            temperature_k=293.6,
            isotopes=[(hf177, 1.0e-5)],
            max_iter=100,
            background=False,
            flight_path_m=25.0,
            delta_t_us=0.5,
            delta_l_m=0.005,
        )

        # Baseline captured on the fixed code (macOS Accelerate).
        # Tolerance is loose enough to absorb BLAS-ordering noise across
        # Linux CI / macOS dev / any future backend, but orders of
        # magnitude tighter than the ~33 % MLBW dispatch error this gate
        # is meant to catch.
        #
        # Baseline regenerated after NV-1 SLBW/MLBW velocity-factor fix
        # (issue #568): the buggy MLBW had Γ_n(E) ∝ E for s-wave instead
        # of ∝ √E per ENDF-6 §D.1.1 eq D.7. After removing the extra
        # √(E/E_r) factor at slbw.rs:314 / :427, MLBW σ in the wings is
        # larger, so the LM fit on aggregated Hf-177 converges at a
        # ~28 % lower density (1.122e-4 → 8.110e-5 atoms/barn) — the
        # physically correct direction. Iteration count also dropped
        # (27 → 14) because the corrected objective is less ill-conditioned.
        #
        # Baseline regenerated again after the exact-FGM Doppler kernel
        # fix (Eq. III B1.7's w² integrand weight, was w¹ — a first-order
        # flank skew) plus the always-on low-edge velocity padding. On
        # this Hf-177 workload the net effect is small (the antisymmetric
        # skew largely cancels in the fitted density): density −3.6e-5
        # rel, χ²_r −2.4e-6 rel — the exact kernel fits the measured data
        # marginally BETTER — iteration count unchanged.
        #
        # Baseline regenerated after #635's analytic-Jacobian availability
        # fix: no-temperature fits without precomputed sigma previously fell
        # through to a model with NO analytical_jacobian, so LM ran on FD
        # columns. The model itself is UNCHANGED (bit-exact parity between
        # the old forward_model path and the new working-grid precompute was
        # verified before re-anchoring); only the optimizer's stopping point
        # inside the same flat basin moved: chi2_r agrees with the old
        # anchor to 1e-9 RELATIVE (219657.2440 vs .2437) while the density
        # shifts -0.07 % and the iteration count halves (14 -> 7) because
        # analytic steps satisfy the relative-chi2 tolerance sooner.
        #
        # These pinned values are machine-generated regression anchors
        # (produced by the code under test); the correctness burden is
        # carried by the SAMMY-oracle suites (samtry, ex001) and the
        # analytic kernel pins in doppler.rs.
        EXPECTED_DENSITY = 8.10458528518008e-05
        EXPECTED_CHI2_R = 219657.2439575215
        EXPECTED_ITERATIONS = 7

        FLOAT_TOL = pytest.approx
        assert float(result.densities[0]) == FLOAT_TOL(EXPECTED_DENSITY, rel=1e-6), (
            f"density drifted: got {float(result.densities[0])!r}, "
            f"expected {EXPECTED_DENSITY!r} (±1e-6 rel)"
        )
        assert float(result.reduced_chi_squared) == FLOAT_TOL(
            EXPECTED_CHI2_R, rel=1e-6
        ), (
            f"chi2_r drifted: got {float(result.reduced_chi_squared)!r}, "
            f"expected {EXPECTED_CHI2_R!r} (±1e-6 rel)"
        )
        # LM step acceptance can shift iteration count by ±1 or ±2 on
        # different BLAS backends; assert "close enough" rather than
        # exact.  Any dispatch regression would move this by orders of
        # magnitude (or hit max_iter without converging).
        assert abs(int(result.iterations) - EXPECTED_ITERATIONS) <= 3, (
            f"iteration count drifted: got {int(result.iterations)}, "
            f"expected ~{EXPECTED_ITERATIONS} (±3)"
        )
        assert bool(result.converged) is True, (
            f"fit did not converge: got converged={bool(result.converged)}. "
            f"A dispatch regression can prevent convergence entirely — investigate."
        )

    def test_counts_kl_fit_matches_baseline(self, venus_data):
        """Counts-KL (joint-Poisson) fit on the same real VENUS spectrum.

        This is the real-data regression gate for the counts-path solver:
        it substantiates, in-tree, the docs' claim that the joint-Poisson
        deviance path is exercised against real VENUS counts — the
        synthetic counts-KL tests elsewhere use NEREIDS-generated
        observations and cannot do that.

        Two properties are pinned:

        * The fit converges with the anchored density.  As with the LM
          gate above, the pinned values are machine-generated regression
          anchors (produced by the code under test); correctness of the
          deviance math is carried by the analytic joint-Poisson unit
          tests in nereids-fitting.
        * ``deviance_per_dof`` lands in the >> 1 regime (measured ~3.1e4).
          Real VENUS counts carry un-modelled upstream physics, so D/dof
          saturates at 10^4-10^5 — exactly the regime documented on
          ``JointPoissonFitConfig::enable_polish`` (and the reason polish
          is off by default).  A sudden drop to O(1) would mean the gate
          silently switched to a synthetic-like input, not that the model
          got better.

        The KL density (~2.9e-5) deliberately differs from the LM gate's
        (~8.1e-5): with a mis-specified no-background single-isotope model
        on real data, the transmission-domain least-squares and the
        counts-domain deviance weight bins differently and converge to
        different biased optima.  Both anchors move only when their
        respective solver paths change.

        Tolerances follow the LM gate's cross-backend rationale: anchors
        were captured on macOS (Accelerate); ``rel=1e-6`` absorbs
        BLAS/libm sum-ordering differences on Linux CI while staying
        orders of magnitude tighter than any real dispatch regression.
        If this gate ever flaps across backends, relax the deviance
        anchor first — the sum over ~4e3 bins amplifies bin-level libm
        differences far more than the converged density does.
        """
        E, S_agg, O_agg, c, hf177 = venus_data

        result = nereids.fit_counts_spectrum_typed(
            S_agg,
            O_agg,
            E,
            isotopes=[(hf177, 1.0e-5)],
            solver="kl",
            temperature_k=293.6,
            max_iter=200,
            background=False,
            c=c,
            flight_path_m=25.0,
            delta_t_us=0.5,
            delta_l_m=0.005,
        )

        # Anchors regenerated after #635's analytic-Jacobian availability
        # fix. The old anchor was captured at an identity-Fisher
        # gradient-descent STALL: without an analytic transmission Jacobian
        # the joint-Poisson stage 1 silently degraded to projected gradient
        # descent, which stopped 1.7 % away (in density) from the true
        # optimum of the SAME objective. With the analytic Fisher the fit
        # reaches a strictly BETTER minimum (deviance/dof 31445.853 <
        # 31445.957) in 3 iterations. The model is unchanged (bit-exact
        # parity verified); only the optimum actually attained improved.
        EXPECTED_DENSITY = 2.9596692297867937e-05
        EXPECTED_DEVIANCE_PER_DOF = 31445.852761391532

        assert bool(result.converged) is True, (
            f"counts-KL fit did not converge on the real VENUS fixture "
            f"(converged={bool(result.converged)})"
        )
        assert float(result.densities[0]) == pytest.approx(
            EXPECTED_DENSITY, rel=1e-6
        ), (
            f"counts-KL density drifted: got {float(result.densities[0])!r}, "
            f"expected {EXPECTED_DENSITY!r} (±1e-6 rel)"
        )
        # Coarse physical bracket, independent of the machine-generated
        # anchor above: both solver families land in (2.9-8.1)e-5
        # atoms/barn on this measured Hf spectrum, so any value outside
        # [1e-5, 1e-4] means solver breakage, not sample physics.  This
        # prevents a future wholesale re-anchoring commit from silently
        # absorbing an order-of-magnitude regression.
        assert 1e-5 < float(result.densities[0]) < 1e-4, (
            f"counts-KL density {float(result.densities[0])!r} fell outside "
            f"the physical bracket [1e-5, 1e-4] for this measured sample"
        )
        assert result.deviance_per_dof is not None, (
            "counts-KL dispatch must populate deviance_per_dof (primary GOF)"
        )
        assert float(result.deviance_per_dof) == pytest.approx(
            EXPECTED_DEVIANCE_PER_DOF, rel=1e-6
        ), (
            f"deviance/dof drifted: got {float(result.deviance_per_dof)!r}, "
            f"expected {EXPECTED_DEVIANCE_PER_DOF!r} (±1e-6 rel)"
        )
        assert float(result.deviance_per_dof) > 1e3, (
            "real-data regime check: D/dof should be >> 1 on raw VENUS "
            "counts (un-modelled upstream physics); an O(1) value means "
            "the gate is no longer fitting real data"
        )


# ===========================================================================
# fit_energy_range Python parameter (#514)
# ===========================================================================


class TestFitEnergyRangeBindingParameter:
    """Tests for the SAMMY EMIN/EMAX-equivalent `fit_energy_range` parameter
    on `fit_spectrum_typed`.  The other two binding entry points
    (`fit_counts_spectrum_typed`, `spatial_map_typed`) share the same
    config-builder code path (`with_fit_energy_range(...).map_err(...)?`
    after the rest of the chain), so a regression in the validation /
    plumbing surfaces here too.  Locks in the behaviour the perf
    scripts now depend on for SoftwareX-paper SAMMY parity.
    """

    def test_fit_spectrum_typed_full_grid_with_range_matches_pre_cropped(
        self, u238_data
    ):
        """`fit_spectrum_typed` called with the full grid +
        `fit_energy_range=(low, high)` must produce the same density as
        the equivalent pre-cropped fit (within tight numerical tolerance).
        This is the SAMMY-parity contract: the model is evaluated on the
        full grid so resolution broadening at the boundaries is correct,
        and the LM cost path masks residuals to the inner range.
        """
        # Synthetic noisy transmission across a wide energy grid with a
        # single resonance at 6.67 eV.
        energies = np.linspace(1.0, 30.0, 600)
        true_density = 0.0008
        t_clean = np.asarray(
            nereids.forward_model(energies, [(u238_data, true_density)])
        )
        rng = np.random.default_rng(20260430)
        t_obs = t_clean + rng.normal(0.0, 0.005, size=t_clean.shape)
        sigma = np.full_like(t_obs, 0.005)

        e_min, e_max = 4.0, 12.0  # window around the 6.67 eV resonance

        # Approach 1: pre-crop in Python (legacy pattern).
        mask = (energies >= e_min) & (energies <= e_max)
        r_cropped = nereids.fit_spectrum_typed(
            transmission=np.ascontiguousarray(t_obs[mask]),
            uncertainty=np.ascontiguousarray(sigma[mask]),
            energies=np.ascontiguousarray(energies[mask]),
            isotopes=[(u238_data, true_density)],
            solver="lm",
            temperature_k=293.6,
            max_iter=200,
        )

        # Approach 2: full grid + fit_energy_range (the new SAMMY-parity
        # path).
        r_ranged = nereids.fit_spectrum_typed(
            transmission=t_obs,
            uncertainty=sigma,
            energies=energies,
            isotopes=[(u238_data, true_density)],
            solver="lm",
            temperature_k=293.6,
            max_iter=200,
            fit_energy_range=(e_min, e_max),
        )

        # Both should converge.
        assert bool(r_cropped.converged) is True
        assert bool(r_ranged.converged) is True

        # No resolution broadening is configured in this test, so the
        # only difference between "pre-crop" and "full grid + range" is
        # whether the LM cost path sums residuals over the same set of
        # active bins — which it does (the mask matches the explicit
        # crop bin-for-bin).  Densities therefore agree to numerical-
        # noise tolerance.  We enforce 1e-4 — much tighter than any
        # broadening-effect tolerance, but loose enough to absorb BLAS
        # ordering noise across platforms.  Tightening below 1e-4 risks
        # platform-flapping; loosening above 1e-4 lets a real masking
        # regression slip through.
        rel_err = abs(r_cropped.densities[0] - r_ranged.densities[0]) / abs(
            r_ranged.densities[0]
        )
        assert rel_err < 1e-4, (
            f"density disagreement: cropped={r_cropped.densities[0]:.6e}, "
            f"ranged={r_ranged.densities[0]:.6e}, rel_err={rel_err:.3e}"
        )

    def test_fit_spectrum_typed_rejects_invalid_range(self, u238_data):
        """Invalid `fit_energy_range` (reversed, non-finite, or empty)
        must raise `ValueError` from the binding — the underlying
        `with_fit_energy_range` setter validates these and the binding
        propagates the error with `?`.
        """
        energies = np.linspace(1.0, 30.0, 200)
        t = np.full_like(energies, 0.95)
        sigma = np.full_like(energies, 0.01)

        # Reversed range.
        with pytest.raises(ValueError, match="strictly less than max"):
            nereids.fit_spectrum_typed(
                transmission=t,
                uncertainty=sigma,
                energies=energies,
                isotopes=[(u238_data, 0.001)],
                fit_energy_range=(20.0, 5.0),
            )

        # Non-finite bound.
        with pytest.raises(ValueError, match="finite"):
            nereids.fit_spectrum_typed(
                transmission=t,
                uncertainty=sigma,
                energies=energies,
                isotopes=[(u238_data, 0.001)],
                fit_energy_range=(float("nan"), 20.0),
            )

    def test_fit_spectrum_typed_default_none_unchanged(self, u238_data):
        """Calling without `fit_energy_range` must produce the same
        result as before the parameter was added.  Backward-compat
        regression against accidentally treating `None` as "narrow to
        the smallest range" or other surprising semantics.
        """
        energies = np.linspace(1.0, 30.0, 400)
        true_density = 0.001
        # Match temperatures: `forward_model` uses 293.6 K by default,
        # so the fit must too — otherwise Doppler-broadening mismatch
        # produces a multi-percent density bias unrelated to the
        # parameter under test.
        t = np.asarray(nereids.forward_model(energies, [(u238_data, true_density)]))
        sigma = np.full_like(t, 0.005)

        # Seed the fit away from the truth (5× too high) so a do-nothing
        # solver path (e.g. a regression that treats `fit_energy_range
        # = None` as "no active bins" → returns the initial parameters
        # unchanged) is detectable: the assertion below would fire on
        # the seed value `5e-3`, not on the true `1e-3`.
        r = nereids.fit_spectrum_typed(
            transmission=t,
            uncertainty=sigma,
            energies=energies,
            isotopes=[(u238_data, 5.0 * true_density)],
            solver="lm",
            temperature_k=293.6,
            max_iter=100,
        )
        # Recovery on noiseless data should be tight; with the wrong
        # seed, this catches both "degenerate range" regressions AND
        # any regression that prevents convergence outright.
        assert bool(r.converged) is True
        assert abs(r.densities[0] - true_density) / true_density < 1e-3, (
            f"density did not converge to truth: got {r.densities[0]:.6e}, "
            f"expected {true_density:.6e} (initial seed was "
            f"{5.0 * true_density:.6e}, so a do-nothing solver path "
            f"would leave the param at the seed)"
        )


class TestFixDensities:
    """Issue #633: freeze known densities (calibration-foil thermometry).

    These verify the freeze *plumbing* — which parameters vary, how frozen
    slots map through the free-only covariance/uncertainty vector, and the
    reported degrees of freedom. Spectra are generated with
    ``nereids.forward_model`` and fitted through the same stack (loop
    closure): appropriate here because the physics oracle (Doppler /
    temperature sensitivity, Beer–Lambert) is validated independently against
    SAMMY in the ``nereids-physics`` crate — these tests deliberately do not
    re-validate it. Non-vacuity is guaranteed by seeding every fit away from
    truth (temperature 50 K off, densities offset), so a no-op fit fails.
    """

    def test_fix_densities_holds_density_and_recovers_temperature(self, u238_data):
        """Synthetic spectrum at a known (n, T); freeze n at truth and fit
        temperature only. The frozen density is held EXACTLY at its initial
        value and the temperature is recovered."""
        energies = np.linspace(1.0, 30.0, 400)
        true_density = 8.0e-4
        true_temp = 350.0
        t = np.asarray(
            nereids.forward_model(
                energies, [(u238_data, true_density)], temperature_k=true_temp
            )
        )
        sigma = np.full_like(t, 0.005)

        r = nereids.fit_spectrum_typed(
            transmission=t,
            uncertainty=sigma,
            energies=energies,
            isotopes=[(u238_data, true_density)],
            solver="lm",
            temperature_k=300.0,  # seeded 50 K off
            fit_temperature=True,
            fix_densities=True,
            max_iter=200,
        )
        assert bool(r.converged) is True
        # Frozen density held bit-exactly at its initial value.
        assert r.densities[0] == true_density
        # Temperature (the sole free parameter) recovered.
        assert r.temperature_k is not None
        assert abs(r.temperature_k - true_temp) < 2.0, (
            f"temperature not recovered: got {r.temperature_k}, want {true_temp}"
        )
        # #633 P0 regression: with the density frozen, temperature is the sole
        # free parameter. Its 1-σ must be finite and positive (mapped from the
        # correct free slot); the frozen density reports NaN (no covariance
        # column). Before the free-index mapping fix the binding returned a NaN
        # temperature σ and a misassigned density σ, since uncertainties were
        # indexed by the full parameter layout.
        assert r.temperature_k_unc is not None
        assert np.isfinite(r.temperature_k_unc) and r.temperature_k_unc > 0.0, (
            "frozen-density thermometry must report a finite positive "
            f"temperature σ, got {r.temperature_k_unc}"
        )
        assert np.isnan(float(r.uncertainties[0])), (
            f"frozen density must report NaN σ, got {r.uncertainties[0]}"
        )

    def test_fix_densities_kl_reports_temperature_uncertainty(self, u238_data):
        """The KL transmission solver (``solver="kl"``) must also report a
        finite temperature 1-σ with a density frozen — the parallel path the
        LM tests don't exercise. Pre-fix a leftover full-index overwrite
        clobbered ``temperature_k_unc`` to None on this path (review R2 P0)."""
        energies = np.linspace(1.0, 30.0, 400)
        true_density = 8.0e-4
        true_temp = 350.0
        t = np.asarray(
            nereids.forward_model(
                energies, [(u238_data, true_density)], temperature_k=true_temp
            )
        )
        sigma = np.full_like(t, 0.005)
        r = nereids.fit_spectrum_typed(
            transmission=t,
            uncertainty=sigma,
            energies=energies,
            isotopes=[(u238_data, true_density)],
            solver="kl",
            temperature_k=300.0,
            fit_temperature=True,
            fix_densities=True,
            max_iter=200,
        )
        assert bool(r.converged) is True
        assert r.densities[0] == true_density
        assert r.temperature_k is not None
        # The regression: temperature_k_unc was None pre-fix on the KL path.
        assert r.temperature_k_unc is not None
        assert np.isfinite(r.temperature_k_unc) and r.temperature_k_unc > 0.0, (
            "KL frozen-density temperature σ must be finite positive, "
            f"got {r.temperature_k_unc}"
        )

    def test_fix_densities_counts_path_holds_density_and_reports_uncertainty(
        self, u238_data
    ):
        """The counts KL fitter (``fit_counts_spectrum_typed``) must also hold
        a frozen density, report NaN for its 1-σ (no covariance column), and
        report a finite temperature σ. Exercises the counts uncertainty loop
        directly — the transmission tests never touch it (the CHANGELOG's
        "every fitter" claim)."""
        energies = np.linspace(1.0, 30.0, 300)
        true_density = 8.0e-4
        true_temp = 350.0
        flux = 5000.0
        t_1d = np.asarray(
            nereids.forward_model(
                energies, [(u238_data, true_density)], temperature_k=true_temp
            )
        )
        rng = np.random.default_rng(20260703)
        open_beam = np.maximum(
            rng.poisson(np.full_like(t_1d, flux)).astype(float), 1.0
        )
        sample = rng.poisson(flux * t_1d).astype(float)
        r = nereids.fit_counts_spectrum_typed(
            sample_counts=sample,
            open_beam_counts=open_beam,
            energies=energies,
            isotopes=[(u238_data, true_density)],
            solver="kl",
            c=1.0,
            temperature_k=300.0,
            fit_temperature=True,
            fix_densities=True,
            max_iter=200,
        )
        assert bool(r.converged) is True
        assert r.densities[0] == true_density, "frozen density must not move"
        # R1 counts-loop regression: frozen density → NaN σ, not a
        # neighbouring free parameter's error bar.
        assert np.isnan(float(r.uncertainties[0])), (
            f"frozen density must report NaN σ, got {r.uncertainties[0]}"
        )
        assert r.temperature_k_unc is not None
        assert np.isfinite(r.temperature_k_unc) and r.temperature_k_unc > 0.0, (
            "counts frozen-density temperature σ must be finite positive, "
            f"got {r.temperature_k_unc}"
        )

    def test_fix_densities_spatial_map_holds_density(self, u238_data):
        """``spatial_map_typed`` must freeze densities per pixel: the frozen
        density map stays bit-exactly at the (offset) seed instead of fitting
        toward the data. Exercises the per-pixel frozen path end to end."""
        energies = np.linspace(1.0, 30.0, 200)
        true_density = 2.0e-3
        ny, nx = 2, 2
        # Cube at 350 K; the fit seeds temperature at 300 K and fits it while
        # the density is frozen at an offset seed (≠ truth).
        t_1d = np.asarray(
            nereids.forward_model(
                energies, [(u238_data, true_density)], temperature_k=350.0
            )
        )
        trans = np.tile(t_1d[:, None, None], (1, ny, nx))
        unc = np.full_like(trans, 0.005)
        data = nereids.from_transmission(trans, unc)
        seed = 1.0e-3  # deliberately far from the 2e-3 truth
        result = nereids.spatial_map_typed(
            data,
            energies,
            [u238_data],
            initial_densities=[seed],
            temperature_k=300.0,
            fit_temperature=True,  # keep ≥1 free param so pixels converge
            fix_densities=True,
            max_iter=60,
        )
        dmap = np.asarray(result.density_maps[0])
        assert dmap.shape == (ny, nx)
        # Every pixel frozen at the seed, NOT driven toward true_density.
        np.testing.assert_array_equal(dmap, seed)

    def test_density_free_mask_freezes_selected_density(self, u238_data):
        """A per-density ``density_free`` mask freezes only the marked density
        parameter while the unmarked one is still fitted. Two spectrally
        distinct isotopes: index 0 frozen at its seed (held bit-exactly, NaN
        σ), index 1 free and recovered (finite σ). This also exercises the
        leading-frozen uncertainty mapping — the free density's σ must come
        from free slot 0, not full index 1 (the R1 index-compression fix)."""
        # A second, well-separated resonance (≈21 eV) so the two densities are
        # distinguishable and the free one can be recovered.
        iso_b = _make_single_resonance(
            z=90, a=232, awr=230.045, scattering_radius=9.0,
            energy=21.0, j=0.5, gn=0.002, gg=0.02,
        )
        energies = np.linspace(1.0, 30.0, 500)
        d_a_true, d_b_true = 8.0e-4, 1.2e-3
        t = np.asarray(
            nereids.forward_model(
                energies, [(u238_data, d_a_true), (iso_b, d_b_true)]
            )
        )
        # Deterministic ~0.2 % pseudo-noise: the covariance is scaled by
        # chi2/nu, so EXACTLY noise-free data drives chi2 -> 0 once the
        # analytic-Jacobian fit (#635) converges machine-exactly, collapsing
        # the free density's sigma to NaN — a degenerate oracle, not an
        # index-mapping regression. Real data always has chi2 > 0.
        t = t * (1.0 + 0.002 * np.sin(7.3 * np.arange(t.size)))
        sigma = np.full_like(t, 0.003)
        r = nereids.fit_spectrum_typed(
            transmission=t,
            uncertainty=sigma,
            energies=energies,
            # index 0 frozen at truth; index 1 seeded off and left free.
            isotopes=[(u238_data, d_a_true), (iso_b, 5.0e-4)],
            solver="lm",
            temperature_k=293.6,
            density_free=[False, True],
            max_iter=200,
        )
        assert bool(r.converged) is True
        # Masked-false density held bit-exactly; free density recovered.
        assert r.densities[0] == d_a_true, "masked-false density must not move"
        assert abs(r.densities[1] - d_b_true) / d_b_true < 0.05, (
            f"free density should recover: got {r.densities[1]}, want {d_b_true}"
        )
        # Frozen density → NaN σ; free density → finite positive σ.
        assert np.isnan(float(r.uncertainties[0])), (
            f"frozen density σ must be NaN, got {r.uncertainties[0]}"
        )
        assert np.isfinite(float(r.uncertainties[1])) and r.uncertainties[1] > 0.0, (
            f"free density σ must be finite positive, got {r.uncertainties[1]}"
        )

    def test_all_frozen_no_free_param_rejected(self, u238_data):
        """Freezing the only density with no other free parameter is rejected
        up front — the all-fixed solver fast path would otherwise report
        ``converged=true`` from a fit that varied nothing (review R4)."""
        energies = np.linspace(1.0, 30.0, 200)
        t = np.asarray(nereids.forward_model(energies, [(u238_data, 8.0e-4)]))
        sigma = np.full_like(t, 0.005)
        # ValueError (not RuntimeError) since review R2: config-class
        # errors (PipelineError::InvalidParameter) map to ValueError on
        # every fitter, matching the spatial API convention.
        with pytest.raises(ValueError, match="no free parameters"):
            nereids.fit_spectrum_typed(
                transmission=t,
                uncertainty=sigma,
                energies=energies,
                isotopes=[(u238_data, 8.0e-4)],
                solver="lm",
                temperature_k=293.6,
                fix_densities=True,  # freeze the only param, nothing else free
            )

    def test_fix_densities_and_density_free_are_mutually_exclusive(self, u238_data):
        """Supplying both `fix_densities` and `density_free` is rejected."""
        energies = np.linspace(1.0, 30.0, 100)
        t = np.full_like(energies, 0.95)
        sigma = np.full_like(energies, 0.01)
        with pytest.raises(ValueError, match="not both"):
            nereids.fit_spectrum_typed(
                transmission=t,
                uncertainty=sigma,
                energies=energies,
                isotopes=[(u238_data, 0.001)],
                fix_densities=True,
                density_free=[True],
            )

    def test_density_free_wrong_length_rejected(self, u238_data):
        """A `density_free` mask of the wrong length is rejected."""
        energies = np.linspace(1.0, 30.0, 100)
        t = np.full_like(energies, 0.95)
        sigma = np.full_like(energies, 0.01)
        with pytest.raises(ValueError):
            nereids.fit_spectrum_typed(
                transmission=t,
                uncertainty=sigma,
                energies=energies,
                isotopes=[(u238_data, 0.001)],
                density_free=[True, False],  # 2 masks for 1 density
            )


# ===========================================================================
# fit_energy_scale recovery (#531 — single-spectrum LM, SAMMY TZERO)
# ===========================================================================


# SAMMY TZERO time-of-flight constant: TOF_FACTOR = sqrt(m_n / (2·eV)) · 1e6.
# Same formula as `EnergyScaleTransmissionModel::new` in
# `crates/nereids-fitting/src/transmission_model.rs`, using the CODATA-2018
# constants from `nereids-core::constants` (NEUTRON_MASS_KG = 1.674927498e-27,
# EV_TO_JOULES = 1.602176634e-19) so this test stays bit-identical to the
# Rust side; hardcoding 72.297 would drift by ~1e-5 relative on the energy
# inversion and silently bias the recovered t0.
_NEUTRON_MASS_KG = 1.674_927_498_04e-27
_EV_TO_JOULES = 1.602_176_634e-19
_TOF_FACTOR = np.sqrt(0.5 * _NEUTRON_MASS_KG / _EV_TO_JOULES) * 1.0e6


def _measured_energies_for_known_tzero(
    e_true: np.ndarray,
    *,
    t0_true_us: float,
    l_scale_true: float,
    l_nom_m: float,
) -> np.ndarray:
    """Inverse of `EnergyScaleTransmissionModel::corrected_energies`.

    Given the *true* corrected energies the physics produced and the
    instrument's true (t0, L_scale), return the *measured* (nominal)
    energy grid the data would be reported on so that
    `corrected_energies(E_meas, t0_true, L_scale_true) ≈ E_true`.

    Forward map (Rust):
        tof_meas[i] = TOF_FACTOR · L_nom / √E_meas[i]
        E_corr[i]   = (TOF_FACTOR · L_eff / (tof_meas[i] − t0))²
                    where L_eff = L_nom · L_scale.
    Inverting for E_meas given E_true ≡ E_corr:
        tof_meas[i] = t0 + TOF_FACTOR · L_nom · L_scale_true / √E_true[i]
        E_meas[i]   = (TOF_FACTOR · L_nom / tof_meas[i])²
    """
    tof_meas = t0_true_us + _TOF_FACTOR * l_nom_m * l_scale_true / np.sqrt(e_true)
    return (_TOF_FACTOR * l_nom_m / tof_meas) ** 2


class TestFitEnergyScaleRecovery:
    """Exercises the `fit_energy_scale=True` branch of `fit_spectrum_typed`
    end-to-end.  This is the only Python-level test that does so — the
    existing coverage in `crates/nereids-pipeline/src/spatial.rs` runs
    the spatial pipeline (a different binding entry point) and would not
    catch a silent regression in `py_fit_spectrum_typed`'s
    `config.with_energy_scale(...)` branch.

    Gate strategy (per #531).  The test injects a known (t0, L_scale)
    calibration offset, then verifies the fit's *physically observable*
    output — not the individual parameter values — for three reasons:

    1.  (t0, L_scale) are coupled.  At a single resonance, position alone
        gives one constraint for two unknowns; multiple resonances lift
        the degeneracy, but the LM cost surface still has a shallow
        valley along the line  ``t0 ≈ const - L_scale × something(E)``.
        Tightening individual-parameter tolerances below the valley width
        produces flaky tests without catching the regression we care
        about (silent disable).
    2.  At realistic neutron-imaging temperatures (293 K) the Doppler
        kernel applied to the *corrected* energies is not bit-identical
        to the kernel applied during data synthesis on the true grid;
        this contributes a small residual bias (~few percent) to the
        recovered density even on noiseless data, distinct from the
        bug under test.
    3.  The bug to catch is a silent ``fit_energy_scale=True`` disable.
        That is detected by  (a) the χ² ratio between baseline (flag off)
        and calibrated (flag on),  (b) the presence of finite t0_us /
        l_scale on the FitResult,  and  (c) the recovered *effective*
        energy mapping agreeing with truth.  All three would fail if the
        flag silently became a no-op.
    """

    def test_recovers_injected_t0_and_l_scale(self):
        L_NOM = 25.0
        T0_TRUE = 0.5  # μs
        L_SCALE_TRUE = 1.005
        TRUE_DENSITY = 3.0e-4

        # Multi-resonance U-238 fixture.  Three well-separated resonances
        # (6.67 / 20.87 / 36.68 eV, standard SAMMY benchmark values)
        # supply three position constraints — enough to lift the
        # (t0, L_scale) degeneracy a single peak would leave (see
        # class docstring point 1).
        u238 = nereids.create_resonance_data(
            z=92,
            a=238,
            awr=236.006,
            scattering_radius=9.48,
            resonances=[
                (6.67, 0.5, 0.0015, 0.023),
                (20.87, 0.5, 0.0103, 0.026),
                (36.68, 0.5, 0.0344, 0.027),
            ],
            target_spin=0.0,
        )

        # True (corrected) energy grid spanning all three resonances.
        # Linear-in-TOF sampling so peak widths are sampled uniformly
        # in the data's native coordinate (matches real instruments).
        tof_lo = _TOF_FACTOR * L_NOM / np.sqrt(45.0)
        tof_hi = _TOF_FACTOR * L_NOM / np.sqrt(4.0)
        tof_grid = np.linspace(tof_lo, tof_hi, 800)
        e_true = np.sort((_TOF_FACTOR * L_NOM / tof_grid) ** 2)

        # Clean transmission at the TRUE energies.  Match forward_model's
        # default 293.6 K Doppler temperature on the fit side.
        t_clean = np.asarray(
            nereids.forward_model(e_true, [(u238, TRUE_DENSITY)])
        )

        # Measured energy grid: what the instrument reports given the
        # unknown (T0_TRUE, L_SCALE_TRUE) calibration offset.  Without
        # the energy-scale fit branch, the solver evaluates σ at this
        # *wrong* grid and density / χ² blow up.
        e_meas = _measured_energies_for_known_tzero(
            e_true,
            t0_true_us=T0_TRUE,
            l_scale_true=L_SCALE_TRUE,
            l_nom_m=L_NOM,
        )

        rng = np.random.default_rng(20260514)
        sigma_noise = 0.005
        t_obs = t_clean + rng.normal(0.0, sigma_noise, size=t_clean.shape)
        sigma = np.full_like(t_obs, sigma_noise)

        baseline_kwargs = dict(
            transmission=t_obs,
            uncertainty=sigma,
            energies=e_meas,
            isotopes=[(u238, TRUE_DENSITY)],
            solver="lm",
            temperature_k=293.6,
            max_iter=200,
        )

        # Baseline: no energy-scale fit.  σ is evaluated at E_meas
        # (shifted by the injected calibration), so the model peaks
        # never line up with the data and χ² stays large.
        r_baseline = nereids.fit_spectrum_typed(**baseline_kwargs)

        # Energy-scale fit ON.  Solver maps E_meas → estimated E_true
        # via internal `corrected_energies(t0, L_scale)`.
        r_calibrated = nereids.fit_spectrum_typed(
            **baseline_kwargs,
            fit_energy_scale=True,
            t0_init_us=0.0,
            l_scale_init=1.0,
            energy_scale_flight_path_m=L_NOM,
        )

        # 1. The calibrated fit must converge.
        assert bool(r_calibrated.converged) is True, (
            f"calibrated fit did not converge: "
            f"iters={int(r_calibrated.iterations)}, "
            f"chi2_r={float(r_calibrated.reduced_chi_squared):.4e}"
        )

        # 2. t0_us and l_scale must be populated (not None) on the
        # FitResult.  A silent regression that dropped
        # `config.with_energy_scale(...)` would leave these as None.
        assert r_calibrated.t0_us is not None, (
            "result.t0_us is None — fit_energy_scale=True was likely "
            "silently disabled in py_fit_spectrum_typed"
        )
        assert r_calibrated.l_scale is not None, (
            "result.l_scale is None — fit_energy_scale=True was likely "
            "silently disabled in py_fit_spectrum_typed"
        )
        assert np.isfinite(float(r_calibrated.t0_us))
        assert np.isfinite(float(r_calibrated.l_scale))

        # 3. χ²-ratio gate — the primary silent-disable detector.  With
        # the injected ~0.5 % calibration offset the baseline fit
        # produces χ²_r ~ O(10²-10³); the calibrated fit drops it by
        # at least 10× (typically 100×+).  If the flag is silently
        # ignored, both fits return the same χ² and this fires.
        chi2_baseline = float(r_baseline.reduced_chi_squared)
        chi2_calibrated = float(r_calibrated.reduced_chi_squared)
        assert chi2_calibrated < chi2_baseline / 10.0, (
            f"calibrated χ²_r={chi2_calibrated:.4e} not ≥10× better than "
            f"baseline χ²_r={chi2_baseline:.4e}; fit_energy_scale=True "
            f"may be silently disabled"
        )

        # 4. Effective energy-mapping recovery.  Recompute the fit's
        # corrected energies in Python using the SAME formula as
        # `EnergyScaleTransmissionModel::corrected_energies`, then
        # compare against E_true.  This gates the *physics* without
        # being sensitive to which point in the (t0, L_scale) valley
        # the LM solver settled at.  1 % relative is well inside the
        # ~0.5 % calibration offset injected — a silent disable would
        # leave the mapping at identity (E_corr = E_meas), producing
        # max-rel-err ~5 % at the low-energy end of the grid.
        tof_meas = _TOF_FACTOR * L_NOM / np.sqrt(e_meas)
        l_eff_fit = L_NOM * float(r_calibrated.l_scale)
        e_corr_fit = (_TOF_FACTOR * l_eff_fit / (tof_meas - float(r_calibrated.t0_us))) ** 2
        max_rel_err = float(np.max(np.abs(e_corr_fit - e_true) / e_true))
        assert max_rel_err < 1.0e-2, (
            f"effective energy mapping not recovered: max rel err "
            f"{max_rel_err:.3e}, fit_(t0, L_scale)="
            f"({float(r_calibrated.t0_us):.4f}, "
            f"{float(r_calibrated.l_scale):.6f})"
        )

        # 5. L_scale recovery is empirically stable across LM noise to
        # within 1 % of truth — tighter than the (t0, L_scale) valley
        # width but loose enough to absorb single-precision-noise-level
        # LM step variation.  Catches regressions that scramble the L
        # parameter without dropping the whole flag (where the χ²
        # gate above would still pass).
        l_scale_rel_err = abs(float(r_calibrated.l_scale) - L_SCALE_TRUE) / L_SCALE_TRUE
        assert l_scale_rel_err < 1.0e-2, (
            f"L_scale rel err {l_scale_rel_err:.3e} exceeds 1 %: "
            f"got {float(r_calibrated.l_scale):.6f}, truth {L_SCALE_TRUE}"
        )

    def test_recovers_injected_t0_and_l_scale_counts_kl(self):
        """Issue #608: the KL/counts energy-scale path must also recover an
        injected (t0, L_scale) calibration from the production cold start
        (t0=0, L_scale=1).  The physically-exact true-σ EnergyScale model makes
        the calibration χ² razor-thin around the truth (a cold-start LM lands in
        a wrong minimum), so the resonance peak-matching seed in `pipeline.rs`
        puts the optimizer in the global-min basin.  This is the KL analogue of
        `test_recovers_injected_t0_and_l_scale` (the LM path); the GUI uses KL
        for counts data, so this path must be robust too.

        Gated on the recovered *density* — a clean physical observable that only
        recovers when the calibration does (at the wrong cold-start minimum the
        density is tens of percent off) — plus a finite, near-truth L_scale.
        """
        L_NOM = 25.0
        T0_TRUE = 0.5  # μs
        L_SCALE_TRUE = 1.005
        TRUE_DENSITY = 3.0e-4
        u238 = nereids.create_resonance_data(
            z=92,
            a=238,
            awr=236.006,
            scattering_radius=9.48,
            resonances=[
                (6.67, 0.5, 0.0015, 0.023),
                (20.87, 0.5, 0.0103, 0.026),
                (36.68, 0.5, 0.0344, 0.027),
            ],
            target_spin=0.0,
        )
        tof_lo = _TOF_FACTOR * L_NOM / np.sqrt(45.0)
        tof_hi = _TOF_FACTOR * L_NOM / np.sqrt(4.0)
        e_true = np.sort((_TOF_FACTOR * L_NOM / np.linspace(tof_lo, tof_hi, 800)) ** 2)
        t_clean = np.asarray(nereids.forward_model(e_true, [(u238, TRUE_DENSITY)]))
        e_meas = _measured_energies_for_known_tzero(
            e_true, t0_true_us=T0_TRUE, l_scale_true=L_SCALE_TRUE, l_nom_m=L_NOM
        )

        rng = np.random.default_rng(20260601)
        flux = 5000.0
        open_beam = np.maximum(
            rng.poisson(np.full_like(t_clean, flux)).astype(float), 1.0
        )
        sample = rng.poisson(flux * t_clean).astype(float)

        r = nereids.fit_counts_spectrum_typed(
            sample_counts=sample,
            open_beam_counts=open_beam,
            energies=e_meas,
            isotopes=[(u238, TRUE_DENSITY)],
            solver="kl",
            c=1.0,
            temperature_k=293.6,
            max_iter=200,
            fit_energy_scale=True,
            t0_init_us=0.0,
            l_scale_init=1.0,
            energy_scale_flight_path_m=L_NOM,
        )

        assert bool(r.converged) is True, (
            f"KL calibrated fit did not converge: iters={int(r.iterations)}"
        )
        assert r.t0_us is not None and np.isfinite(float(r.t0_us))
        assert r.l_scale is not None and np.isfinite(float(r.l_scale))
        # Density recovers only if the calibration does; at the wrong
        # cold-start minimum it is tens of percent off, so this fires if the
        # peak-match seed fails to reach the global-min basin.
        dens_rel_err = abs(float(r.densities[0]) - TRUE_DENSITY) / TRUE_DENSITY
        assert dens_rel_err < 0.05, (
            f"recovered density {float(r.densities[0]):.4e} is {dens_rel_err:.1%} from "
            f"truth {TRUE_DENSITY:.1e} — energy-scale calibration not recovered from "
            f"the cold start (peak-match seed regression?)"
        )
        l_scale_rel_err = abs(float(r.l_scale) - L_SCALE_TRUE) / L_SCALE_TRUE
        assert l_scale_rel_err < 1.0e-2, (
            f"L_scale rel err {l_scale_rel_err:.3e} exceeds 1 %: "
            f"got {float(r.l_scale):.6f}, truth {L_SCALE_TRUE}"
        )

    def test_recovers_t0_l_scale_and_temperature_jointly(self):
        """Issue #634: fit_energy_scale + fit_temperature recover the injected
        (t0, L_scale, T) in ONE fit through the Python binding — the flag
        combination the binding used to reject. Gate on the observables
        (L_scale + temperature), seeding T 60 K off so a no-op cannot pass."""
        L_NOM = 25.0
        T0_TRUE = 0.5
        L_SCALE_TRUE = 1.005
        TRUE_DENSITY = 3.0e-4
        TRUE_TEMP = 450.0

        u238 = nereids.create_resonance_data(
            z=92,
            a=238,
            awr=236.006,
            scattering_radius=9.48,
            resonances=[
                (6.67, 0.5, 0.0015, 0.023),
                (20.87, 0.5, 0.0103, 0.026),
                (36.68, 0.5, 0.0344, 0.027),
            ],
            target_spin=0.0,
        )
        tof_lo = _TOF_FACTOR * L_NOM / np.sqrt(45.0)
        tof_hi = _TOF_FACTOR * L_NOM / np.sqrt(4.0)
        tof_grid = np.linspace(tof_lo, tof_hi, 800)
        e_true = np.sort((_TOF_FACTOR * L_NOM / tof_grid) ** 2)
        # Clean transmission at the TRUE energies AND the true temperature.
        t_clean = np.asarray(
            nereids.forward_model(
                e_true, [(u238, TRUE_DENSITY)], temperature_k=TRUE_TEMP
            )
        )
        e_meas = _measured_energies_for_known_tzero(
            e_true,
            t0_true_us=T0_TRUE,
            l_scale_true=L_SCALE_TRUE,
            l_nom_m=L_NOM,
        )
        rng = np.random.default_rng(20260704)
        sigma_noise = 0.004
        t_obs = t_clean + rng.normal(0.0, sigma_noise, size=t_clean.shape)
        sigma = np.full_like(t_obs, sigma_noise)

        r = nereids.fit_spectrum_typed(
            transmission=t_obs,
            uncertainty=sigma,
            energies=e_meas,
            isotopes=[(u238, TRUE_DENSITY)],
            solver="lm",
            temperature_k=TRUE_TEMP - 60.0,  # seeded 60 K off
            fit_temperature=True,
            fit_energy_scale=True,
            t0_init_us=0.0,
            l_scale_init=1.0,
            energy_scale_flight_path_m=L_NOM,
            max_iter=300,
        )
        assert bool(r.converged) is True
        assert r.t0_us is not None and np.isfinite(r.t0_us)
        assert r.l_scale is not None and np.isfinite(r.l_scale)
        assert r.temperature_k is not None
        # L_scale + temperature are the well-constrained observables (t0/L
        # share a shallow valley; see the class docstring).
        assert abs(float(r.l_scale) - L_SCALE_TRUE) / L_SCALE_TRUE < 1e-2
        assert abs(float(r.temperature_k) - TRUE_TEMP) < 15.0, (
            f"temperature not recovered jointly: got {r.temperature_k}, "
            f"want {TRUE_TEMP}"
        )
        # Non-identity accessor oracle (#634 review): at a genuinely shifted
        # calibration, corrected_energies(e_meas) must land on the TRUE
        # energy axis the data was synthesized on (to within the recovered
        # parameters' accuracy) — a no-op accessor returns e_meas instead,
        # which differs from e_true by ~0.5-1 % here.
        corr = np.asarray(r.corrected_energies(e_meas))
        med_rel = np.median(np.abs(corr - e_true) / e_true)
        noop_rel = np.median(np.abs(e_meas - e_true) / e_true)
        assert med_rel < 0.2 * noop_rel, (
            f"corrected axis (med rel err {med_rel:.2e}) should be far closer "
            f"to truth than the uncorrected axis ({noop_rel:.2e})"
        )

    def test_corrected_energies_accessor(self):
        """Issue #634: FitResult.corrected_energies maps a nominal grid through
        the fitted energy scale (finite ndarray), and returns None when the
        energy scale was not fitted."""
        L_NOM = 25.0
        u238 = nereids.create_resonance_data(
            z=92,
            a=238,
            awr=236.006,
            scattering_radius=9.48,
            resonances=[(6.67, 0.5, 0.0015, 0.023), (20.87, 0.5, 0.0103, 0.026)],
            target_spin=0.0,
        )
        energies = np.linspace(4.0, 30.0, 400)
        t_clean = np.asarray(nereids.forward_model(energies, [(u238, 3.0e-4)]))
        sigma = np.full_like(t_clean, 0.005)

        # Energy-scale fit → corrected_energies is a finite ndarray, ascending.
        r_es = nereids.fit_spectrum_typed(
            transmission=t_clean,
            uncertainty=sigma,
            energies=energies,
            isotopes=[(u238, 3.0e-4)],
            solver="lm",
            fit_energy_scale=True,
            t0_init_us=0.0,
            l_scale_init=1.0,
            energy_scale_flight_path_m=L_NOM,
            max_iter=100,
        )
        corr = r_es.corrected_energies(energies)
        assert corr is not None
        corr = np.asarray(corr)
        assert corr.shape == energies.shape
        assert np.all(np.isfinite(corr)) and np.all(np.diff(corr) > 0)
        # Non-circular numeric oracle (#634 review): hand-compute the SAMMY
        # −t0 transform in numpy from the FITTED (t0, l_scale) and the fit's
        # flight path — a no-op accessor, a sign flip, a t0/l_scale swap, or
        # a wrong flight-path source all break this equality, none of which
        # the shape/finite asserts above can see.
        kl = _TOF_FACTOR * L_NOM
        tof = kl / np.sqrt(energies)
        expected = (kl * float(r_es.l_scale) / (tof - float(r_es.t0_us))) ** 2
        np.testing.assert_allclose(corr, expected, rtol=1e-12)

        # No energy-scale fit → None (distinguishes "unfit" from "identity").
        r_plain = nereids.fit_spectrum_typed(
            transmission=t_clean,
            uncertainty=sigma,
            energies=energies,
            isotopes=[(u238, 3.0e-4)],
            solver="lm",
            max_iter=50,
        )
        assert r_plain.corrected_energies(energies) is None

        # Invalid nominal grids are rejected with the binding's standard
        # energy-grid validation (#634 review) — NaN, non-positive, and
        # non-ascending grids raise instead of passing through.
        with pytest.raises(ValueError):
            r_es.corrected_energies(np.array([np.nan, 2.0, 3.0]))
        with pytest.raises(ValueError):
            r_es.corrected_energies(np.array([0.0, 1.0, 2.0]))
        with pytest.raises(ValueError):
            r_es.corrected_energies(np.array([3.0, 1.0, 2.0]))


# ===========================================================================
# Issue #558 — energy-grid validation at the PyO3 boundary
# ===========================================================================
#
# The SLBW / RML / URR / Reich-Moore leaves carry release-mode
# `assert!(energy_ev.is_finite() && energy_ev > 0.0)` guards.  Every PyO3
# entry that takes an `energies` argument must validate the grid before
# any energy reaches those leaves so callers see a clean
# ``ValueError`` instead of a ``pyo3_runtime.PanicException`` (which is
# not a subclass of ``ValueError`` and bypasses normal Python error
# handling).  These tests cover the most recently hardened entries
# — ``forward_model``, ``spatial_map_typed``, and
# ``calibrate_energy`` — and re-cover the previously hardened entries to
# document the contract.

class TestEnergyGridValidation:
    """Every PyO3 entry that takes ``energies`` must reject malformed grids
    with ``ValueError`` rather than leaking a ``PanicException`` from the
    physics leaves' release-mode asserts (issue #558)."""

    @pytest.mark.parametrize(
        "bad_energies",
        [
            np.array([np.nan, 2.0, 3.0]),
            np.array([1.0, np.inf, 3.0]),
            np.array([0.0, 1.0, 2.0]),  # non-positive
            np.array([-1.0, 1.0, 2.0]),
            np.array([3.0, 1.0, 2.0]),  # not ascending
            np.array([1.0, 1.0, 2.0]),  # not strictly ascending
        ],
    )
    def test_cross_sections_rejects_invalid_grid(self, u238_data, bad_energies):
        with pytest.raises(ValueError):
            nereids.cross_sections(bad_energies, u238_data)

    @pytest.mark.parametrize(
        "bad_energies",
        [
            np.array([np.nan, 2.0, 3.0]),
            np.array([1.0, np.inf, 3.0]),
            np.array([0.0, 1.0, 2.0]),
            np.array([-1.0, 1.0, 2.0]),
            np.array([3.0, 1.0, 2.0]),
        ],
    )
    def test_forward_model_rejects_invalid_grid(self, u238_data, bad_energies):
        with pytest.raises(ValueError):
            nereids.forward_model(bad_energies, [(u238_data, 0.001)])

    @pytest.mark.parametrize(
        "bad_energies",
        [
            np.array([np.nan, 2.0, 3.0]),
            np.array([1.0, np.inf, 3.0]),
            np.array([0.0, 1.0, 2.0]),
            np.array([-1.0, 1.0, 2.0]),
        ],
    )
    def test_spatial_map_typed_rejects_invalid_grid(self, u238_data, bad_energies):
        # Build a tiny matching-shape transmission cube so the
        # length check passes and we hit the energy-grid validator.
        n_e = len(bad_energies)
        t = np.ones((n_e, 1, 1), dtype=np.float64) * 0.9
        u = np.ones((n_e, 1, 1), dtype=np.float64) * 0.01
        data = nereids.from_transmission(t, u)
        with pytest.raises(ValueError):
            nereids.spatial_map_typed(
                data, bad_energies, [u238_data], max_iter=2
            )

    @pytest.mark.parametrize(
        "bad_energies",
        [
            np.array([np.nan, 2.0, 3.0]),
            np.array([1.0, np.inf, 3.0]),
            np.array([0.0, 1.0, 2.0]),
            np.array([-1.0, 1.0, 2.0]),
            np.array([3.0, 1.0, 2.0]),
        ],
    )
    def test_calibrate_energy_rejects_invalid_grid(self, u238_data, bad_energies):
        n_e = len(bad_energies)
        t = np.full(n_e, 0.9, dtype=np.float64)
        s = np.full(n_e, 0.01, dtype=np.float64)
        with pytest.raises(ValueError):
            nereids.calibrate_energy(
                bad_energies,
                t,
                s,
                [u238_data],
                [1.0],
                25.0,
            )

    def test_calibrate_energy_rejects_empty_grid(self, u238_data):
        e = np.array([], dtype=np.float64)
        t = np.array([], dtype=np.float64)
        s = np.array([], dtype=np.float64)
        with pytest.raises(ValueError):
            nereids.calibrate_energy(e, t, s, [u238_data], [1.0], 25.0)


# ===========================================================================
# Type-stub conformance — regression for issue #555 (M5)
# ===========================================================================


class TestStubConformance:
    """Sanity checks that the runtime PyO3 export surface is reflected in
    ``bindings/python/python/nereids/__init__.pyi``.

    The original M5 drift (``from_counts_with_nuisance`` exported by Rust
    but missing from the stub) slipped past ``scripts/check_python_api_drift.py``
    because that checker compares stub vs the curated narrative docs, not
    stub vs the compiled extension.  These tests close that gap on the
    runtime side.
    """

    def test_from_counts_with_nuisance_importable_and_introspectable(self):
        """M5 regression: the symbol must be importable from ``nereids``,
        callable, AND its signature must be introspectable via
        ``inspect.signature``.

        Tooling and users that rely on runtime signature introspection
        (``inspect.signature``) — e.g. IDE REPL completions, Sphinx
        ``autodoc``, ``help()`` — need the runtime signature to stay
        consistent with the stub.  Static type checkers (mypy / pyright)
        read the ``.pyi`` directly and don't use runtime introspection,
        but a divergence between the stub and the compiled extension
        still misleads either audience.  This test catches stub/runtime
        drift in both directions.
        """
        import inspect

        from nereids import from_counts_with_nuisance  # noqa: F401

        # Stub fix means a static type-checker would now accept this
        # `getattr` path too; the runtime check below is the regression
        # gate against future re-drift.
        assert callable(nereids.from_counts_with_nuisance)
        sig = inspect.signature(nereids.from_counts_with_nuisance)
        params = list(sig.parameters.keys())
        # Parameter names must match the PyO3 binding signature so the
        # stub stays in lock-step with runtime introspection.
        assert params == ["sample_counts", "flux", "background"], (
            f"from_counts_with_nuisance signature drifted from stub: "
            f"got {params!r}"
        )

    def test_runtime_exports_present_in_stub(self):
        """Every public (non-underscore) attribute on the compiled
        ``nereids`` module must be either defined in ``__init__.pyi`` OR
        documented as an intentional omission below.

        This catches the inverse of the python-api.md drift check: a
        PyO3 export that was never added to the stub at all.  Failure
        mode prior to this regression: `from_counts_with_nuisance`
        worked at runtime but was rejected by mypy / pyright.
        """
        import ast
        from pathlib import Path

        # Resolve the stub relative to the installed package (works
        # whether nereids is installed editable from the source tree
        # or as a wheel that carries the .pyi alongside the .so).
        nereids_dir = Path(nereids.__file__).resolve().parent
        stub_path = nereids_dir / "__init__.pyi"
        assert stub_path.exists(), f"missing type stub: {stub_path}"

        tree = ast.parse(stub_path.read_text(encoding="utf-8"))
        stub_names: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):
                    stub_names.add(node.name)

        # Dunders + the ``__version__`` style attributes don't belong
        # in the stub; pyo3's auto-injected ``annotations`` import isn't
        # an export either.  We also exclude submodules (e.g. PyO3
        # re-exports the inner ``nereids.nereids`` extension module
        # itself as an attribute on the outer package) because those
        # are package plumbing, not typed-surface exports.
        #
        # No ``Py*``-prefix filter: every current PyO3 export carries
        # a ``#[pyclass(name = "...")]`` / ``#[pyo3(name = "...")]``
        # attribute that strips the prefix on the Python side, so a
        # runtime symbol that *did* start with ``Py`` would be a real
        # surface (e.g. an unrenamed ``#[pyclass]``) and should be
        # gated, not silently skipped.
        import types

        runtime_names = {
            name
            for name in dir(nereids)
            if not name.startswith("_")
            and not isinstance(getattr(nereids, name), types.ModuleType)
        }

        # Intentional omissions: re-exported third-party modules or
        # convenience aliases that aren't part of the typed surface.
        # Keep this set tight — every entry is a drift loophole.
        intentional_omissions: set[str] = set()

        missing = runtime_names - stub_names - intentional_omissions
        assert not missing, (
            f"runtime symbols missing from __init__.pyi: {sorted(missing)!r}. "
            "Either add a `def`/`class` to the stub OR justify the omission "
            "in `intentional_omissions`."
        )

    def test_runtime_function_parameter_names_match_stub(self):
        """Every top-level callable exported by the compiled ``nereids``
        module must have parameter NAMES that match the stub signature
        in ``__init__.pyi``.

        This is the sibling drift gate to
        ``test_runtime_exports_present_in_stub``: that test ensures the
        *name* of every export is present; this one ensures the
        *parameter names* of every function-shaped export are present.

        Class methods are out of scope (PyO3 ``#[getter]`` /
        ``#[pymethods]`` introspection is noisier and rarely drifts the
        same way).  This guards against the M5/P2-1 failure mode where
        a kwarg like ``resolution=`` was added to the PyO3 export but
        not to the stub, so static type checkers rejected valid
        runtime calls.
        """
        import ast
        import inspect
        from pathlib import Path

        nereids_dir = Path(nereids.__file__).resolve().parent
        stub_path = nereids_dir / "__init__.pyi"
        tree = ast.parse(stub_path.read_text(encoding="utf-8"))

        # Map of top-level function name -> ordered list of parameter
        # names as declared in the stub.  We include positional-only,
        # positional-or-keyword, the ``*`` marker is skipped, and
        # keyword-only parameters are appended in declaration order.
        # ``self`` / ``cls`` cannot appear at module scope; ignore
        # ``*args`` / ``**kwargs`` because none of our exports use them
        # and including them would only weaken the comparison.
        stub_params: dict[str, list[str]] = {}
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name.startswith("_"):
                continue
            names: list[str] = []
            args = node.args
            for arg in args.posonlyargs:
                names.append(arg.arg)
            for arg in args.args:
                names.append(arg.arg)
            for arg in args.kwonlyargs:
                names.append(arg.arg)
            stub_params[node.name] = names

        mismatches: list[str] = []
        for name, expected in stub_params.items():
            rt_obj = getattr(nereids, name, None)
            if rt_obj is None or not callable(rt_obj):
                # Stub declares a function but runtime export is
                # missing or non-callable: that is a different drift
                # class, covered by the test above.
                continue
            try:
                sig = inspect.signature(rt_obj)
            except (TypeError, ValueError):
                # PyO3 builtins can refuse signature introspection on
                # rare overload shapes.  Skip rather than fail — the
                # name-presence gate above still applies.
                continue
            # Drop *args / **kwargs entries on the runtime side too;
            # we compare the named-parameter spine only.
            actual = [
                p.name
                for p in sig.parameters.values()
                if p.kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ]
            if actual != expected:
                mismatches.append(
                    f"  {name}:\n"
                    f"    stub:    {expected!r}\n"
                    f"    runtime: {actual!r}"
                )

        assert not mismatches, (
            "parameter-name drift between PyO3 runtime exports and "
            "__init__.pyi:\n" + "\n".join(mismatches)
        )


# ===========================================================================
# calibrate_energy — regression smoke test for issue #555 (M13)
# ===========================================================================


class TestCalibrateEnergySmoke:
    """Smoke test for the ``calibrate_energy`` PyO3 binding.

    The M13 fix replaced ``as_slice()?`` borrows into NumPy memory with
    ``as_slice()?.to_vec()`` copies so the closure passed to ``py.detach``
    no longer holds dangling references when the GIL is released and
    another Python thread mutates the input arrays.

    A reliable concurrency exploit test for the old borrow pattern is
    racy and difficult to write deterministically; the value here is
    confirming the mechanical owned-Vec rewrite did not break the
    happy path on a small synthetic case."""

    def test_calibrate_energy_runs_on_synthetic_u238(self, u238_data):
        # Use the U-238-like single-resonance fixture (resonance at 6.67 eV).
        # Generate a clean transmission spectrum and feed it back through
        # ``calibrate_energy``.  The fitter does a coarse grid over
        # (L, t0, n_total); we only need a finite, non-NaN return.
        assumed_l = 25.0
        energies = np.linspace(1.0, 30.0, 400)
        transmission = np.asarray(
            nereids.forward_model(energies, [(u238_data, 5.0e-4)])
        )
        uncertainty = np.full_like(transmission, 0.01)

        result = nereids.calibrate_energy(
            energies,
            transmission,
            uncertainty,
            [u238_data],
            [1.0],
            assumed_l,
            293.6,
        )

        # All returned scalars must be finite — a dangling-borrow / GIL
        # corruption regression typically surfaces as NaN, junk values,
        # or a crash inside `py.detach`.
        assert np.isfinite(result.flight_path_m)
        assert np.isfinite(result.t0_us)
        assert np.isfinite(result.total_density)
        assert np.isfinite(result.reduced_chi_squared)
        # The corrected energy grid must round-trip with the input length
        # and be strictly ascending (basic sanity for the fitted mapping).
        e_corr = np.asarray(result.energies_corrected)
        assert e_corr.shape == energies.shape
        assert np.all(np.diff(e_corr) > 0.0)

    def test_calibrate_energy_input_arrays_unchanged(self, u238_data):
        """Regression gate for the borrow→to_vec fix: after the closure
        captures owned copies, the original NumPy arrays must round-trip
        unchanged across the call (no in-place mutation, no aliased
        write-back).  This is a structural rather than concurrency
        check, but it catches the simplest class of aliasing
        regressions if someone reverts the ``.to_vec()`` later."""
        energies = np.linspace(1.0, 30.0, 200)
        transmission = np.asarray(
            nereids.forward_model(energies, [(u238_data, 5.0e-4)])
        )
        uncertainty = np.full_like(transmission, 0.01)

        e_before = energies.copy()
        t_before = transmission.copy()
        s_before = uncertainty.copy()

        _ = nereids.calibrate_energy(
            energies,
            transmission,
            uncertainty,
            [u238_data],
            [1.0],
            25.0,
            293.6,
        )

        np.testing.assert_array_equal(energies, e_before)
        np.testing.assert_array_equal(transmission, t_before)
        np.testing.assert_array_equal(uncertainty, s_before)


class TestCalibrateResolution:
    """Resolution calibration binding: position is PINNED by default; the shared
    SAMMY energy-scale ``(t0, L_scale)`` is an explicit, prior-constrained opt-in
    (replacing the retired per-family ``position_nuisance_us``)."""

    @staticmethod
    def _calibrant():
        # Synthetic IC-broadened, non-black calibrant with TWO well-separated
        # resonances (15 + 45 eV) so the width is identifiable (no width<->position
        # ridge) — mirrors the Rust erosion test, so a chi2 change is attributable to
        # the 1/sqrt(E) lag / L_scale confounding, not mere over-parameterization.
        iso = nereids.create_resonance_data(
            72,
            177,
            175.0,
            0.7,
            [(15.0, 3.5, 0.05, 0.06), (45.0, 3.5, 0.05, 0.06)],
            target_spin=3.5,
        )
        flight = 25.0
        energies = np.linspace(5.0, 60.0, 1000)
        ic = nereids.IkedaCarpenter(
            flight_path_m=flight,
            e_min_ev=0.5e-3,
            e_max_ev=1000.0,
            alpha=nereids.EnergyLaw.sqrt_e(0.30, 0.0),
            beta=0.10,
            r=nereids.EnergyLaw.exp_mev(25.0),
        )
        t = np.asarray(
            nereids.forward_model(
                energies,
                [(iso, 5.0e-4)],
                temperature_k=300.0,
                resolution=ic.as_tabulated(),
            )
        )
        unc = np.full_like(t, 0.004)
        return iso, energies, t, unc

    def test_pins_position_by_default(self):
        iso, e, t, unc = self._calibrant()
        cal = nereids.calibrate_resolution(
            e, t, unc, "ic", isotopes=[(iso, 5.0e-4)], temperature_k=300.0
        )
        assert np.isfinite(cal.chi2)
        # Default config pins position at its center and incurs no prior penalty.
        assert cal.position_t0_us == 0.0
        assert cal.position_l_scale == 1.0
        assert cal.prior_penalty == 0.0

    def test_fit_position_kwargs_accepted_with_prior(self):
        iso, e, t, unc = self._calibrant()
        cal = nereids.calibrate_resolution(
            e,
            t,
            unc,
            "ic",
            isotopes=[(iso, 5.0e-4)],
            temperature_k=300.0,
            fit_t0=True,
            fit_l_scale=True,
            t0_prior_us=0.5,
            l_scale_prior=0.002,
            restarts=2,
        )
        assert np.isfinite(cal.chi2)
        assert np.isfinite(cal.position_t0_us)
        assert np.isfinite(cal.position_l_scale)
        assert cal.prior_penalty >= 0.0
        # L_scale must respect the ±2% guard rail.
        assert 0.98 <= cal.position_l_scale <= 1.02
        # t0 within the ±5 µs guard rail.
        assert abs(cal.position_t0_us) <= 5.0

    def test_result_exposes_new_position_fields_not_old_nuisance(self):
        iso, e, t, unc = self._calibrant()
        cal = nereids.calibrate_resolution(
            e, t, unc, "gaussian", isotopes=[(iso, 5.0e-4)], temperature_k=300.0
        )
        for attr in ("position_t0_us", "position_l_scale", "prior_penalty"):
            assert hasattr(cal, attr), f"missing new result field {attr}"
        # The retired per-family nuisance getter must be gone.
        assert not hasattr(cal, "position_nuisance_us")

    def test_udr_corr_requires_base(self):
        iso, e, t, unc = self._calibrant()
        with pytest.raises(ValueError, match="base_udr"):
            nereids.calibrate_resolution(
                e,
                t,
                unc,
                "udr_corr",
                isotopes=[(iso, 5.0e-4)],
                temperature_k=300.0,
            )

    def test_free_l_scale_erodes_wrong_family_penalty(self):
        # Binding-level mirror of the Rust test, on a TWO-resonance calibrant (width
        # identifiable): a Gaussian fitting an IC calibrant fits better with a free
        # physical (t0, L_scale) than pinned, because the asymmetric-kernel lag
        # shares the 1/sqrt(E) basis of an L_scale error — not just more parameters.
        iso, e, t, unc = self._calibrant()
        kw = dict(isotopes=[(iso, 5.0e-4)], temperature_k=300.0, restarts=2)
        pinned = nereids.calibrate_resolution(e, t, unc, "gaussian", **kw)
        free = nereids.calibrate_resolution(
            e, t, unc, "gaussian", fit_t0=True, fit_l_scale=True, **kw
        )
        assert free.chi2 < pinned.chi2

    def test_invalid_position_prior_rejected(self):
        iso, e, t, unc = self._calibrant()
        with pytest.raises(ValueError):
            nereids.calibrate_resolution(
                e,
                t,
                unc,
                "ic",
                isotopes=[(iso, 5.0e-4)],
                temperature_k=300.0,
                fit_t0=True,
                t0_prior_us=-1.0,  # σ must be > 0 when set
            )

    def test_invalid_flight_path_rejected_not_panicked(self):
        # Regression for the fit_t0 panic path: a non-positive flight_path_m must
        # raise ValueError (graceful) rather than panic via the inverted t0 clamp
        # bound (min > max). Covers both the pinned default and the fit_t0 opt-in.
        iso, e, t, unc = self._calibrant()
        for kw in ({}, {"fit_t0": True}):
            with pytest.raises(ValueError):
                nereids.calibrate_resolution(
                    e,
                    t,
                    unc,
                    "ic",
                    isotopes=[(iso, 5.0e-4)],
                    temperature_k=300.0,
                    flight_path_m=-1.0,
                    **kw,
                )

    def test_ic_params_decoded_and_bounds_reported(self):
        # The bounded "ic" family (#642) exposes decoded physical parameters
        # (single source of truth: the calibrated resolution, not the
        # ln/box-encoded theta) plus the degeneracy report.
        iso, e, t, unc = self._calibrant()
        cal = nereids.calibrate_resolution(
            e, t, unc, "ic", isotopes=[(iso, 5.0e-4)], temperature_k=300.0
        )
        p = cal.params()
        for key in ("a0", "a1", "beta", "r", "psr_fwhm_us"):
            assert key in p, f"missing decoded param {key}"
        assert p["a0"] > 0.0
        assert p["a1"] > 0.0  # alpha(E) positive by construction
        assert p["beta"] > 0.0
        assert 0.0 <= p["r"] <= 1.0
        assert p["psr_fwhm_us"] >= 0.0
        assert cal.n_free_params == 4
        assert len(cal.theta) == 4
        assert isinstance(cal.bounds_hit, list)
        assert all(isinstance(s, str) for s in cal.bounds_hit)
        assert "n_free_params=4" in repr(cal)

    def test_fit_psr_appends_fifth_parameter(self):
        iso, e, t, unc = self._calibrant()
        cal = nereids.calibrate_resolution(
            e,
            t,
            unc,
            "ic",
            isotopes=[(iso, 5.0e-4)],
            temperature_k=300.0,
            fit_psr=True,
        )
        assert cal.n_free_params == 5
        assert len(cal.theta) == 5
        # The fitted PSR FWHM respects its box (0.05-1 us).
        assert 0.05 <= cal.params()["psr_fwhm_us"] <= 1.0

    def test_fit_psr_requires_ic_family(self):
        iso, e, t, unc = self._calibrant()
        with pytest.raises(ValueError, match="fit_psr"):
            nereids.calibrate_resolution(
                e,
                t,
                unc,
                "gaussian",
                isotopes=[(iso, 5.0e-4)],
                temperature_k=300.0,
                fit_psr=True,
            )

    def test_invalid_psr_fwhm_rejected(self):
        iso, e, t, unc = self._calibrant()
        for bad in (-1.0, float("nan")):
            with pytest.raises(ValueError, match="psr_fwhm_ns"):
                nereids.calibrate_resolution(
                    e,
                    t,
                    unc,
                    "ic",
                    isotopes=[(iso, 5.0e-4)],
                    temperature_k=300.0,
                    psr_fwhm_ns=bad,
                )

    def test_absurd_pinned_psr_width_rejected(self):
        # Review #645 round 2, F1: psr_fwhm_ns is NANOSECONDS (FTS convention
        # 350 ns); kernel-synthesis cost is quadratic in the fold width, so a
        # us-as-ns unit slip (350 meaning us -> 350_000 ns) previously passed
        # the finite/sign check and became a multi-hour silent hang behind a
        # fictitious 350 us fold. Nonzero widths above the 10_000 ns (10 us)
        # ceiling must raise up front; the message names the ns unit and the
        # 350-ns convention. (Boundary acceptance at exactly 10_000 ns is
        # pinned Rust-side: rejects_absurd_pinned_psr_width.)
        iso, e, t, unc = self._calibrant()
        with pytest.raises(ValueError, match="NANOSECONDS.*350 ns"):
            nereids.calibrate_resolution(
                e,
                t,
                unc,
                "ic",
                isotopes=[(iso, 5.0e-4)],
                temperature_k=300.0,
                psr_fwhm_ns=350_000.0,
            )

    def test_infeasible_start_psr_width_rejected(self):
        # Review #645 round 3, F1: a pinned PSR width in (0, ~58.6 ns) passes
        # the finite/sign/ceiling checks but cannot be SYNTHESIZED at the
        # optimizer's default beta/R start (beta = 0.1 spans a 160 us storage
        # tail, capping the tau-step at ~19.5 ns > fwhm/3). Previously every
        # initial simplex vertex was infinite and the calibration burned
        # max_iter before a generic "no finite-objective" error blaming the
        # forward model. The Rust pre-flight must reject the start up front,
        # and the informative message (naming psr_fwhm_ns and the tau-cap
        # cause) must propagate through the binding as a ValueError.
        iso, e, t, unc = self._calibrant()
        with pytest.raises(
            ValueError, match="starting parameter vector.*psr_fwhm_ns"
        ):
            nereids.calibrate_resolution(
                e,
                t,
                unc,
                "ic",
                isotopes=[(iso, 5.0e-4)],
                temperature_k=300.0,
                psr_fwhm_ns=55.0,
            )

    def test_fit_psr_with_zero_width_rejected(self):
        # psr_fwhm_ns=0 is documented as "no PSR fold"; fit_psr=True would
        # silently clamp the 0 start into the [0.05, 1] us fit box. The
        # contradiction must raise, not fit a phantom fold.
        iso, e, t, unc = self._calibrant()
        with pytest.raises(ValueError, match="fit_psr"):
            nereids.calibrate_resolution(
                e,
                t,
                unc,
                "ic",
                isotopes=[(iso, 5.0e-4)],
                temperature_k=300.0,
                psr_fwhm_ns=0.0,
                fit_psr=True,
            )

    def test_psr_parameters_trail_the_signature(self):
        # Review #645 F7: psr_fwhm_ns / fit_psr were added AFTER the original
        # signature froze, so they must sit at the END — inserting them
        # mid-signature would silently shift every pre-existing call passing
        # >= 14 positional arguments.
        sig = nereids.calibrate_resolution.__text_signature__
        assert sig is not None
        assert (
            sig.index("l_scale_prior")
            < sig.index("psr_fwhm_ns")
            < sig.index("fit_psr")
        ), f"psr_fwhm_ns/fit_psr must trail the signature: {sig}"



# ===========================================================================
# Bounded multiplicative baseline (#635)
# ===========================================================================


# Truth baseline shared by the closed loops below: a few % off unity,
# curved, strictly positive on the test grids, inside the DEFAULT bounds.
_BL_TRUE = (1.02, -0.03, 0.01)


def _baseline_curve(energies):
    """B(E) = b0 + b1·ln(E/E_ref) + b2·ln²(E/E_ref) at the truth
    coefficients, with E_ref the geometric midpoint of the grid — the same
    convention the Rust fitter uses (``baseline_reference_energy``)."""
    e = np.asarray(energies, dtype=float)
    e_ref = float(np.sqrt(e[0] * e[-1]))
    z = np.log(e / e_ref)
    return _BL_TRUE[0] + _BL_TRUE[1] * z + _BL_TRUE[2] * z * z, e_ref


class TestMultiplicativeBaseline:
    """Issue #635: bounded multiplicative baseline B(E) applied OUTERMOST.

    These verify the *plumbing* — kwargs, result fields, mode routing, and
    boundary rejections. The deep closed loops (Jacobians, solver behavior,
    stage-1 aggregation, non-vacuity controls) live in the Rust suites
    (``nereids-fitting`` / ``nereids-pipeline``). Non-vacuity here comes
    from seeding every fit at the identity baseline (1, 0, 0) while the
    data carries a distinctly different truth, so a no-op fit fails.
    """

    def test_baseline_composes_with_energy_scale_and_temperature(self):
        """Merge guard (#635 x #634): baseline + fit_energy_scale + fit_temperature
        jointly — the one combination neither PR's suite covered alone. The
        baseline B is a function of the NOMINAL (measured) grid while the
        physics is evaluated on the corrected grid, matching the wrapper's
        construction; truth is injected accordingly and all three parameter
        groups must be recovered from deliberately-off seeds.

        THREE well-separated resonances: a single line cannot pin
        (t0, L_scale, T) simultaneously — the offset-seeded fit then walks
        to a distant (t0, T) basin (observed: t0 = −4.2 µs, T = 1220 K,
        baseline still recovered) — so identifiability, not plumbing,
        requires the multi-line scene, as in the #634 joint-recovery tests."""
        data = nereids.create_resonance_data(
            92, 238, 236.006, 9.4285,
            [(6.674, 0.5, 1.493e-3, 2.3e-2),
             (14.0, 0.5, 3.0e-3, 2.3e-2),
             (24.5, 0.5, 5.0e-3, 2.3e-2)],
            target_spin=0.0,
        )
        energies = np.linspace(1.0, 30.0, 400)
        true_density, true_temp = 8.0e-4, 350.0
        t0_true, ls_true, flight_path = 0.4, 1.003, 25.0
        k = nereids.energy_to_tof(1.0, 1.0)
        tof = k * flight_path / np.sqrt(energies)
        e_corr = (k * flight_path * ls_true / (tof - t0_true)) ** 2
        t_phys = np.asarray(
            nereids.forward_model(
                e_corr, [(data, true_density)], temperature_k=true_temp
            )
        )
        curve, e_ref = _baseline_curve(energies)
        t = t_phys * curve
        sigma = np.full_like(t, 0.005)

        r = nereids.fit_spectrum_typed(
            transmission=t,
            uncertainty=sigma,
            energies=energies,
            isotopes=[(data, true_density)],
            solver="lm",
            temperature_k=300.0,           # 50 K off
            fit_temperature=True,
            fit_energy_scale=True,         # seeds t0=0, l_scale=1
            energy_scale_flight_path_m=flight_path,
            fix_densities=True,
            baseline=True,                 # seeds identity (1, 0, 0)
            max_iter=300,
        )
        assert bool(r.converged) is True
        assert r.temperature_k is not None
        assert abs(r.temperature_k - true_temp) < 5.0
        assert abs(r.t0_us - t0_true) < 0.05
        assert abs(r.l_scale - ls_true) < 5e-4
        assert r.baseline is not None
        for i, (fitted, truth) in enumerate(zip(r.baseline, _BL_TRUE)):
            assert abs(fitted - truth) < 1.5e-2, (
                f"baseline[{i}] = {fitted} vs truth {truth}"
            )
        assert r.baseline_e_ref_ev == pytest.approx(e_ref, rel=1e-12)

    def test_baseline_lm_recovers_coefficients_and_temperature(self, u238_data):
        energies = np.linspace(1.0, 30.0, 400)
        true_density = 8.0e-4
        true_temp = 350.0
        t_clean = np.asarray(
            nereids.forward_model(
                energies, [(u238_data, true_density)], temperature_k=true_temp
            )
        )
        curve, e_ref = _baseline_curve(energies)
        t = t_clean * curve
        sigma = np.full_like(t, 0.005)

        r = nereids.fit_spectrum_typed(
            transmission=t,
            uncertainty=sigma,
            energies=energies,
            isotopes=[(u238_data, true_density)],
            solver="lm",
            temperature_k=300.0,  # seeded 50 K off
            fit_temperature=True,
            fix_densities=True,
            baseline=True,
            max_iter=200,
        )
        assert bool(r.converged) is True
        assert r.baseline is not None
        for i, (fitted, truth) in enumerate(zip(r.baseline, _BL_TRUE)):
            assert abs(fitted - truth) < 1e-2, (
                f"baseline[{i}] = {fitted} vs truth {truth}"
            )
        assert r.baseline_e_ref_ev == pytest.approx(e_ref, rel=1e-12)
        assert r.temperature_k is not None
        assert abs(r.temperature_k - true_temp) < 5.0
        assert r.warnings == []

    def test_baseline_counts_kl_recovers_coefficients(self, u238_data):
        energies = np.linspace(1.0, 30.0, 300)
        true_density = 8.0e-4
        flux = 5000.0
        t_clean = np.asarray(
            nereids.forward_model(energies, [(u238_data, true_density)])
        )
        curve, _ = _baseline_curve(energies)
        rng = np.random.default_rng(20260705)
        open_beam = np.maximum(
            rng.poisson(np.full_like(t_clean, flux)).astype(float), 1.0
        )
        sample = rng.poisson(flux * t_clean * curve).astype(float)

        r = nereids.fit_counts_spectrum_typed(
            sample_counts=sample,
            open_beam_counts=open_beam,
            energies=energies,
            isotopes=[(u238_data, true_density)],
            solver="kl",
            fix_densities=True,
            baseline=True,
            max_iter=500,
        )
        assert bool(r.converged) is True
        assert r.deviance_per_dof is not None
        assert r.baseline is not None
        # Poisson noise at 5000 counts/bin: coefficients recover to a few
        # times the shot-noise floor.
        for i, (fitted, truth) in enumerate(zip(r.baseline, _BL_TRUE)):
            assert abs(fitted - truth) < 2e-2, (
                f"baseline[{i}] = {fitted} vs truth {truth}"
            )

    def test_baseline_none_when_disabled(self, u238_data):
        energies = np.linspace(1.0, 30.0, 200)
        t = np.asarray(nereids.forward_model(energies, [(u238_data, 8.0e-4)]))
        r = nereids.fit_spectrum_typed(
            transmission=t,
            uncertainty=np.full_like(t, 0.005),
            energies=energies,
            isotopes=[(u238_data, 5.0e-4)],
            solver="lm",
        )
        assert r.baseline is None
        assert r.baseline_e_ref_ev is None
        assert r.warnings == []

    def test_baseline_with_default_background_rejected_on_all_fitters(
        self, u238_data
    ):
        """``background=True`` frees Anorm by default, and b0/Anorm are
        degenerate normalizations — every fitter rejects the combination at
        the binding boundary with the fix in the message."""
        energies = np.linspace(1.0, 30.0, 100)
        t = np.full_like(energies, 0.95)
        sigma = np.full_like(energies, 0.01)
        with pytest.raises(ValueError, match="fit_anorm=False"):
            nereids.fit_spectrum_typed(
                transmission=t,
                uncertainty=sigma,
                energies=energies,
                isotopes=[(u238_data, 0.001)],
                background=True,
                baseline=True,
            )
        with pytest.raises(ValueError, match="fit_anorm=False"):
            nereids.fit_counts_spectrum_typed(
                sample_counts=np.full_like(energies, 900.0),
                open_beam_counts=np.full_like(energies, 1000.0),
                energies=energies,
                isotopes=[(u238_data, 0.001)],
                background=True,
                baseline=True,
            )
        trans = np.tile(t[:, None, None], (1, 2, 2))
        unc = np.full_like(trans, 0.01)
        with pytest.raises(ValueError, match="fit_anorm=False"):
            nereids.spatial_map_typed(
                nereids.from_transmission(trans, unc),
                energies,
                [u238_data],
                initial_densities=[0.001],
                background=True,
                baseline=True,
            )

    def test_baseline_with_background_fit_anorm_false_accepted(self, u238_data):
        """The sanctioned combination: additive ABC background with Anorm
        HELD FIXED alongside the baseline."""
        energies = np.linspace(1.0, 30.0, 400)
        true_density = 8.0e-4
        t_clean = np.asarray(
            nereids.forward_model(energies, [(u238_data, true_density)])
        )
        curve, _ = _baseline_curve(energies)
        t = t_clean * curve
        r = nereids.fit_spectrum_typed(
            transmission=t,
            uncertainty=np.full_like(t, 0.005),
            energies=energies,
            isotopes=[(u238_data, true_density)],
            solver="lm",
            fix_densities=True,
            background=True,
            fit_anorm=False,
            baseline=True,
            max_iter=300,
        )
        assert bool(r.converged) is True
        assert r.anorm == 1.0, "Anorm was frozen at its init"
        assert r.baseline is not None
        assert abs(r.baseline[0] - _BL_TRUE[0]) < 2e-2

    def test_baseline_suboptions_require_baseline_true(self, u238_data):
        energies = np.linspace(1.0, 30.0, 100)
        t = np.full_like(energies, 0.95)
        sigma = np.full_like(energies, 0.01)
        with pytest.raises(ValueError, match="require baseline=True"):
            nereids.fit_spectrum_typed(
                transmission=t,
                uncertainty=sigma,
                energies=energies,
                isotopes=[(u238_data, 0.001)],
                fit_b1=False,  # baseline sub-option without baseline=True
            )

    def test_fit_anorm_false_requires_background(self, u238_data):
        energies = np.linspace(1.0, 30.0, 100)
        t = np.full_like(energies, 0.95)
        sigma = np.full_like(energies, 0.01)
        with pytest.raises(ValueError, match="requires background=True"):
            nereids.fit_spectrum_typed(
                transmission=t,
                uncertainty=sigma,
                energies=energies,
                isotopes=[(u238_data, 0.001)],
                fit_anorm=False,
            )

    def test_baseline_bounds_kwargs_widen_the_box(self, u238_data):
        """An init outside the DEFAULT box is rejected by the core config
        validation; supplying a wider explicit box makes the same init
        legal — the bounds kwargs genuinely reach the optimizer."""
        energies = np.linspace(1.0, 30.0, 300)
        true_density = 8.0e-4
        t_clean = np.asarray(
            nereids.forward_model(energies, [(u238_data, true_density)])
        )
        curve, _ = _baseline_curve(energies)
        t = t_clean * curve
        sigma = np.full_like(t, 0.005)
        common = dict(
            transmission=t,
            uncertainty=sigma,
            energies=energies,
            isotopes=[(u238_data, true_density)],
            solver="lm",
            fix_densities=True,
            baseline=True,
            b0_init=1.3,  # outside the default (0.9, 1.1) box
            max_iter=300,
        )
        # ValueError since review R2: the out-of-bounds init is rejected by
        # the core config validation (PipelineError::InvalidParameter),
        # which now maps to ValueError on every fitter — the same exception
        # type the spatial API raises for the identical bad input.
        with pytest.raises(ValueError, match="outside"):
            nereids.fit_spectrum_typed(**common)
        r = nereids.fit_spectrum_typed(**common, b0_bounds=(0.5, 1.5))
        assert bool(r.converged) is True
        assert abs(r.baseline[0] - _BL_TRUE[0]) < 2e-2

    def test_degenerate_trio_warning_roundtrip(self, u238_data):
        """Free Anorm + free temperature + free density surfaces the
        structured warning on FitResult.warnings (the silent field failure:
        T ran to 4471 K with chi2/nu 932 and no diagnostic)."""
        energies = np.linspace(1.0, 30.0, 200)
        t = np.asarray(nereids.forward_model(energies, [(u238_data, 8.0e-4)]))
        r = nereids.fit_spectrum_typed(
            transmission=t,
            uncertainty=np.full_like(t, 0.005),
            energies=energies,
            isotopes=[(u238_data, 5.0e-4)],
            solver="lm",
            fit_temperature=True,
            background=True,  # fit_anorm defaults True
            max_iter=5,
        )
        assert any("degenerate" in w for w in r.warnings), r.warnings

    def test_baseline_spatial_global_roundtrip(self, u238_data):
        energies = np.linspace(1.0, 30.0, 300)
        true_density = 2.0e-3
        t_1d = np.asarray(
            nereids.forward_model(energies, [(u238_data, true_density)])
        )
        curve, e_ref = _baseline_curve(energies)
        trans = np.tile((t_1d * curve)[:, None, None], (1, 2, 2))
        unc = np.full_like(trans, 0.005)
        result = nereids.spatial_map_typed(
            nereids.from_transmission(trans, unc),
            energies,
            [u238_data],
            initial_densities=[true_density],
            fix_densities=True,
            fit_temperature=True,  # keep >=1 free param per pixel
            temperature_k=300.0,
            baseline=True,
            max_iter=200,
        )
        assert result.baseline_global is not None
        for i, (fitted, truth) in enumerate(zip(result.baseline_global, _BL_TRUE)):
            assert abs(fitted - truth) < 1e-2, (
                f"baseline_global[{i}] = {fitted} vs truth {truth}"
            )
        assert result.baseline_e_ref_ev == pytest.approx(e_ref, rel=1e-12)
        assert result.baseline_maps is None, "global mode has no per-pixel maps"
        assert result.warnings == []
        assert result.n_converged == 4

    def test_baseline_spatial_per_pixel_maps(self, u238_data):
        energies = np.linspace(1.0, 30.0, 300)
        true_density = 2.0e-3
        t_1d = np.asarray(
            nereids.forward_model(energies, [(u238_data, true_density)])
        )
        curve, _ = _baseline_curve(energies)
        trans = np.tile((t_1d * curve)[:, None, None], (1, 2, 2))
        unc = np.full_like(trans, 0.005)
        result = nereids.spatial_map_typed(
            nereids.from_transmission(trans, unc),
            energies,
            [u238_data],
            initial_densities=[true_density],
            fix_densities=True,
            fit_temperature=True,
            temperature_k=300.0,
            baseline=True,
            baseline_global=False,
            max_iter=200,
        )
        assert result.baseline_global is None, "per-pixel mode has no global triple"
        maps = result.baseline_maps
        assert maps is not None and len(maps) == 3
        b0_map = np.asarray(maps[0])
        assert b0_map.shape == (2, 2)
        converged = np.asarray(result.converged_map)
        assert converged.any(), "at least one pixel must converge"
        assert np.all(
            np.abs(b0_map[converged] - _BL_TRUE[0]) < 5e-2
        ), f"per-pixel b0 off truth: {b0_map}"

def _create_synthetic_nxevent_bank(
    path, pulse_times_s, events_per_pulse_us, tof_units="microsecond",
    etz_units="second",
):
    """Create a facility-schema NXevent_data bank (issue #637).

    Mirrors the SNS layout: ``/entry/<bank>/{event_time_offset,
    event_index, event_time_zero}`` where ``event_index`` is the
    cumulative first-event index per pulse and ``event_time_zero`` is
    seconds since run start.  Also writes a matching ``pause`` DASlogs
    transition log and ``/entry/duration`` so the full
    read_run_log -> intervals_where -> load_nexus_bank_spectrum chain
    can be exercised on one file.
    """
    import h5py

    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        entry.create_dataset(
            "duration", data=np.array([len(pulse_times_s)], dtype=np.float32)
        )
        bank = entry.create_group("monitor1")
        index, tofs = [], []
        for evs in events_per_pulse_us:
            index.append(len(tofs))
            tofs.extend(evs)
        etz = bank.create_dataset(
            "event_time_zero", data=np.asarray(pulse_times_s, dtype=np.float64)
        )
        # fixed-length ASCII attrs, exactly like SNS/ADARA facility files
        if etz_units is not None:
            etz.attrs["units"] = np.bytes_(etz_units)
        etz.attrs["offset"] = np.bytes_("2026-06-22T19:01:07.183368667-04:00")
        bank.create_dataset("event_index", data=np.asarray(index, dtype=np.uint64))
        # f32 like real SNS files
        eto = bank.create_dataset(
            "event_time_offset", data=np.asarray(tofs, dtype=np.float32)
        )
        if tof_units is not None:
            eto.attrs["units"] = np.bytes_(tof_units)
        # pause transition log: paused (value 1) during the middle third
        n = len(pulse_times_s)
        logs = entry.create_group("DASlogs")
        pause = logs.create_group("pause")
        t = pause.create_dataset(
            "time", data=np.array([0.0, n / 3.0, 2.0 * n / 3.0])
        )
        t.attrs["start"] = np.bytes_("2026-06-22T19:01:07.183368667-04:00")
        # uint16, like the real SNS pause PV
        pause.create_dataset("value", data=np.array([0, 1, 0], dtype=np.uint16))


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
class TestRunLogAndBankSpectrum:
    """Beam-state filtering: DASlogs intervals + NXevent_data banks (#637)."""

    def test_read_run_log_and_step_semantics(self, tmp_path):
        path = str(tmp_path / "bank.h5")
        _create_synthetic_nxevent_bank(path, [0.0, 1.0, 2.0], [[100.0]] * 3)
        log = nereids.read_run_log(path, "pause")
        assert isinstance(log, nereids.RunLog)
        assert log.times.shape == (3,)
        assert log.n_dropped_corrupt == 0
        assert "Some(" not in repr(log)
        assert log.duration_s == pytest.approx(3.0)
        assert log.offset_iso.startswith("2026-06-22")
        # Step semantics: pause==0 on [0, 1) and [2, 3) (last value
        # persists to duration) — the middle third is paused.
        live = nereids.intervals_where(
            log.times, log.values, log.duration_s, max_value=0.5
        )
        assert len(live) == 2
        assert live[0] == (0.0, pytest.approx(1.0))
        assert live[1][0] == pytest.approx(2.0)
        assert live[1][1] >= 3.0  # final segment padded past f32 duration
        with pytest.raises((IOError, ValueError)):
            nereids.read_run_log(path, "no_such_pv")

    def test_corrupt_reconnect_record_dropped(self, tmp_path):
        # Mirror of VENUS run 19383 BL10:SE:ND1:CH1:PV — a reconnect
        # appends (time=0.0, value=denormal garbage) mid-log.
        import h5py

        path = str(tmp_path / "reconnect.h5")
        with h5py.File(path, "w") as f:
            entry = f.create_group("entry")
            entry.create_dataset("duration", data=np.array([2000.0], dtype=np.float32))
            g = entry.create_group("DASlogs/ch1")
            g.create_dataset(
                "time", data=np.array([0.0, 2.0, 1194.96, 1226.96, 0.0, 1228.99])
            )
            g.create_dataset(
                "value", data=np.array([6.9e-310, 27.7, 27.75, 27.79, 6.9e-310, 27.78])
            )
        log = nereids.read_run_log(path, "ch1")
        assert log.n_dropped_corrupt == 2
        assert log.times.shape == (4,)
        assert not np.any(log.values < 1.0)
        # Cleaned log feeds intervals_where without error.  The state on
        # [0, 2) was recorded only by the garbage record, so the interval
        # starts at the first clean entry; the end is the f32-ULP-padded
        # run end.
        iv = nereids.intervals_where(log.times, log.values, log.duration_s, min_value=27.0)
        assert len(iv) == 1
        assert iv[0][0] == pytest.approx(2.0)
        assert iv[0][1] >= 2000.0

    def test_intervals_where_entry_mean_trap(self):
        # The motivating case: entry-mean of the log reads 3/7 = 0.43
        # "paused" while the time-weighted pause fraction is ~0.90.
        times = [0.0, 1000.0, 20000.0, 21500.0, 42589.0, 44100.0, 44338.0]
        values = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        live = nereids.intervals_where(times, values, 44339.0, max_value=0.5)
        live_s = sum(b - a for a, b in live)
        assert np.mean(values) == pytest.approx(3.0 / 7.0)
        assert live_s / 44339.0 < 0.11
        with pytest.raises(ValueError):
            nereids.intervals_where(times, values, 44339.0, min_value=2.0, max_value=1.0)

    def test_intervals_intersect(self):
        keep = nereids.intervals_intersect(
            [(0.0, 10.0), (20.0, 30.0)], [(5.0, 25.0)]
        )
        assert keep == [(5.0, 10.0), (20.0, 25.0)]
        assert nereids.intervals_intersect([(0.0, 5.0)], [(5.0, 9.0)]) == []
        # Unsorted input is normalised, not silently corrupted.
        assert nereids.intervals_intersect(
            [(20.0, 30.0), (0.0, 10.0)], [(5.0, 25.0)]
        ) == [(5.0, 10.0), (20.0, 25.0)]
        with pytest.raises(ValueError):
            nereids.intervals_intersect([(5.0, 5.0)], [(0.0, 1.0)])

    def test_load_bank_spectrum_unfiltered_and_filtered(self, tmp_path):
        path = str(tmp_path / "bank.h5")
        # 6 pulses at t = 0..5 s; pulse p carries p events at TOF 500 µs.
        _create_synthetic_nxevent_bank(
            path, [float(t) for t in range(6)], [[500.0] * p for p in range(6)]
        )
        s = nereids.load_nexus_bank_spectrum(
            path, "monitor1", n_bins=2, tof_min_us=0.0, tof_max_us=1000.0
        )
        assert isinstance(s, nereids.BankSpectrum)
        assert s.pulses_total == 6 and s.pulses_kept == 6
        assert s.events_total == 15 and s.events_kept == 15
        assert list(s.counts) == [0, 15]
        assert s.tof_edges_us.shape == (3,)
        assert s.pulse_time_offset_iso.startswith("2026-06-22")
        # Filter to the live intervals derived from the pause log:
        # [0, 2) and [4, 6) keep pulses 0, 1, 4, 5 -> 0+1+4+5 = 10 events.
        log = nereids.read_run_log(path, "pause")
        live = nereids.intervals_where(
            log.times, log.values, log.duration_s, max_value=0.5
        )
        sf = nereids.load_nexus_bank_spectrum(
            path,
            "monitor1",
            n_bins=2,
            tof_min_us=0.0,
            tof_max_us=1000.0,
            keep_intervals=live,
        )
        assert sf.pulses_kept == 4
        assert sf.events_kept == 10
        assert list(sf.counts) == [0, 10]

    def test_empty_bank_grace(self, tmp_path):
        # The VENUS reality: pulses recorded, zero events (frame-mode tpx1).
        path = str(tmp_path / "empty.h5")
        _create_synthetic_nxevent_bank(path, [0.0, 1.0, 2.0], [[], [], []])
        s = nereids.load_nexus_bank_spectrum(
            path, "monitor1", n_bins=4, tof_min_us=0.0, tof_max_us=1000.0,
            keep_intervals=[(0.5, 2.5)],
        )
        assert s.pulses_total == 3 and s.pulses_kept == 2
        assert s.events_total == 0 and s.events_kept == 0
        assert list(s.counts) == [0, 0, 0, 0]

    def test_missing_units_is_an_error_on_both_event_datasets(self, tmp_path):
        # event_time_offset without units
        path = str(tmp_path / "nounits.h5")
        _create_synthetic_nxevent_bank(path, [0.0], [[250.0]], tof_units=None)
        with pytest.raises(ValueError, match="units"):
            nereids.load_nexus_bank_spectrum(
                path, "monitor1", n_bins=2, tof_min_us=0.0, tof_max_us=1000.0
            )
        # event_time_zero without units (same #554 policy, separate check)
        path2 = str(tmp_path / "nounits_etz.h5")
        _create_synthetic_nxevent_bank(path2, [0.0], [[250.0]], etz_units=None)
        with pytest.raises(ValueError, match="event_time_zero"):
            nereids.load_nexus_bank_spectrum(
                path2, "monitor1", n_bins=2, tof_min_us=0.0, tof_max_us=1000.0
            )

    def test_bad_keep_intervals_rejected(self, tmp_path):
        path = str(tmp_path / "bank.h5")
        _create_synthetic_nxevent_bank(path, [0.0], [[250.0]])
        for bad in [(5.0, 5.0), (5.0, 1.0), (float("nan"), 1.0)]:
            with pytest.raises(ValueError):
                nereids.load_nexus_bank_spectrum(
                    path, "monitor1", n_bins=2, tof_min_us=0.0,
                    tof_max_us=1000.0, keep_intervals=[bad],
                )
