"""Comprehensive pytest suite for the NEREIDS Python bindings.

All tests use synthetic data built with ``nereids.create_resonance_data()``
so no network access or ENDF downloads are required.
"""

import math
import os
import tempfile
import warnings

import numpy as np
import pytest

import nereids

from _fixtures import _make_single_resonance

_CACHE_FOLLOWS_HOME = pytest.mark.skipif(
    os.name == "nt", reason="the ENDF cache on Windows ignores HOME and XDG_CACHE_HOME"
)


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

    def test_create_resonance_data_rejects_empty_resonances(self):
        """An empty resonance list raises instead of building inert data.

        A resolved range with no resonances evaluates to zero cross-section
        everywhere (transmission = 1) — the same silently-inert shape the
        ENDF load path rejects with its no-evaluable-ranges error.
        """
        with pytest.raises(ValueError, match="no resonances"):
            nereids.create_resonance_data(
                z=92,
                a=238,
                awr=236.006,
                scattering_radius=9.48,
                resonances=[],
            )
        # The explicit-l_groups arm is guarded the same way.
        with pytest.raises(ValueError, match="no resonances"):
            nereids.create_resonance_data(
                z=92,
                a=238,
                awr=236.006,
                scattering_radius=9.48,
                resonances=[],
                l_groups=[(0, [])],
            )

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

    def test_pure_lrf7_file_rejected_no_evaluable_ranges(self):
        """A pure-LRF=7 evaluation is rejected at load, not silently zeroed.

        NEREIDS removed the LRF=7 cross-section physics (its closed-channel
        treatment was incomplete — the Coulomb/SHF=1 closed-channel shift was
        unimplemented — and it was never validated against SAMMY), so LRF=7
        ranges are parse-and-skip placeholders. A
        file whose ONLY range is LRF=7 would produce zero cross-sections
        everywhere (transmission = 1) — the parser now rejects it loudly
        instead of loading it with ``n_resonances == 0``.
        """
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fixture = os.path.join(
            root, "tests/data/synthetic/lrf7_krm3_resonance_column_order.endf"
        )
        # The fixture is committed to the repo, so a missing file is a
        # packaging/path regression that MUST fail — not a skip that would
        # silently disable this no-evaluable-content guard.
        assert os.path.exists(fixture), (
            f"vendored LRF=7 regression fixture must be present at {fixture} "
            "(committed test data)"
        )

        with pytest.raises(ValueError, match="No evaluable resonance ranges"):
            nereids.load_endf_file(fixture)

    def test_mixed_evaluation_loads_with_unevaluated_range_warning(self):
        """A mixed evaluation (resolved + URR) loads, warns, and is flagged.

        U-238 (ENDF/B-VIII.0, SAMMY ex027) carries a resolved Reich-Moore
        range plus an unresolved (LRU=2) range. The resolved range keeps the
        file loadable; the skipped URR range must surface as a ``UserWarning``
        at load time and as ``has_unevaluated_ranges == True``, because its
        span contributes zero cross-section.
        """
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fixture = os.path.join(
            root, "crates/nereids-endf/tests/data/u238_ex027.endf"
        )
        assert os.path.exists(fixture), (
            f"vendored U-238 fixture must be present at {fixture} "
            "(committed test data)"
        )

        with pytest.warns(UserWarning, match="parsed but NOT evaluated"):
            data = nereids.load_endf_file(fixture)
        assert data.has_unevaluated_ranges, (
            "the skipped URR range must be flagged as unevaluated"
        )
        # The resolved Reich-Moore range is fully usable. The vendored tape
        # is byte-pinned, so the count is exact: a lower bound would still
        # pass if the parser silently dropped most of the resolved range.
        # Update this number only when the fixture is deliberately replaced.
        assert data.n_resonances == 1653
        # Explicit alias must agree with n_resonances.
        assert data.total_resonance_count == data.n_resonances

    def test_mixed_lru0_getters_prefer_evaluable_range(self):
        """``target_spin`` / ``scattering_radius`` read the first EVALUABLE range.

        The mixed fixture's leading LRU=0 placeholder deliberately carries
        different SPI/AP (1.5 / 5.0 fm) than the resolved range (2.5 /
        9.6931 fm), so this test can see which range the getters read: a
        placeholder's SPI/AP describes a span the cross-section code never
        evaluates, and the getters must skip it.
        """
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fixture = os.path.join(
            root, "tests/data/synthetic/lru0_plus_resolved_mixed.endf"
        )
        assert os.path.exists(fixture), (
            f"vendored mixed LRU=0 fixture must be present at {fixture} "
            "(committed test data)"
        )
        with pytest.warns(UserWarning, match="parsed but NOT evaluated"):
            data = nereids.load_endf_file(fixture)
        assert data.target_spin == 2.5
        assert data.scattering_radius == pytest.approx(9.6931)

    @staticmethod
    def _fe56_cache_files(tmp_path, monkeypatch):
        """The Fe-56 ENDF/B-VIII.1 cache file under ``tmp_path`` as HOME, on macOS and on Linux."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        return [
            tmp_path / root / "nereids" / "endf" / "ENDF-B-VIII.1" / "Fe-56.endf"
            for root in ("Library/Caches", ".")
        ]

    @_CACHE_FOLLOWS_HOME
    def test_unreachable_download_raises_connection_error(self, tmp_path, monkeypatch):
        """A download through an unreachable proxy into an empty cache raises ``ConnectionError``."""
        self._fe56_cache_files(tmp_path, monkeypatch)
        for name in ("HTTPS_PROXY", "https_proxy"):
            monkeypatch.setenv(name, "http://127.0.0.1:9")
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.delenv(name, raising=False)
        with pytest.raises(ConnectionError):
            nereids.load_endf(26, 56)

    @_CACHE_FOLLOWS_HOME
    def test_unreadable_cache_file_raises_os_error_naming_it(self, tmp_path, monkeypatch):
        """A cache file that cannot be read raises an ``OSError`` naming it, not ``ConnectionError``."""
        for cached in self._fe56_cache_files(tmp_path, monkeypatch):
            cached.mkdir(parents=True)
        with pytest.raises(OSError, match="Fe-56.endf") as raised:
            nereids.load_endf(26, 56)
        assert not isinstance(raised.value, ConnectionError)

    @_CACHE_FOLLOWS_HOME
    def test_group_refuses_cached_file_for_another_isotope(self, tmp_path, monkeypatch):
        """A group member whose cached file holds another isotope raises ``ValueError``."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "crates/nereids-endf/tests/data/u238_ex027.endf"), "rb") as f:
            u238 = f.read()
        for cached in self._fe56_cache_files(tmp_path, monkeypatch):
            cached.parent.mkdir(parents=True)
            cached.write_bytes(u238)
        group = nereids.IsotopeGroup.custom("fe56", [(26, 56, 1.0)])
        with pytest.raises(ValueError, match="requested Z=26 A=56 but file contains Z=92 A=238"):
            group.load_endf()


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


class TestTabulatedDetectorBins:
    """The public tabulated-response API integrates the loaded UDR directly
    into measured detector-time bins."""

    TOF_FACTOR = 72.298254398292800
    L = 25.0

    @staticmethod
    def _write_kernel(path):
        path.write_text(
            "\n".join(
                [
                    "synthetic asymmetric triangular response",
                    "-----",
                    "   2.50000e+001   0.00000e+000",
                    "-1.0 0.0",
                    "0.0 1.0",
                    "2.0 0.0",
                    "",
                ]
            )
        )

    def test_exact_bin_probabilities_without_window_renormalization(self, tmp_path):
        kernel_path = tmp_path / "synthetic_exact_bins.txt"
        self._write_kernel(kernel_path)
        response = nereids.load_resolution(str(kernel_path), self.L)
        arrival = self.TOF_FACTOR * self.L / np.sqrt(25.0)

        probabilities = np.asarray(
            response.detector_bin_probabilities(
                25.0,
                [arrival - 2.0, arrival - 1.0, arrival, arrival + 1.0],
                0.0,
            )
        )

        np.testing.assert_allclose(probabilities, [0.0, 1.0 / 3.0, 1.0 / 2.0])
        assert probabilities.sum() == pytest.approx(5.0 / 6.0)

    def test_invalid_flight_path_is_rejected_when_loading(self, tmp_path):
        kernel_path = tmp_path / "synthetic_invalid_path.txt"
        self._write_kernel(kernel_path)

        with pytest.raises(ValueError, match="flight_path"):
            nereids.load_resolution(str(kernel_path), float("nan"))




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




# ===========================================================================
# Spatial mapping (Poisson)
# ===========================================================================






# ===========================================================================
# Counts-only fit API
# ===========================================================================


class TestCountsApi:
    """Construction and input validation of the counts-only fit API."""

    @staticmethod
    def _calibration():
        return nereids.Calibration(
            t0_us=nereids.Value.known(3.0),
            flight_path_m=nereids.Value.known(25.0),
            pulse=nereids.Pulse(
                alpha0=nereids.Value.known(0.35),
                alpha1=nereids.Value.known(0.05),
                beta0=nereids.Value.known(0.25),
                beta1=nereids.Value.known(0.0),
                r=nereids.Value.known(0.15),
                fwhm_squared_us2=nereids.Value.known(0.0),
                energy_span_ev=(1.0, 200.0),
            ),
        )

    def test_value_kinds(self):
        known = nereids.Value.known(2.0)
        assert known.kind == "known" and known.value == 2.0
        assert nereids.Value.fitted(1.0).kind == "fitted"
        measured = nereids.Value.measured(1.0, 0.1)
        assert measured.kind == "measured" and measured.sd == 0.1
        within = nereids.Value.within(1.0, 0.5, 1.5)
        assert (within.kind, within.start, within.lower, within.upper) == (
            "within",
            1.0,
            0.5,
            1.5,
        )

    def test_no_time_background(self):
        background = nereids.NO_TIME_BACKGROUND
        assert len(background) == 3
        assert [value.value for value in background] == [0.0, 0.0, 1.0]

    def test_measurement_round_trip(self, u238_data):
        material = nereids.Material(
            isotopes=[(u238_data, nereids.Value.fitted(0.001))],
            temperature_k=nereids.Value.known(293.6),
        )
        region = nereids.Region(
            open_counts=[100.0] * 4,
            sample_counts=[80.0] * 4,
            pixels=1.0,
            background=[nereids.Value.known(0.0)] * 3,
            gamma_transmission=nereids.Value.known(1.0),
            material=material,
        )
        measurement = nereids.Measurement(
            time_edges_us=[1.0, 2.0, 3.0, 4.0, 5.0],
            charge_ratio=1.0,
            normalization=nereids.Value.fitted(1.0),
            regions=[region],
        )
        assert measurement.time_edges_us == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert measurement.charge_ratio == 1.0
        assert measurement.regions[0].pixels == 1.0
        assert measurement.regions[0].material.isotopes[0][0].z == u238_data.z
        assert measurement.room_background is None
        room = nereids.RoomBackground(nereids.Value.fitted(0.5), (3.0e4, 2.5e4))
        measurement = nereids.Measurement(
            time_edges_us=[1.0, 2.0, 3.0, 4.0, 5.0],
            charge_ratio=1.0,
            normalization=nereids.Value.fitted(1.0),
            regions=[region],
            room_background=room,
        )
        assert measurement.room_background.rate.value == 0.5
        assert measurement.room_background.frames == (3.0e4, 2.5e4)
        assert material.thickness_variance.value == 0.0
        uneven = nereids.Material(
            isotopes=[(u238_data, nereids.Value.fitted(0.001))],
            temperature_k=nereids.Value.known(293.6),
            thickness_variance=nereids.Value.fitted(0.05),
        )
        assert uneven.thickness_variance.value == 0.05
        assert region.capture_gammas.value == 0.0
        assert region.in_scatter.value == 0.0
        assert measurement.spread is None
        spread = nereids.Spread(
            nereids.Value.fitted(0.1), [(1.0, 2.5)], (4, 4), [[(0, 0), (1, 1)]]
        )
        spread_measurement = nereids.Measurement(
            time_edges_us=[1.0, 2.0, 3.0, 4.0, 5.0],
            charge_ratio=1.0,
            normalization=nereids.Value.fitted(1.0),
            regions=[region],
            spread=spread,
        )
        assert spread_measurement.spread.share.value == 0.1
        assert spread_measurement.spread.kernel == [(1.0, 2.5)]

    def test_fit_counts_requires_a_material(self):
        measurement = nereids.Measurement(
            time_edges_us=[1.0, 2.0, 3.0],
            charge_ratio=1.0,
            normalization=nereids.Value.known(1.0),
            regions=[
                nereids.Region(
                    open_counts=[10.0, 10.0],
                    sample_counts=[5.0, 5.0],
                    pixels=1.0,
                    background=[nereids.Value.known(0.0)] * 3,
                    gamma_transmission=nereids.Value.known(1.0),
                )
            ],
        )
        with pytest.raises(ValueError):
            nereids.fit_counts(measurement, self._calibration())

    def test_map_measurement_patches(self, u238_data):
        material = nereids.Material(
            isotopes=[(u238_data, nereids.Value.fitted(0.001))],
            temperature_k=nereids.Value.known(293.6),
        )
        counts = np.ones((3, 2, 2), dtype=np.float64)
        map_measurement = nereids.MapMeasurement(
            time_edges_us=[1.0, 2.0, 3.0, 4.0],
            open_counts=counts,
            sample_counts=counts,
            excluded=np.zeros((2, 2), dtype=bool),
            sample=np.ones((2, 2), dtype=bool),
            empty=np.zeros((2, 2), dtype=bool),
            material=material,
            charge_ratio=1.0,
            normalization=nereids.Value.fitted(1.0),
            background=[nereids.Value.known(0.0)] * 3,
            binning=1,
        )
        patches = map_measurement.patches()
        assert len(patches) == 2 and len(patches[0]) == 2
        assert all(patch.kind == "sample" for row in patches for patch in row)
        assert map_measurement.binning == 1

    def test_default_empty_background_predicts_equal_empty_run_counts(self, u238_data):
        counts = np.full((12, 1, 2), 100.0)
        measurement = nereids.MapMeasurement(
            time_edges_us=np.linspace(350.0, 440.0, 13),
            open_counts=counts,
            sample_counts=counts,
            excluded=np.zeros((1, 2), dtype=bool),
            sample=np.array([[True, False]]),
            empty=np.array([[False, True]]),
            material=nereids.Material(
                isotopes=[(u238_data, nereids.Value.known(0.0))],
                temperature_k=nereids.Value.known(300.0),
            ),
            charge_ratio=1.0,
            normalization=nereids.Value.known(1.0),
            background=[nereids.Value.known(0.0)] * 3,
        )
        result = nereids.fit_map(measurement, self._calibration())
        assert result.empty is not None
        np.testing.assert_allclose(
            result.empty.predicted_sample, result.empty.predicted_open,
            rtol=1e-12, atol=1e-10,
        )




# ===========================================================================
# Normalization
# ===========================================================================




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

    @staticmethod
    def _write_chunked_folder(tmpdir, n_frames=4, h=2, w=3):
        """Write a 2-chunk VENUS-style folder (run_764_* / run_765_*).

        Frame f of chunk c holds the constant value ``(c - 763) * 100 + f``
        so per-element sums are easy to assert.  Returns the expected
        summed stack.
        """
        import tifffile

        expected = np.zeros((n_frames, h, w))
        for c in (764, 765):
            for f in range(n_frames):
                value = (c - 763) * 100 + f
                frame = np.full((h, w), value, dtype=np.uint16)
                tifffile.imwrite(
                    os.path.join(tmpdir, f"run_{c}_{f:04d}.tif"), frame
                )
                expected[f] += value
        return expected

    def test_load_tiff_folder_chunked_sums(self):
        """T40: a 2-chunk folder sums element-wise by default; opting out
        loads the 2x lexicographic concatenation."""
        pytest.importorskip("tifffile")
        n_frames, h, w = 4, 2, 3

        with tempfile.TemporaryDirectory() as tmpdir:
            expected = self._write_chunked_folder(tmpdir, n_frames, h, w)

            summed = np.asarray(nereids.load_tiff_folder(tmpdir))
            assert summed.shape == (n_frames, h, w)
            np.testing.assert_allclose(summed, expected)

            legacy = np.asarray(nereids.load_tiff_folder(tmpdir, sum_chunks=False))
            assert legacy.shape == (2 * n_frames, h, w)
            # Lexicographic order: all of chunk 764, then all of chunk 765.
            assert legacy[0, 0, 0] == 100.0
            assert legacy[n_frames, 0, 0] == 200.0

    def test_pixel_policy_negative_int16(self):
        """T41: a negative signed pixel rejects by default (naming
        detect_bad_pixels), clips to 0 under "clip", passes under "allow"."""
        tifffile = pytest.importorskip("tifffile")
        frame = np.array([[10, -32554], [30, 40]], dtype=np.int16)

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = f.name
        try:
            tifffile.imwrite(path, frame)

            with pytest.raises(ValueError, match="detect_bad_pixels"):
                nereids.load_tiff_stack(path)

            clipped = np.asarray(nereids.load_tiff_stack(path, pixel_policy="clip"))
            assert clipped[0, 0, 1] == 0.0
            assert clipped[0, 0, 0] == 10.0

            allowed = np.asarray(nereids.load_tiff_stack(path, pixel_policy="allow"))
            assert allowed[0, 0, 1] == -32554.0
        finally:
            os.unlink(path)

    def test_pixel_policy_nan_float32(self):
        """T42: a NaN float pixel rejects by default and passes under
        "allow"."""
        tifffile = pytest.importorskip("tifffile")
        frame = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = f.name
        try:
            tifffile.imwrite(path, frame)

            with pytest.raises(ValueError):
                nereids.load_tiff_stack(path)

            allowed = np.asarray(nereids.load_tiff_stack(path, pixel_policy="allow"))
            assert np.isnan(allowed[0, 0, 1])
        finally:
            os.unlink(path)

    def test_venus_run_folder_three_calls(self):
        """T43 (acceptance): a 2-chunk VENUS run folder with a
        *_Spectra.txt sidecar loads to stack + energy axis in 3 calls."""
        pytest.importorskip("tifffile")
        n_frames, h, w = 4, 2, 3

        with tempfile.TemporaryDirectory() as tmpdir:
            expected = self._write_chunked_folder(tmpdir, n_frames, h, w)
            # Evidence-shaped sidecar mirroring measured IPTS-37432 VENUS
            # autoreduce output (synthetic counts): header
            # "shutter_time,counts", first start time 1.12e-6 s (7 x the
            # 160 ns bin -- NOT zero; autoreduce drops pre-trigger bins),
            # uniform 1.6e-7 s steps.
            sidecar = os.path.join(tmpdir, "run_Spectra.txt")
            with open(sidecar, "w") as fh:
                fh.write("shutter_time,counts\n")
                for f in range(n_frames):
                    fh.write(f"{1.12e-6 + f * 1.6e-7:.10e},{100 + f}\n")

            # Call 1: chunk-aware stack (the sidecar .txt is ignored by
            # the TIFF extension filter).
            counts = np.asarray(nereids.load_tiff_folder(tmpdir))
            # Call 2: sidecar -> N+1 ascending µs edges, validated.
            edges_us = np.asarray(
                nereids.read_tof_sidecar(sidecar, n_frames=counts.shape[0])
            )
            # Call 3: energy centers.
            energies = np.asarray(
                nereids.tof_to_energy_centers(edges_us, 25.0)
            )

            assert counts.shape == (n_frames, h, w)
            np.testing.assert_allclose(counts, expected)
            assert edges_us.shape == (n_frames + 1,)
            # 1.12 us first edge, 0.16 us steps (seconds -> us conversion).
            np.testing.assert_allclose(edges_us[0], 1.12, rtol=1e-12)
            np.testing.assert_allclose(np.diff(edges_us), 0.16, rtol=1e-9)
            assert np.all(np.diff(edges_us) > 0)
            assert energies.shape == (n_frames,)
            assert np.all(np.diff(energies) > 0)

    def test_read_tof_sidecar_errors(self):
        """T44: frame-count mismatch raises ValueError; a missing sidecar
        raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar = os.path.join(tmpdir, "run_Spectra.txt")
            with open(sidecar, "w") as fh:
                fh.write("0.001,10\n0.002,20\n0.003,30\n")

            with pytest.raises(ValueError):
                nereids.read_tof_sidecar(sidecar, n_frames=7)

            with pytest.raises(FileNotFoundError):
                nereids.read_tof_sidecar(
                    os.path.join(tmpdir, "missing_Spectra.txt")
                )

    def test_invalid_pixel_policy_string(self):
        """T45: an invalid pixel_policy lists the accepted options."""
        tifffile = pytest.importorskip("tifffile")
        frame = np.ones((2, 2), dtype=np.uint16)

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = f.name
        try:
            tifffile.imwrite(path, frame)
            with pytest.raises(ValueError, match="reject"):
                nereids.load_tiff_stack(path, pixel_policy="bogus")
            with pytest.raises(ValueError, match="allow"):
                nereids.load_tiff_folder(
                    os.path.dirname(path), pixel_policy="bogus"
                )
        finally:
            os.unlink(path)

    def test_chunk_sum_warns_and_return_info(self):
        """T46: summing chunks emits a UserWarning naming the chunk count,
        ids, and the sum_chunks=False escape hatch; return_info=True
        returns (array, info) with the full provenance dict."""
        pytest.importorskip("tifffile")
        n_frames, h, w = 4, 2, 3

        with tempfile.TemporaryDirectory() as tmpdir:
            expected = self._write_chunked_folder(tmpdir, n_frames, h, w)

            with pytest.warns(
                UserWarning,
                match=r"summed 2 DAQ chunks \(764, 765\).*sum_chunks=False",
            ):
                arr, info = nereids.load_tiff_folder(tmpdir, return_info=True)

            np.testing.assert_allclose(np.asarray(arr), expected)
            assert info == {
                "n_files": 2 * n_frames,
                "n_chunks": 2,
                "chunk_ids": [764, 765],
                "chunks_summed": True,
                "n_clipped_pixels": 0,
                "n_unrecognized_files": 0,
                "unrecognized_examples": [],
                "chunk_inconsistent": False,
            }

    def test_clip_warns(self):
        """T47: pixel_policy="clip" emits a UserWarning counting the
        clipped pixels."""
        tifffile = pytest.importorskip("tifffile")
        frame = np.array([[10, -5], [-3, 40]], dtype=np.int16)

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = f.name
        try:
            tifffile.imwrite(path, frame)
            with pytest.warns(UserWarning, match=r"2 negative pixel\(s\) clipped"):
                clipped = np.asarray(
                    nereids.load_tiff_stack(path, pixel_policy="clip")
                )
            assert clipped[0, 0, 1] == 0.0
            assert clipped[0, 1, 0] == 0.0
        finally:
            os.unlink(path)

    def test_plain_folder_no_warning(self):
        """T48: a plain (non-chunked) clean folder loads with no warning;
        return_info reports the un-chunked layout."""
        tifffile = pytest.importorskip("tifffile")
        n_frames, h, w = 3, 2, 2

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(n_frames):
                tifffile.imwrite(
                    os.path.join(tmpdir, f"frame_{i:04d}.tif"),
                    np.ones((h, w), dtype=np.uint16),
                )
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                arr, info = nereids.load_tiff_folder(tmpdir, return_info=True)
            assert np.asarray(arr).shape == (n_frames, h, w)
            assert info == {
                "n_files": n_frames,
                "n_chunks": 0,
                "chunk_ids": [],
                "chunks_summed": False,
                "n_clipped_pixels": 0,
                "n_unrecognized_files": 0,
                "unrecognized_examples": [],
                "chunk_inconsistent": False,
            }

    def test_warning_attributed_to_caller(self):
        """T50: TIFF-load warnings are attributed to this file (the
        Python call site).  pyo3 functions execute in the caller's frame,
        so stacklevel=1 already points there; stacklevel=2 would blame
        the caller's caller (pytest internals in this suite).  pytest.warns
        checks only category/message, so assert the filename explicitly."""
        pytest.importorskip("tifffile")
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_chunked_folder(tmpdir)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                nereids.load_tiff_folder(tmpdir)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert w[0].filename == __file__

    def test_missing_tiff_raises_filenotfound(self):
        """T51: a nonexistent path raises FileNotFoundError from
        load_tiff_stack (aligned with load_tiff_folder and
        read_tof_sidecar).  Only a genuine NotFound maps there; other
        open failures (e.g. permission denied) stay plain OSError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = os.path.join(tmpdir, "no_such.tif")
            with pytest.raises(FileNotFoundError):
                nereids.load_tiff_stack(missing)

    def test_return_info_is_keyword_only(self):
        """T52: return_info is keyword-only at runtime, matching the
        .pyi overloads; passing it positionally raises TypeError."""
        tifffile = pytest.importorskip("tifffile")
        with tempfile.TemporaryDirectory() as tmpdir:
            tifffile.imwrite(
                os.path.join(tmpdir, "frame_0000.tif"),
                np.ones((2, 2), dtype=np.uint16),
            )
            arr, info = nereids.load_tiff_folder(
                tmpdir, None, True, "reject", return_info=True
            )
            assert np.asarray(arr).shape == (1, 2, 2)
            assert info["n_files"] == 1
            with pytest.raises(TypeError):
                nereids.load_tiff_folder(tmpdir, None, True, "reject", True)

    def test_mixed_folder_warns_and_counts_unrecognized(self):
        """T53: a mixed folder (chunk-patterned files plus a stray TIFF)
        falls back to legacy lexicographic loading with a UserWarning
        counting and naming the non-conforming files; return_info reports
        the same via n_unrecognized_files / unrecognized_examples."""
        tifffile = pytest.importorskip("tifffile")
        n_frames, h, w = 4, 2, 3

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_chunked_folder(tmpdir, n_frames, h, w)
            tifffile.imwrite(
                os.path.join(tmpdir, "overview.tif"),
                np.full((h, w), 7, dtype=np.uint16),
            )

            with pytest.warns(
                UserWarning,
                match=(
                    r"1 file\(s\) did not match the chunk naming pattern "
                    r"\(e\.g\. overview\.tif\).*chunk detection disabled.*"
                    r"lexicographic order.*pattern="
                ),
            ):
                arr, info = nereids.load_tiff_folder(tmpdir, return_info=True)

            # Legacy concatenation of all files — never a k-fold chunk sum.
            assert np.asarray(arr).shape == (2 * n_frames + 1, h, w)
            assert info["n_chunks"] == 0
            assert info["chunks_summed"] is False
            assert info["n_unrecognized_files"] == 1
            assert info["unrecognized_examples"] == ["overview.tif"]

    def test_ragged_chunks_opt_out_concatenates(self):
        """T55: a ragged chunked folder (chunk 764 has 3 frames, chunk 765
        has 2) is a ValueError on the default summing path — summing ragged
        chunks would corrupt counts — but with sum_chunks=False it honors
        the opt-out contract: it loads as the lexicographic concatenation of
        all files (frame count = the sum of all files), emits a UserWarning
        naming the inconsistency, and return_info sets
        chunk_inconsistent=True instead of raising."""
        tifffile = pytest.importorskip("tifffile")
        h, w = 2, 3

        with tempfile.TemporaryDirectory() as tmpdir:
            for f in range(3):
                tifffile.imwrite(
                    os.path.join(tmpdir, f"run_764_{f:04d}.tif"),
                    np.full((h, w), 100 + f, dtype=np.uint16),
                )
            for f in range(2):
                tifffile.imwrite(
                    os.path.join(tmpdir, f"run_765_{f:04d}.tif"),
                    np.full((h, w), 200 + f, dtype=np.uint16),
                )

            # Default summing path: still a hard ValueError (regression guard).
            with pytest.raises(ValueError):
                nereids.load_tiff_folder(tmpdir)

            # Opt-out: concatenation of all 5 files + inconsistency signal,
            # never a ValueError.
            with pytest.warns(
                UserWarning,
                match=r"DAQ chunks are internally inconsistent.*sum_chunks=False",
            ):
                arr, info = nereids.load_tiff_folder(
                    tmpdir, sum_chunks=False, return_info=True
                )

            assert np.asarray(arr).shape == (5, h, w)
            assert info["chunk_inconsistent"] is True
            assert info["chunks_summed"] is False
            assert info["n_chunks"] == 2
            assert info["chunk_ids"] == [764, 765]

    def test_missing_folder_vs_not_a_directory(self):
        """T54: load_tiff_folder raises FileNotFoundError for a
        nonexistent path (matching the docstring) and NotADirectoryError
        only for a path that exists but is a file."""
        tifffile = pytest.importorskip("tifffile")
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                nereids.load_tiff_folder(os.path.join(tmpdir, "no_such_dir"))

            file_path = os.path.join(tmpdir, "frame_0000.tif")
            tifffile.imwrite(file_path, np.ones((2, 2), dtype=np.uint16))
            with pytest.raises(NotADirectoryError):
                nereids.load_tiff_folder(file_path)


# ===========================================================================
# Error handling / validation
# ===========================================================================


class TestErrorHandling:
    """Tests for input validation and error messages."""



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


    def test_run_health_daslogs(self, tmp_path):
        """T49: run_health reads DASlogs pause/power via last-value-held
        time-weighted integration; absent PVs yield None."""
        import h5py

        path = str(tmp_path / "run.h5")
        with h5py.File(path, "w") as f:
            entry = f.create_group("entry")
            entry.create_dataset("duration", data=100.0)
            das = entry.create_group("DASlogs")
            pause = das.create_group("pause")
            pause.create_dataset("time", data=np.array([0.0, 10.0, 20.0]))
            pause.create_dataset("value", data=np.array([0.0, 1.0, 0.0]))
            power = das.create_group("proton_charge")
            power.create_dataset(
                "time", data=np.array([0.0, 10.0, 20.0, 30.0])
            )
            power.create_dataset(
                "value", data=np.array([10.0, 1.0, 10.0, 10.0])
            )

        health = nereids.run_health(path)
        assert isinstance(health, nereids.RunHealth)
        # Paused (value 1) over [10, 20) of a 100 s run.
        assert health.pause_fraction == pytest.approx(0.1)
        # Power 1 < 0.5 * median(10) over [10, 20) of 100 s.
        assert health.median_power == pytest.approx(10.0)
        assert health.beam_dip_fraction == pytest.approx(0.1)
        assert health.duration_s == pytest.approx(100.0)
        assert health.n_pause_entries == 3
        assert health.n_power_entries == 4

        # Custom PV name that is absent -> None fields, not an error.
        other = nereids.run_health(path, pause_pv="no_such_pv")
        assert other.pause_fraction is None
        assert other.n_pause_entries == 0
        assert other.beam_dip_fraction is not None


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




# ===========================================================================
# fit_energy_range Python parameter (#514)
# ===========================================================================






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





# ===========================================================================
# Type-stub conformance
# ===========================================================================


class TestStubConformance:
    """Sanity checks that the runtime PyO3 export surface is reflected in
    ``bindings/python/python/nereids/__init__.pyi``.

    """


    def test_runtime_exports_present_in_stub(self):
        """Every public (non-underscore) attribute on the compiled
        ``nereids`` module must be either defined in ``__init__.pyi`` OR
        documented as an intentional omission below.

        This catches the inverse of the python-api.md drift check: a
        PyO3 export that was never added to the stub at all.  Failure
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
        intentional_omissions: set[str] = {"NO_TIME_BACKGROUND"}

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
# Ikeda-Carpenter causal response
# ===========================================================================




class TestIkedaCarpenterCausalResponse:
    def test_source_pulse_keeps_physical_delay_origin(self):
        ic = nereids.IkedaCarpenter(
            flight_path_m=25.0,
            e_min_ev=1.0,
            e_max_ev=10.0,
            alpha=nereids.EnergyLaw.const(2.0),
            beta=0.5,
            r=nereids.EnergyLaw.const(0.3),
        )

        delay_us, density = map(np.asarray, ic.source_pulse_at(4.0))

        assert delay_us[0] == 0.0
        assert delay_us[np.argmax(density)] > 0.0
        assert np.all(np.diff(delay_us) > 0.0)

    def test_detector_bin_probabilities_preserve_probability_and_clock(self):
        flight_path_m = 25.0
        energy_ev = 4.0
        arrival_us = nereids.energy_to_tof(energy_ev, flight_path_m)
        ic = nereids.IkedaCarpenter(
            flight_path_m=flight_path_m,
            e_min_ev=1.0,
            e_max_ev=10.0,
            alpha=nereids.EnergyLaw.const(2.0),
            beta=0.5,
            r=nereids.EnergyLaw.const(0.3),
        )

        probabilities = np.asarray(
            ic.detector_bin_probabilities(
                energy_ev,
                [arrival_us - 1.0, arrival_us, arrival_us + 1.0, arrival_us + 80.0],
                0.0,
            )
        )

        assert probabilities.shape == (3,)
        assert probabilities[0] == 0.0
        assert np.all(probabilities[1:] > 0.0)
        assert 0.999 < probabilities.sum() <= 1.0

    def test_partial_window_reports_known_missing_mass(self):
        # A deliberately partial window whose true mass is the analytic
        # Gamma(3, alpha) CDF at the cut: a binding that silently renormalized
        # the window would report 1.0 instead.
        flight_path_m = 25.0
        energy_ev = 4.0
        alpha = 2.0
        arrival_us = nereids.energy_to_tof(energy_ev, flight_path_m)
        ic = nereids.IkedaCarpenter(
            flight_path_m=flight_path_m,
            e_min_ev=1.0,
            e_max_ev=10.0,
            alpha=nereids.EnergyLaw.const(alpha),
            beta=0.5,
            r=nereids.EnergyLaw.const(0.0),
        )

        probabilities = np.asarray(
            ic.detector_bin_probabilities(
                energy_ev, [arrival_us, arrival_us + 1.0], 0.0
            )
        )

        x = alpha * 1.0
        gamma3_at_cut = 1.0 - math.exp(-x) * (1.0 + x + 0.5 * x * x)
        assert probabilities.shape == (1,)
        assert probabilities[0] == pytest.approx(gamma3_at_cut, abs=1e-9)
        assert probabilities[0] < 1.0

    def test_energy_dependent_beta_law_scales_storage_tail(self):
        # With alpha and beta both proportional to sqrt(E), the pulse is a
        # pure time rescaling: P(delay <= 2 us at 1 eV) equals
        # P(delay <= 1 us at 4 eV). A binding that ignored beta_law (or
        # evaluated it at a fixed reference energy) breaks the equality.
        flight_path_m = 25.0
        ic = nereids.IkedaCarpenter(
            flight_path_m=flight_path_m,
            e_min_ev=0.5,
            e_max_ev=10.0,
            alpha=nereids.EnergyLaw.sqrt_e(2.0, 0.0),
            beta=999.0,  # overridden by beta_law
            r=nereids.EnergyLaw.const(0.4),
            beta_law=nereids.EnergyLaw.sqrt_e(0.5, 0.0),
        )

        def mass_within(energy_ev, delay_us):
            arrival = nereids.energy_to_tof(energy_ev, flight_path_m)
            (p,) = ic.detector_bin_probabilities(
                energy_ev, [arrival, arrival + delay_us], 0.0
            )
            return p

        slow = mass_within(1.0, 2.0)
        fast = mass_within(4.0, 1.0)
        assert 0.0 < slow < 1.0
        assert slow == pytest.approx(fast, abs=1e-9)







# ===========================================================================
# Bounded multiplicative baseline (#635)
# ===========================================================================


# Truth baseline shared by the closed loops below: a few % off unity,
# curved, strictly positive on the test grids, inside the DEFAULT bounds.
_BL_TRUE = (1.02, -0.03, 0.01)





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


class TestResolutionWidthConvention:
    """The width convention at the Python surface.

    ``delta_t_us`` / ``delta_l_m`` are W-parameters (the width in
    ``exp(-x^2/W^2)``), not standard deviations. The docstrings said 1-sigma
    for a long time, which invited a kernel sqrt(2) too narrow; a too-narrow
    resolution is absorbed into a fitted temperature that is too high. These
    tests pin the convention at the boundary where a user supplies the number.
    """

    # Grid half-span in units of the nominal width; beyond 6 the remaining
    # Gaussian mass is below 1e-16, so 12 is generous.
    SPAN_IN_WIDTHS = 12
    # Grid points per nominal width. The trapezoidal second-moment error and
    # the one-bin impulse's own variance (step^2/12) both scale as (step/W)^2;
    # at 50 they are ~1e-4 of the measured sigma, two orders below the
    # tolerances asserted below, while convolution cost scales as the square of
    # the point count.
    SAMPLES_PER_WIDTH = 50

    @classmethod
    def _measured_sigma_ev(cls, flight_path_m, delta_t_us, delta_l_m, center_ev, nominal_width_ev):
        """Read the kernel's standard deviation off the broadened impulse.

        ``nominal_width_ev`` only sizes the grid; the measurement never uses
        it, so the result cannot agree with the library by construction.
        """
        n = 2 * cls.SPAN_IN_WIDTHS * cls.SAMPLES_PER_WIDTH + 1
        half_span_ev = cls.SPAN_IN_WIDTHS * nominal_width_ev
        energies = np.linspace(center_ev - half_span_ev, center_ev + half_span_ev, n)
        step = float(energies[1] - energies[0])
        impulse = np.zeros(n)
        impulse[(n - 1) // 2] = 1.0 / step

        kernel = nereids.resolution_broaden(
            energies, impulse, flight_path_m, delta_t_us, delta_l_m
        )
        mass = float(kernel.sum() * step)
        assert abs(mass - 1.0) < 1e-3, f"kernel not normalised on this span: {mass}"
        mean = float((kernel * energies).sum() * step / mass)
        var = float((kernel * (energies - mean) ** 2).sum() * step / mass)
        return math.sqrt(var)

    def test_width_from_sigma_produces_the_requested_standard_deviation(self):
        """A measured 1-sigma, converted, gives a kernel with that sigma."""
        flight_path_m = 25.0
        sigma_t_us = 1.0
        w_t_us = nereids.width_from_sigma(sigma_t_us)
        assert w_t_us == pytest.approx(sigma_t_us * math.sqrt(2.0), rel=1e-15)

        center = 10.0
        # sigma_E = 2 * sigma_t * E^(3/2) / (TOF_FACTOR * L), ordinary propagation
        # through E = (1/2) m (L/t)^2.
        tof_factor = 72.2982543982928
        expected = 2.0 * sigma_t_us * center**1.5 / (tof_factor * flight_path_m)

        measured = self._measured_sigma_ev(flight_path_m, w_t_us, 0.0, center, expected)
        assert measured == pytest.approx(expected, rel=3e-3), (
            f"converted width gives sigma {measured!r}, asked for {expected!r}"
        )

    def test_passing_sigma_unconverted_is_narrower_by_root_two(self):
        """The error the old docstring invited, measured rather than asserted."""
        flight_path_m = 25.0
        sigma_t_us = 1.0
        center = 10.0
        tof_factor = 72.2982543982928
        # Both grids are sized from the same nominal width, so the two
        # measurements are made on identical abscissae and the ratio below is
        # a property of the kernels, not of the sampling.
        nominal = 2.0 * sigma_t_us * center**1.5 / (tof_factor * flight_path_m)

        converted = self._measured_sigma_ev(
            flight_path_m, nereids.width_from_sigma(sigma_t_us), 0.0, center, nominal
        )
        unconverted = self._measured_sigma_ev(
            flight_path_m, sigma_t_us, 0.0, center, nominal
        )
        assert converted / unconverted == pytest.approx(math.sqrt(2.0), rel=5e-3), (
            "unconverted 1-sigma input is not exactly sqrt(2) too narrow: "
            f"{converted!r} vs {unconverted!r}"
        )

    def test_width_from_fwhm_matches_the_sammy_deltag_conversion(self):
        """SAMMY's Deltag is a FWHM; the same divisor must apply here."""
        delta_g = 0.022  # tr007's BROADENING card timing value
        assert nereids.width_from_fwhm(delta_g) == pytest.approx(
            delta_g / (2.0 * math.sqrt(math.log(2.0))), rel=1e-15
        )
