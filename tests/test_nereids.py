"""Comprehensive pytest suite for the NEREIDS Python bindings.

All tests use synthetic data built with ``nereids.create_resonance_data()``
so no network access or ENDF downloads are required.
"""

import os
import tempfile

import numpy as np
import pytest

import nereids


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_single_resonance(
    z=92,
    a=238,
    awr=236.006,
    scattering_radius=9.48,
    energy=6.67,
    j=0.5,
    gn=0.0015,
    gg=0.023,
    target_spin=0.0,
    formalism=None,
):
    """Build a minimal single-resonance isotope for testing."""
    return nereids.create_resonance_data(
        z=z,
        a=a,
        awr=awr,
        scattering_radius=scattering_radius,
        resonances=[(energy, j, gn, gg)],
        target_spin=target_spin,
        formalism=formalism,
    )


@pytest.fixture
def u238_data():
    """Single-resonance U-238-like isotope (6.67 eV resonance)."""
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


# ===========================================================================
# LM fitting
# ===========================================================================


class TestFitSpectrumLM:
    """Tests for fit_spectrum() with the LM (default) fitter."""

    def test_fit_single_isotope(self, u238_data):
        """Fit synthetic transmission with known density, recover it."""
        energies = np.linspace(1.0, 30.0, 500)
        true_density = 0.002

        # Generate synthetic "measured" transmission
        t = np.asarray(
            nereids.forward_model(energies, [(u238_data, true_density)])
        )
        sigma = np.full_like(t, 0.005)  # 0.5% constant uncertainty

        result = nereids.fit_spectrum(
            t, sigma, energies, [u238_data], max_iter=200
        )
        assert result.converged
        densities = np.asarray(result.densities)
        assert len(densities) == 1
        assert abs(densities[0] - true_density) / true_density < 0.05

    def test_fit_result_properties(self, u238_data):
        """Check that FitResult has all expected properties."""
        energies = np.linspace(1.0, 30.0, 200)
        t = np.asarray(
            nereids.forward_model(energies, [(u238_data, 0.001)])
        )
        sigma = np.full_like(t, 0.01)

        result = nereids.fit_spectrum(t, sigma, energies, [u238_data])

        assert isinstance(result.converged, bool)
        assert isinstance(result.iterations, int)
        assert isinstance(result.reduced_chi_squared, float)
        assert len(np.asarray(result.densities)) == 1
        assert len(np.asarray(result.uncertainties)) == 1

    def test_fit_repr(self, u238_data):
        energies = np.linspace(1.0, 30.0, 200)
        t = np.asarray(
            nereids.forward_model(energies, [(u238_data, 0.001)])
        )
        sigma = np.full_like(t, 0.01)
        result = nereids.fit_spectrum(t, sigma, energies, [u238_data])
        r = repr(result)
        assert "FitResult" in r

    def test_fit_temperature(self, u238_data):
        """Fit with fit_temperature=True should return temperature fields."""
        energies = np.linspace(1.0, 30.0, 500)
        t = np.asarray(
            nereids.forward_model(
                energies, [(u238_data, 0.001)], temperature_k=300.0
            )
        )
        sigma = np.full_like(t, 0.005)

        result = nereids.fit_spectrum(
            t,
            sigma,
            energies,
            [u238_data],
            temperature_k=300.0,
            fit_temperature=True,
        )
        assert result.temperature_k is not None
        assert result.temperature_k_unc is not None

    def test_fit_noisy_data(self, u238_data):
        """Fit with small noise should still converge."""
        rng = np.random.default_rng(42)
        energies = np.linspace(1.0, 30.0, 500)
        true_density = 0.002

        t_clean = np.asarray(
            nereids.forward_model(energies, [(u238_data, true_density)])
        )
        noise = rng.normal(0, 0.002, size=t_clean.shape)
        t_noisy = np.clip(t_clean + noise, 0.01, 1.0)
        sigma = np.full_like(t_noisy, 0.005)

        result = nereids.fit_spectrum(
            t_noisy, sigma, energies, [u238_data], max_iter=200
        )
        assert result.converged
        densities = np.asarray(result.densities)
        # Within 20% of true value (noisy data)
        assert abs(densities[0] - true_density) / true_density < 0.20


# ===========================================================================
# Poisson fitting
# ===========================================================================


class TestFitSpectrumPoisson:
    """Tests for fit_spectrum() with the Poisson fitter."""

    def test_poisson_basic(self, u238_data):
        """Poisson fitter with synthetic count data."""
        energies = np.linspace(1.0, 30.0, 300)
        true_density = 0.002
        flux = 10000.0  # high counts for easy fitting

        # Build clean transmission
        t_clean = np.asarray(
            nereids.forward_model(energies, [(u238_data, true_density)])
        )

        # Simulate counts: sample_counts ~ Poisson(flux * T), open_beam ~ flux
        rng = np.random.default_rng(123)
        open_beam = np.full(len(energies), flux)
        sample_counts = rng.poisson(flux * t_clean).astype(float)

        result = nereids.fit_spectrum(
            sample_counts,
            open_beam,
            energies,
            [u238_data],
            fitter="poisson",
            max_iter=200,
        )
        assert result.converged
        densities = np.asarray(result.densities)
        assert abs(densities[0] - true_density) / true_density < 0.20

    def test_poisson_result_has_nll(self, u238_data):
        """The reduced_chi_squared field contains NLL for Poisson fits."""
        energies = np.linspace(1.0, 30.0, 200)
        t = np.asarray(
            nereids.forward_model(energies, [(u238_data, 0.001)])
        )
        open_beam = np.full(len(energies), 5000.0)
        sample_counts = (t * 5000.0).astype(float)

        result = nereids.fit_spectrum(
            sample_counts,
            open_beam,
            energies,
            [u238_data],
            fitter="poisson",
        )
        assert isinstance(result.reduced_chi_squared, float)

    def test_invalid_fitter_raises(self, u238_data):
        energies = np.linspace(1.0, 30.0, 100)
        t = np.ones(100)
        s = np.ones(100)
        with pytest.raises(ValueError, match="fitter must be"):
            nereids.fit_spectrum(t, s, energies, [u238_data], fitter="bogus")


# ===========================================================================
# Spatial mapping (LM)
# ===========================================================================


class TestSpatialMapLM:
    """Tests for spatial_map() with the LM fitter."""

    def test_basic_spatial_map(self, u238_data):
        """3x3 spatial map should return correct shapes."""
        energies = np.linspace(1.0, 30.0, 200)
        true_density = 0.002
        ny, nx = 3, 3

        # Build a (n_e, ny, nx) transmission cube
        t_1d = np.asarray(
            nereids.forward_model(energies, [(u238_data, true_density)])
        )
        trans = np.tile(t_1d[:, None, None], (1, ny, nx))
        unc = np.full_like(trans, 0.005)

        result = nereids.spatial_map(
            trans, unc, energies, [u238_data], max_iter=50
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

        result = nereids.spatial_map(
            trans, unc, energies, [u238_data], max_iter=20
        )
        r = repr(result)
        assert "SpatialResult" in r


# ===========================================================================
# Spatial mapping (Poisson)
# ===========================================================================


class TestSpatialMapPoisson:
    """Tests for spatial_map() with the Poisson fitter."""

    def test_poisson_spatial_map(self, u238_data):
        """Poisson spatial map with synthetic count data."""
        energies = np.linspace(1.0, 30.0, 150)
        true_density = 0.002
        flux = 5000.0
        n_e = len(energies)
        ny, nx = 2, 2

        t_1d = np.asarray(
            nereids.forward_model(energies, [(u238_data, true_density)])
        )

        rng = np.random.default_rng(999)
        open_beam = np.full((n_e, ny, nx), flux)
        sample = np.zeros((n_e, ny, nx))
        for y in range(ny):
            for x in range(nx):
                sample[:, y, x] = rng.poisson(flux * t_1d).astype(float)

        result = nereids.spatial_map(
            sample,
            open_beam,
            energies,
            [u238_data],
            fitter="poisson",
            max_iter=50,
        )
        # Should return SparseResult
        assert hasattr(result, "density_maps")
        assert hasattr(result, "nll_map")
        assert hasattr(result, "converged_map")

        density_maps = result.density_maps
        assert len(density_maps) == 1
        dmap = np.asarray(density_maps[0])
        assert dmap.shape == (ny, nx)

        # Density recovery: Poisson is noisier, so use wider tolerance
        np.testing.assert_allclose(dmap, true_density, rtol=0.5)


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

    def test_empty_energies_fit(self, u238_data):
        """Empty measurement array should raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            nereids.fit_spectrum(
                np.array([]),
                np.array([]),
                np.array([]),
                [u238_data],
            )

    def test_length_mismatch_fit(self, u238_data):
        """Mismatched array lengths should raise ValueError."""
        e = np.linspace(1.0, 10.0, 100)
        t = np.ones(50)  # wrong length
        s = np.ones(50)
        with pytest.raises(ValueError, match="must match"):
            nereids.fit_spectrum(t, s, e, [u238_data])

    def test_sigma_mismatch_fit(self, u238_data):
        e = np.linspace(1.0, 10.0, 100)
        t = np.ones(100)
        s = np.ones(50)  # wrong length
        with pytest.raises(ValueError, match="must match"):
            nereids.fit_spectrum(t, s, e, [u238_data])

    def test_empty_isotopes_fit(self):
        e = np.linspace(1.0, 10.0, 100)
        t = np.ones(100)
        s = np.ones(100)
        with pytest.raises(ValueError, match="isotopes"):
            nereids.fit_spectrum(t, s, e, [])

    def test_invalid_temperature_fit(self, u238_data):
        e = np.linspace(1.0, 10.0, 100)
        t = np.ones(100)
        s = np.ones(100)
        with pytest.raises(ValueError, match="temperature_k"):
            nereids.fit_spectrum(
                t, s, e, [u238_data], temperature_k=-1.0
            )

    def test_fit_temperature_too_low(self, u238_data):
        """fit_temperature=True with temperature_k < 1.0 should raise."""
        e = np.linspace(1.0, 10.0, 100)
        t = np.ones(100)
        s = np.ones(100)
        with pytest.raises(ValueError, match="temperature_k"):
            nereids.fit_spectrum(
                t, s, e, [u238_data], temperature_k=0.5, fit_temperature=True
            )

    def test_spatial_map_empty_energies(self, u238_data):
        trans = np.ones((0, 2, 2))
        unc = np.ones((0, 2, 2))
        with pytest.raises(ValueError, match="energies"):
            nereids.spatial_map(trans, unc, np.array([]), [u238_data])

    def test_spatial_map_shape_mismatch(self, u238_data):
        e = np.linspace(1.0, 10.0, 5)
        trans = np.ones((5, 3, 3))
        unc = np.ones((5, 3, 4))  # width mismatch
        with pytest.raises(ValueError, match="shape"):
            nereids.spatial_map(trans, unc, e, [u238_data])

    def test_spatial_map_empty_isotopes(self):
        e = np.linspace(1.0, 10.0, 5)
        trans = np.ones((5, 3, 3))
        unc = np.ones((5, 3, 3))
        with pytest.raises(ValueError, match="isotopes"):
            nereids.spatial_map(trans, unc, e, [])

    def test_initial_densities_mismatch(self, u238_data):
        e = np.linspace(1.0, 10.0, 100)
        t = np.ones(100)
        s = np.ones(100)
        with pytest.raises(ValueError, match="initial_densities"):
            nereids.fit_spectrum(
                t, s, e, [u238_data], initial_densities=[0.001, 0.002]
            )

    def test_doppler_broaden_invalid_awr(self):
        e = np.linspace(1.0, 10.0, 100)
        xs = np.ones(100)
        with pytest.raises(ValueError, match="awr"):
            nereids.doppler_broaden(e, xs, -1.0, 300.0)

    def test_doppler_broaden_invalid_temperature(self):
        e = np.linspace(1.0, 10.0, 100)
        xs = np.ones(100)
        with pytest.raises(ValueError, match="temperature_k"):
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
