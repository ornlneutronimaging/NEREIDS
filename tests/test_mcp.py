"""Tests for the NEREIDS MCP server tools.

Tests call tool functions directly (not via the MCP protocol) to avoid
async complexity and FastMCP version-specific call_tool behavior.
The functions are plain Python under the @mcp.tool() decorator, so
direct invocation validates all business logic.
"""

import pytest

fastmcp = pytest.importorskip("fastmcp")

from nereids.mcp.server import (
    _registry,
    compute_cross_sections,
    compute_transmission,
    detect_isotopes,
    forward_model,
    get_resonance_parameters,
    list_isotopes,
    load_endf,
)


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear the isotope registry before each test."""
    _registry.clear()
    yield
    _registry.clear()


# ---------------------------------------------------------------------------
# list_isotopes
# ---------------------------------------------------------------------------


class TestListIsotopes:
    def test_iron(self):
        result = list_isotopes(z=26)
        assert len(result) >= 4  # Fe has 4 stable isotopes
        symbols = [entry["symbol"] for entry in result]
        assert "Fe-56" in symbols

    def test_uranium(self):
        result = list_isotopes(z=92)
        assert len(result) >= 3  # U-234, U-235, U-238
        mass_numbers = [entry["a"] for entry in result]
        assert 238 in mass_numbers

    def test_return_structure(self):
        result = list_isotopes(z=1)
        assert len(result) >= 1
        entry = result[0]
        assert "z" in entry
        assert "a" in entry
        assert "symbol" in entry
        assert "abundance" in entry
        assert entry["z"] == 1


# ---------------------------------------------------------------------------
# load_endf
# ---------------------------------------------------------------------------


class TestLoadEndf:
    def test_load_fe56(self):
        result = load_endf(isotope="Fe-56")
        assert result["z"] == 26
        assert result["a"] == 56
        assert result["n_resonances"] > 0
        assert "Fe-56" in _registry

    def test_load_invalid_isotope(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            load_endf(isotope="invalid")

    def test_load_stores_in_registry(self):
        assert "U-238" not in _registry
        load_endf(isotope="U-238")
        assert "U-238" in _registry

    def test_return_structure(self):
        result = load_endf(isotope="Fe-56")
        for key in ("isotope", "z", "a", "n_resonances", "scattering_radius",
                     "target_spin", "l_values"):
            assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# get_resonance_parameters
# ---------------------------------------------------------------------------


class TestGetResonanceParameters:
    def test_loaded_isotope(self):
        load_endf(isotope="Fe-56")
        result = get_resonance_parameters(isotope="Fe-56")
        assert result["z"] == 26
        assert result["a"] == 56
        assert result["awr"] > 50  # ~55.845

    def test_not_loaded(self):
        with pytest.raises(ValueError, match="not loaded"):
            get_resonance_parameters(isotope="U-238")


# ---------------------------------------------------------------------------
# compute_cross_sections
# ---------------------------------------------------------------------------


class TestComputeCrossSections:
    def test_basic(self):
        load_endf(isotope="Fe-56")
        result = compute_cross_sections(
            isotope="Fe-56", energy_min=1.0, energy_max=100.0, n_points=100,
        )
        assert len(result["energies"]) == 100
        assert len(result["total"]) == 100
        assert len(result["elastic"]) == 100
        assert len(result["capture"]) == 100
        assert len(result["fission"]) == 100
        assert all(v >= 0 for v in result["total"])

    def test_not_loaded(self):
        with pytest.raises(ValueError, match="not loaded"):
            compute_cross_sections(isotope="Fe-56")


# ---------------------------------------------------------------------------
# compute_transmission
# ---------------------------------------------------------------------------


class TestComputeTransmission:
    def test_basic(self):
        load_endf(isotope="Fe-56")
        result = compute_transmission(
            isotope="Fe-56", thickness=0.01,
            energy_min=1.0, energy_max=100.0, n_points=50,
        )
        assert len(result["transmission"]) == 50
        # Transmission should be between 0 and 1
        assert all(0 <= v <= 1 for v in result["transmission"])

    def test_zero_thickness(self):
        load_endf(isotope="Fe-56")
        result = compute_transmission(
            isotope="Fe-56", thickness=0.0,
            energy_min=1.0, energy_max=100.0, n_points=50,
        )
        # Zero thickness -> transmission = 1.0
        assert all(abs(v - 1.0) < 1e-12 for v in result["transmission"])

    def test_not_loaded(self):
        with pytest.raises(ValueError, match="not loaded"):
            compute_transmission(isotope="Fe-56", thickness=0.01)


# ---------------------------------------------------------------------------
# forward_model (multi-isotope)
# ---------------------------------------------------------------------------


class TestForwardModel:
    def test_single_isotope(self):
        load_endf(isotope="Fe-56")
        result = forward_model(
            isotopes=[{"isotope": "Fe-56", "thickness": 0.01}],
            energy_min=1.0, energy_max=100.0, n_points=50,
        )
        assert len(result["transmission"]) == 50
        assert all(0 <= v <= 1 for v in result["transmission"])

    def test_multi_isotope(self):
        load_endf(isotope="Fe-56")
        load_endf(isotope="U-238")
        result = forward_model(
            isotopes=[
                {"isotope": "Fe-56", "thickness": 0.01},
                {"isotope": "U-238", "thickness": 0.001},
            ],
            energy_min=1.0, energy_max=50.0, n_points=50,
        )
        assert len(result["transmission"]) == 50

    def test_not_loaded(self):
        with pytest.raises(ValueError, match="not loaded"):
            forward_model(
                isotopes=[{"isotope": "Fe-56", "thickness": 0.01}],
            )


# ---------------------------------------------------------------------------
# detect_isotopes
# ---------------------------------------------------------------------------


class TestDetectIsotopes:
    def test_basic(self):
        load_endf(isotope="Fe-56")
        load_endf(isotope="W-182")
        result = detect_isotopes(
            matrix_isotope="Fe-56",
            matrix_density=0.01,
            trace_isotopes=["W-182"],
            trace_ppm=1000.0,
            energy_min=1.0, energy_max=100.0, n_points=200,
            i0=1e6,
        )
        assert len(result) == 1
        entry = result[0]
        assert isinstance(entry["detectable"], bool)
        assert isinstance(entry["peak_snr"], float)
        assert isinstance(entry["peak_energy_ev"], float)
        assert isinstance(entry["peak_delta_t_per_ppm"], float)
        assert isinstance(entry["opaque_fraction"], float)

    def test_matrix_not_loaded(self):
        with pytest.raises(ValueError, match="not loaded"):
            detect_isotopes(
                matrix_isotope="Fe-56",
                matrix_density=0.01,
                trace_isotopes=["W-182"],
            )

    def test_trace_not_loaded(self):
        load_endf(isotope="Fe-56")
        with pytest.raises(ValueError, match="not loaded"):
            detect_isotopes(
                matrix_isotope="Fe-56",
                matrix_density=0.01,
                trace_isotopes=["W-182"],
            )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_compute_cross_sections_invalid_energy(self):
        """Energy validation catches bad inputs."""
        load_endf("Fe-56")
        with pytest.raises(ValueError, match="energy_min"):
            compute_cross_sections("Fe-56", energy_min=-1.0)
        with pytest.raises(ValueError, match="n_points"):
            compute_cross_sections("Fe-56", n_points=0)
        with pytest.raises(ValueError, match="energy_min.*less than"):
            compute_cross_sections("Fe-56", energy_min=100.0, energy_max=1.0)

    def test_compute_transmission_invalid_thickness(self):
        load_endf("Fe-56")
        with pytest.raises(ValueError, match="thickness"):
            compute_transmission("Fe-56", thickness=-1.0)

    def test_forward_model_missing_key(self):
        load_endf("Fe-56")
        with pytest.raises(ValueError, match="thickness"):
            forward_model([{"isotope": "Fe-56"}])  # missing thickness

    def test_detect_isotopes_empty_traces(self):
        load_endf("Fe-56")
        with pytest.raises(ValueError, match="empty"):
            detect_isotopes("Fe-56", 0.01, [])

    def test_list_isotopes_invalid_z(self):
        with pytest.raises(ValueError, match="z must be"):
            list_isotopes(0)
