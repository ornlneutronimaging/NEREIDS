"""Tests for the NEREIDS MCP server tools.

Tests call tool functions directly (not via the MCP protocol) to avoid
async complexity and FastMCP version-specific call_tool behavior.
The functions are plain Python under the @mcp.tool() decorator, so
direct invocation validates all business logic.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

fastmcp = pytest.importorskip("fastmcp")

import nereids
from nereids.mcp.server import (
    _fit_result_summary,
    _json_safe,
    _registry,
    compute_cross_sections,
    compute_transmission,
    detect_isotopes,
    extract_resonance_manifest,
    forward_model,
    get_resonance_parameters,
    list_isotopes,
    load_endf,
    process_resonance_dataset,
    validate_resonance_dataset,
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


# ---------------------------------------------------------------------------
# Manifest-driven workflow tools
# ---------------------------------------------------------------------------


def _synthetic_u238_entry(initial_density=0.001):
    return {
        "isotope": "U-238",
        "initial_density": initial_density,
        "synthetic_resonance": {
            "z": 92,
            "a": 238,
            "awr": 236.006,
            "scattering_radius": 9.48,
            "target_spin": 0.0,
            "resonances": [[6.67, 0.5, 0.0015, 0.023]],
        },
    }


def _synthetic_u238_data():
    return nereids.create_resonance_data(
        z=92,
        a=238,
        awr=236.006,
        scattering_radius=9.48,
        resonances=[(6.67, 0.5, 0.0015, 0.023)],
        target_spin=0.0,
    )


def _write_json_frontmatter_manifest(tmp_path, analysis):
    frontmatter = {
        "name": "synthetic-u238-mcp-demo",
        "description": "Synthetic U-238 resonance processing test",
        "tool": "nereids",
        "physics": "neutron-resonance",
        "version": "0.1.0",
        "analysis": analysis,
    }
    path = tmp_path / "manifest_intermediate.md"
    path.write_text(
        "---\n"
        + json.dumps(frontmatter, indent=2)
        + "\n---\n"
        + "# Synthetic NEREIDS MCP workflow\n"
    )
    return path


class TestManifestWorkflowTools:
    def test_extract_and_validate_manifest(self, tmp_path):
        np.savez(
            tmp_path / "spectrum.npz",
            energies_ev=np.linspace(1.0, 30.0, 20),
            transmission=np.ones(20),
            uncertainty=np.full(20, 0.01),
        )
        manifest_path = _write_json_frontmatter_manifest(
            tmp_path,
            {
                "mode": "single_spectrum",
                "data": {"kind": "transmission_npz", "path": "spectrum.npz"},
                "isotopes": [_synthetic_u238_entry()],
                "fit": {"solver": "lm", "max_iter": 5},
                "resolution": {"kind": "none"},
            },
        )

        extracted = extract_resonance_manifest(str(tmp_path))
        assert extracted["manifest_path"] == str(manifest_path)
        assert extracted["frontmatter"]["tool"] == "nereids"

        validation = validate_resonance_dataset(str(tmp_path))
        assert validation["valid"] is True
        assert validation["mode"] == "single_spectrum"
        assert validation["data_paths"][0]["exists"] is True

    def test_process_single_spectrum_manifest(self, tmp_path):
        energies = np.linspace(1.0, 30.0, 160)
        true_density = 0.002
        isotope = _synthetic_u238_data()
        transmission = np.asarray(
            nereids.forward_model(energies, [(isotope, true_density)])
        )
        np.savez(
            tmp_path / "spectrum.npz",
            energies_ev=energies,
            transmission=transmission,
            uncertainty=np.full_like(transmission, 0.005),
        )
        _write_json_frontmatter_manifest(
            tmp_path,
            {
                "mode": "single_spectrum",
                "data": {"kind": "transmission_npz", "path": "spectrum.npz"},
                "isotopes": [_synthetic_u238_entry(initial_density=0.001)],
                "fit": {"solver": "lm", "max_iter": 50},
                "resolution": {"kind": "none"},
                "output": {"directory": "output"},
            },
        )

        result = process_resonance_dataset(str(tmp_path))

        assert result["success"] is True
        fit = result["results"]["density_fits"][0]
        assert fit["isotope"] == "U-238"
        assert fit["density_atoms_per_barn"] == pytest.approx(true_density, rel=0.15)
        assert (tmp_path / "output" / "nereids_spectrum_fit.npz").exists()
        assert (tmp_path / "output" / "nereids_mcp_result.json").exists()

    def test_process_density_map_manifest(self, tmp_path):
        energies = np.linspace(1.0, 30.0, 120)
        true_density = 0.002
        isotope = _synthetic_u238_data()
        spectrum = np.asarray(
            nereids.forward_model(energies, [(isotope, true_density)])
        )
        transmission = np.tile(spectrum[:, None, None], (1, 2, 3))
        np.savez(
            tmp_path / "density-map.npz",
            energies_ev=energies,
            transmission=transmission,
            uncertainty=np.full_like(transmission, 0.005),
        )
        _write_json_frontmatter_manifest(
            tmp_path,
            {
                "mode": "density_map",
                "data": {"kind": "transmission_npz", "path": "density-map.npz"},
                "isotopes": [_synthetic_u238_entry(initial_density=0.001)],
                "fit": {"solver": "lm", "max_iter": 50},
                "resolution": {"kind": "none"},
                "output": {"directory": "output"},
            },
        )

        result = process_resonance_dataset(str(tmp_path))

        assert result["success"] is True
        assert result["results"]["n_converged"] == 6
        stats = result["results"]["density_maps"][0]["density_atoms_per_barn"]
        assert stats["mean"] == pytest.approx(true_density, rel=0.15)
        assert (tmp_path / "output" / "nereids_density_map.npz").exists()

    def test_nexus_density_map_aligns_counts_with_ascending_energy(
        self, tmp_path, monkeypatch
    ):
        sample_path = tmp_path / "sample.nxs"
        open_beam_path = tmp_path / "open_beam.nxs"
        sample_path.touch()
        open_beam_path.touch()
        captured = {}

        def fake_load_nexus_histogram(path):
            if path == str(sample_path):
                counts = np.asarray([[[90.0]], [[40.0]], [[10.0]]])
            else:
                counts = np.asarray([[[100.0]], [[100.0]], [[100.0]]])
            return SimpleNamespace(
                counts=counts,
                tof_edges_us=np.asarray([1.0, 2.0, 3.0, 4.0]),
                flight_path_m=25.0,
            )

        def fake_from_counts(sample, open_beam):
            captured["sample"] = np.asarray(sample).copy()
            captured["open_beam"] = np.asarray(open_beam).copy()
            return {"sample": sample, "open_beam": open_beam}

        def fake_spatial_map_typed(
            input_data, energies, isotope_data, initial_densities, **kwargs
        ):
            captured["energies"] = np.asarray(energies).copy()
            return SimpleNamespace(
                chi_squared_map=np.zeros((1, 1)),
                converged_map=np.ones((1, 1), dtype=bool),
                deviance_per_dof_map=None,
                density_maps=[np.full((1, 1), 0.002)],
                uncertainty_maps=[np.full((1, 1), 0.0001)],
                n_converged=1,
                n_total=1,
                n_failed=0,
            )

        monkeypatch.setattr(nereids, "load_nexus_histogram", fake_load_nexus_histogram)
        monkeypatch.setattr(
            nereids,
            "tof_to_energy_centers",
            lambda edges, flight_path, delay_us=0.0: np.asarray([1.0, 4.0, 9.0]),
        )
        monkeypatch.setattr(nereids, "from_counts", fake_from_counts)
        monkeypatch.setattr(nereids, "spatial_map_typed", fake_spatial_map_typed)

        _write_json_frontmatter_manifest(
            tmp_path,
            {
                "mode": "density_map",
                "data": {
                    "kind": "nexus",
                    "sample_path": sample_path.name,
                    "open_beam_path": open_beam_path.name,
                },
                "isotopes": [_synthetic_u238_entry(initial_density=0.001)],
                "fit": {"solver": "lm", "max_iter": 5},
                "resolution": {"kind": "none"},
                "output": {"directory": "output"},
            },
        )

        result = process_resonance_dataset(str(tmp_path))

        assert result["success"] is True
        np.testing.assert_allclose(captured["energies"], [1.0, 4.0, 9.0])
        np.testing.assert_allclose(captured["sample"][:, 0, 0], [10.0, 40.0, 90.0])

    def test_process_rejects_mismatched_counts_shapes(self, tmp_path):
        np.savez(
            tmp_path / "counts.npz",
            energies_ev=np.linspace(1.0, 5.0, 5),
            sample_counts=np.ones(5),
            open_beam_counts=np.ones(6),
        )
        _write_json_frontmatter_manifest(
            tmp_path,
            {
                "mode": "single_spectrum",
                "data": {"kind": "counts_npz", "path": "counts.npz"},
                "isotopes": [_synthetic_u238_entry()],
                "fit": {"solver": "lm", "max_iter": 5},
                "resolution": {"kind": "none"},
            },
        )

        result = process_resonance_dataset(str(tmp_path))

        assert result["success"] is False
        assert "matching shapes" in result["error"]

    def test_process_reports_missing_counts_npz_key(self, tmp_path):
        np.savez(
            tmp_path / "counts.npz",
            energies_ev=np.linspace(1.0, 5.0, 5),
            sample_counts=np.ones(5),
        )
        _write_json_frontmatter_manifest(
            tmp_path,
            {
                "mode": "single_spectrum",
                "data": {"kind": "counts_npz", "path": "counts.npz"},
                "isotopes": [_synthetic_u238_entry()],
                "fit": {"solver": "lm", "max_iter": 5},
                "resolution": {"kind": "none"},
            },
        )

        result = process_resonance_dataset(str(tmp_path))

        assert result["success"] is False
        assert "missing required key(s): open_beam_counts" in result["error"]
        assert "Available keys: energies_ev, sample_counts" in result["error"]

    def test_process_rejects_mismatched_density_uncertainty_shapes(self, tmp_path):
        np.savez(
            tmp_path / "density-map.npz",
            energies_ev=np.linspace(1.0, 5.0, 5),
            transmission=np.ones((5, 2, 2)),
            uncertainty=np.ones((5, 2, 3)),
        )
        _write_json_frontmatter_manifest(
            tmp_path,
            {
                "mode": "density_map",
                "data": {"kind": "transmission_npz", "path": "density-map.npz"},
                "isotopes": [_synthetic_u238_entry()],
                "fit": {"solver": "lm", "max_iter": 5},
                "resolution": {"kind": "none"},
            },
        )

        result = process_resonance_dataset(str(tmp_path))

        assert result["success"] is False
        assert "matching shapes" in result["error"]

    def test_process_reports_missing_density_npz_key(self, tmp_path):
        np.savez(
            tmp_path / "density-map.npz",
            energies_ev=np.linspace(1.0, 5.0, 5),
            sample_counts=np.ones((5, 2, 2)),
        )
        _write_json_frontmatter_manifest(
            tmp_path,
            {
                "mode": "density_map",
                "data": {"kind": "counts_npz", "path": "density-map.npz"},
                "isotopes": [_synthetic_u238_entry()],
                "fit": {"solver": "lm", "max_iter": 5},
                "resolution": {"kind": "none"},
            },
        )

        result = process_resonance_dataset(str(tmp_path))

        assert result["success"] is False
        assert "density_map counts data is missing required key(s)" in result["error"]
        assert "open_beam_counts" in result["error"]

    def test_process_reports_missing_density_transmission_key(self, tmp_path):
        np.savez(
            tmp_path / "density-map.npz",
            energies_ev=np.linspace(1.0, 5.0, 5),
            not_transmission=np.ones((5, 2, 2)),
        )
        _write_json_frontmatter_manifest(
            tmp_path,
            {
                "mode": "density_map",
                "data": {"kind": "transmission_npz", "path": "density-map.npz"},
                "isotopes": [_synthetic_u238_entry()],
                "fit": {"solver": "lm", "max_iter": 5},
                "resolution": {"kind": "none"},
            },
        )

        result = process_resonance_dataset(str(tmp_path))

        assert result["success"] is False
        assert "density_map transmission data is missing required key(s)" in result["error"]
        assert "transmission" in result["error"]

    def test_process_rejects_out_of_bounds_roi(self, tmp_path):
        np.savez(
            tmp_path / "density-map.npz",
            energies_ev=np.linspace(1.0, 5.0, 5),
            transmission=np.ones((5, 2, 2)),
            uncertainty=np.ones((5, 2, 2)),
        )
        _write_json_frontmatter_manifest(
            tmp_path,
            {
                "mode": "density_map",
                "data": {
                    "kind": "transmission_npz",
                    "path": "density-map.npz",
                    "roi": {"x0": 1, "y0": 0, "width": 2, "height": 1},
                },
                "isotopes": [_synthetic_u238_entry()],
                "fit": {"solver": "lm", "max_iter": 5},
                "resolution": {"kind": "none"},
            },
        )

        result = process_resonance_dataset(str(tmp_path))

        assert result["success"] is False
        assert "exceeds cube bounds" in result["error"]

    def test_process_rejects_mismatched_nexus_tof_edges(self, tmp_path, monkeypatch):
        sample_path = tmp_path / "sample.nxs"
        open_beam_path = tmp_path / "open_beam.nxs"
        sample_path.touch()
        open_beam_path.touch()

        def fake_load_nexus_histogram(path):
            tof_edges = (
                np.asarray([1.0, 2.0, 3.0, 4.0])
                if path == str(sample_path)
                else np.asarray([1.0, 2.0, 3.2, 4.0])
            )
            return SimpleNamespace(
                counts=np.ones((3, 1, 1)),
                tof_edges_us=tof_edges,
                flight_path_m=25.0,
            )

        monkeypatch.setattr(nereids, "load_nexus_histogram", fake_load_nexus_histogram)
        _write_json_frontmatter_manifest(
            tmp_path,
            {
                "mode": "density_map",
                "data": {
                    "kind": "nexus",
                    "sample_path": sample_path.name,
                    "open_beam_path": open_beam_path.name,
                },
                "isotopes": [_synthetic_u238_entry()],
                "fit": {"solver": "lm", "max_iter": 5},
                "resolution": {"kind": "none"},
            },
        )

        result = process_resonance_dataset(str(tmp_path))

        assert result["success"] is False
        assert "TOF bin edges must match" in result["error"]

    def test_validation_messages_match_allowed_values(self, tmp_path):
        frontmatter = {
            "name": "invalid-validation-demo",
            "tool": "pleiades",
            "analysis": {
                "mode": "bad-mode",
                "data": {"kind": "transmission_npz"},
                "isotopes": [_synthetic_u238_entry()],
            },
        }
        (tmp_path / "manifest_intermediate.md").write_text(
            "---\n" + json.dumps(frontmatter) + "\n---\n"
        )

        result = validate_resonance_dataset(str(tmp_path))

        assert result["valid"] is False
        assert any("'nereids-mcp'" in error for error in result["errors"])
        assert any("spectrum" in error for error in result["errors"])

    def test_dry_run_rejects_missing_required_data_path(self, tmp_path):
        _write_json_frontmatter_manifest(
            tmp_path,
            {
                "mode": "single_spectrum",
                "data": {"kind": "transmission_npz"},
                "isotopes": [_synthetic_u238_entry()],
                "fit": {"solver": "lm", "max_iter": 5},
                "resolution": {"kind": "none"},
            },
        )

        result = process_resonance_dataset(str(tmp_path), dry_run=True)

        assert result["success"] is False
        assert "single_spectrum workflow requires data.path" in result["validation"]["errors"]

    def test_fit_summary_is_strict_json_safe(self):
        result = SimpleNamespace(
            densities=np.asarray([np.nan]),
            uncertainties=np.asarray([np.inf]),
            reduced_chi_squared=np.nan,
            deviance_per_dof=np.inf,
            converged=False,
            iterations=0,
            temperature_k=np.nan,
            anorm=np.inf,
            background=[np.nan],
        )

        summary = _fit_result_summary(result, ["U-238"])

        assert summary["density_fits"][0]["density_atoms_per_barn"] is None
        assert summary["density_fits"][0]["uncertainty_atoms_per_barn"] is None
        assert summary["reduced_chi_squared"] is None
        assert summary["deviance_per_dof"] is None
        json.dumps(_json_safe(summary), allow_nan=False)

    def test_fit_summary_emits_energy_scale_and_background_fields(self):
        """Issue #530: when the LM solver fits t0/L_scale/back_d/back_f,
        those values must appear in the result summary.  Before the fix,
        `_fit_result_summary` only emitted `anorm` and `background`;
        the four extra scalars were silently dropped, making it
        impossible to tell whether the manifest's fit flags were
        exercised and impossible to reconstruct the model curve.
        """
        result = SimpleNamespace(
            densities=np.asarray([0.001]),
            uncertainties=np.asarray([1.0e-5]),
            reduced_chi_squared=1.2,
            deviance_per_dof=None,
            converged=True,
            iterations=17,
            temperature_k=293.6,
            anorm=0.99,
            background=[0.01, -1e-4, 5e-7],
            t0_us=0.4662,
            l_scale=1.005273,
            back_d=0.0796,
            back_f=1.10e-4,
        )

        summary = _fit_result_summary(result, ["U-238"])

        assert summary["t0_us"] == pytest.approx(0.4662)
        assert summary["l_scale"] == pytest.approx(1.005273)
        assert summary["back_d"] == pytest.approx(0.0796)
        assert summary["back_f"] == pytest.approx(1.10e-4)
        json.dumps(_json_safe(summary), allow_nan=False)

    def test_fit_summary_omits_energy_scale_fields_when_unset(self):
        """Backwards-compat: fits without the new flags emit a summary
        with the four extra keys *absent* (not present-with-null), so
        downstream consumers can keep using `"key" in summary` to
        detect whether each flag was active.  Matches the existing
        `deviance_per_dof` / `temperature_k` pattern.
        """
        result = SimpleNamespace(
            densities=np.asarray([0.001]),
            uncertainties=np.asarray([1.0e-5]),
            reduced_chi_squared=1.2,
            deviance_per_dof=None,
            converged=True,
            iterations=12,
            temperature_k=293.6,
            anorm=1.0,
            background=[0.0, 0.0, 0.0],
            t0_us=None,
            l_scale=None,
            back_d=None,
            back_f=None,
        )

        summary = _fit_result_summary(result, ["U-238"])

        assert "t0_us" not in summary
        assert "l_scale" not in summary
        assert "back_d" not in summary
        assert "back_f" not in summary
        # Existing keys must still be present.
        assert "anorm" in summary
        assert "background" in summary

    def test_fit_summary_emits_energy_scale_via_real_fit(self):
        """End-to-end variant of the SimpleNamespace gates above: drive a
        real `nereids.fit_spectrum_typed` LM fit with `fit_energy_scale=True`
        and confirm the new keys make it into the summary returned to MCP
        callers.  Catches regressions where the binding stops populating
        `result.t0_us` / `result.l_scale` on the actual `PyFitResult`.
        """
        energies = np.linspace(4.0, 30.0, 400)
        isotope = _synthetic_u238_data()
        t = np.asarray(nereids.forward_model(energies, [(isotope, 1.0e-3)]))
        sigma = np.full_like(t, 0.005)

        result = nereids.fit_spectrum_typed(
            transmission=t,
            uncertainty=sigma,
            energies=energies,
            isotopes=[(isotope, 1.0e-3)],
            solver="lm",
            temperature_k=293.6,
            max_iter=100,
            background=True,
            fit_back_d=True,
            fit_back_f=True,
            fit_energy_scale=True,
            t0_init_us=0.0,
            l_scale_init=1.0,
            energy_scale_flight_path_m=25.0,
        )

        summary = _fit_result_summary(result, ["U-238"])

        for key in ("t0_us", "l_scale", "back_d", "back_f"):
            assert key in summary, f"summary missing {key}: {sorted(summary)}"
            value = summary[key]
            assert isinstance(value, float) and np.isfinite(value), (
                f"summary[{key!r}] not a finite float: {value!r}"
            )
        # Round-trip strict-JSON-safe so downstream MCP serialization
        # cannot break on the new fields.
        json.dumps(_json_safe(summary), allow_nan=False)

    def test_process_single_spectrum_manifest_emits_energy_scale_keys(self, tmp_path):
        """Manifest-driven path: enabling `fit_energy_scale` / `fit_back_d`
        / `fit_back_f` in the manifest's analysis.fit block must surface
        the corresponding scalars in the `results` block of the JSON the
        MCP server returns.  Closes the #530 acceptance criterion.
        """
        energies = np.linspace(4.0, 30.0, 200)
        true_density = 1.0e-3
        isotope = _synthetic_u238_data()
        transmission = np.asarray(
            nereids.forward_model(energies, [(isotope, true_density)])
        )
        np.savez(
            tmp_path / "spectrum.npz",
            energies_ev=energies,
            transmission=transmission,
            uncertainty=np.full_like(transmission, 0.005),
        )
        _write_json_frontmatter_manifest(
            tmp_path,
            {
                "mode": "single_spectrum",
                "data": {"kind": "transmission_npz", "path": "spectrum.npz"},
                "isotopes": [_synthetic_u238_entry(initial_density=true_density)],
                "fit": {
                    "solver": "lm",
                    "max_iter": 80,
                    "background": True,
                    "fit_back_d": True,
                    "fit_back_f": True,
                    "fit_energy_scale": True,
                    "t0_init_us": 0.0,
                    "l_scale_init": 1.0,
                    "energy_scale_flight_path_m": 25.0,
                },
                "resolution": {"kind": "none"},
                "output": {"directory": "output"},
            },
        )

        result = process_resonance_dataset(str(tmp_path))

        assert result["success"] is True
        results_block = result["results"]
        for key in ("t0_us", "l_scale", "back_d", "back_f"):
            assert key in results_block, (
                f"results block missing {key}: keys={sorted(results_block)}"
            )
            assert isinstance(results_block[key], float)
            assert np.isfinite(results_block[key])

    def test_process_density_map_manifest_emits_fit_param_stats(self, tmp_path):
        """Spatial counterpart of #530: `_process_density_map` previously
        emitted only `density_maps` — no `anorm` / `background` / `t0` /
        `L_scale` info, even though `SpatialResult` exposes those as
        per-pixel maps when `fit_energy_scale` is on.  Backwards-compat:
        the key is absent when no energy-scale-related maps are populated.
        """
        energies = np.linspace(4.0, 30.0, 120)
        true_density = 1.0e-3
        isotope = _synthetic_u238_data()
        spectrum = np.asarray(
            nereids.forward_model(energies, [(isotope, true_density)])
        )
        # Tiny 2x2 cube so the test stays fast.
        cube = np.tile(spectrum[:, None, None], (1, 2, 2))
        np.savez(
            tmp_path / "density-map.npz",
            energies_ev=energies,
            transmission=cube,
            uncertainty=np.full_like(cube, 0.005),
        )
        _write_json_frontmatter_manifest(
            tmp_path,
            {
                "mode": "density_map",
                "data": {"kind": "transmission_npz", "path": "density-map.npz"},
                "isotopes": [_synthetic_u238_entry(initial_density=true_density)],
                "fit": {
                    "solver": "lm",
                    "max_iter": 80,
                    "background": True,
                    "fit_energy_scale": True,
                    "t0_init_us": 0.0,
                    "l_scale_init": 1.0,
                    "energy_scale_flight_path_m": 25.0,
                },
                "resolution": {"kind": "none"},
                "output": {"directory": "output"},
            },
        )

        result = process_resonance_dataset(str(tmp_path))

        assert result["success"] is True
        stats = result["results"].get("fit_param_stats")
        assert stats is not None, (
            f"fit_param_stats absent when fit_energy_scale=True; "
            f"keys={sorted(result['results'])}"
        )
        # Per-pixel anorm + 3-term background + t0/L_scale maps must all
        # be summarised — the four scalars `SpatialResult` exposes.
        # `back_d` / `back_f` are NOT in the spatial result type yet
        # (per-pixel back_d_map / back_f_map are not exposed on
        # SpatialResult); tracked as a follow-up.
        for key in ("anorm", "background", "t0_us", "l_scale"):
            assert key in stats, f"fit_param_stats missing {key}: {sorted(stats)}"
        # The npz must carry the raw arrays so downstream consumers
        # can reconstruct the model curve per pixel.
        npz = np.load(tmp_path / "output" / "nereids_density_map.npz")
        try:
            for arr_key in ("anorm_map", "t0_us_map", "l_scale_map"):
                assert arr_key in npz.files, (
                    f"density-map npz missing {arr_key}: {npz.files}"
                )
        finally:
            npz.close()

    def test_process_density_map_manifest_omits_fit_param_stats_by_default(
        self, tmp_path
    ):
        """When no fit-energy-scale / per-pixel background flags are
        set, the spatial summary must NOT carry a `fit_param_stats`
        key — matching the pre-fix schema for unrelated consumers.
        """
        energies = np.linspace(4.0, 30.0, 120)
        true_density = 1.0e-3
        isotope = _synthetic_u238_data()
        spectrum = np.asarray(
            nereids.forward_model(energies, [(isotope, true_density)])
        )
        cube = np.tile(spectrum[:, None, None], (1, 2, 2))
        np.savez(
            tmp_path / "density-map.npz",
            energies_ev=energies,
            transmission=cube,
            uncertainty=np.full_like(cube, 0.005),
        )
        _write_json_frontmatter_manifest(
            tmp_path,
            {
                "mode": "density_map",
                "data": {"kind": "transmission_npz", "path": "density-map.npz"},
                "isotopes": [_synthetic_u238_entry(initial_density=true_density)],
                "fit": {"solver": "lm", "max_iter": 50},
                "resolution": {"kind": "none"},
                "output": {"directory": "output"},
            },
        )

        result = process_resonance_dataset(str(tmp_path))

        assert result["success"] is True
        assert "fit_param_stats" not in result["results"]
