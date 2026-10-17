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
    _registry,
    compute_cross_sections,
    compute_transmission,
    extract_resonance_manifest,
    forward_model,
    get_resonance_parameters,
    list_isotopes,
    load_endf,
    process_resonance_dataset,
    validate_resonance_dataset,
)

from _fixtures import _synthetic_u238_entry


def _load_endf_or_skip(**kwargs):
    """Call the load_endf tool, skipping the test when the IAEA site cannot be reached."""
    try:
        return load_endf(**kwargs)
    except ConnectionError as exc:
        pytest.skip(f"IAEA ENDF service unreachable: {exc}")


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
        result = _load_endf_or_skip(isotope="Fe-56")
        assert result["z"] == 26
        assert result["a"] == 56
        assert result["n_resonances"] > 0
        assert "Fe-56" in _registry

    def test_load_invalid_isotope(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            load_endf(isotope="invalid")

    def test_load_stores_in_registry(self):
        assert "U-238" not in _registry
        _load_endf_or_skip(isotope="U-238")
        assert "U-238" in _registry

    def test_return_structure(self):
        result = _load_endf_or_skip(isotope="Fe-56")
        for key in ("isotope", "z", "a", "n_resonances", "scattering_radius",
                     "target_spin", "l_values", "has_unevaluated_ranges"):
            assert key in result, f"Missing key: {key}"

    def test_pure_lrf7_isotope_rejected(self):
        """W-182 (pure LRF=7 + URR) has no evaluable range and must be rejected.

        LRF=7/URR evaluation was removed; a file whose every range is a
        parse-and-skip placeholder fails to load with a ValueError rather than
        loading as a silent zero-cross-section isotope.

        The library is pinned so this asserts against a fixed evaluation
        (W-182 is pure LRF=7 + URR in ENDF/B-VIII.1) rather than whatever the
        tool's default library becomes; requires network on a cold cache, like
        the other live-download tests in this suite.
        """
        with pytest.raises(ValueError, match="No evaluable resonance ranges"):
            _load_endf_or_skip(isotope="W-182", library="endf8.1")

    def test_mixed_evaluation_reports_skipped_ranges(self):
        """A mixed evaluation (U-238: resolved RM + URR) flags its skipped span.

        The URR range is parsed-and-skipped, so load_endf must set
        has_unevaluated_ranges and list the skipped URR span in skipped_ranges.
        """
        result = _load_endf_or_skip(isotope="U-238")
        assert result["has_unevaluated_ranges"] is True
        assert isinstance(result["skipped_ranges"], list)
        assert result["skipped_ranges"], "skipped_ranges must be non-empty"
        assert any("LRU=2" in span for span in result["skipped_ranges"])


# ---------------------------------------------------------------------------
# get_resonance_parameters
# ---------------------------------------------------------------------------


class TestGetResonanceParameters:
    def test_loaded_isotope(self):
        _load_endf_or_skip(isotope="Fe-56")
        result = get_resonance_parameters(isotope="Fe-56")
        assert result["z"] == 26
        assert result["a"] == 56
        assert result["awr"] > 50  # ~55.845

    def test_not_loaded(self):
        with pytest.raises(ValueError, match="not loaded"):
            get_resonance_parameters(isotope="U-238")

    def test_reports_unevaluated_ranges(self):
        _load_endf_or_skip(isotope="U-238")
        result = get_resonance_parameters(isotope="U-238")
        assert result["has_unevaluated_ranges"] is True
        assert any("LRU=2" in span for span in result["skipped_ranges"])


# ---------------------------------------------------------------------------
# compute_cross_sections
# ---------------------------------------------------------------------------


class TestComputeCrossSections:
    def test_basic(self):
        _load_endf_or_skip(isotope="Fe-56")
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
        _load_endf_or_skip(isotope="Fe-56")
        result = compute_transmission(
            isotope="Fe-56", thickness=0.01,
            energy_min=1.0, energy_max=100.0, n_points=50,
        )
        assert len(result["transmission"]) == 50
        # Transmission should be between 0 and 1
        assert all(0 <= v <= 1 for v in result["transmission"])

    def test_zero_thickness(self):
        _load_endf_or_skip(isotope="Fe-56")
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
        _load_endf_or_skip(isotope="Fe-56")
        result = forward_model(
            isotopes=[{"isotope": "Fe-56", "thickness": 0.01}],
            energy_min=1.0, energy_max=100.0, n_points=50,
        )
        assert len(result["transmission"]) == 50
        assert all(0 <= v <= 1 for v in result["transmission"])

    def test_multi_isotope(self):
        _load_endf_or_skip(isotope="Fe-56")
        _load_endf_or_skip(isotope="U-238")
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
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_compute_cross_sections_invalid_energy(self):
        """Energy validation catches bad inputs."""
        _load_endf_or_skip(isotope="Fe-56")
        with pytest.raises(ValueError, match="energy_min"):
            compute_cross_sections("Fe-56", energy_min=-1.0)
        with pytest.raises(ValueError, match="n_points"):
            compute_cross_sections("Fe-56", n_points=0)
        with pytest.raises(ValueError, match="energy_min.*less than"):
            compute_cross_sections("Fe-56", energy_min=100.0, energy_max=1.0)

    def test_compute_transmission_invalid_thickness(self):
        _load_endf_or_skip(isotope="Fe-56")
        with pytest.raises(ValueError, match="thickness"):
            compute_transmission("Fe-56", thickness=-1.0)

    def test_forward_model_missing_key(self):
        _load_endf_or_skip(isotope="Fe-56")
        with pytest.raises(ValueError, match="thickness"):
            forward_model([{"isotope": "Fe-56"}])  # missing thickness

    def test_list_isotopes_invalid_z(self):
        with pytest.raises(ValueError, match="z must be"):
            list_isotopes(0)




# ---------------------------------------------------------------------------
# Manifest-driven workflow tools
# ---------------------------------------------------------------------------


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


class TestCountsManifest:
    """Routing oracle for the counts-only process_resonance_dataset workflow."""

    @staticmethod
    def _calibration_block():
        return {
            "t0_us": 3.0,
            "flight_path_m": 25.0,
            "pulse": {
                "alpha0": 0.35,
                "alpha1": 0.05,
                "beta0": 0.25,
                "beta1": 0.0,
                "r": 0.15,
                "fwhm_squared_us2": 0.0,
                "energy_span_ev": [1.0, 200.0],
            },
        }

    @staticmethod
    def _counts_npz(tmp_path, bins=6):
        edges = np.linspace(200.0, 400.0, bins + 1)
        np.savez(
            tmp_path / "counts.npz",
            time_edges_us=edges,
            open_beam_counts=np.full(bins, 100.0),
            sample_counts=np.full(bins, 80.0),
        )

    def _analysis(self, mode="single_spectrum", **fit):
        block = {
            "mode": mode,
            "data": {"kind": "counts_npz", "path": "counts.npz"},
            "calibration": self._calibration_block(),
            "isotopes": [_synthetic_u238_entry()],
            "fit": {"fit_domain": "counts", "normalization": 1.0},
        }
        block["fit"].update(fit)
        return block

    @staticmethod
    def _fake_fit():
        region = SimpleNamespace(
            densities=[0.001],
            temperature_k=293.6,
            thickness_variance=0.04,
            capture_gammas=0.2,
            in_scatter=0.6,
            timing_offset_us=0.25,
            background=[0.01, 0.02, 0.0, 0.0, 0.0],
            gamma_transmission=0.7,
            overdispersion=[1.0, None],
        )
        return SimpleNamespace(
            regions=[region],
            covariance=None,
            deviance=1.0,
            converged=True,
            normalization=1.0,
            t0_us=3.0,
            flight_path_m=25.0,
            time_background=[0.0, 0.0, 1.0],
            room_background=0.0,
            alpha=[0.35, 0.05],
            beta=[0.25, 0.0],
            r=0.15,
            fwhm_squared_us2=0.0,
            r_high=0.3,
            detector_tail=(0.1, 1.5),
            spread_share=0.05,
        )

    def test_extract_and_validate_manifest(self, tmp_path):
        self._counts_npz(tmp_path)
        manifest = _write_json_frontmatter_manifest(tmp_path, self._analysis())
        extracted = extract_resonance_manifest(str(manifest))
        assert extracted["manifest_path"]
        assert validate_resonance_dataset(str(manifest))["valid"] is True

    def test_counts_with_resolution_rejected(self, tmp_path):
        self._counts_npz(tmp_path)
        analysis = self._analysis()
        analysis["resolution"] = {
            "kind": "gaussian",
            "flight_path_m": 25.0,
            "delta_t_us": 1.0,
            "delta_l_m": 0.0,
        }
        manifest = _write_json_frontmatter_manifest(tmp_path, analysis)
        report = validate_resonance_dataset(str(manifest))
        assert report["valid"] is False
        assert any(
            "counts input with instrument resolution" in error
            for error in report["errors"]
        )

    def test_transmission_kind_rejected(self, tmp_path):
        self._counts_npz(tmp_path)
        analysis = self._analysis()
        analysis["data"]["kind"] = "transmission_npz"
        manifest = _write_json_frontmatter_manifest(tmp_path, analysis)
        assert validate_resonance_dataset(str(manifest))["valid"] is False

    def test_process_single_spectrum_routes_to_fit_counts(self, tmp_path, monkeypatch):
        self._counts_npz(tmp_path)
        analysis = self._analysis(debye_temperature_k=240.0)
        analysis["calibration"]["pulse"]["detector_tail"] = [
            0.1,
            {"kind": "fitted", "value": 2.0},
        ]
        manifest = _write_json_frontmatter_manifest(tmp_path, analysis)
        calls = {}

        def fake_fit_counts(measurement, calibration):
            calls["measurement"] = measurement
            calls["calibration"] = calibration
            return self._fake_fit()

        monkeypatch.setattr(nereids, "fit_counts", fake_fit_counts)
        result = process_resonance_dataset(
            str(manifest), output_dir=str(tmp_path / "out")
        )
        assert result["success"] is True, result
        assert isinstance(calls["measurement"], nereids.Measurement)
        assert isinstance(calls["calibration"], nereids.Calibration)
        assert calls["measurement"].charge_ratio == 1.0
        lattice = calls["measurement"].regions[0].material.lattice
        assert lattice.effective_temperature_k(300.0) > 300.0
        assert result["results"]["mode"] == "single_spectrum"
        assert result["results"]["converged"] is True
        assert result["results"]["thickness_variance"] == 0.04
        assert result["results"]["capture_gammas"] == 0.2
        assert result["results"]["in_scatter"] == 0.6
        assert result["results"]["timing_offset_us"] == 0.25
        assert result["results"]["background"] == [0.01, 0.02, 0.0, 0.0, 0.0]
        assert result["results"]["gamma_transmission"] == 0.7
        assert result["results"]["overdispersion"] == [1.0, None]
        assert result["results"]["spread_share"] == 0.05
        assert result["results"]["r_high"] == 0.3
        assert calls["calibration"].pulse.r_high is None
        f, decay = calls["calibration"].pulse.detector_tail
        assert (f.kind, f.value, decay.kind, decay.value) == ("known", 0.1, "fitted", 2.0)
        assert result["results"]["detector_tail"] == [0.1, 1.5]

    def test_single_spectrum_preserves_time_background(self, tmp_path, monkeypatch):
        self._counts_npz(tmp_path)
        manifest = _write_json_frontmatter_manifest(
            tmp_path, self._analysis(time_background=[10.0, 50.0, 25.0])
        )
        captured = []

        def fit(measurement, calibration):
            captured.extend(v.value for v in measurement.time_background)
            return self._fake_fit()

        monkeypatch.setattr(nereids, "fit_counts", fit)
        result = process_resonance_dataset(str(manifest))
        assert result["success"], result
        assert captured == [10.0, 50.0, 25.0]

    def test_room_background_reaches_the_counts_fit(self, tmp_path, monkeypatch):
        room = {"rate": 0.5, "frames": [3.0e4, 2.5e4]}
        self._counts_npz(tmp_path)
        manifest = _write_json_frontmatter_manifest(
            tmp_path, self._analysis(room_background=room)
        )
        captured = []

        def fit(measurement, calibration):
            captured.append(
                (measurement.room_background.rate.value, measurement.room_background.frames)
            )
            return self._fake_fit()

        monkeypatch.setattr(nereids, "fit_counts", fit)
        assert process_resonance_dataset(str(manifest))["success"]
        assert captured == [(0.5, (3.0e4, 2.5e4))]
        bad = _write_json_frontmatter_manifest(
            tmp_path, self._analysis(room_background={"rate": 0.5})
        )
        report = validate_resonance_dataset(str(bad))
        assert not report["valid"]
        assert any("room background" in e for e in report["errors"])

    def test_counts_require_original_time_bin_edges(self, tmp_path, monkeypatch):
        np.savez(
            tmp_path / "counts.npz",
            energies_ev=np.linspace(5.0, 10.0, 6),
            open_beam_counts=np.arange(100.0, 106.0),
            sample_counts=np.arange(80.0, 86.0),
        )
        manifest = _write_json_frontmatter_manifest(tmp_path, self._analysis())
        monkeypatch.setattr(nereids, "fit_counts", lambda *args: self._fake_fit())
        result = process_resonance_dataset(str(manifest))
        assert not result["success"]
        assert "time_edges_us" in str(result)

    def test_map_preserves_time_background(self, tmp_path, monkeypatch):
        bins = 4
        np.savez(
            tmp_path / "cube.npz",
            time_edges_us=np.linspace(200.0, 400.0, bins + 1),
            open_beam_counts=np.full((bins, 2, 2), 100.0),
            sample_counts=np.full((bins, 2, 2), 80.0),
        )
        analysis = self._analysis(
            mode="density_map",
            time_background=[10.0, 50.0, 25.0],
            gamma_transmission=0.6,
        )
        analysis["data"] = {"kind": "counts_npz", "path": "cube.npz", "binning": 1}
        manifest = _write_json_frontmatter_manifest(tmp_path, analysis)
        captured = []

        def fit(map_measurement, calibration):
            captured.extend(v.value for v in map_measurement.time_background)
            captured.append(map_measurement.gamma_transmission.value)
            raise RuntimeError("measurement captured")

        monkeypatch.setattr(nereids, "fit_map", fit)
        assert validate_resonance_dataset(str(manifest))["valid"]
        process_resonance_dataset(str(manifest))
        assert captured == [10.0, 50.0, 25.0, 0.6]

    def test_unknown_fit_settings_are_rejected(self, tmp_path):
        self._counts_npz(tmp_path)
        manifest = _write_json_frontmatter_manifest(
            tmp_path, self._analysis(fix_densities=True, fit_alpha_1=True)
        )
        report = validate_resonance_dataset(str(manifest))
        assert not report["valid"]
        assert any("unsupported fit settings" in e for e in report["errors"])

    def test_material_retains_the_isotope_library(self, tmp_path, monkeypatch):
        from nereids.mcp.server import _material

        config = self._analysis()
        config["isotopes"][0]["library"] = "jendl5"
        original = nereids.Material
        libraries = []

        def material(**kwargs):
            libraries.append(kwargs["library"])
            return original(**kwargs)

        monkeypatch.setattr(nereids, "Material", material)
        _material(config, {}, tmp_path, config["fit"])
        assert libraries == ["jendl5"]

    def test_mixed_library_material_is_rejected(self, tmp_path):
        from nereids.mcp.server import _material

        config = self._analysis()
        entry = _synthetic_u238_entry()
        entry["library"] = "jendl5"
        config["isotopes"].append(entry)
        with pytest.raises(ValueError, match="one ENDF library"):
            _material(config, {}, tmp_path, config["fit"])

    def test_map_masks_follow_the_counts_roi(self, tmp_path, monkeypatch):
        counts = np.full((6, 2, 4), 100.0)
        empty = np.array([[False, False, True, False]] * 2)
        np.savez(
            tmp_path / "counts.npz", time_edges_us=np.linspace(200.0, 400.0, 7),
            open_beam_counts=counts, sample_counts=counts, empty=empty,
        )
        config = self._analysis(mode="density_map")
        config["data"]["roi"] = {"x0": 1, "y0": 0, "width": 2, "height": 2}
        manifest = _write_json_frontmatter_manifest(tmp_path, config)
        captured = []

        def fit(measurement, calibration):
            captured.extend([[p.kind for p in row] for row in measurement.patches()])
            raise RuntimeError("measurement captured")

        monkeypatch.setattr(nereids, "fit_map", fit)
        process_resonance_dataset(str(manifest))
        assert captured == [["sample", "outside"], ["sample", "outside"]]

    def test_process_density_map_routes_to_fit_map(self, tmp_path, monkeypatch):
        bins, height, width = 4, 2, 2
        np.savez(
            tmp_path / "cube.npz",
            time_edges_us=np.linspace(200.0, 400.0, bins + 1),
            open_beam_counts=np.full((bins, height, width), 100.0),
            sample_counts=np.full((bins, height, width), 80.0),
        )
        analysis = self._analysis(mode="density_map")
        analysis["data"] = {"kind": "counts_npz", "path": "cube.npz", "binning": 1}
        manifest = _write_json_frontmatter_manifest(tmp_path, analysis)
        calls = {}

        def fake_fit_map(map_measurement, calibration):
            calls["map"] = map_measurement
            zeros = np.zeros((height, width))
            return SimpleNamespace(
                patches=[[SimpleNamespace(kind="sample")] * width] * height,
                fits=[
                    [
                        SimpleNamespace(
                            thickness_variance=0.03,
                            capture_gammas=0.1,
                            in_scatter=0.5,
                            timing_offset_us=0.0,
                            gamma_transmission=0.8,
                            background=[0.01, 0.02, 0.03],
                        ),
                        None,
                    ],
                    [
                        SimpleNamespace(
                            thickness_variance=None,
                            capture_gammas=0.0,
                            in_scatter=0.0,
                            timing_offset_us=0.0,
                            gamma_transmission=1.0,
                            background=[0.0, 0.0, 0.0],
                        ),
                        None,
                    ],
                ],
                failed=[[None] * width] * height,
                empty=None,
                densities=[zeros],
                density_sd=[zeros],
                temperature_k=zeros,
                temperature_sd_k=zeros,
                overdispersion=[zeros, zeros],
                trusted=np.ones((height, width), dtype=bool),
                normalization=1.0,
                time_background=[0.0, 0.0, 1.0],
                room_background=0.0,
                t0_us=3.0,
                flight_path_m=25.0,
                alpha=[0.35, 0.05],
                beta=[0.25, 0.0],
                r=0.15,
                fwhm_squared_us2=0.0,
                shared=[],
                shared_covariance=None,
                measured_pulls=None,
                pulse_consistency=None,
                deviance=1.0,
                converged=True,
                unconverged=None,
                steps=1,
                n_fitted=height * width,
            )

        monkeypatch.setattr(nereids, "fit_map", fake_fit_map)
        result = process_resonance_dataset(
            str(manifest), output_dir=str(tmp_path / "out")
        )
        assert result["success"] is True, result
        assert isinstance(calls["map"], nereids.MapMeasurement)
        assert result["results"]["n_fitted"] == height * width
        saved = np.load(result["results"]["output_npz"], allow_pickle=True)
        np.testing.assert_array_equal(
            saved["thickness_variance"], [[0.03, np.nan], [np.nan, np.nan]]
        )
        np.testing.assert_array_equal(
            saved["capture_gammas"], [[0.1, np.nan], [0.0, np.nan]]
        )
        np.testing.assert_array_equal(
            saved["in_scatter"], [[0.5, np.nan], [0.0, np.nan]]
        )
        np.testing.assert_array_equal(
            saved["gamma_transmission"], [[0.8, np.nan], [1.0, np.nan]]
        )
        np.testing.assert_array_equal(saved["background"][0, 0], [0.01, 0.02, 0.03])
        assert np.isnan(saved["background"][0, 1]).all()
