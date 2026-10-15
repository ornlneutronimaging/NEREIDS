"""FastMCP server exposing NEREIDS nuclear data tools."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
try:
    from fastmcp import FastMCP
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "fastmcp is required for the MCP server. "
        "Install it with: pip install 'nereids[mcp]'"
    ) from exc

import nereids

mcp = FastMCP("nereids")

# In-memory registry of loaded ResonanceData objects.
# PyO3 objects are not JSON-serializable, so we keep them here
# and reference by isotope key (e.g., "Fe-56").
_registry: dict[str, nereids.ResonanceData] = {}

_MANIFEST_NAMES = (
    "manifest_intermediate.md",
    "smcp_manifest.md",
    "nereids_manifest.md",
    "nereids_mcp.json",
    "analysis.json",
)
_COUNTS_RESOLUTION_UNSUPPORTED = (
    "counts input with instrument resolution is not available: the counts fit "
    "uses the Ikeda-Carpenter pulse, not a tabulated resolution. Remove the "
    "resolution block."
)

_SAFE_KEY = re.compile(r"[^A-Za-z0-9_]+")




def _json_safe(value: Any) -> Any:
    """Convert numpy/path values to JSON-serializable Python objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _parse_scalar(value: str) -> Any:
    """Parse a small YAML-ish scalar without making PyYAML a hard dependency."""
    value = value.strip()
    if value == "":
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_frontmatter(raw: str) -> dict[str, Any]:
    """Parse JSON frontmatter, PyYAML frontmatter if available, or flat YAML."""
    text = raw.strip()
    if not text:
        return {}
    if text.startswith("{"):
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("manifest frontmatter JSON must be an object")
        return data

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        yaml = None
    if yaml is not None:
        data = yaml.safe_load(text)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError("manifest YAML frontmatter must be a mapping")
        return data

    parsed: dict[str, Any] = {}
    unsupported_nested = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line[:1].isspace() or stripped.startswith("- "):
            unsupported_nested = True
            continue
        if ":" not in stripped:
            raise ValueError(f"cannot parse manifest frontmatter line: {line!r}")
        key, value = stripped.split(":", 1)
        parsed[key.strip()] = _parse_scalar(value)
    if unsupported_nested:
        raise ValueError(
            "nested YAML frontmatter requires PyYAML, or use JSON frontmatter"
        )
    return parsed


def _find_manifest_path(dataset_path: Path) -> Path:
    """Find a supported sMCP/NEREIDS manifest file."""
    if dataset_path.is_file() and dataset_path.name in _MANIFEST_NAMES:
        return dataset_path
    root = dataset_path if dataset_path.is_dir() else dataset_path.parent
    for name in _MANIFEST_NAMES:
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No NEREIDS sMCP manifest found in {root}. "
        f"Expected one of: {', '.join(_MANIFEST_NAMES)}"
    )


def _read_manifest(dataset_path: str | Path) -> dict[str, Any]:
    """Read a manifest file and return metadata, frontmatter, and body."""
    path = Path(dataset_path).expanduser().resolve()
    manifest_path = _find_manifest_path(path)
    dataset_root = manifest_path.parent

    if manifest_path.suffix == ".json":
        frontmatter = json.loads(manifest_path.read_text())
        if not isinstance(frontmatter, dict):
            raise ValueError(f"{manifest_path} must contain a JSON object")
        body = ""
    else:
        content = manifest_path.read_text()
        if not content.lstrip().startswith("---"):
            raise ValueError(
                f"{manifest_path} must start with YAML/JSON frontmatter delimited by ---"
            )
        parts = content.split("---", 2)
        if len(parts) < 3:
            raise ValueError(f"{manifest_path} is missing closing frontmatter ---")
        frontmatter = _parse_frontmatter(parts[1])
        body = parts[2].strip()

    return {
        "dataset_path": str(dataset_root),
        "manifest_path": str(manifest_path),
        "frontmatter": frontmatter,
        "body": body,
    }


def _workflow_config(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the workflow config object from a parsed manifest."""
    frontmatter = manifest["frontmatter"]
    config = (
        frontmatter.get("analysis")
        or frontmatter.get("workflow")
        or frontmatter.get("processing")
        or frontmatter
    )
    if not isinstance(config, dict):
        raise ValueError("manifest analysis/workflow section must be an object")
    return config


def _resolve_path(base: Path, value: str | Path | None) -> Path | None:
    """Resolve a manifest path relative to the dataset root."""
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _normalise_mode(mode: Any) -> str:
    return str(mode or "").strip().lower().replace("-", "_")


def _normalise_kind(kind: Any, default: str) -> str:
    return str(kind or default).strip().lower().replace("-", "_")


def _get_data_config(config: dict[str, Any]) -> dict[str, Any]:
    data = config.get("data") or config.get("input") or {}
    if isinstance(data, str):
        return {"path": data}
    if not isinstance(data, dict):
        raise ValueError("workflow data/input section must be an object or path string")
    return data


def _get_fit_config(config: dict[str, Any]) -> dict[str, Any]:
    fit = config.get("fit") or config.get("fitting") or {}
    if not isinstance(fit, dict):
        raise ValueError("workflow fit/fitting section must be an object")
    return fit


def _get_isotope_entries(config: dict[str, Any], frontmatter: dict[str, Any]) -> list[Any]:
    entries = config.get("isotopes") or config.get("nuclides")
    if entries is None and "isotope" in config:
        entries = [{"isotope": config["isotope"]}]
    if entries is None and "isotope" in frontmatter:
        entries = [{"isotope": frontmatter["isotope"]}]
    if entries is None:
        return []
    if isinstance(entries, (str, dict)):
        return [entries]
    if not isinstance(entries, list):
        raise ValueError("isotopes must be a string, object, or list")
    return entries


def _array_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    stats: dict[str, Any] = {
        "shape": list(arr.shape),
        "finite_count": int(finite.size),
        "nan_count": int(np.isnan(arr).sum()),
    }
    if finite.size:
        stats.update(
            {
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite)),
            }
        )
    return stats


def _validate_array(name: str, values: Any, ndim: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(arr)


def _require_same_shape(
    reference_name: str,
    reference: np.ndarray,
    *others: tuple[str, np.ndarray],
) -> None:
    """Require arrays that share a physical axis/layout to have equal shape."""
    for name, arr in others:
        if arr.shape != reference.shape:
            raise ValueError(
                f"{reference_name} and {name} must have matching shapes, "
                f"got {reference.shape} and {arr.shape}"
            )


def _finite_float_or_none(value: Any) -> float | None:
    scalar = float(value)
    return scalar if math.isfinite(scalar) else None




def _npz_arrays(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"data file does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _require_npz_keys(
    arrays: dict[str, np.ndarray],
    keys: list[str],
    *,
    context: str,
) -> None:
    missing = [key for key in keys if key not in arrays]
    if missing:
        available = ", ".join(sorted(arrays)) or "<none>"
        required = ", ".join(keys)
        missing_text = ", ".join(missing)
        raise ValueError(
            f"{context} is missing required key(s): {missing_text}. "
            f"Required keys: {required}. Available keys: {available}"
        )




def _load_isotopes(
    entries: list[Any],
    base: Path,
    default_library: str = "endf8.1",
) -> tuple[list[tuple[nereids.ResonanceData, float]], list[str]]:
    """Load isotope entries from ENDF files, ENDF cache/retrieval, or synthetic specs."""
    loaded: list[tuple[nereids.ResonanceData, float]] = []
    names: list[str] = []
    if not entries:
        raise ValueError("at least one isotope must be configured")

    for raw_entry in entries:
        entry = {"isotope": raw_entry} if isinstance(raw_entry, str) else dict(raw_entry)
        isotope = entry.get("isotope") or entry.get("name")
        library = str(entry.get("library", default_library))
        initial_density = float(
            entry.get("initial_density", entry.get("density", 0.001))
        )
        if not math.isfinite(initial_density) or initial_density < 0:
            raise ValueError(
                f"initial density for {isotope or entry!r} must be finite and non-negative"
            )

        synthetic = entry.get("synthetic_resonance") or entry.get("resonance_data")
        endf_file = entry.get("endf_file") or entry.get("endf_path")
        if synthetic is not None:
            spec = dict(synthetic)
            if isotope and ("z" not in spec or "a" not in spec):
                parsed = nereids.parse_isotope_str(str(isotope))
                if parsed is not None:
                    spec.setdefault("z", parsed[0])
                    spec.setdefault("a", parsed[1])
            resonances = spec.get("resonances")
            if resonances is None:
                resonance = spec.get("resonance")
                resonances = [resonance] if resonance is not None else None
            if resonances is None:
                raise ValueError(f"synthetic isotope {isotope!r} needs resonances")
            data = nereids.create_resonance_data(
                z=int(spec["z"]),
                a=int(spec["a"]),
                awr=float(spec["awr"]),
                scattering_radius=float(spec["scattering_radius"]),
                resonances=[tuple(map(float, r)) for r in resonances],
                target_spin=float(spec.get("target_spin", 0.0)),
                formalism=spec.get("formalism"),
            )
        elif endf_file is not None:
            path = _resolve_path(base, endf_file)
            if path is None or not path.exists():
                raise FileNotFoundError(f"ENDF file not found for {isotope}: {path}")
            data = nereids.load_endf_file(str(path))
        else:
            parsed = None
            if isotope:
                parsed = nereids.parse_isotope_str(str(isotope))
            if parsed is None and "z" in entry and "a" in entry:
                parsed = (int(entry["z"]), int(entry["a"]))
            if parsed is None:
                raise ValueError(
                    f"Cannot parse isotope entry {entry!r}; provide isotope, z/a, "
                    "endf_file, or synthetic_resonance"
                )
            data = nereids.load_endf(parsed[0], parsed[1], library=library)

        name = str(isotope or _isotope_key(int(data.z), int(data.a)))
        loaded.append((data, initial_density))
        names.append(name)
    return loaded, names


def _as_value(spec: Any, default_kind: str) -> nereids.Value:
    """Build a Value from a manifest entry.

    A number is a known value.  A mapping takes 'kind' ("known", "fitted",
    "within" or "measured") and the matching fields: 'value' for known and
    fitted, 'value'/'lower'/'upper' for within, and 'value'/'sd' for measured.
    """
    if isinstance(spec, bool) or not isinstance(spec, (int, float, dict)):
        raise ValueError("Value spec must be a number or an object")
    if isinstance(spec, (int, float)):
        return nereids.Value.known(float(spec))
    kind = _normalise_kind(spec.get("kind", default_kind), default_kind)
    value = spec.get("value")
    if kind == "known":
        return nereids.Value.known(float(value))
    if kind == "fitted":
        return nereids.Value.fitted(float(value))
    if kind == "within":
        return nereids.Value.within(
            float(value), float(spec["lower"]), float(spec["upper"])
        )
    if kind == "measured":
        return nereids.Value.measured(float(value), float(spec["sd"]))
    raise ValueError(f"unknown Value kind: {kind}")


def _value_or_number(spec: Any, default: float) -> nereids.Value:
    if spec is None:
        return nereids.Value.known(default)
    return _as_value(spec, "known")


def _background_values(spec: Any) -> list:
    """A [b0, b1, b2, d, f] background, each a Value; absent means none."""
    if spec is None:
        return [nereids.Value.known(0.0)] * 5
    if isinstance(spec, dict):
        missing = [key for key in ("b0", "b1", "b2") if key not in spec]
        if missing:
            raise ValueError(f"background is missing {', '.join(missing)}")
        return [
            _as_value(spec.get(key, 0.0), "known") for key in ("b0", "b1", "b2", "d", "f")
        ]
    if isinstance(spec, (list, tuple)) and len(spec) in (3, 5):
        values = [_as_value(value, "known") for value in spec]
        return values + [nereids.Value.known(0.0)] * (5 - len(values))
    raise ValueError(
        "background must be [b0, b1, b2], [b0, b1, b2, d, f] or an object with "
        "b0/b1/b2 and optional d/f"
    )


def _time_background_values(spec: Any) -> list | None:
    """The additive time background's [C, A, tau], or none."""
    if spec is None:
        return None
    if not isinstance(spec, (list, tuple)) or len(spec) != 3:
        raise ValueError("time_background must be [C, A, tau]")
    return [_as_value(value, "known") for value in spec]


def _room_background(spec: Any) -> Any:
    """The room background {rate, frames: [open-beam, sample]}, or none."""
    if spec is None:
        return None
    if not isinstance(spec, dict) or "rate" not in spec or "frames" not in spec:
        raise ValueError("room_background must be an object with rate and frames")
    frames = spec["frames"]
    if not isinstance(frames, (list, tuple)) or len(frames) != 2:
        raise ValueError("room_background frames must be [open-beam, sample]")
    return nereids.RoomBackground(
        _as_value(spec["rate"], "known"), (float(frames[0]), float(frames[1]))
    )


def _temperature_value(fit_config: dict[str, Any]) -> nereids.Value:
    """The material temperature; a number is fitted only with fit_temperature."""
    spec = fit_config.get("temperature")
    if spec is None:
        spec = fit_config.get("temperature_k", 293.6)
    if isinstance(spec, bool):
        raise ValueError("temperature must be a number or an object")
    if isinstance(spec, (int, float)):
        if bool(fit_config.get("fit_temperature", False)):
            return nereids.Value.fitted(float(spec))
        return nereids.Value.known(float(spec))
    return _as_value(spec, "fitted")


def _calibration(config: dict[str, Any], data_config: dict[str, Any]) -> nereids.Calibration:
    """Build the timing offset, flight path and Ikeda-Carpenter pulse."""
    block = config.get("calibration") or data_config.get("calibration") or {}
    if not isinstance(block, dict):
        raise ValueError("calibration must be an object")
    pulse_config = block.get("pulse") or {}
    if not isinstance(pulse_config, dict):
        raise ValueError("calibration.pulse must be an object")
    span = pulse_config.get("energy_span_ev")
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        raise ValueError("calibration.pulse.energy_span_ev must be [low_ev, high_ev]")
    line_span = pulse_config.get("line_span_ev")
    pulse = nereids.Pulse(
        alpha0=_value_or_number(pulse_config.get("alpha0"), 0.0),
        alpha1=_value_or_number(pulse_config.get("alpha1"), 0.0),
        beta0=_value_or_number(pulse_config.get("beta0"), 0.0),
        beta1=_value_or_number(pulse_config.get("beta1"), 0.0),
        r=_value_or_number(pulse_config.get("r"), 0.0),
        fwhm_squared_us2=_value_or_number(pulse_config.get("fwhm_squared_us2"), 0.0),
        energy_span_ev=(float(span[0]), float(span[1])),
        n_tau=int(pulse_config.get("n_tau", 600)),
        line_span_ev=None if line_span is None else (float(line_span[0]), float(line_span[1])),
    )
    return nereids.Calibration(
        t0_us=_value_or_number(block.get("t0_us"), 0.0),
        flight_path_m=_value_or_number(block.get("flight_path_m"), 25.0),
        pulse=pulse,
    )


def _load_time_edges(
    base: Path,
    data_config: dict[str, Any],
    arrays: dict[str, np.ndarray] | None,
    n_bins: int,
) -> np.ndarray:
    """Original detector time-bin edges in microseconds, in the counts' order."""
    key = str(data_config.get("time_edges_key", "time_edges_us"))
    if arrays is None or key not in arrays:
        raise ValueError(f"raw counts require original time_edges_us (NPZ key {key!r})")
    edges = _validate_array("time_edges_us", arrays[key], ndim=1)
    if edges.size != n_bins + 1:
        raise ValueError(
            f"time_edges_us must have {n_bins + 1} edges for {n_bins} bins, got {edges.size}"
        )
    if not np.all(np.diff(edges) > 0):
        raise ValueError("time_edges_us must be strictly increasing in the counts' order")
    return np.ascontiguousarray(edges, dtype=np.float64)


def _material(
    config: dict[str, Any],
    frontmatter: dict[str, Any],
    base: Path,
    fit_config: dict[str, Any],
) -> tuple[nereids.Material, list[str], list[float]]:
    """The fitted material and its isotope names and initial densities."""
    entries = _get_isotope_entries(config, frontmatter)
    default_library = str(config.get("library", "endf8.1"))
    libraries = {
        str(entry.get("library", default_library))
        if isinstance(entry, dict) else default_library
        for entry in entries
    }
    if len(libraries) > 1:
        raise ValueError("a counts material must use one ENDF library for every isotope")
    library = next(iter(libraries), default_library)
    isotopes, names = _load_isotopes(
        entries, base, default_library=default_library
    )
    initial = [density for _, density in isotopes]
    material = nereids.Material(
        isotopes=[(data, nereids.Value.fitted(density)) for data, density in isotopes],
        temperature_k=_temperature_value(fit_config),
        thickness_variance=_value_or_number(fit_config.get("thickness_variance"), 0.0),
        library=library,
    )
    return material, names, initial



def _safe_npz_key(name: str) -> str:
    key = _SAFE_KEY.sub("_", name).strip("_")
    return key or "isotope"


def _apply_roi(cube: np.ndarray, roi: dict[str, Any] | None) -> np.ndarray:
    if not roi:
        return cube
    x0 = int(roi.get("x0", roi.get("x", 0)))
    y0 = int(roi.get("y0", roi.get("y", 0)))
    width = int(roi.get("width", cube.shape[2] - x0))
    height = int(roi.get("height", cube.shape[1] - y0))
    if x0 < 0 or y0 < 0 or width <= 0 or height <= 0:
        raise ValueError(f"invalid ROI: {roi}")
    if x0 + width > cube.shape[2] or y0 + height > cube.shape[1]:
        raise ValueError(f"ROI {roi} exceeds cube bounds {cube.shape[1:]}")
    return np.ascontiguousarray(cube[:, y0 : y0 + height, x0 : x0 + width])


def _limit_pixels(
    cube: np.ndarray, max_pixels: int | None
) -> tuple[np.ndarray, dict[str, Any] | None]:
    if max_pixels is None or cube.shape[1] * cube.shape[2] <= max_pixels:
        return cube, None
    if max_pixels < 1:
        raise ValueError("max_pixels must be positive")
    h, w = cube.shape[1], cube.shape[2]
    side = max(1, int(math.sqrt(max_pixels)))
    new_h = min(h, side)
    new_w = min(w, max(1, max_pixels // new_h))
    cropped = np.ascontiguousarray(cube[:, :new_h, :new_w])
    return cropped, {
        "original_shape": [int(cube.shape[0]), int(h), int(w)],
        "cropped_shape": [int(cropped.shape[0]), int(new_h), int(new_w)],
        "reason": f"max_pixels={max_pixels}",
    }


def _patch_values(fits: list[list[Any]], name: str) -> np.ndarray:
    """Each patch's fitted `name`, NaN where the patch has no fit or no value."""
    return np.asarray(
        [
            [
                np.nan if fit is None or getattr(fit, name) is None else float(getattr(fit, name))
                for fit in row
            ]
            for row in fits
        ],
        dtype=float,
    )


def _fit_result_summary(result: Any, names: list[str]) -> dict[str, Any]:
    region = result.regions[0] if result.regions else None
    densities = np.asarray(region.densities if region is not None else [], dtype=float)
    covariance = result.covariance
    deviations = None
    if covariance is not None and np.asarray(covariance).size:
        deviations = np.sqrt(np.diag(np.asarray(covariance, dtype=float)))
    entries = []
    for index, name in enumerate(names):
        entries.append(
            {
                "isotope": name,
                "density_atoms_per_barn": _finite_float_or_none(densities[index])
                if index < densities.size
                else None,
                "uncertainty_atoms_per_barn": _finite_float_or_none(deviations[index])
                if deviations is not None and index < deviations.size
                else None,
            }
        )
    return {
        "density_fits": entries,
        "deviance": _finite_float_or_none(result.deviance),
        "converged": bool(result.converged),
        "normalization": _finite_float_or_none(result.normalization),
        "t0_us": _finite_float_or_none(result.t0_us),
        "flight_path_m": _finite_float_or_none(result.flight_path_m),
        "temperature_k": None
        if region is None or region.temperature_k is None
        else _finite_float_or_none(region.temperature_k),
        "thickness_variance": None
        if region is None or region.thickness_variance is None
        else _finite_float_or_none(region.thickness_variance),
        "time_background": [_finite_float_or_none(v) for v in result.time_background],
        "room_background": _finite_float_or_none(result.room_background),
        "alpha": _json_safe(list(result.alpha)),
        "beta": _json_safe(list(result.beta)),
        "r": _finite_float_or_none(result.r),
        "fwhm_squared_us2": _finite_float_or_none(result.fwhm_squared_us2),
    }



def _process_single_spectrum(
    base: Path,
    manifest: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    frontmatter = manifest["frontmatter"]
    config = _workflow_config(manifest)
    data_config = _get_data_config(config)
    fit_config = _get_fit_config(config)
    data_path = _resolve_path(base, data_config.get("path"))
    if data_path is None:
        raise ValueError("single_spectrum workflow requires data.path")
    if data_path.suffix != ".npz":
        raise ValueError("single_spectrum counts data must be a .npz file")
    arrays = _npz_arrays(data_path)
    sample_key = data_config.get("sample_key", "sample_counts")
    open_beam_key = data_config.get("open_beam_key", "open_beam_counts")
    _require_npz_keys(
        arrays, [sample_key, open_beam_key], context="single_spectrum counts data"
    )
    sample = _validate_array("sample_counts", arrays[sample_key])
    open_beam = _validate_array("open_beam_counts", arrays[open_beam_key])
    _require_same_shape("sample_counts", sample, ("open_beam_counts", open_beam))
    pixels = float(data_config.get("pixels", 1.0))
    if sample.ndim == 3:
        pixels = float(data_config.get("pixels", sample.shape[1] * sample.shape[2]))
        sample = np.ascontiguousarray(sample.sum(axis=(1, 2)))
        open_beam = np.ascontiguousarray(open_beam.sum(axis=(1, 2)))
    if sample.ndim != 1 or open_beam.ndim != 1:
        raise ValueError("single_spectrum counts arrays must be 1D or 3D")
    calibration = _calibration(config, data_config)
    edges = _load_time_edges(base, data_config, arrays, sample.size)
    material, names, _ = _material(config, frontmatter, base, fit_config)
    charge_ratio = float(data_config.get("pc_ratio", arrays.get("pc_ratio", 1.0)))
    region = nereids.Region(
        open_counts=open_beam,
        sample_counts=sample,
        pixels=pixels,
        background=_background_values(fit_config.get("background")),
        gamma_transmission=_value_or_number(fit_config.get("gamma_transmission"), 1.0),
        material=material,
    )
    measurement = nereids.Measurement(
        time_edges_us=edges,
        charge_ratio=charge_ratio,
        normalization=_value_or_number(fit_config.get("normalization"), 1.0),
        regions=[region],
        time_background=_time_background_values(fit_config.get("time_background")),
        room_background=_room_background(fit_config.get("room_background")),
    )
    result = nereids.fit_counts(measurement, calibration)

    output_dir.mkdir(parents=True, exist_ok=True)
    result_npz = output_dir / "nereids_spectrum_fit.npz"
    fitted = result.regions[0].densities if result.regions else []
    np.savez_compressed(
        result_npz,
        time_edges_us=edges,
        sample_counts=sample,
        open_beam_counts=open_beam,
        fitted_densities=np.asarray(fitted, dtype=float),
        isotope_names=np.asarray(names),
    )
    summary = _fit_result_summary(result, names)
    summary.update(
        {
            "mode": "single_spectrum",
            "input_kind": "counts",
            "data_path": str(data_path),
            "output_npz": str(result_npz),
            "time_range_us": [float(edges[0]), float(edges[-1])],
            "n_bins": int(sample.size),
        }
    )
    return summary



def _process_density_map(
    base: Path,
    manifest: dict[str, Any],
    output_dir: Path,
    max_pixels: int | None,
) -> dict[str, Any]:
    frontmatter = manifest["frontmatter"]
    config = _workflow_config(manifest)
    data_config = _get_data_config(config)
    fit_config = _get_fit_config(config)
    kind = _normalise_kind(data_config.get("kind"), "counts_npz")
    path = _resolve_path(base, data_config.get("path"))
    arrays = _npz_arrays(path) if path is not None and path.suffix == ".npz" else None
    crop: dict[str, Any] | None = None

    if kind in {"counts_npz", "counts"}:
        if arrays is None:
            raise ValueError("counts_npz data requires a .npz data.path")
        sample_key = data_config.get("sample_key", "sample_counts")
        open_beam_key = data_config.get("open_beam_key", "open_beam_counts")
        _require_npz_keys(
            arrays, [sample_key, open_beam_key], context="density_map counts data"
        )
        sample = _validate_array("sample_counts", arrays[sample_key], ndim=3)
        open_beam = _validate_array("open_beam_counts", arrays[open_beam_key], ndim=3)
        _require_same_shape("sample_counts", sample, ("open_beam_counts", open_beam))
        roi = data_config.get("roi") or config.get("roi")
        sample = _apply_roi(sample, roi)
        open_beam = _apply_roi(open_beam, roi)
        sample, crop = _limit_pixels(sample, max_pixels)
        open_beam = open_beam[:, : sample.shape[1], : sample.shape[2]]
        edges = _load_time_edges(base, data_config, arrays, sample.shape[0])
    elif kind in {"nexus_histogram", "nexus"}:
        sample_path = _resolve_path(base, data_config.get("sample_path"))
        open_beam_path = _resolve_path(base, data_config.get("open_beam_path"))
        if sample_path is None or open_beam_path is None:
            raise ValueError("nexus data requires sample_path and open_beam_path")
        sample_data = nereids.load_nexus_histogram(str(sample_path))
        open_data = nereids.load_nexus_histogram(str(open_beam_path))
        sample = _validate_array("sample_counts", sample_data.counts, ndim=3)
        open_beam = _validate_array("open_beam_counts", open_data.counts, ndim=3)
        _require_same_shape("sample_counts", sample, ("open_beam_counts", open_beam))
        sample_edges = _validate_array(
            "sample_tof_edges_us", sample_data.tof_edges_us, ndim=1
        )
        open_edges = _validate_array(
            "open_beam_tof_edges_us", open_data.tof_edges_us, ndim=1
        )
        if sample_edges.shape != open_edges.shape or not np.allclose(
            sample_edges, open_edges
        ):
            raise ValueError("nexus sample and open_beam TOF bin edges must match")
        roi = data_config.get("roi") or config.get("roi")
        sample = _apply_roi(sample, roi)
        open_beam = _apply_roi(open_beam, roi)
        sample, crop = _limit_pixels(sample, max_pixels)
        open_beam = open_beam[:, : sample.shape[1], : sample.shape[2]]
        edges = np.ascontiguousarray(sample_edges, dtype=np.float64)
    else:
        raise ValueError(
            f"unsupported density_map data kind: {kind}; raw counts are required"
        )

    if edges.size != sample.shape[0] + 1:
        raise ValueError(
            f"time_edges_us must have {sample.shape[0] + 1} edges, got {edges.size}"
        )
    height, width = sample.shape[1], sample.shape[2]
    excluded_key = str(data_config.get("excluded_key", "excluded"))
    sample_mask_key = str(data_config.get("sample_mask_key", "sample"))
    empty_key = str(data_config.get("empty_key", "empty"))

    def mask(key: str, default: np.ndarray) -> np.ndarray:
        if arrays is None or key not in arrays:
            return default
        original = np.asarray(arrays[key], dtype=bool)
        if original.shape != arrays[sample_key].shape[1:]:
            raise ValueError("excluded/sample/empty masks must match the counts spatial shape")
        selected = _apply_roi(original[np.newaxis], roi)[0]
        return np.ascontiguousarray(selected[:height, :width])

    excluded = mask(excluded_key, np.zeros((height, width), dtype=bool))
    empty = mask(empty_key, np.zeros((height, width), dtype=bool))
    behind = mask(sample_mask_key, ~empty)

    charge_ratio = float(
        data_config.get("pc_ratio", arrays.get("pc_ratio", 1.0) if arrays is not None else 1.0)
    )
    material, names, _ = _material(config, frontmatter, base, fit_config)
    calibration = _calibration(config, data_config)
    map_measurement = nereids.MapMeasurement(
        time_edges_us=edges,
        open_counts=open_beam,
        sample_counts=sample,
        excluded=np.ascontiguousarray(excluded),
        sample=np.ascontiguousarray(behind),
        empty=np.ascontiguousarray(empty),
        material=material,
        charge_ratio=charge_ratio,
        normalization=_value_or_number(fit_config.get("normalization"), 1.0),
        background=_background_values(fit_config.get("background")),
        binning=int(data_config.get("binning", 1)),
        empty_background=_background_values(fit_config.get("empty_background")),
        time_background=_time_background_values(fit_config.get("time_background")),
        gamma_transmission=_value_or_number(fit_config.get("gamma_transmission"), 1.0),
        room_background=_room_background(fit_config.get("room_background")),
    )
    result = nereids.fit_map(map_measurement, calibration)

    output_dir.mkdir(parents=True, exist_ok=True)
    result_npz = output_dir / "nereids_density_map.npz"
    patch_kinds = [[patch.kind for patch in row] for row in result.patches]
    arrays_to_save: dict[str, Any] = {
        "time_edges_us": edges,
        "patch_kinds": np.asarray(patch_kinds, dtype=object),
        "trusted": np.asarray(result.trusted),
        "temperature_k": np.asarray(result.temperature_k),
        "thickness_variance": _patch_values(result.fits, "thickness_variance"),
        "isotope_names": np.asarray(names),
    }
    density_stats = []
    for name, density_map, uncertainty_map in zip(
        names, result.densities, result.density_sd
    ):
        key = _safe_npz_key(name)
        density_arr = np.asarray(density_map)
        uncertainty_arr = np.asarray(uncertainty_map)
        arrays_to_save[f"density_{key}"] = density_arr
        arrays_to_save[f"uncertainty_{key}"] = uncertainty_arr
        density_stats.append(
            {
                "isotope": name,
                "density_atoms_per_barn": _array_stats(density_arr),
                "uncertainty_atoms_per_barn": _array_stats(uncertainty_arr),
            }
        )
    np.savez_compressed(result_npz, **arrays_to_save)
    return {
        "mode": "density_map",
        "input_kind": "counts",
        "data_path": None if path is None else str(path),
        "output_npz": str(result_npz),
        "time_range_us": [float(edges[0]), float(edges[-1])],
        "n_bins": int(sample.shape[0]),
        "map_shape": list(np.asarray(result.trusted).shape),
        "n_fitted": int(result.n_fitted),
        "time_background": [_finite_float_or_none(v) for v in result.time_background],
        "room_background": _finite_float_or_none(result.room_background),
        "converged": bool(result.converged),
        "unconverged": result.unconverged,
        "density_maps": density_stats,
        "crop": crop,
    }



def _validate_workflow(manifest: dict[str, Any]) -> dict[str, Any]:
    base = Path(manifest["dataset_path"])
    frontmatter = manifest["frontmatter"]
    config = _workflow_config(manifest)
    data_config = _get_data_config(config)
    fit_config = _get_fit_config(config)
    errors: list[str] = []
    warnings: list[str] = []

    tool = str(frontmatter.get("tool", config.get("tool", "nereids"))).lower()
    if tool not in {"nereids", "nereids-mcp", "smcp-nereids"}:
        errors.append(
            "manifest tool must be one of 'nereids', 'nereids-mcp', "
            f"or 'smcp-nereids', got {tool!r}"
        )

    mode = _normalise_mode(config.get("mode", config.get("analysis_mode")))
    if mode not in {"single_spectrum", "fit_spectrum", "spectrum", "density_map", "spatial_map"}:
        errors.append(
            "analysis mode must be one of single_spectrum, fit_spectrum, "
            "spectrum, density_map, or spatial_map"
        )

    effective_kind = _normalise_kind(data_config.get("kind"), "counts_npz")
    if effective_kind not in {"counts_npz", "counts", "nexus_histogram", "nexus"}:
        errors.append(
            f"unsupported data kind: {effective_kind}; the fit is counts-only"
        )

    has_path = data_config.get("path") not in (None, "")
    has_sample_path = data_config.get("sample_path") not in (None, "")
    has_open_beam_path = data_config.get("open_beam_path") not in (None, "")
    if mode in {"single_spectrum", "fit_spectrum", "spectrum"} and not has_path:
        errors.append("single_spectrum workflow requires data.path")
    elif mode in {"density_map", "spatial_map"}:
        if effective_kind in {"nexus_histogram", "nexus"}:
            if not has_sample_path or not has_open_beam_path:
                errors.append(
                    "nexus density_map workflow requires sample_path and open_beam_path"
                )
        elif not has_path:
            errors.append(f"{effective_kind} density_map workflow requires data.path")

    data_paths = []
    for key in ("path", "sample_path", "open_beam_path"):
        value = data_config.get(key)
        if value in (None, ""):
            continue
        path = _resolve_path(base, value)
        data_paths.append({"key": key, "path": str(path), "exists": path.exists()})
        if not path.exists():
            errors.append(f"{key} does not exist: {path}")

    entries = _get_isotope_entries(config, frontmatter)
    if not entries:
        errors.append("no isotopes configured")
    for entry in entries:
        if isinstance(entry, str):
            continue
        endf_file = entry.get("endf_file") or entry.get("endf_path")
        if endf_file is not None:
            path = _resolve_path(base, endf_file)
            if path is None or not path.exists():
                errors.append(f"ENDF file does not exist: {path}")
        elif not (
            entry.get("synthetic_resonance")
            or entry.get("resonance_data")
            or entry.get("isotope")
            or ("z" in entry and "a" in entry)
        ):
            errors.append(f"isotope entry is missing isotope/z/a/endf_file: {entry}")

    fit_domain = str(fit_config.get("fit_domain", "counts")).lower()
    if fit_domain != "counts":
        errors.append("the fit is counts-only: fit_domain must be 'counts'")
    supported_fit = {
        "fit_domain", "normalization", "background", "empty_background",
        "temperature", "temperature_k", "fit_temperature",
        "gamma_transmission", "time_background", "room_background",
        "thickness_variance",
    }
    unsupported_fit = sorted(set(fit_config) - supported_fit)
    if unsupported_fit:
        errors.append(f"unsupported fit settings: {', '.join(unsupported_fit)}")

    try:
        _time_background_values(fit_config.get("time_background"))
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"time background is invalid: {exc}")
    try:
        _room_background(fit_config.get("room_background"))
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"room background is invalid: {exc}")

    resolution = config.get("resolution")
    resolution_kind: str | None = None
    if isinstance(resolution, dict):
        resolution_kind = _normalise_kind(resolution.get("kind", "none"), "none")
    elif isinstance(resolution, str):
        resolution_kind = _normalise_kind(resolution, "none")
    elif resolution:
        errors.append(
            "resolution must be an object or kind string, got "
            f"{type(resolution).__name__}"
        )
    if resolution_kind not in {None, "none", "disabled", "false"}:
        errors.append(_COUNTS_RESOLUTION_UNSUPPORTED)

    try:
        _calibration(config, data_config)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"calibration is invalid: {exc}")

    if effective_kind in {"counts_npz", "counts"} and has_path:
        resolved = _resolve_path(base, data_config.get("path"))
        if resolved is not None and resolved.suffix != ".npz":
            errors.append("counts_npz data.path must be a .npz file")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "manifest_path": manifest["manifest_path"],
        "dataset_path": manifest["dataset_path"],
        "mode": mode or None,
        "data_kind": effective_kind,
        "fit_domain": "counts",
        "data_paths": data_paths,
        "isotopes": _json_safe(entries),
    }



def _isotope_key(z: int, a: int) -> str:
    """Build a registry key like 'Fe-56'."""
    symbol = nereids.element_symbol(z) or f"Z{z}"
    return f"{symbol}-{a}"


def _validate_energy_grid(energy_min: float, energy_max: float, n_points: int) -> None:
    """Validate energy grid parameters."""
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")
    if not math.isfinite(energy_min) or not math.isfinite(energy_max):
        raise ValueError("energy_min and energy_max must be finite")
    if energy_min <= 0 or energy_max <= 0:
        raise ValueError("energy_min and energy_max must be positive (neutron energies)")
    if energy_min >= energy_max:
        raise ValueError(
            f"energy_min ({energy_min}) must be less than energy_max ({energy_max})"
        )


@mcp.tool()
def list_isotopes(z: int) -> list[dict]:
    """List naturally occurring isotopes for element Z with their abundances.

    Args:
        z: Atomic number (e.g., 26 for iron).

    Returns:
        List of dicts with keys: z, a, symbol, abundance.
    """
    if not isinstance(z, int) or z < 1 or z > 118:
        raise ValueError(f"z must be an integer between 1 and 118, got {z}")
    symbol = nereids.element_symbol(z) or f"Z{z}"
    isotopes = nereids.natural_isotopes(z)
    return [
        {"z": z, "a": za[1], "symbol": f"{symbol}-{za[1]}", "abundance": ab}
        for (za, ab) in isotopes
    ]


@mcp.tool()
def load_endf(
    isotope: str,
    library: str = "endf8.1",
) -> dict:
    """Load ENDF resonance data for an isotope and store it in the registry.

    Args:
        isotope: Isotope string like "Fe-56", "U-238", "Pu-239".
        library: ENDF library name (default: "endf8.1").

    Returns:
        Summary dict with keys: isotope, z, a, n_resonances, scattering_radius,
        target_spin, l_values, has_unevaluated_ranges. When
        has_unevaluated_ranges is True (mixed evaluations, e.g. U-238), a
        skipped_ranges list of the parsed-but-not-evaluated spans is included.
    """
    parsed = nereids.parse_isotope_str(isotope)
    if parsed is None:
        raise ValueError(f"Cannot parse isotope string: {isotope!r}")
    z, a = parsed
    data = nereids.load_endf(z, a, library=library)
    key = _isotope_key(z, a)
    _registry[key] = data
    result = {
        "isotope": key,
        "z": data.z,
        "a": data.a,
        "n_resonances": data.n_resonances,
        "scattering_radius": data.scattering_radius,
        "target_spin": data.target_spin,
        "l_values": data.l_values,
        "has_unevaluated_ranges": data.has_unevaluated_ranges,
    }
    if data.has_unevaluated_ranges:
        result["skipped_ranges"] = data.skipped_ranges
    return result


@mcp.tool()
def get_resonance_parameters(isotope: str) -> dict:
    """Get resonance parameters for a loaded isotope.

    Args:
        isotope: Isotope key (e.g., "Fe-56"). Must be loaded first via load_endf.

    Returns:
        Dict with keys: isotope, z, a, awr, target_spin, scattering_radius,
        n_resonances, l_values, has_unevaluated_ranges. When
        has_unevaluated_ranges is True, a skipped_ranges list of the
        parsed-but-not-evaluated spans is included.
    """
    data = _registry.get(isotope)
    if data is None:
        raise ValueError(f"Isotope {isotope!r} not loaded. Call load_endf first.")
    result = {
        "isotope": isotope,
        "z": data.z,
        "a": data.a,
        "awr": data.awr,
        "target_spin": data.target_spin,
        "scattering_radius": data.scattering_radius,
        "n_resonances": data.n_resonances,
        "l_values": data.l_values,
        "has_unevaluated_ranges": data.has_unevaluated_ranges,
    }
    if data.has_unevaluated_ranges:
        result["skipped_ranges"] = data.skipped_ranges
    return result


@mcp.tool()
def compute_cross_sections(
    isotope: str,
    energy_min: float = 1.0,
    energy_max: float = 100.0,
    n_points: int = 1000,
) -> dict:
    """Compute unbroadened cross-sections for a loaded isotope.

    Args:
        isotope: Isotope key (e.g., "Fe-56"). Must be loaded first.
        energy_min: Minimum energy in eV.
        energy_max: Maximum energy in eV.
        n_points: Number of energy points.

    Returns:
        Dict with keys: energies, total, elastic, capture, fission (all as lists).
    """
    _validate_energy_grid(energy_min, energy_max, n_points)
    data = _registry.get(isotope)
    if data is None:
        raise ValueError(f"Isotope {isotope!r} not loaded. Call load_endf first.")
    energies = np.linspace(energy_min, energy_max, n_points)
    xs = nereids.cross_sections(energies, data)
    return {
        "energies": energies.tolist(),
        "total": xs["total"].tolist(),
        "elastic": xs["elastic"].tolist(),
        "capture": xs["capture"].tolist(),
        "fission": xs["fission"].tolist(),
    }


@mcp.tool()
def compute_transmission(
    isotope: str,
    thickness: float,
    energy_min: float = 1.0,
    energy_max: float = 100.0,
    n_points: int = 1000,
    temperature_k: float = 0.0,
) -> dict:
    """Compute transmission spectrum for a single isotope.

    Args:
        isotope: Isotope key (e.g., "Fe-56"). Must be loaded first.
        thickness: Areal density in atoms/barn.
        energy_min: Minimum energy in eV.
        energy_max: Maximum energy in eV.
        n_points: Number of energy points.
        temperature_k: Sample temperature in Kelvin (0 = no Doppler broadening).

    Returns:
        Dict with keys: energies, transmission (as lists).
    """
    _validate_energy_grid(energy_min, energy_max, n_points)
    if not math.isfinite(thickness) or thickness < 0:
        raise ValueError(f"thickness must be non-negative and finite, got {thickness}")
    if not math.isfinite(temperature_k) or temperature_k < 0:
        raise ValueError(
            f"temperature_k must be non-negative and finite, got {temperature_k}"
        )
    data = _registry.get(isotope)
    if data is None:
        raise ValueError(f"Isotope {isotope!r} not loaded. Call load_endf first.")
    energies = np.linspace(energy_min, energy_max, n_points)
    t = nereids.forward_model(
        energies, [(data, thickness)], temperature_k=temperature_k
    )
    return {
        "energies": energies.tolist(),
        "transmission": t.tolist(),
    }


@mcp.tool()
def forward_model(
    isotopes: list[dict],
    energy_min: float = 1.0,
    energy_max: float = 100.0,
    n_points: int = 1000,
    temperature_k: float = 0.0,
) -> dict:
    """Compute multi-isotope transmission forward model.

    Args:
        isotopes: List of dicts, each with keys "isotope" (str) and "thickness"
                  (float in atoms/barn).
                  Example: [{"isotope": "Fe-56", "thickness": 0.01}]
        energy_min: Minimum energy in eV.
        energy_max: Maximum energy in eV.
        n_points: Number of energy points.
        temperature_k: Sample temperature in Kelvin (0 = no Doppler broadening).

    Returns:
        Dict with keys: energies, transmission (as lists).
    """
    _validate_energy_grid(energy_min, energy_max, n_points)
    if not math.isfinite(temperature_k) or temperature_k < 0:
        raise ValueError(
            f"temperature_k must be non-negative and finite, got {temperature_k}"
        )
    iso_list = []
    for entry in isotopes:
        key = entry.get("isotope")
        if key is None:
            raise ValueError(f"Missing 'isotope' key in isotope entry: {entry}")
        thickness = entry.get("thickness")
        if thickness is None:
            raise ValueError(f"Missing 'thickness' key in isotope entry: {entry}")
        if not math.isfinite(thickness) or thickness < 0:
            raise ValueError(
                f"thickness must be non-negative and finite, got {thickness}"
            )
        data = _registry.get(key)
        if data is None:
            raise ValueError(f"Isotope {key!r} not loaded. Call load_endf first.")
        iso_list.append((data, thickness))
    energies = np.linspace(energy_min, energy_max, n_points)
    t = nereids.forward_model(energies, iso_list, temperature_k=temperature_k)
    return {
        "energies": energies.tolist(),
        "transmission": t.tolist(),
    }



@mcp.tool()
def extract_resonance_manifest(dataset_path: str) -> dict:
    """Extract a NEREIDS sMCP manifest without running analysis.

    The manifest may be `manifest_intermediate.md`, `smcp_manifest.md`,
    `nereids_manifest.md`, `nereids_mcp.json`, or `analysis.json`.
    Markdown manifests use `---` frontmatter. JSON frontmatter is preferred
    because it works without optional YAML dependencies.

    Args:
        dataset_path: Dataset directory or manifest path.

    Returns:
        Dict containing the manifest path, parsed frontmatter, and body preview.
    """
    manifest = _read_manifest(dataset_path)
    body = manifest["body"]
    return {
        "dataset_path": manifest["dataset_path"],
        "manifest_path": manifest["manifest_path"],
        "frontmatter": _json_safe(manifest["frontmatter"]),
        "body_preview": body[:800] + "..." if len(body) > 800 else body,
    }


@mcp.tool()
def validate_resonance_dataset(dataset_path: str) -> dict:
    """Validate a NEREIDS sMCP dataset before processing.

    Checks the manifest, data file paths, isotope definitions, and whether
    instrument resolution is configured. Synthetic demo data may intentionally
    omit resolution; real VENUS data should normally include Gaussian or
    tabulated resolution settings.

    Args:
        dataset_path: Dataset directory or manifest path.

    Returns:
        Validation report with errors, warnings, inferred mode, and data paths.
    """
    try:
        manifest = _read_manifest(dataset_path)
        return _validate_workflow(manifest)
    except Exception as exc:
        return {
            "valid": False,
            "errors": [str(exc)],
            "warnings": [],
            "dataset_path": str(Path(dataset_path).expanduser()),
        }


@mcp.tool()
def process_resonance_dataset(
    dataset_path: str,
    output_dir: str | None = None,
    max_pixels: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Run a manifest-driven NEREIDS resonance analysis workflow.

    This is the high-level demo tool intended for AI-agent orchestration:
    the agent can call it when a user says "help me process the data here".
    Supported workflows:

    - `single_spectrum`: fit one region's raw open-beam and sample counts to
      its isotope areal densities and temperature.
    - `density_map` / `spatial_map`: fit every patch of a raw counts cube and
      write density-map outputs.

    Supported inputs are raw counts: a `.npz` cube of open-beam and sample
    counts (with time-bin edges), or a NeXus histogram sample/open-beam pair.
    The manifest must give a `calibration` block for the time bins.

    Args:
        dataset_path: Dataset directory or manifest path.
        output_dir: Optional output directory. Defaults to `<dataset>/output`.
        max_pixels: Optional safety crop for spatial workflows.
        dry_run: If true, only validate and return the intended plan.

    Returns:
        Small JSON-safe summary plus paths to `.npz` result artifacts.
    """
    try:
        manifest = _read_manifest(dataset_path)
        validation = _validate_workflow(manifest)
        if dry_run or not validation["valid"]:
            return {
                "success": validation["valid"],
                "dry_run": dry_run,
                "validation": validation,
            }

        base = Path(manifest["dataset_path"])
        config = _workflow_config(manifest)
        mode = _normalise_mode(config.get("mode", config.get("analysis_mode")))
        out = _resolve_path(base, output_dir) if output_dir else None
        if out is None:
            output_config = config.get("output")
            configured = config.get("output_dir")
            if configured is None and isinstance(output_config, dict):
                configured = output_config.get("directory")
            out = _resolve_path(base, configured) if configured else base / "output"
        out = out.resolve()

        if mode in {"single_spectrum", "fit_spectrum", "spectrum"}:
            result = _process_single_spectrum(base, manifest, out)
        elif mode in {"density_map", "spatial_map"}:
            result = _process_density_map(base, manifest, out, max_pixels=max_pixels)
        else:
            raise ValueError(f"unsupported analysis mode: {mode}")

        summary_path = out / "nereids_mcp_result.json"
        summary = _json_safe({
            "success": True,
            "message": "NEREIDS resonance workflow completed",
            "manifest_path": manifest["manifest_path"],
            "validation_warnings": validation["warnings"],
            "results": result,
        })
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
        summary["summary_json"] = str(summary_path)
        return summary
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc)[:2000],
            "dataset_path": str(Path(dataset_path).expanduser()),
        }
