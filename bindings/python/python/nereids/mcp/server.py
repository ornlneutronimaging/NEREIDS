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
    "counts input with instrument resolution is not available through the MCP "
    "manifest: an exact count fit requires incident fluence weights and measured "
    "detector-time bin edges, which this manifest schema does not carry. Use the "
    "direct Python fit_counts_spectrum_typed exact-count arguments, supply "
    "pre-normalized transmission, or disable instrument resolution."
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


def _ascending_energy_grid(
    energies: np.ndarray, *aligned: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Ensure energies are strictly ascending, reversing aligned arrays if needed."""
    grid = _validate_array("energies", energies, ndim=1)
    arrays = tuple(np.ascontiguousarray(a) for a in aligned)
    for arr in arrays:
        if arr.shape[0] != grid.size:
            raise ValueError(
                "aligned data first dimension must match energies, "
                f"got {arr.shape[0]} and {grid.size}"
            )
    diffs = np.diff(grid)
    if np.all(diffs > 0):
        return (grid, *arrays)
    if np.all(diffs < 0):
        reversed_arrays = tuple(np.ascontiguousarray(a[::-1, ...]) for a in arrays)
        return (np.ascontiguousarray(grid[::-1]), *reversed_arrays)
    raise ValueError("energies must be strictly monotonic")


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


def _load_energy_grid(
    base: Path,
    data_config: dict[str, Any],
    arrays: dict[str, np.ndarray] | None,
    n_expected: int | None = None,
) -> np.ndarray:
    """Load or synthesize an energy grid from manifest config and data arrays."""
    grid_config = data_config.get("energy_grid") or data_config.get("energies") or {}
    if isinstance(grid_config, str):
        grid_config = {"path": grid_config}
    if not isinstance(grid_config, dict):
        raise ValueError("energy_grid/energies must be an object or path string")

    key = (
        grid_config.get("key")
        or data_config.get("energy_key")
        or data_config.get("energies_key")
    )
    if key and arrays is not None:
        if key not in arrays:
            raise ValueError(f"energy key {key!r} not found in data file")
        return _validate_array("energies", arrays[key], ndim=1)

    if arrays is not None and "energies_ev" in arrays:
        return _validate_array("energies", arrays["energies_ev"], ndim=1)

    path = _resolve_path(base, grid_config.get("path"))
    if path is not None:
        return _validate_array("energies", np.loadtxt(path), ndim=1)

    start = grid_config.get("start_ev", data_config.get("energy_min_ev"))
    stop = grid_config.get("stop_ev", data_config.get("energy_max_ev"))
    n_points = grid_config.get("n_points", data_config.get("n_points", n_expected))
    if start is None or stop is None or n_points is None:
        raise ValueError(
            "energy grid is required: provide energies_ev in the data file or "
            "energy_grid with start_ev, stop_ev, and n_points"
        )
    return np.linspace(float(start), float(stop), int(n_points), dtype=np.float64)


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


def _resolution_kwargs(base: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Build NEREIDS resolution keyword arguments from manifest config."""
    resolution_config = config.get("resolution") or {}
    if not resolution_config:
        return {}
    if isinstance(resolution_config, str):
        resolution_config = {"kind": resolution_config}
    if not isinstance(resolution_config, dict):
        raise ValueError("resolution must be an object")

    kind = _normalise_kind(resolution_config.get("kind", "none"), "none")
    if kind in {"none", "disabled", "false"}:
        return {}
    if kind in {"gaussian", "sammy_gaussian"}:
        return {
            "flight_path_m": float(resolution_config["flight_path_m"]),
            "delta_t_us": float(resolution_config["delta_t_us"]),
            "delta_l_m": float(resolution_config["delta_l_m"]),
        }
    if kind in {"tabulated", "file", "resolution_file"}:
        path = _resolve_path(base, resolution_config.get("path"))
        if path is None:
            raise ValueError("tabulated resolution requires a path")
        flight_path_m = float(resolution_config["flight_path_m"])
        return {"resolution": nereids.load_resolution(str(path), flight_path_m)}
    raise ValueError(f"unknown resolution kind: {kind}")


def _fit_energy_range(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = (value.get("min_ev"), value.get("max_ev"))
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("fit_energy_range must be [min_ev, max_ev]")
    return (float(value[0]), float(value[1]))


def _single_fit_kwargs(
    fit_config: dict[str, Any],
    resolution: dict[str, Any],
    counts: bool,
) -> dict[str, Any]:
    keys = {
        "temperature_k",
        "fit_temperature",
        "max_iter",
        "solver",
        "background",
        "fit_back_d",
        "fit_back_f",
        "back_d_init",
        "back_f_init",
        "fit_energy_scale",
        "t0_init_us",
        "l_scale_init",
        "energy_scale_flight_path_m",
        "tzero_jacobian",
    }
    if counts:
        keys.update(
            {
                "fit_alpha_1",
                "fit_alpha_2",
                "alpha_1_init",
                "alpha_2_init",
                "enable_polish",
            }
        )
    kwargs = {key: fit_config[key] for key in keys if key in fit_config}
    if "fit_energy_range" in fit_config:
        kwargs["fit_energy_range"] = _fit_energy_range(fit_config["fit_energy_range"])
    kwargs.update(resolution)
    return kwargs


def _spatial_fit_kwargs(
    fit_config: dict[str, Any],
    resolution: dict[str, Any],
) -> dict[str, Any]:
    keys = {
        "temperature_k",
        "fit_temperature",
        "max_iter",
        "solver",
        "background",
        # Surface the SAMMY exponential background tail
        # (`BackD` / `BackF`) at the MCP boundary so callers can request
        # per-pixel `back_d_map` / `back_f_map`.
        "fit_back_d",
        "fit_back_f",
        "back_d_init",
        "back_f_init",
        "fit_alpha_1",
        "fit_alpha_2",
        "alpha_1_init",
        "alpha_2_init",
        "c",
        "enable_polish",
        "fit_energy_scale",
        "t0_init_us",
        "l_scale_init",
        "energy_scale_flight_path_m",
        "tzero_jacobian",
    }
    kwargs = {key: fit_config[key] for key in keys if key in fit_config}
    if "fit_energy_range" in fit_config:
        kwargs["fit_energy_range"] = _fit_energy_range(fit_config["fit_energy_range"])
    kwargs.update(resolution)
    return kwargs


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


def _fit_result_summary(result: Any, names: list[str]) -> dict[str, Any]:
    densities = np.asarray(result.densities, dtype=float)
    uncertainties = np.asarray(result.uncertainties, dtype=float)
    entries = []
    for i, name in enumerate(names):
        entries.append(
            {
                "isotope": name,
                "density_atoms_per_barn": _finite_float_or_none(densities[i]),
                "uncertainty_atoms_per_barn": _finite_float_or_none(uncertainties[i])
                if i < len(uncertainties)
                else None,
            }
        )
    summary: dict[str, Any] = {
        "density_fits": entries,
        "reduced_chi_squared": _finite_float_or_none(result.reduced_chi_squared),
        "deviance_per_dof": None
        if getattr(result, "deviance_per_dof", None) is None
        else _finite_float_or_none(result.deviance_per_dof),
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "temperature_k": None
        if result.temperature_k is None
        else _finite_float_or_none(result.temperature_k),
        "anorm": _finite_float_or_none(result.anorm),
        "background": [_finite_float_or_none(v) for v in result.background],
    }
    # Fitted TZERO (t0_us, l_scale) and exponential-background (back_d, back_f)
    # parameters are populated on FitResult when the corresponding fit flags
    # are set in the manifest's analysis.fit block. Emit them only when present
    # so the schema stays backwards-compatible for fits that don't enable
    # those flags (key absent rather than null, matching deviance_per_dof).
    for attr in ("t0_us", "l_scale", "back_d", "back_f"):
        value = getattr(result, attr, None)
        if value is not None:
            summary[attr] = _finite_float_or_none(value)
    return summary


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

    kind = _normalise_kind(data_config.get("kind"), "counts_npz")
    arrays = _npz_arrays(data_path) if data_path.suffix == ".npz" else None
    if arrays is None:
        loaded = np.loadtxt(data_path, delimiter=data_config.get("delimiter", None))
        if loaded.ndim != 2 or loaded.shape[1] < 2:
            raise ValueError("spectrum text input must have at least two columns")
        arrays = {
            "energies_ev": loaded[:, int(data_config.get("energy_column", 0))],
            "transmission": loaded[:, int(data_config.get("transmission_column", 1))],
        }
        if loaded.shape[1] > 2:
            arrays["uncertainty"] = loaded[:, int(data_config.get("uncertainty_column", 2))]
        kind = "transmission_npz"

    entries = _get_isotope_entries(config, frontmatter)
    isotopes, names = _load_isotopes(
        entries, base, default_library=str(config.get("library", "endf8.1"))
    )
    resolution = _resolution_kwargs(base, config)
    if kind in {"counts_npz", "counts"}:
        sample_key = data_config.get("sample_key", "sample_counts")
        open_beam_key = data_config.get("open_beam_key", "open_beam_counts")
        _require_npz_keys(
            arrays,
            [sample_key, open_beam_key],
            context="single_spectrum counts data",
        )
        sample = _validate_array("sample_counts", arrays[sample_key])
        open_beam = _validate_array("open_beam_counts", arrays[open_beam_key])
        _require_same_shape("sample_counts", sample, ("open_beam_counts", open_beam))
        energies = _load_energy_grid(base, data_config, arrays, n_expected=sample.shape[0])
        energies, sample, open_beam = _ascending_energy_grid(energies, sample, open_beam)
        if sample.ndim == 3:
            sample = np.ascontiguousarray(sample.sum(axis=(1, 2)))
            open_beam = np.ascontiguousarray(open_beam.sum(axis=(1, 2)))
        if sample.ndim != 1 or open_beam.ndim != 1:
            raise ValueError("single_spectrum counts arrays must be 1D or 3D")
        c = float(data_config.get("pc_ratio", arrays.get("pc_ratio", 1.0)))
        fit_domain = str(fit_config.get("fit_domain", "counts")).lower()
        if fit_domain != "counts":
            raise ValueError(
                "raw count input cannot be fit in the transmission domain: "
                "the conversion discards open-beam count uncertainty; use "
                "fit_domain='counts' with solver='auto' or solver='kl'"
            )
        kwargs = _single_fit_kwargs(fit_config, resolution, counts=True)
        kwargs["c"] = float(fit_config.get("c", c))
        result = nereids.fit_counts_spectrum_typed(
            sample_counts=sample,
            open_beam_counts=open_beam,
            energies=energies,
            isotopes=isotopes,
            **kwargs,
        )
        transmission = sample / np.maximum(kwargs["c"] * open_beam, 1.0)
        uncertainty = transmission * np.sqrt(
            1.0 / np.maximum(sample, 1.0) + 1.0 / np.maximum(open_beam, 1.0)
        )
    elif kind in {"transmission_npz", "transmission", "spectrum"}:
        fit_domain = str(fit_config.get("fit_domain", "transmission")).lower()
        if fit_domain != "transmission":
            raise ValueError(
                "normalized transmission input must use fit_domain='transmission' "
                "with solver='auto' or solver='lm'"
            )
        trans_key = data_config.get("transmission_key", "transmission")
        unc_key = data_config.get("uncertainty_key", "uncertainty")
        _require_npz_keys(
            arrays,
            [trans_key],
            context="single_spectrum transmission data",
        )
        transmission = _validate_array("transmission", arrays[trans_key])
        if unc_key in arrays:
            uncertainty = _validate_array("uncertainty", arrays[unc_key])
        else:
            uncertainty = np.full_like(
                transmission, float(data_config.get("uncertainty", 0.01))
            )
        _require_same_shape("transmission", transmission, ("uncertainty", uncertainty))
        energies = _load_energy_grid(base, data_config, arrays, n_expected=transmission.shape[0])
        energies, transmission, uncertainty = _ascending_energy_grid(
            energies, transmission, uncertainty
        )
        if transmission.ndim == 3:
            transmission = np.ascontiguousarray(np.nanmean(transmission, axis=(1, 2)))
            uncertainty = np.ascontiguousarray(
                np.sqrt(np.nanmean(np.square(uncertainty), axis=(1, 2)))
            )
        if transmission.ndim != 1:
            raise ValueError("single_spectrum transmission arrays must be 1D or 3D")
        kwargs = _single_fit_kwargs(fit_config, resolution, counts=False)
        result = nereids.fit_spectrum_typed(
            transmission=transmission,
            uncertainty=uncertainty,
            energies=energies,
            isotopes=isotopes,
            **kwargs,
        )
    else:
        raise ValueError(f"unsupported single_spectrum data kind: {kind}")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_npz = output_dir / "nereids_spectrum_fit.npz"
    np.savez_compressed(
        result_npz,
        energies_ev=energies,
        transmission=transmission,
        uncertainty=uncertainty,
        fitted_densities=np.asarray(result.densities),
        density_uncertainties=np.asarray(result.uncertainties),
        isotope_names=np.asarray(names),
    )
    summary = _fit_result_summary(result, names)
    summary.update(
        {
            "mode": "single_spectrum",
            "data_path": str(data_path),
            "output_npz": str(result_npz),
            "energy_range_ev": [float(energies[0]), float(energies[-1])],
            "n_energy": int(energies.size),
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
    kind = _normalise_kind(data_config.get("kind"), "transmission_npz")

    arrays: dict[str, np.ndarray] | None = None
    path = _resolve_path(base, data_config.get("path"))
    if path is not None and path.suffix == ".npz":
        arrays = _npz_arrays(path)

    if kind in {"transmission_npz", "transmission"}:
        if arrays is None:
            raise ValueError("transmission_npz data requires a .npz data.path")
        trans_key = data_config.get("transmission_key", "transmission")
        unc_key = data_config.get("uncertainty_key", "uncertainty")
        _require_npz_keys(
            arrays,
            [trans_key],
            context="density_map transmission data",
        )
        transmission = _validate_array("transmission", arrays[trans_key], ndim=3)
        if unc_key in arrays:
            uncertainty = _validate_array("uncertainty", arrays[unc_key], ndim=3)
        else:
            uncertainty = np.full_like(
                transmission, float(data_config.get("uncertainty", 0.01))
            )
        _require_same_shape("transmission", transmission, ("uncertainty", uncertainty))
        energies = _load_energy_grid(
            base, data_config, arrays, n_expected=transmission.shape[0]
        )
        energies, transmission, uncertainty = _ascending_energy_grid(
            energies, transmission, uncertainty
        )
        transmission = _apply_roi(transmission, data_config.get("roi") or config.get("roi"))
        uncertainty = _apply_roi(uncertainty, data_config.get("roi") or config.get("roi"))
        transmission, crop = _limit_pixels(transmission, max_pixels)
        uncertainty = uncertainty[:, : transmission.shape[1], : transmission.shape[2]]
        input_data = nereids.from_transmission(transmission, uncertainty)
        input_kind = "transmission"
        c = None
    elif kind in {"transmission_tiff", "tiff"}:
        if path is None:
            raise ValueError("transmission_tiff data requires data.path")
        # Pre-normalized transmission: noise around zero legitimately
        # produces small negative values, so bypass the raw-counts
        # pixel-value guard ("allow" rather than the default "reject").
        transmission = _validate_array(
            "transmission",
            nereids.load_tiff_stack(str(path), pixel_policy="allow"),
            ndim=3,
        )
        uncertainty_path = _resolve_path(base, data_config.get("uncertainty_path"))
        if uncertainty_path is not None:
            uncertainty = _validate_array(
                "uncertainty",
                nereids.load_tiff_stack(str(uncertainty_path), pixel_policy="allow"),
                ndim=3,
            )
        else:
            uncertainty = np.full_like(
                transmission, float(data_config.get("uncertainty", 0.01))
            )
        _require_same_shape("transmission", transmission, ("uncertainty", uncertainty))
        energies = _load_energy_grid(base, data_config, None, n_expected=transmission.shape[0])
        energies, transmission, uncertainty = _ascending_energy_grid(
            energies, transmission, uncertainty
        )
        transmission = _apply_roi(transmission, data_config.get("roi") or config.get("roi"))
        uncertainty = _apply_roi(uncertainty, data_config.get("roi") or config.get("roi"))
        transmission, crop = _limit_pixels(transmission, max_pixels)
        uncertainty = uncertainty[:, : transmission.shape[1], : transmission.shape[2]]
        input_data = nereids.from_transmission(transmission, uncertainty)
        input_kind = "transmission"
        c = None
    elif kind in {"counts_npz", "counts"}:
        if arrays is None:
            raise ValueError("counts_npz data requires a .npz data.path")
        sample_key = data_config.get("sample_key", "sample_counts")
        open_beam_key = data_config.get("open_beam_key", "open_beam_counts")
        _require_npz_keys(
            arrays,
            [sample_key, open_beam_key],
            context="density_map counts data",
        )
        sample = _validate_array(
            "sample_counts", arrays[sample_key], ndim=3
        )
        open_beam = _validate_array(
            "open_beam_counts",
            arrays[open_beam_key],
            ndim=3,
        )
        _require_same_shape("sample_counts", sample, ("open_beam_counts", open_beam))
        energies = _load_energy_grid(base, data_config, arrays, n_expected=sample.shape[0])
        energies, sample, open_beam = _ascending_energy_grid(energies, sample, open_beam)
        sample = _apply_roi(sample, data_config.get("roi") or config.get("roi"))
        open_beam = _apply_roi(open_beam, data_config.get("roi") or config.get("roi"))
        sample, crop = _limit_pixels(sample, max_pixels)
        open_beam = open_beam[:, : sample.shape[1], : sample.shape[2]]
        input_data = nereids.from_counts(sample, open_beam)
        input_kind = "counts"
        c = float(data_config.get("pc_ratio", arrays.get("pc_ratio", 1.0)))
    elif kind in {"nexus_histogram", "nexus"}:
        sample_path = _resolve_path(base, data_config.get("sample_path"))
        open_beam_path = _resolve_path(base, data_config.get("open_beam_path"))
        if sample_path is None or open_beam_path is None:
            raise ValueError("nexus data requires sample_path and open_beam_path")
        sample_data = nereids.load_nexus_histogram(str(sample_path))
        ob_data = nereids.load_nexus_histogram(str(open_beam_path))
        sample = _validate_array("sample_counts", sample_data.counts, ndim=3)
        open_beam = _validate_array("open_beam_counts", ob_data.counts, ndim=3)
        _require_same_shape("sample_counts", sample, ("open_beam_counts", open_beam))
        sample_tof_edges = _validate_array(
            "sample_tof_edges_us", sample_data.tof_edges_us, ndim=1
        )
        open_beam_tof_edges = _validate_array(
            "open_beam_tof_edges_us", ob_data.tof_edges_us, ndim=1
        )
        if (
            sample_tof_edges.shape != open_beam_tof_edges.shape
            or not np.allclose(sample_tof_edges, open_beam_tof_edges)
        ):
            raise ValueError("nexus sample and open_beam TOF bin edges must match")
        flight_path = float(
            data_config.get(
                "flight_path_m",
                sample_data.flight_path_m or ob_data.flight_path_m or 25.0,
            )
        )
        delay_us = float(data_config.get("delay_us", 0.0))
        energies = np.asarray(
            nereids.tof_to_energy_centers(sample_tof_edges, flight_path, delay_us),
            dtype=np.float64,
        )
        # NeXus counts are loaded in TOF-bin order. The Python binding returns
        # ascending energy centers, so reverse the counts to energy order first.
        if np.all(np.diff(energies) > 0):
            sample = np.ascontiguousarray(sample[::-1, ...])
            open_beam = np.ascontiguousarray(open_beam[::-1, ...])
        energies, sample, open_beam = _ascending_energy_grid(energies, sample, open_beam)
        sample = _apply_roi(sample, data_config.get("roi") or config.get("roi"))
        open_beam = _apply_roi(open_beam, data_config.get("roi") or config.get("roi"))
        sample, crop = _limit_pixels(sample, max_pixels)
        open_beam = open_beam[:, : sample.shape[1], : sample.shape[2]]
        input_data = nereids.from_counts(sample, open_beam)
        input_kind = "counts"
        c = float(data_config.get("pc_ratio", 1.0))
    else:
        raise ValueError(f"unsupported density_map data kind: {kind}")

    entries = _get_isotope_entries(config, frontmatter)
    isotopes, names = _load_isotopes(
        entries, base, default_library=str(config.get("library", "endf8.1"))
    )
    initial_densities = [density for _, density in isotopes]
    isotope_data = [data for data, _ in isotopes]
    kwargs = _spatial_fit_kwargs(fit_config, _resolution_kwargs(base, config))
    requested_solver = str(fit_config.get("solver", "auto")).lower()
    if input_kind == "counts" and requested_solver == "lm":
        raise ValueError(
            "raw count maps cannot use solver='lm': use solver='auto' or "
            "solver='kl' so the separate sample/open-beam arms are preserved"
        )
    if input_kind == "transmission" and requested_solver in {
        "kl",
        "poisson",
        "poisson_kl",
        "joint_poisson",
    }:
        raise ValueError(
            "normalized transmission maps cannot use a Poisson/KL count "
            "likelihood; use solver='auto' or solver='lm'"
        )
    if c is not None and "c" not in kwargs:
        kwargs["c"] = c
    result = nereids.spatial_map_typed(
        input_data,
        energies,
        isotope_data,
        initial_densities=initial_densities,
        **kwargs,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result_npz = output_dir / "nereids_density_map.npz"
    arrays_to_save: dict[str, Any] = {
        "energies_ev": energies,
        "chi_squared_map": np.asarray(result.chi_squared_map),
        "converged_map": np.asarray(result.converged_map),
        "isotope_names": np.asarray(names),
    }
    if result.deviance_per_dof_map is not None:
        arrays_to_save["deviance_per_dof_map"] = np.asarray(result.deviance_per_dof_map)
    density_stats = []
    for name, density_map, uncertainty_map in zip(
        names, result.density_maps, result.uncertainty_maps
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

    # SpatialResult exposes per-pixel anorm / background / back_d /
    # back_f / t0 / l_scale maps whenever the spatial pipeline ran with
    # the corresponding *feature* enabled (e.g. `background=True`
    # materialises all three terms of `background_maps` — including
    # NaN-per-pixel entries for terms that were not actually fit — and
    # `fit_energy_scale=True` materialises both t0/L scale maps).  Save
    # the raw arrays into the NPZ (so downstream consumers can
    # reconstruct the model curve per-pixel) and surface aggregate
    # stats in the JSON summary.  Per-pixel ``back_d_map`` / ``back_f_map``
    # require ``background=True`` AND ``fit_back_d=True`` /
    # ``fit_back_f=True`` — counts-KL runs always have them as ``None``
    # because the joint-Poisson dispatch never fits the SAMMY
    # exponential tail.
    fit_param_stats: dict[str, Any] = {}
    anorm_map = getattr(result, "anorm_map", None)
    if anorm_map is not None:
        anorm_arr = np.asarray(anorm_map)
        arrays_to_save["anorm_map"] = anorm_arr
        fit_param_stats["anorm"] = _array_stats(anorm_arr)
    background_maps = getattr(result, "background_maps", None)
    if background_maps is not None:
        bm_stats = []
        for term_idx, bm in enumerate(background_maps):
            bm_arr = np.asarray(bm)
            arrays_to_save[f"background_term_{term_idx}_map"] = bm_arr
            bm_stats.append(_array_stats(bm_arr))
        fit_param_stats["background"] = bm_stats
    back_d_map = getattr(result, "back_d_map", None)
    if back_d_map is not None:
        back_d_arr = np.asarray(back_d_map)
        arrays_to_save["back_d_map"] = back_d_arr
        fit_param_stats["back_d"] = _array_stats(back_d_arr)
    back_f_map = getattr(result, "back_f_map", None)
    if back_f_map is not None:
        back_f_arr = np.asarray(back_f_map)
        arrays_to_save["back_f_map"] = back_f_arr
        fit_param_stats["back_f"] = _array_stats(back_f_arr)
    t0_us_map = getattr(result, "t0_us_map", None)
    if t0_us_map is not None:
        t0_arr = np.asarray(t0_us_map)
        arrays_to_save["t0_us_map"] = t0_arr
        fit_param_stats["t0_us"] = _array_stats(t0_arr)
    l_scale_map = getattr(result, "l_scale_map", None)
    if l_scale_map is not None:
        l_arr = np.asarray(l_scale_map)
        arrays_to_save["l_scale_map"] = l_arr
        fit_param_stats["l_scale"] = _array_stats(l_arr)

    np.savez_compressed(result_npz, **arrays_to_save)

    summary: dict[str, Any] = {
        "mode": "density_map",
        "input_kind": input_kind,
        "data_path": None if path is None else str(path),
        "output_npz": str(result_npz),
        "energy_range_ev": [float(energies[0]), float(energies[-1])],
        "n_energy": int(energies.size),
        "map_shape": list(np.asarray(result.converged_map).shape),
        "n_converged": int(result.n_converged),
        "n_total": int(result.n_total),
        "n_failed": int(result.n_failed),
        "density_maps": density_stats,
        "crop": crop,
    }
    if fit_param_stats:
        summary["fit_param_stats"] = fit_param_stats
    return summary


def _validate_workflow(manifest: dict[str, Any]) -> dict[str, Any]:
    base = Path(manifest["dataset_path"])
    frontmatter = manifest["frontmatter"]
    config = _workflow_config(manifest)
    data_config = _get_data_config(config)
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

    effective_kind = _normalise_kind(
        data_config.get("kind"),
        "counts_npz"
        if mode in {"single_spectrum", "fit_spectrum", "spectrum"}
        else "transmission_npz",
    )
    has_path = data_config.get("path") is not None and data_config.get("path") != ""
    has_sample_path = (
        data_config.get("sample_path") is not None and data_config.get("sample_path") != ""
    )
    has_open_beam_path = (
        data_config.get("open_beam_path") is not None
        and data_config.get("open_beam_path") != ""
    )
    if mode in {"single_spectrum", "fit_spectrum", "spectrum"} and not has_path:
        errors.append("single_spectrum workflow requires data.path")
    elif mode in {"density_map", "spatial_map"}:
        if effective_kind in {"nexus_histogram", "nexus"}:
            if not has_sample_path or not has_open_beam_path:
                errors.append("nexus density_map workflow requires sample_path and open_beam_path")
        elif effective_kind in {
            "transmission_npz",
            "transmission",
            "transmission_tiff",
            "tiff",
            "counts_npz",
            "counts",
        }:
            if not has_path:
                errors.append(f"{effective_kind} density_map workflow requires data.path")
        else:
            errors.append(f"unsupported density_map data kind: {effective_kind}")

    data_paths = []
    for key in ("path", "sample_path", "open_beam_path", "uncertainty_path"):
        value = data_config.get(key)
        if value is None or value == "":
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

    resolution = config.get("resolution")
    resolution_kind: str | None = None
    if isinstance(resolution, dict):
        resolution_kind = _normalise_kind(resolution.get("kind", "none"), "none")
        if resolution_kind in {"tabulated", "file", "resolution_file"}:
            path = _resolve_path(base, resolution.get("path"))
            if path is None or not path.exists():
                errors.append(f"resolution file does not exist: {path}")
        if resolution_kind in {"none", "disabled", "false"}:
            warnings.append("resolution disabled; appropriate for synthetic data")
    elif isinstance(resolution, str):
        resolution_kind = _normalise_kind(resolution, "none")
    elif resolution is None and mode in {"density_map", "spatial_map"}:
        warnings.append("no resolution configured; OK for synthetic/demo data")

    # Raw count inputs need two separately evaluated response arms. The direct
    # Python fitter supports this when source weights and detector-time edges
    # are supplied, but the MCP manifest schema does not carry those inputs.
    counts_input = effective_kind in {
        "counts_npz",
        "counts",
        "nexus_histogram",
        "nexus",
    }
    resolution_active = resolution_kind not in {None, "none", "disabled", "false"}
    if counts_input and resolution_active:
        errors.append(_COUNTS_RESOLUTION_UNSUPPORTED)

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "manifest_path": manifest["manifest_path"],
        "dataset_path": manifest["dataset_path"],
        "mode": mode or None,
        "data_kind": effective_kind,
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
        target_spin, l_values.
    """
    parsed = nereids.parse_isotope_str(isotope)
    if parsed is None:
        raise ValueError(f"Cannot parse isotope string: {isotope!r}")
    z, a = parsed
    data = nereids.load_endf(z, a, library=library)
    key = _isotope_key(z, a)
    _registry[key] = data
    return {
        "isotope": key,
        "z": data.z,
        "a": data.a,
        "n_resonances": data.n_resonances,
        "scattering_radius": data.scattering_radius,
        "target_spin": data.target_spin,
        "l_values": data.l_values,
    }


@mcp.tool()
def get_resonance_parameters(isotope: str) -> dict:
    """Get resonance parameters for a loaded isotope.

    Args:
        isotope: Isotope key (e.g., "Fe-56"). Must be loaded first via load_endf.

    Returns:
        Dict with keys: isotope, z, a, awr, target_spin, scattering_radius,
        n_resonances, l_values.
    """
    data = _registry.get(isotope)
    if data is None:
        raise ValueError(f"Isotope {isotope!r} not loaded. Call load_endf first.")
    return {
        "isotope": isotope,
        "z": data.z,
        "a": data.a,
        "awr": data.awr,
        "target_spin": data.target_spin,
        "scattering_radius": data.scattering_radius,
        "n_resonances": data.n_resonances,
        "l_values": data.l_values,
    }


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
def detect_isotopes(
    matrix_isotope: str,
    matrix_density: float,
    trace_isotopes: list[str],
    trace_ppm: float = 100.0,
    energy_min: float = 1.0,
    energy_max: float = 100.0,
    n_points: int = 1000,
    i0: float = 1e6,
    temperature_k: float = 293.6,
    snr_threshold: float = 3.0,
) -> list[dict]:
    """Analyze detectability of trace isotopes in a matrix.

    Args:
        matrix_isotope: Matrix isotope key (e.g., "Fe-56"). Must be loaded.
        matrix_density: Matrix areal density in atoms/barn.
        trace_isotopes: List of trace isotope keys. Must be loaded.
        trace_ppm: Trace concentration in ppm.
        energy_min: Minimum energy in eV.
        energy_max: Maximum energy in eV.
        n_points: Number of energy points.
        i0: Neutron fluence (counts per bin).
        temperature_k: Sample temperature in Kelvin.
        snr_threshold: Minimum SNR for "detectable" verdict.

    Returns:
        List of dicts with keys: isotope, detectable, peak_snr, peak_energy_ev,
        peak_delta_t_per_ppm, opaque_fraction.
    """
    _validate_energy_grid(energy_min, energy_max, n_points)
    if not math.isfinite(matrix_density) or matrix_density <= 0:
        raise ValueError(
            f"matrix_density must be positive and finite, got {matrix_density}"
        )
    if not trace_isotopes:
        raise ValueError("trace_isotopes must not be empty")
    if not math.isfinite(trace_ppm) or trace_ppm < 0:
        raise ValueError(f"trace_ppm must be non-negative and finite, got {trace_ppm}")
    if not math.isfinite(i0) or i0 <= 0:
        raise ValueError(f"i0 must be positive and finite, got {i0}")
    if not math.isfinite(snr_threshold) or snr_threshold < 0:
        raise ValueError(
            f"snr_threshold must be non-negative and finite, got {snr_threshold}"
        )
    if not math.isfinite(temperature_k) or temperature_k < 0:
        raise ValueError(
            f"temperature_k must be non-negative and finite, got {temperature_k}"
        )

    matrix = _registry.get(matrix_isotope)
    if matrix is None:
        raise ValueError(f"Matrix isotope {matrix_isotope!r} not loaded.")

    traces = []
    for key in trace_isotopes:
        data = _registry.get(key)
        if data is None:
            raise ValueError(f"Trace isotope {key!r} not loaded.")
        traces.append(data)

    energies = np.linspace(energy_min, energy_max, n_points)
    results = nereids.trace_detectability_survey(
        matrix,
        matrix_density,
        traces,
        trace_ppm,
        energies,
        i0,
        temperature_k=temperature_k,
        snr_threshold=snr_threshold,
    )
    return [
        {
            "isotope": name,
            "detectable": report.detectable,
            "peak_snr": report.peak_snr,
            "peak_energy_ev": report.peak_energy_ev,
            "peak_delta_t_per_ppm": report.peak_delta_t_per_ppm,
            "opaque_fraction": report.opaque_fraction,
        }
        for name, report in results
    ]


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

    - `single_spectrum`: fit one transmission/counts spectrum and estimate
      isotope areal densities.
    - `density_map` / `spatial_map`: fit every pixel in a 3D transmission or
      counts cube and write density-map outputs.

    Supported inputs include `.npz` spectra/cubes, multi-frame TIFF
    transmission stacks, and NeXus histogram sample/open-beam pairs.

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
