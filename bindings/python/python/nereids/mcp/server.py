"""FastMCP server exposing NEREIDS nuclear data tools."""

from __future__ import annotations

import numpy as np
from fastmcp import FastMCP

import nereids

mcp = FastMCP("nereids")

# In-memory registry of loaded ResonanceData objects.
# PyO3 objects are not JSON-serializable, so we keep them here
# and reference by isotope key (e.g., "Fe-56").
_registry: dict[str, nereids.ResonanceData] = {}


def _isotope_key(z: int, a: int) -> str:
    """Build a registry key like 'Fe-56'."""
    symbol = nereids.element_symbol(z) or f"Z{z}"
    return f"{symbol}-{a}"


@mcp.tool()
def list_isotopes(z: int) -> list[dict]:
    """List naturally occurring isotopes for element Z with their abundances.

    Args:
        z: Atomic number (e.g., 26 for iron).

    Returns:
        List of dicts with keys: z, a, symbol, abundance.
    """
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
    iso_list = []
    for entry in isotopes:
        key = entry["isotope"]
        data = _registry.get(key)
        if data is None:
            raise ValueError(f"Isotope {key!r} not loaded. Call load_endf first.")
        iso_list.append((data, entry["thickness"]))
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
