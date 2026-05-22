"""Shared synthetic-fixture builders for the Python test suite.

Energy 6.674 aligns to the Rust-side ``u238_single_resonance()`` in
``nereids_endf::resonance::test_support`` (architecture-audit F13:
fixes the historical Python ``6.67`` vs Rust ``6.674`` drift).
"""

import nereids


def _make_single_resonance(
    z=92,
    a=238,
    awr=236.006,
    scattering_radius=9.48,
    energy=6.674,
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


def _synthetic_u238_data():
    """Single-resonance U-238 ``ResonanceData`` at the 6.674 eV anchor."""
    return nereids.create_resonance_data(
        z=92,
        a=238,
        awr=236.006,
        scattering_radius=9.48,
        resonances=[(6.674, 0.5, 0.0015, 0.023)],
        target_spin=0.0,
    )


def _synthetic_u238_entry(initial_density=0.001):
    """MCP-manifest entry dict carrying a single-resonance U-238 fixture."""
    return {
        "isotope": "U-238",
        "initial_density": initial_density,
        "synthetic_resonance": {
            "z": 92,
            "a": 238,
            "awr": 236.006,
            "scattering_radius": 9.48,
            "target_spin": 0.0,
            "resonances": [[6.674, 0.5, 0.0015, 0.023]],
        },
    }
