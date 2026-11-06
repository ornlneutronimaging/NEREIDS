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


def _write_tpx1_run(folder, frames, triggers, starts_s=None):
    """Write a raw Timepix-1 run: 16-bit FITS frames and one shutter's sidecars."""
    import numpy as np

    folder.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames, dtype=">i2")
    height, width = frames.shape[1:]
    for k, frame in enumerate(frames):
        cards = [
            "SIMPLE  =                    T",
            "BITPIX  =                   16",
            "NAXIS   =                    2",
            f"NAXIS1  = {width:>20}",
            f"NAXIS2  = {height:>20}",
            "END",
        ]
        header = "".join(f"{card:<80}" for card in cards).encode()
        header += b" " * (-len(header) % 2880)
        data = frame.tobytes()
        data += b"\0" * (-len(data) % 2880)
        (folder / f"run_0_114_{k:05d}.fits").write_bytes(header + data)
    if starts_s is None:
        starts_s = [1e-6 * (k + 1) for k in range(len(frames))]
    width_s = starts_s[-1] - starts_s[-2]
    (folder / "run_0_114_Spectra.txt").write_text(
        "".join(f"{t}\t0\r\n" for t in starts_s)
    )
    length = starts_s[-1] + 2 * width_s - starts_s[0]
    (folder / "run_0_114_ShutterTimes.txt").write_text(
        f"0\t{starts_s[0]}\t{length}\r\n1\t0\t0\r\n"
    )
    (folder / "run_0_114_ShutterCount.txt").write_text(f"0\t{triggers}\r\n1\t0\r\n")
