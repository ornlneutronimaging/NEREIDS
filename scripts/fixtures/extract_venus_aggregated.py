"""Extract aggregated VENUS Hf 120min spectrum from the research HDF5.

Produces `tests/data/venus/aggregated_hf_120min.npz` — a small
(~85 KB uncompressed, ~30 KB compressed) fixture holding the
energy grid, the summed sample/open-beam counts, and the proton-charge
normalization ratio. This lets the MLBW regression test run in CI
without needing the 13 GB VENUS cube or the gitignored PLEIADES
resolution file.
"""
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/Users/chenzhang/github.com/NEREIDS/NEREIDS")
H5 = ROOT / ".research/spatial-regularization/data/counts/resonance_data_2cm.h5"
OUT = ROOT / "tests/data/venus/aggregated_hf_120min.npz"

ENERGY_MIN, ENERGY_MAX = 7.0, 200.0

with h5py.File(H5, "r") as f:
    E_full = f["Hf/120min/transmission/spectrum/energy_eV"][:]
    S3d = f["Hf/120min/counts/sample"][:][::-1, :, :]
    O3d = f["Hf/open_beam/counts"][:][::-1, :, :]
    Q_s = float(f["Hf/120min/proton_charges/sample_values_uC"][:].sum())
    Q_ob = float(f["Hf/120min/proton_charges/ob_values_uC"][0])

mask = (E_full >= ENERGY_MIN) & (E_full <= ENERGY_MAX)
E = np.ascontiguousarray(E_full[mask]).astype(np.float64)
S_agg = S3d[mask].astype(np.float64).sum(axis=(1, 2))
O_agg = O3d[mask].astype(np.float64).sum(axis=(1, 2))
c = Q_s / Q_ob

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    OUT,
    energies_ev=E,
    sample_counts=S_agg,
    open_beam_counts=O_agg,
    pc_ratio=np.float64(c),
    energy_min_ev=np.float64(ENERGY_MIN),
    energy_max_ev=np.float64(ENERGY_MAX),
    description=np.array(
        "VENUS Hf 120 min aggregated (summed over 512x512). "
        "Energy range 7..200 eV (3471 bins after mask). "
        "Produced by scripts/extract_venus_fixture.py from "
        ".research/spatial-regularization/data/counts/resonance_data_2cm.h5 "
        "(private dataset, not redistributable). The aggregated spectra are "
        "used by the #465 MLBW regression test to lock fit behaviour against "
        "real-world neutron-resonance-imaging data.",
        dtype="S",
    ),
)

print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
print(f"  E shape   = {E.shape}, range [{E.min():.3f}, {E.max():.3f}] eV")
print(f"  S_agg sum = {S_agg.sum():.3e}")
print(f"  O_agg sum = {O_agg.sum():.3e}")
print(f"  c         = {c:.6f}")
