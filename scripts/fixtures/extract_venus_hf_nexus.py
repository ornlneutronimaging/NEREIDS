"""Extract VENUS Hf 120min sample + open-beam to NeXus-format HDF5 fixtures.

Produces two fixture files for the GUI's NeXus loader and the per-pixel
fitting workflow:

  - tests/data/pleiades_data/venus_hf_120min_sample.h5
  - tests/data/pleiades_data/venus_hf_open_beam.h5

Both follow the rustpix NeXus schema documented in
`crates/nereids-io/src/nexus.rs`:

  /entry                                  attrs: flight_path_m, tof_offset_ns, title
  /entry/histogram/counts                 u64  4D [rot_angle=1, y, x, tof]
  /entry/histogram/time_of_flight         f64  1D, ns, ASCENDING TOF (length tof+1, edges)
  /entry/pixel_masks/dead                 u8   2D [y, x]   (omitted; full sensor live)

Source: the private 38 GB cube
  .research/spatial-regularization/data/counts/resonance_data_2cm.h5
which holds Au/Hf/W × 20min/120min × 512×512 spatial × 4367 TOF bins.
This script extracts ONLY Hf 120min and the matching open-beam, takes
a centered 256×256 spatial crop (matching the LANL-ORNL synthetic test
TIFF), preserves the full 4367-bin TOF axis, and stores u64 counts with
gzip=9 + shuffle compression.

Output is committed to the PLEIADES test-data submodule via Git LFS
(see `tests/data/pleiades_data/.gitattributes`).

Notes on axis ordering
----------------------
The source `Hf/120min/counts/sample` has axis 0 in **ASCENDING TOF**
order (verified empirically: the deepest count-rate dip lands at
axis-0 index 4061, which corresponds to E_full[N-1-4061] = 7.71 eV =
Hf-178 ground-state resonance). The bin-0 ~1e9 count rate is the
gamma flash from the spallation pulse at shortest TOF, not a flux
peak. The sister harness scripts in `.research/algo_design_489_*`
apply `[::-1, :, :]` to align counts with `energies_eV` (which is in
ASCENDING ENERGY = DESCENDING TOF order) for fitting; the conversion
here goes the OTHER direction — straight to ASC-TOF NeXus convention.
"""
from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / ".research/spatial-regularization/data/counts/resonance_data_2cm.h5"
OUT_DIR = ROOT / "tests/data/pleiades_data"

CROP_SIZE = 256
CROP_Y0 = (512 - CROP_SIZE) // 2
CROP_X0 = (512 - CROP_SIZE) // 2

# VENUS instrument constants.
FLIGHT_PATH_M = 25.0
TOF_OFFSET_NS = 0.0

# Physical constants — same source values used throughout the codebase
# (CODATA 2018 via nereids-core::constants).
NEUTRON_MASS_KG = 1.674_927_498_05e-27
EV_TO_JOULES = 1.602_176_634e-19
TOF_FACTOR = (0.5 * NEUTRON_MASS_KG / EV_TO_JOULES) ** 0.5 * 1.0e6  # μs·√eV/m


def tof_edges_ns_from_centers_us(tof_centers_us: np.ndarray) -> np.ndarray:
    """Convert ASCENDING TOF bin centers (μs) to ASCENDING bin edges (ns).

    For N centers we produce N+1 edges: interior edges at the
    midpoint of consecutive centers, outer edges by half-bin
    extrapolation. Output unit: nanoseconds.
    """
    n = len(tof_centers_us)
    edges_us = np.empty(n + 1, dtype=np.float64)
    edges_us[1:-1] = 0.5 * (tof_centers_us[:-1] + tof_centers_us[1:])
    edges_us[0] = tof_centers_us[0] - 0.5 * (tof_centers_us[1] - tof_centers_us[0])
    edges_us[-1] = tof_centers_us[-1] + 0.5 * (tof_centers_us[-1] - tof_centers_us[-2])
    return edges_us * 1000.0  # ns


def write_nexus(
    out: Path,
    counts_yxtof_u64: np.ndarray,
    tof_edges_ns: np.ndarray,
    flight_path_m: float,
    tof_offset_ns: float,
    title: str,
) -> None:
    """Write a single NeXus-schema HDF5 file with gzip=9 + shuffle.

    counts_yxtof_u64 must already be in (y, x, tof) ASC-TOF order;
    this function adds the rot_angle=1 axis at position 0.
    """
    if out.exists():
        out.unlink()
    counts_4d = counts_yxtof_u64[np.newaxis, ...]  # (1, y, x, tof)
    with h5py.File(out, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["title"] = title
        entry.attrs["flight_path_m"] = float(flight_path_m)
        entry.attrs["tof_offset_ns"] = float(tof_offset_ns)

        hist = entry.create_group("histogram")
        hist.attrs["NX_class"] = "NXdata"
        # Chunk shape: keep TOF axis whole per spatial tile so per-pixel
        # reads stream one chunk = one spatial tile of full TOF.
        chunks = (1, 32, 32, counts_4d.shape[3])
        chunks = tuple(min(c, s) for c, s in zip(chunks, counts_4d.shape))
        hist.create_dataset(
            "counts",
            data=counts_4d,
            chunks=chunks,
            compression="gzip",
            compression_opts=9,
            shuffle=True,
        )
        hist.create_dataset(
            "time_of_flight",
            data=tof_edges_ns,
            compression="gzip",
            compression_opts=9,
            shuffle=True,
        )


def main() -> None:
    print(f"[load] {SRC}  ({SRC.stat().st_size / 1e9:.1f} GB)")
    t0 = time.time()
    with h5py.File(SRC, "r") as f:
        ys = slice(CROP_Y0, CROP_Y0 + CROP_SIZE)
        xs = slice(CROP_X0, CROP_X0 + CROP_SIZE)
        # Source axis 0 is ASCENDING TOF (verified empirically — see
        # module docstring). Crop spatial; keep TOF axis 0 as-is.
        sample_tof_first = f["Hf/120min/counts/sample"][:, ys, xs]  # (4367, 256, 256)
        ob_tof_first = f["Hf/open_beam/counts"][:, ys, xs]
        # energies_eV is in ASCENDING ENERGY = DESCENDING TOF order.
        # Reverse once so the per-bin TOF computation comes out in
        # ASCENDING TOF order matching counts axis 0.
        energies_eV = f["Hf/120min/transmission/spectrum/energy_eV"][:]
    print(f"  load wall: {time.time()-t0:.1f}s")
    print(f"  sample shape (tof, y, x): {sample_tof_first.shape}  dtype: {sample_tof_first.dtype}")
    print(f"  ob     shape:             {ob_tof_first.shape}")
    assert energies_eV[0] < energies_eV[-1], "Expected ASCENDING energy in source"

    # Convert float32 counts to u64 (round + clamp non-negative).
    print("[convert] float32 → u64 (rint, clamp ≥ 0)")
    sample_u64 = np.maximum(np.rint(sample_tof_first), 0.0).astype(np.uint64)
    ob_u64 = np.maximum(np.rint(ob_tof_first), 0.0).astype(np.uint64)
    print(f"  sample: min={sample_u64.min()} max={sample_u64.max()} sum={sample_u64.sum():,}")
    print(f"  ob:     min={ob_u64.min()} max={ob_u64.max()} sum={ob_u64.sum():,}")

    # Reshape to (y, x, tof) in ASC-TOF order. Source is (tof, y, x);
    # moveaxis tof to last preserves bin order.
    sample_yxt = np.moveaxis(sample_u64, 0, -1)  # (y, x, tof)
    ob_yxt = np.moveaxis(ob_u64, 0, -1)
    assert sample_yxt.shape == (CROP_SIZE, CROP_SIZE, energies_eV.size)

    # Build TOF axis. energies_eV is ASC energy (= DESC TOF); reverse
    # once so tof_centers comes out ASC TOF, aligned with counts axis 3.
    energies_desc_E = energies_eV[::-1]  # DESC energy = ASC TOF
    tof_centers_us = TOF_FACTOR * FLIGHT_PATH_M / np.sqrt(energies_desc_E)
    assert tof_centers_us[0] < tof_centers_us[-1], (
        "TOF centers should be ASCENDING; got first > last"
    )
    tof_edges_ns = tof_edges_ns_from_centers_us(tof_centers_us)
    print(f"  tof_centers_us: range [{tof_centers_us[0]:.3f}, {tof_centers_us[-1]:.1f}] μs")
    print(f"  tof_edges_ns:   shape {tof_edges_ns.shape}, "
          f"range [{tof_edges_ns[0]:.1f}, {tof_edges_ns[-1]:.1f}] ns")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = OUT_DIR / "venus_hf_120min_sample.h5"
    ob_path = OUT_DIR / "venus_hf_open_beam.h5"

    print(f"\n[write] {sample_path.name} (gzip=9 + shuffle)")
    t0 = time.time()
    write_nexus(
        sample_path, sample_yxt, tof_edges_ns,
        FLIGHT_PATH_M, TOF_OFFSET_NS,
        title="VENUS Hf 120min sample (256x256 centered crop, full 4367 TOF bins)",
    )
    print(f"  done in {time.time()-t0:.1f}s; size {sample_path.stat().st_size / 1e6:.2f} MB")

    print(f"[write] {ob_path.name} (gzip=9 + shuffle)")
    t0 = time.time()
    write_nexus(
        ob_path, ob_yxt, tof_edges_ns,
        FLIGHT_PATH_M, TOF_OFFSET_NS,
        title="VENUS Hf open-beam (256x256 centered crop, full 4367 TOF bins)",
    )
    print(f"  done in {time.time()-t0:.1f}s; size {ob_path.stat().st_size / 1e6:.2f} MB")

    total_mb = (sample_path.stat().st_size + ob_path.stat().st_size) / 1e6
    print(f"\nTotal committed: {total_mb:.2f} MB across two files")


if __name__ == "__main__":
    main()
