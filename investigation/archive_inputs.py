"""Locate and verify the user-supplied research archive for investigation probes.

By default the scripts read ``/Users/chenzhang/Downloads/Archive.zip`` and
extract its verified contents into a deterministic directory under ``/tmp``.
Set ``NEREIDS_ARCHIVE_ZIP`` to use the same archive at another path, or set
``NEREIDS_ARCHIVE_ROOT`` to use an already extracted tree.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import zipfile


EXPECTED_SHA256 = "8b5afb3f1a4efcb5b822a955a633403afe48b62baa05f0b49c0d55567ab9e0d2"
DEFAULT_ZIP = Path("/Users/chenzhang/Downloads/Archive.zip")
REQUIRED_MEMBER = Path("01_spectral_lineshape_bias/data/region_counts.npz")


def archive_zip() -> Path:
    return Path(os.environ.get("NEREIDS_ARCHIVE_ZIP", DEFAULT_ZIP))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_root() -> Path:
    configured = os.environ.get("NEREIDS_ARCHIVE_ROOT")
    if configured:
        root = Path(configured)
        if not (root / REQUIRED_MEMBER).is_file():
            raise FileNotFoundError(
                f"NEREIDS_ARCHIVE_ROOT lacks {REQUIRED_MEMBER}: {root}"
            )
        return root

    source = archive_zip()
    observed = sha256(source)
    if observed != EXPECTED_SHA256:
        raise ValueError(
            f"Archive SHA-256 mismatch: expected {EXPECTED_SHA256}, got {observed}"
        )

    root = Path("/tmp") / f"nereids-archive-{observed[:12]}"
    if not (root / REQUIRED_MEMBER).is_file():
        root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source) as archive:
            archive.extractall(root)
    if not (root / REQUIRED_MEMBER).is_file():
        raise FileNotFoundError(f"Archive extraction lacks {REQUIRED_MEMBER}: {root}")
    return root
