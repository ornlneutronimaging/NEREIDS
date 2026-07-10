"""Verify archive identity, integrity, member count, and notebook structure."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from pathlib import PurePosixPath
import zipfile

import nbformat

from archive_inputs import EXPECTED_SHA256, archive_root, archive_zip, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extract-images",
        type=Path,
        help="optional directory for decoded notebook image/png outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = archive_zip()
    observed = sha256(source)
    with zipfile.ZipFile(source) as archive:
        bad_member = archive.testzip()
        members = archive.infolist()
    unsafe_members = [
        member.filename
        for member in members
        if PurePosixPath(member.filename).is_absolute()
        or ".." in PurePosixPath(member.filename).parts
    ]

    root = archive_root()
    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    notebook_summary: dict[str, dict[str, int]] = {}
    extracted_images: list[str] = []
    if args.extract_images:
        args.extract_images.mkdir(parents=True, exist_ok=True)
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        code = [cell for cell in notebook.cells if cell.cell_type == "code"]
        png_outputs = []
        for cell_index, cell in enumerate(code):
            for output_index, output in enumerate(cell.get("outputs", [])):
                png = output.get("data", {}).get("image/png")
                if png:
                    png_outputs.append(png)
                    if args.extract_images:
                        target = args.extract_images / (
                            f"{path.stem}-cell{cell_index:02d}-output{output_index:02d}.png"
                        )
                        target.write_bytes(base64.b64decode(png))
                        extracted_images.append(str(target))
        notebook_summary[path.name] = {
            "cells": len(notebook.cells),
            "code_cells": len(code),
            "executed_code_cells": sum(cell.execution_count is not None for cell in code),
            "embedded_png_outputs": len(png_outputs),
        }

    print(
        json.dumps(
            {
                "archive": str(source),
                "sha256": observed,
                "sha256_matches": observed == EXPECTED_SHA256,
                "members": len(members),
                "integrity_bad_member": bad_member,
                "uncompressed_bytes": sum(member.file_size for member in members),
                "unsafe_members": unsafe_members,
                "extraction_root": str(root),
                "extracted_images": extracted_images,
                "notebooks": notebook_summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
