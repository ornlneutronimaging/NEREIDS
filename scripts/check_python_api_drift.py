#!/usr/bin/env python3
"""Check drift between the PyO3 type stubs and the curated Python API page.

Compares the public symbols (top-level ``def`` / ``class`` not prefixed with
``_``) declared in ``bindings/python/python/nereids/__init__.pyi`` against
mentions in ``docs/guide/src/python-api.md``.  A symbol counts as documented
if it appears in a markdown heading, an inline ``code`` span, or as
``nereids.symbol(...)`` / ``symbol(...)`` in a code block.

Symbols intentionally omitted from the curated narrative (e.g. those covered
in ``data-io.md`` or low-level utilities not yet given a section) are listed
in ``scripts/python_api_allowlist.txt``.  The allowlist is the drift ratchet:
prefer moving entries OUT of it by documenting them, rather than letting it
grow unboundedly.

Exit codes:
  0 - no drift; every public stub symbol is either documented or allowlisted.
  1 - one or more public stub symbols are neither documented nor allowlisted.
  2 - unexpected error (missing input file, parse failure, etc.).

Run via ``pixi run lint-docs`` or directly with ``python``.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STUB_PATH = REPO_ROOT / "bindings" / "python" / "python" / "nereids" / "__init__.pyi"
DOC_PATH = REPO_ROOT / "docs" / "guide" / "src" / "python-api.md"
ALLOWLIST_PATH = REPO_ROOT / "scripts" / "python_api_allowlist.txt"


def collect_stub_symbols(stub_path: Path) -> set[str]:
    """Return public top-level ``def`` / ``class`` names from the stub."""
    tree = ast.parse(stub_path.read_text(encoding="utf-8"), filename=str(stub_path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
    return names


def collect_documented_symbols(doc_path: Path, candidates: set[str]) -> set[str]:
    """Return the subset of ``candidates`` that appears in ``doc_path``."""
    text = doc_path.read_text(encoding="utf-8")
    documented: set[str] = set()
    for name in candidates:
        # Match: heading (`# name`), inline `name`, `name(`, `nereids.name`,
        # or `nereids.name(` — all forms readers actually use to refer to a
        # symbol.  ``\b`` boundaries keep ``normalize`` from matching
        # ``normalized``.
        pattern = re.compile(
            r"(?:^#+[^\n]*\b" + re.escape(name) + r"\b)"
            r"|(?:`" + re.escape(name) + r"[`(])"
            r"|(?:`nereids\." + re.escape(name) + r"[`(])"
            r"|(?:\bnereids\." + re.escape(name) + r"\s*\()"
            r"|(?:\b" + re.escape(name) + r"\s*\()",
            re.MULTILINE,
        )
        if pattern.search(text):
            documented.add(name)
    return documented


def load_allowlist(path: Path) -> set[str]:
    """Read one-symbol-per-line allowlist; ``#`` and blank lines are ignored."""
    if not path.exists():
        return set()
    entries: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            entries.add(line)
    return entries


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    for required in (STUB_PATH, DOC_PATH):
        if not required.exists():
            print(f"error: required input missing: {required}", file=sys.stderr)
            return 2
    stub_symbols = collect_stub_symbols(STUB_PATH)
    documented = collect_documented_symbols(DOC_PATH, stub_symbols)
    allowlist = load_allowlist(ALLOWLIST_PATH)
    drift = sorted(stub_symbols - documented - allowlist)
    if drift:
        print(
            "drift detected: the following public symbols in __init__.pyi "
            "are not documented in docs/guide/src/python-api.md and not in "
            "scripts/python_api_allowlist.txt:"
        )
        for name in drift:
            print(f"  - {name}")
        print(
            "Add a section for each to docs/guide/src/python-api.md, OR if "
            "the symbol is intentionally omitted from the curated narrative "
            "reference, add it to scripts/python_api_allowlist.txt with a "
            "justification comment."
        )
        return 1
    print(
        f"python-api.md drift check passed: {len(stub_symbols)} public "
        f"symbols, {len(documented)} documented, {len(allowlist)} allowlisted"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — exit code 2 is the contract
        print(f"unexpected error: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
