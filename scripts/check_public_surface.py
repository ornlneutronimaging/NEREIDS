#!/usr/bin/env python3
"""Fail when a change adds a public name without changing the pipeline map.

Public names are what rustdoc documents for the crates under ``crates/``: each
item by its module path, with the inherent methods, fields, variants and trait
items on its page, and each module's re-exports; and the public ``def`` /
``class`` names of the Python stub with each class's public methods.  A name
is added when the base commit does not have it.

Usage: ``python3 scripts/check_public_surface.py BASE`` compares the merge base
of BASE and HEAD with HEAD.  When a crate or stub file differs it builds the
docs of both in this checkout, checking out the base and then the original
commit again, so tracked files must have no uncommitted changes.

Exit codes:
  0 - no name added, or the map changed too.
  1 - names added and ``docs/guide/src/pipeline-map.html`` unchanged.
  2 - unexpected error.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

from check_python_api_drift import public_definitions

REPO_ROOT = Path(__file__).resolve().parent.parent
MAP = "docs/guide/src/pipeline-map.html"
STUB = "bindings/python/python/nereids/__init__.pyi"
DOC = ("cargo", "doc", "--workspace", "--no-deps", "--exclude", "nereids-python")
ITEM_PAGE = re.compile(
    r"(?:struct|enum|union|trait|traitalias|fn|type|constant|static|macro|derive|attr)"
    r"\.(\w+)\.html"
)
MEMBER = re.compile(
    r'id="(?:method|tymethod|structfield|variant|associatedconstant|associatedtype)'
    r'\.([\w.]+?)(?:-\d+)?"'
)
REEXPORT = re.compile(r'id="reexport\.(\w+)"')
TRAIT_IMPLS = re.compile(r'id="(?:trait|synthetic|blanket)-implementations"')


def doc_names(doc_dir: Path, crates: list[str]) -> set[str]:
    """``crate::module::Item`` and ``crate::module::Item::member`` for every documented name."""
    names: set[str] = set()
    for crate in crates:
        root = doc_dir / crate
        for page in root.rglob("*.html"):
            rel = page.relative_to(root)
            module = (crate, *rel.parts[:-1])
            text = page.read_text(encoding="utf-8", errors="replace")
            item = ITEM_PAGE.fullmatch(rel.name)
            if item:
                path = "::".join((*module, item.group(1)))
                names.add(path)
                own = TRAIT_IMPLS.split(text, maxsplit=1)[0]
                names |= {f"{path}::{member}" for member in MEMBER.findall(own)}
            elif rel.name == "index.html":
                names |= {"::".join((*module, name)) for name in REEXPORT.findall(text)}
    return names


def stub_names(text: str) -> set[str]:
    """Public top-level names of the stub and ``Class.method`` for each class."""
    names: set[str] = set()
    for node in public_definitions(ast.parse(text).body):
        names.add(node.name)
        if isinstance(node, ast.ClassDef):
            names |= {f"{node.name}.{m.name}" for m in public_definitions(node.body)}
    return names


def violations(base: set[str], head: set[str], changed: set[str]) -> list[str]:
    """The names ``head`` adds to ``base`` when the map is not among ``changed``."""
    return [] if MAP in changed else sorted(head - base)


def run(*args: str) -> str:
    return subprocess.run(args, cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout


def checked_out_names() -> set[str]:
    """Build the docs of the checked-out commit and return its public names."""
    run(*DOC)
    target = Path(json.loads(run("cargo", "metadata", "--format-version", "1", "--no-deps"))[
        "target_directory"
    ])
    crates = sorted(
        p.name.replace("-", "_")
        for p in (REPO_ROOT / "crates").iterdir()
        if (p / "src" / "lib.rs").exists()
    )
    stub = stub_names((REPO_ROOT / STUB).read_text(encoding="utf-8"))
    return doc_names(target / "doc", crates) | {f"nereids.{name}" for name in stub}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        base = run("git", "merge-base", argv[1], "HEAD").strip()
        changed = set(run("git", "diff", "--name-only", base, "HEAD").split())
        if not any(path.startswith("crates/") or path == STUB for path in changed):
            print("check_public_surface: no crate or stub file changed")
            return 0
        if run("git", "status", "--porcelain", "--untracked-files=no").strip():
            raise RuntimeError("tracked files have uncommitted changes; commit them first")
        branch = subprocess.run(
            ("git", "symbolic-ref", "--quiet", "--short", "HEAD"),
            cwd=REPO_ROOT, capture_output=True, text=True,
        ).stdout.strip()
        original = branch or run("git", "rev-parse", "HEAD").strip()
        run("git", "checkout", "--quiet", "--detach", base)
        try:
            before = checked_out_names()
        finally:
            run("git", "checkout", "--quiet", original)
        added = violations(before, checked_out_names(), changed)
    except Exception as exc:
        detail = getattr(exc, "stderr", None) or exc
        print(f"check_public_surface: {str(detail).strip()}", file=sys.stderr)
        return 2
    if added:
        print(f"New public names with {MAP} unchanged;")
        print("a public-surface addition comes with a contract change:")
        print("\n".join(f"  {name}" for name in added))
        return 1
    print("check_public_surface: no public name added without a map change")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
