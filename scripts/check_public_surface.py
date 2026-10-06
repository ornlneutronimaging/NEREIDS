#!/usr/bin/env python3
"""Fail when a change adds a public name without changing the pipeline map.

Public names are what rustdoc documents for the library crates under
``crates/``: each module and item by its path, with the inherent methods,
fields, variants and trait items on the item's page, and each module's
re-exports; and the public ``def`` / ``class`` names of the Python stub with
each class's public methods.  A name is added when the base commit does not
have it.

Usage: ``python3 scripts/check_public_surface.py BASE`` compares the merge base
of BASE and HEAD with HEAD.  When a crate or stub file differs and the map does
not, it builds the docs of HEAD and then of the base in this checkout, checking
out the base and then the original commit again, so tracked files must have no
uncommitted changes; ``target/doc`` then holds the base's docs.  The base is
built with lints capped at warnings.

Exit codes:
  0 - no name added, or the map changed too.
  1 - names added and ``docs/guide/src/pipeline-map.html`` unchanged.
  2 - unexpected error.
"""

from __future__ import annotations

import ast
import json
import os
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
REDIRECT = 'http-equiv="refresh"'


def doc_names(doc_dir: Path, crates: list[str]) -> set[str]:
    """``crate::module::Item`` and ``crate::module::Item::member`` for every documented name."""
    names: set[str] = set()
    for crate in crates:
        root = doc_dir / crate
        if not root.is_dir():
            raise RuntimeError(f"no docs at {root}")
        for page in root.rglob("*.html"):
            text = page.read_text(encoding="utf-8", errors="replace")
            if REDIRECT in text:
                continue
            rel = page.relative_to(root)
            module = (crate, *rel.parts[:-1])
            item = ITEM_PAGE.fullmatch(rel.name)
            if item:
                path = "::".join((*module, item.group(1)))
                names.add(path)
                own = TRAIT_IMPLS.split(text, maxsplit=1)[0]
                names |= {
                    f"{path}::{member}"
                    for member in MEMBER.findall(own)
                    if not member.endswith(".fields")
                }
            elif rel.name == "index.html":
                names.add("::".join(module))
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


def run(*args: str, env: dict[str, str] | None = None) -> str:
    return subprocess.run(
        args, cwd=REPO_ROOT, env=env, check=True, capture_output=True, text=True
    ).stdout


def checked_out_names(rustdocflags: str | None = None) -> set[str]:
    """Build the docs of the checked-out commit and return its public names."""
    env = os.environ.copy()
    if rustdocflags is not None:
        env["RUSTDOCFLAGS"] = rustdocflags
    run(*DOC, env=env)
    metadata = json.loads(run("cargo", "metadata", "--format-version", "1", "--no-deps"))
    crates_dir = (REPO_ROOT / "crates").resolve()
    crates = sorted(
        target["name"]
        for package in metadata["packages"]
        if Path(package["manifest_path"]).resolve().is_relative_to(crates_dir)
        for target in package["targets"]
        if "lib" in target["kind"]
    )
    stub = stub_names((REPO_ROOT / STUB).read_text(encoding="utf-8"))
    doc_dir = Path(metadata["target_directory"]) / "doc"
    return doc_names(doc_dir, crates) | {f"nereids.{name}" for name in stub}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        base = run("git", "merge-base", argv[1], "HEAD").strip()
        changed = set(run("git", "diff", "--name-only", base, "HEAD").split())
        if MAP in changed or not any(p.startswith("crates/") or p == STUB for p in changed):
            print("check_public_surface: the map changed, or no crate or stub file did")
            return 0
        dirty = run("git", "status", "--porcelain", "--untracked-files=no")
        if dirty.strip():
            raise RuntimeError(f"tracked files have uncommitted changes:\n{dirty}")
        after = checked_out_names()
        branch = subprocess.run(
            ("git", "symbolic-ref", "--quiet", "--short", "HEAD"),
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
        original = branch or run("git", "rev-parse", "HEAD").strip()
        capped = f"{os.environ.get('RUSTDOCFLAGS', '')} --cap-lints warn".strip()
        run("git", "checkout", "--quiet", "--detach", base)
        try:
            before = checked_out_names(capped)
        finally:
            run("git", "checkout", "--quiet", "--force", original)
        added = violations(before, after, changed)
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
