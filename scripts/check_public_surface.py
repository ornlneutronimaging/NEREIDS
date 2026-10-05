#!/usr/bin/env python3
"""Fail when a change adds a public name without changing the pipeline map.

Public names are the ``pub`` items of the crates under ``crates/*/src``, with
a method named by the type of its ``impl`` block, the ``pub`` fields of a
``pub struct`` and the variants of a ``pub enum``; and the public ``def`` /
``class`` names of the Python stub with each class's public methods.  A name
is added when no file of its crate, or the stub, has it at the base commit.

Usage: ``python scripts/check_public_surface.py BASE`` compares the merge base
of BASE and HEAD with HEAD.

Exit codes:
  0 - no name added, or the map changed too.
  1 - names added and ``docs/guide/src/pipeline-map.html`` unchanged.
  2 - unexpected error.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

from check_python_api_drift import public_definitions

REPO_ROOT = Path(__file__).resolve().parent.parent
MAP = "docs/guide/src/pipeline-map.html"
STUB = "bindings/python/python/nereids/__init__.pyi"
ITEM = re.compile(
    r'^(\s*)pub\s+(?:(?:unsafe|async|const|extern\s+"[^"]*")\s+)*'
    r"(?:fn|struct|enum|trait|type|const|static|union)\s+([A-Za-z_]\w*)"
)
BLOCK = re.compile(r"^\s*pub\s+(struct|enum)\s+([A-Za-z_]\w*)[^;(]*\{\s*$")
FIELD = re.compile(r"^\s*pub\s+([a-z_]\w*)\s*:")
VARIANT = re.compile(r"^\s*([A-Z]\w*)\s*(?:[({,=]|$)")


def impl_type(line: str) -> str | None:
    """The type an ``impl`` line implements on, or None for any other line."""
    rest = line.strip()
    if not re.match(r"impl\b", rest):
        return None
    rest = rest[4:]
    if rest.startswith("<"):
        depth = 0
        for i, c in enumerate(rest):
            depth += (c == "<") - (c == ">")
            if depth == 0:
                rest = rest[i + 1 :]
                break
    rest = rest.split(" for ", 1)[-1]
    match = re.match(r"\s*(?:[\w]+::)*(\w+)", rest)
    return match.group(1) if match else None


def rust_names(files: dict[str, str]) -> set[str]:
    """``crate::name``, ``crate::Type::member`` and ``crate::Struct.field`` names."""
    names: set[str] = set()
    for path, text in files.items():
        crate = path.split("/")[1]
        blocks: list[tuple[int, str, str]] = []
        for line in text.splitlines():
            indent = len(line) - len(line.lstrip())
            if blocks and line.rstrip() == " " * blocks[-1][0] + "}":
                blocks.pop()
                continue
            item = ITEM.match(line)
            if item:
                owners = [name for i, name, kind in blocks if kind == "impl" and i < indent]
                prefix = f"{owners[-1]}::" if owners else ""
                names.add(f"{crate}::{prefix}{item.group(2)}")
                block = BLOCK.match(line)
                if block:
                    blocks.append((indent, block.group(2), block.group(1)))
                continue
            owner = impl_type(line)
            if owner:
                blocks.append((indent, owner, "impl"))
                continue
            if blocks and indent > blocks[-1][0]:
                outer, name, kind = blocks[-1]
                field = FIELD.match(line) if kind == "struct" else None
                variant = VARIANT.match(line) if kind == "enum" and indent == outer + 4 else None
                if field:
                    names.add(f"{crate}::{name}.{field.group(1)}")
                elif variant:
                    names.add(f"{crate}::{name}::{variant.group(1)}")
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


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout


def surface(rev: str) -> set[str]:
    paths = git("ls-tree", "-r", "--name-only", rev, "--", "crates").split()
    sources = {
        p: git("show", f"{rev}:{p}")
        for p in paths
        if re.fullmatch(r"crates/[^/]+/src/.+\.rs", p)
    }
    stub = stub_names(git("show", f"{rev}:{STUB}"))
    return rust_names(sources) | {f"nereids.{n}" for n in stub}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        base = git("merge-base", argv[1], "HEAD").strip()
        changed = set(git("diff", "--name-only", base, "HEAD").split())
        added = violations(surface(base), surface("HEAD"), changed)
    except subprocess.CalledProcessError as exc:
        print(f"check_public_surface: {exc.stderr.strip() or exc}", file=sys.stderr)
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
