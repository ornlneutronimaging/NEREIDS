#!/usr/bin/env python3
"""Strip, list, delete and report the comments of Rust source files.

    comment_audit.py strip FILE... --out DIR [--root ROOT]
    comment_audit.py bundle FILE... --out DIR [--root ROOT] [--range RANGE --repo REPO]
    comment_audit.py apply FILE --comments JSON --delete ID[,ID...]
    comment_audit.py report RESULT_JSON
    comment_audit.py verify-isolation TRANSCRIPT_DIR

`strip` writes `<out>/<relpath>.stripped.rs` with every comment removed and
`<out>/<relpath>.comments.json` describing each removed comment and the code
line it sits on. `// SAFETY:` comments are kept, as the lint requires them.
`bundle` prints the `files` argument of the comment-audit workflow from those
outputs: the stripped text in windows around the comments in scope, which with
`--range` are the comments touching lines the range added, else all of them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field

EXEMPT_PREFIXES = ("// SAFETY:", "// SAFETY ")


@dataclass
class Span:
    kind: str
    start: int
    end: int


@dataclass
class Comment:
    id: str
    kind: str
    first_line: int
    last_line: int
    trailing: bool
    exempt: bool
    text: list[str]
    anchor: dict | None = None
    spans: list[list[int]] = field(default_factory=list)


def _is_ident(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def lex_comments(text: str) -> list[Span]:
    """Character spans of every comment, respecting strings, raw strings,
    byte strings, char literals and lifetimes."""
    spans: list[Span] = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        two = text[i : i + 2]
        if two == "//":
            end = text.find("\n", i)
            end = n if end < 0 else end
            three = text[i : i + 3]
            if three == "//!":
                kind = "inner_doc"
            elif three == "///" and text[i : i + 4] != "////":
                kind = "doc"
            else:
                kind = "line"
            spans.append(Span(kind, i, end))
            i = end
            continue
        if two == "/*":
            depth = 0
            j = i
            while j < n:
                if text[j : j + 2] == "/*":
                    depth += 1
                    j += 2
                elif text[j : j + 2] == "*/":
                    depth -= 1
                    j += 2
                    if depth == 0:
                        break
                else:
                    j += 1
            spans.append(Span("block", i, j))
            i = j
            continue
        prev_ident = i > 0 and _is_ident(text[i - 1])
        if ch in "rbc" and not prev_ident:
            j = i + 1
            if ch in "bc" and j < n and text[j] == "r":
                j += 1
            if j < n and (text[j] == '"' or text[j] == "#") and text[j - 1] == "r":
                hashes = 0
                while j < n and text[j] == "#":
                    hashes += 1
                    j += 1
                if j < n and text[j] == '"':
                    closer = '"' + "#" * hashes
                    end = text.find(closer, j + 1)
                    i = n if end < 0 else end + len(closer)
                    continue
            if ch in "bc" and j < n and text[j] == '"' and j == i + 1:
                i = _skip_string(text, j)
                continue
            if ch == "b" and j < n and text[j] == "'" and j == i + 1:
                i = _skip_char_or_lifetime(text, j)
                continue
        if ch == '"':
            i = _skip_string(text, i)
            continue
        if ch == "'" and not prev_ident:
            i = _skip_char_or_lifetime(text, i)
            continue
        i += 1
    return spans


def _skip_string(text: str, i: int) -> int:
    n = len(text)
    j = i + 1
    while j < n:
        if text[j] == "\\":
            j += 2
            continue
        if text[j] == '"':
            return j + 1
        j += 1
    return n


def _skip_char_or_lifetime(text: str, i: int) -> int:
    n = len(text)
    if i + 1 < n and text[i + 1] == "\\":
        j = i + 2
        while j < n and text[j] != "'":
            j += 1
        return min(j + 1, n)
    if i + 2 < n and text[i + 2] == "'":
        return i + 3
    return i + 1


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for k, ch in enumerate(text):
        if ch == "\n":
            starts.append(k + 1)
    return starts


def _line_of(starts: list[int], pos: int) -> int:
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= pos:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


def group_comments(text: str) -> list[Comment]:
    """Comments of `text`, consecutive whole-line comments of one kind
    grouped into one, each with its 1-based line span."""
    spans = lex_comments(text)
    starts = _line_starts(text)
    comments: list[Comment] = []
    for sp in spans:
        first = _line_of(starts, sp.start)
        last = _line_of(starts, max(sp.start, sp.end - 1))
        before = text[starts[first - 1] : sp.start]
        trailing = bool(before.strip())
        chunk = text[sp.start : sp.end]
        chunk_lines = chunk.split("\n")
        prev = comments[-1] if comments else None
        category = "doc" if sp.kind in ("doc", "inner_doc") else sp.kind
        prev_category = None
        if prev is not None:
            prev_category = "doc" if prev.kind in ("doc", "inner_doc") else prev.kind
        if (
            prev is not None
            and not trailing
            and not prev.trailing
            and prev.last_line + 1 == first
            and category == prev_category
            and sp.kind != "block"
            and not prev.exempt
        ):
            prev.last_line = last
            prev.text.extend(chunk_lines)
            prev.spans.append([sp.start, sp.end])
            continue
        exempt = chunk_lines[0].strip().startswith(EXEMPT_PREFIXES)
        comments.append(
            Comment(
                id=f"c{len(comments) + 1}",
                kind=sp.kind,
                first_line=first,
                last_line=last,
                trailing=trailing,
                exempt=exempt,
                text=chunk_lines,
                spans=[[sp.start, sp.end]],
            )
        )
    return comments


def strip_text(text: str, comments: list[Comment]) -> tuple[str, dict[int, int]]:
    """The text with every non-exempt comment removed, and the map from
    original line number to stripped line number for the lines kept."""
    cut = []
    for c in comments:
        if not c.exempt:
            cut.extend(c.spans)
    cut.sort()
    pieces = []
    pos = 0
    for a, b in cut:
        pieces.append(text[pos:a])
        pieces.append("\x00" * (b - a))
        pos = b
    pieces.append(text[pos:])
    marked = "".join(pieces)
    trailing_newline = marked.endswith("\n")
    original_lines = marked[:-1].split("\n") if trailing_newline else marked.split("\n")
    kept: list[str] = []
    line_map: dict[int, int] = {}
    for k, line in enumerate(original_lines, start=1):
        had_comment = "\x00" in line
        clean = line.replace("\x00", "").rstrip()
        if had_comment and not clean.strip():
            continue
        kept.append(clean)
        line_map[k] = len(kept)
    return "\n".join(kept) + ("\n" if trailing_newline else ""), line_map


def attach_anchors(comments: list[Comment], stripped: str, line_map: dict[int, int]) -> None:
    stripped_lines = stripped.rstrip("\n").split("\n")
    for c in comments:
        if c.exempt:
            continue
        candidates = [
            line_map[k]
            for k in sorted(line_map)
            if (k >= c.first_line if c.trailing else k > c.last_line)
        ]
        c.anchor = None
        for s in candidates:
            if stripped_lines[s - 1].strip():
                c.anchor = {"line": s, "text": stripped_lines[s - 1].strip()}
                break


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def cmd_strip(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    for path in args.files:
        abs_path = os.path.abspath(path)
        rel = os.path.relpath(abs_path, root)
        with open(abs_path, encoding="utf-8") as fh:
            text = fh.read()
        comments = group_comments(text)
        stripped, line_map = strip_text(text, comments)
        attach_anchors(comments, stripped, line_map)
        out_base = os.path.join(args.out, rel)
        os.makedirs(os.path.dirname(out_base), exist_ok=True)
        with open(out_base + ".stripped.rs", "w", encoding="utf-8") as fh:
            fh.write(stripped)
        listed = [asdict(c) for c in comments if not c.exempt]
        for c in listed:
            c.pop("spans")
        with open(out_base + ".comments.json", "w", encoding="utf-8") as fh:
            json.dump(
                {"file": rel, "sha256": digest(text), "comments": listed},
                fh,
                indent=1,
            )
        print(f"{rel}: {len(listed)} comments, {len(text.split(chr(10)))} -> "
              f"{len(stripped.split(chr(10)))} lines")
    return 0


def apply_deletions(text: str, comments: list[Comment], ids: set[str]) -> str:
    keep = [c for c in comments if c.id not in ids]
    victims = [c for c in comments if c.id in ids]
    for c in victims:
        c.exempt = False
    for c in keep:
        c.exempt = True
    stripped, _ = strip_text(text, comments)
    return stripped


def cmd_apply(args: argparse.Namespace) -> int:
    with open(args.comments, encoding="utf-8") as fh:
        listed = json.load(fh)
    with open(args.file, encoding="utf-8") as fh:
        text = fh.read()
    if listed["sha256"] != digest(text):
        print(f"{args.file} changed since it was stripped", file=sys.stderr)
        return 2
    ids = {s for s in args.delete.split(",") if s}
    comments = group_comments(text)
    known = {c.id for c in comments if not c.exempt}
    unknown = ids - known
    if unknown:
        print(f"unknown comment ids: {sorted(unknown)}", file=sys.stderr)
        return 2
    new_text = apply_deletions(text, comments, ids)
    with open(args.file, "w", encoding="utf-8") as fh:
        fh.write(new_text)
    print(f"{args.file}: deleted {len(ids)} comments")
    return 0


def added_ranges(diff_text: str) -> list[tuple[int, int]]:
    """Inclusive 1-based line ranges the new side of a `diff -U0` adds."""
    ranges = []
    for m in re.finditer(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", diff_text, re.MULTILINE):
        start = int(m.group(1))
        count = int(m.group(2)) if m.group(2) is not None else 1
        if count > 0:
            ranges.append((start, start + count - 1))
    return ranges


def ids_in_ranges(comments: list[dict], ranges: list[tuple[int, int]]) -> list[str]:
    return [
        c["id"]
        for c in comments
        if any(c["first_line"] <= hi and c["last_line"] >= lo for lo, hi in ranges)
    ]


WINDOW_LINES = 60


def windows_around(lines: list[str], centers: list[int], reach: int = WINDOW_LINES) -> list[dict]:
    """Merged windows of `lines` (1-based `start`/`end`) reaching `reach`
    lines either side of each center."""
    spans = sorted((max(1, c - reach), min(len(lines), c + reach)) for c in centers)
    merged: list[list[int]] = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [{"start": lo, "end": hi, "lines": lines[lo - 1 : hi]} for lo, hi in merged]


def cmd_bundle(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    files = []
    for path in args.files:
        rel = os.path.relpath(os.path.abspath(path), root)
        base = os.path.join(args.out, rel)
        with open(base + ".stripped.rs", encoding="utf-8") as fh:
            stripped_lines = fh.read().rstrip("\n").split("\n")
        with open(base + ".comments.json", encoding="utf-8") as fh:
            listed = json.load(fh)
        comments = listed["comments"]
        if args.range:
            diff = subprocess.run(
                ["git", "-C", args.repo, "diff", "-U0", args.range, "--", rel],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            scope = ids_in_ranges(comments, added_ranges(diff))
        else:
            scope = [c["id"] for c in comments]
        in_scope = [c for c in comments if c["id"] in scope]
        centers = [c["anchor"]["line"] for c in in_scope if c["anchor"]]
        if not centers:
            continue
        files.append(
            {
                "path": rel,
                "windows": windows_around(stripped_lines, centers),
                "comments": in_scope,
            }
        )
    json.dump({"files": files}, sys.stdout)
    return 0


def tool_uses(transcript: str) -> dict[str, int]:
    """Tool calls an agent made, from its transcript."""
    counts: dict[str, int] = {}

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "tool_use" and "name" in node:
                counts[node["name"]] = counts.get(node["name"], 0) + 1
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    with open(transcript, encoding="utf-8") as fh:
        for line in fh:
            try:
                walk(json.loads(line))
            except json.JSONDecodeError:
                continue
    return counts


def cmd_verify_isolation(args: argparse.Namespace) -> int:
    """Every writer and judge may call the structured output tool and nothing
    else; any other call means it could have seen the original."""
    ok = True
    for name in sorted(os.listdir(args.transcript_dir)):
        if not (name.startswith("agent-") and name.endswith(".jsonl")):
            continue
        counts = tool_uses(os.path.join(args.transcript_dir, name))
        others = {k: v for k, v in counts.items() if k != "StructuredOutput"}
        status = "ok" if not others else f"INVALID {others}"
        print(f"{name}: {status}")
        ok = ok and not others
    return 0 if ok else 1


def cmd_report(args: argparse.Namespace) -> int:
    with open(args.result, encoding="utf-8") as fh:
        result = json.load(fh)
    rows = result.get("perFile", [])
    print(f"{'file':60} {'scoped':>6} {'absent':>6} {'same':>5} {'keep':>5} {'flag':>5}")
    for r in rows:
        print(
            f"{r['file']:60} {r['scoped']:6d} {len(r['absent']):6d} {len(r['same']):5d} "
            f"{len(r['keep']):5d} {len(r['flagged']):5d}"
        )
    for r in rows:
        for f in r["flagged"]:
            print(f"\n--- {r['file']} {f['id']} (line {f['line']})")
            print("original:   " + " ".join(f["original"]))
            print("regenerated: " + f["regenerated"])
            print("judge:      " + f["reason"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_strip = sub.add_parser("strip")
    p_strip.add_argument("files", nargs="+")
    p_strip.add_argument("--out", required=True)
    p_strip.add_argument("--root", default=".")
    p_strip.set_defaults(func=cmd_strip)
    p_bundle = sub.add_parser("bundle")
    p_bundle.add_argument("files", nargs="+")
    p_bundle.add_argument("--out", required=True)
    p_bundle.add_argument("--root", default=".")
    p_bundle.add_argument("--range")
    p_bundle.add_argument("--repo", default=".")
    p_bundle.set_defaults(func=cmd_bundle)
    p_apply = sub.add_parser("apply")
    p_apply.add_argument("file")
    p_apply.add_argument("--comments", required=True)
    p_apply.add_argument("--delete", required=True)
    p_apply.set_defaults(func=cmd_apply)
    p_report = sub.add_parser("report")
    p_report.add_argument("result")
    p_report.set_defaults(func=cmd_report)
    p_iso = sub.add_parser("verify-isolation")
    p_iso.add_argument("transcript_dir")
    p_iso.set_defaults(func=cmd_verify_isolation)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
