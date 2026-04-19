"""Parse /usr/bin/sample text output into inclusive + self-time tables.

The sample format is indented call tree; each leaf line represents samples
attributed to that leaf. We walk the tree depth-by-depth and accumulate:
  - inclusive[f]: samples where `f` appears anywhere on the stack
  - self[f]    : samples where `f` is the topmost frame
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path


LINE_RE = re.compile(
    r"""
    ^(?P<indent>[\s+!\|:]*)       # tree-drawing chars
    \s*(?P<count>\d+)\s+          # sample count
    (?P<name>.+?)                  # function name + location
    \s*$
    """,
    re.VERBOSE,
)


def clean_symbol(raw: str) -> str:
    """Normalize a frame name: keep just the rust path or system symbol."""
    raw = raw.strip()
    # Strip " (in module.so) ..." tail.
    if "(in " in raw:
        raw = raw.split("(in ", 1)[0].strip()
    # Collapse rust hash suffix h0123456789abcdef
    raw = re.sub(r"::h[0-9a-f]{16}$", "", raw)
    return raw


def parse(path: Path):
    text = path.read_text().splitlines()
    # Find the "Call graph:" section start
    start = None
    end = None
    for i, line in enumerate(text):
        if line.strip() == "Call graph:":
            start = i + 1
        elif line.strip().startswith("Total number in stack"):
            end = i
            break
        elif line.strip().startswith("Binary Images"):
            end = i
            break
    if start is None:
        raise SystemExit("no Call graph: marker")
    if end is None:
        end = len(text)

    inclusive = defaultdict(int)
    self_time = defaultdict(int)
    total_samples = 0

    # Parse the Call graph region into a flat (depth, count, name) list.
    # The format is pre-order DFS: the first line deeper than a frame is
    # its first child; subsequent deeper lines are either siblings at that
    # same depth or their descendants.
    lines: list[tuple[int, int, str]] = []
    for raw_line in text[start:end]:
        m = LINE_RE.match(raw_line)
        if not m:
            continue
        lines.append(
            (
                len(m.group("indent")),
                int(m.group("count")),
                clean_symbol(m.group("name")),
            )
        )

    # Inclusive time: each frame in the tree gets credited with the samples
    # recorded at its own line (parent-child accounting is already baked
    # into the sample output — a parent line's count always equals the
    # sum of its children's counts plus any self time).
    for _, count, name in lines:
        inclusive[name] += count

    # Self time: a frame's count minus the sum of its immediate children's
    # counts.  In the pre-order sample tree, immediate children are the
    # descendants at the `first_child_depth` level, which is the first
    # deeper indentation encountered after the frame's line.
    for i, (d, c, n) in enumerate(lines):
        if i + 1 >= len(lines) or lines[i + 1][0] <= d:
            # No descendants → all of `c` is self time.
            self_time[n] += c
        else:
            first_child_depth = lines[i + 1][0]
            child_total = 0
            k = i + 1
            while k < len(lines) and lines[k][0] > d:
                if lines[k][0] == first_child_depth:
                    child_total += lines[k][1]
                k += 1
            self_time[n] += max(c - child_total, 0)

    # Total = samples on the root frame (first line at minimum depth).
    if lines:
        min_depth = min(d for d, _, _ in lines)
        for d, c, n in lines:
            if d == min_depth:
                total_samples += c

    return inclusive, self_time, total_samples


def fmt(name: str, maxlen: int = 90) -> str:
    if len(name) > maxlen:
        return "..." + name[-(maxlen - 3):]
    return name


def report(inclusive, self_time, total):
    print(f"total samples: {total}  (~{total / 1000:.2f}s at 1 ms rate)\n")

    print("=== Top 25 INCLUSIVE (any frame in stack) ===")
    for name, ct in sorted(inclusive.items(), key=lambda x: -x[1])[:25]:
        pct = 100.0 * ct / total if total else 0.0
        print(f"  {pct:5.1f}%  {ct:6d}  {fmt(name)}")

    print("\n=== Top 25 SELF (time spent directly in function) ===")
    for name, ct in sorted(self_time.items(), key=lambda x: -x[1])[:25]:
        pct = 100.0 * ct / total if total else 0.0
        print(f"  {pct:5.1f}%  {ct:6d}  {fmt(name)}")

    print("\n=== nereids_* focused views ===")
    print("\n-- SELF time in nereids Rust code --")
    filt = [(n, c) for n, c in self_time.items() if "nereids" in n]
    for name, ct in sorted(filt, key=lambda x: -x[1])[:20]:
        pct = 100.0 * ct / total if total else 0.0
        print(f"  {pct:5.1f}%  {ct:6d}  {fmt(name)}")

    print("\n-- INCLUSIVE time in nereids Rust code --")
    filt = [(n, c) for n, c in inclusive.items() if "nereids" in n]
    for name, ct in sorted(filt, key=lambda x: -x[1])[:20]:
        pct = 100.0 * ct / total if total else 0.0
        print(f"  {pct:5.1f}%  {ct:6d}  {fmt(name)}")


if __name__ == "__main__":
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/a1.sample.txt")
    inc, self_t, total = parse(path)
    report(inc, self_t, total)
