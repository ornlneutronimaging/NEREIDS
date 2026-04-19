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

    # depth → current frame name at that depth. When we see a frame at depth d,
    # the current "deeper" entries are superseded.
    stack: list[tuple[int, str, int]] = []  # (depth, name, count)

    total_samples = 0

    for raw_line in text[start:end]:
        m = LINE_RE.match(raw_line)
        if not m:
            continue
        indent = m.group("indent")
        # depth = number of leading whitespace units (use the tree drawing chars)
        # Every "+" or "|" or ":" or " " etc. contributes to indent width.
        depth = len(indent)
        count = int(m.group("count"))
        name = clean_symbol(m.group("name"))

        # Pop deeper-or-equal frames.
        while stack and stack[-1][0] >= depth:
            stack.pop()
        stack.append((depth, name, count))

        # Inclusive: every frame from root to here gets +count. But the parent
        # will already have added count through its own line (since sample
        # output lists each frame in the tree). So we only count THIS line.
        inclusive[name] += count

    # We need to find the "leaves" — lines whose sample count is not
    # explained by a deeper sibling. Re-walk and track whether a frame
    # has any child lines immediately indented deeper. This is trickier.
    # Simpler: a frame at depth d with count N is a leaf iff the next
    # line is either the same depth or shallower.
    lines = []
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

    for i, (d, c, n) in enumerate(lines):
        # Look ahead: if the next line is deeper, this frame has descendants
        # → those samples belong to the child, so "self" at this frame =
        # count - sum(deeper children directly under it).
        j = i + 1
        child_sum = 0
        while j < len(lines):
            dj, cj, nj = lines[j]
            if dj <= d:
                break
            if dj > d and all(lines[k][0] > d for k in range(i + 1, j)):
                # `j` is a direct child only if every line between i and j
                # is deeper than d. But since sample output is pre-order DFS,
                # the first deeper line after i is always a direct child;
                # subsequent deeper lines may be descendants, not siblings.
                pass
            # Immediate children have depth strictly greater than d AND the
            # path from i to j never dips back to d. Since the tree is
            # pre-order, we just track: immediate child when depth == d+k
            # where k is the fixed indent-per-level increment.
            j += 1
        # Simpler self calc: self = count of this frame minus sum of its
        # immediate children's counts. Immediate children are lines with
        # depth of the *first* descendant.
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
