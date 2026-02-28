#!/usr/bin/env python3
"""Fetch GitHub Copilot PR review comments via `gh api`.

Usage:
    python scripts/fetch_copilot_reviews.py 149 150 151 152
    python scripts/fetch_copilot_reviews.py 149 --json          # machine-readable
    python scripts/fetch_copilot_reviews.py 149 --summary-only  # one-line per comment

Requires: `gh` CLI authenticated with repo access.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field


@dataclass
class ReviewComment:
    pr: int
    path: str
    line: int | None
    body: str
    url: str
    group_id: int | None = field(default=None, repr=False)


def _gh_api(endpoint: str) -> list[dict]:
    """Call gh api with automatic pagination."""
    result = subprocess.run(
        ["gh", "api", "--paginate", endpoint],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error calling gh api {endpoint}: {result.stderr}", file=sys.stderr)
        return []

    # gh --paginate concatenates JSON arrays; parse them robustly
    raw = result.stdout.strip()
    if not raw:
        return []
    # Handle concatenated arrays: ][  ->  ,
    raw = raw.replace("]\n[", ",").replace("][", ",")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON from gh api {endpoint}", file=sys.stderr)
        return []


def fetch_copilot_comments(pr_number: int) -> list[ReviewComment]:
    """Fetch all Copilot-authored review comments for a PR."""
    endpoint = f"repos/ornlneutronimaging/NEREIDS/pulls/{pr_number}/comments"
    all_comments = _gh_api(endpoint)

    comments = []
    for c in all_comments:
        author = c.get("user", {}).get("login", "")
        # Copilot comments appear as user "Copilot" or "copilot-pull-request-reviewer[bot]"
        if "copilot" not in author.lower():
            continue
        comments.append(
            ReviewComment(
                pr=pr_number,
                path=c.get("path", ""),
                line=c.get("line") or c.get("original_line"),
                body=c.get("body", ""),
                url=c.get("html_url", ""),
            )
        )
    return comments


def _normalize_body(body: str) -> str:
    """Strip code suggestions, identifiers, and paths for similarity comparison."""
    # Remove markdown code blocks (```suggestion ... ```)
    body = re.sub(r"```.*?```", "", body, flags=re.DOTALL)
    # Remove all backtick-enclosed identifiers (e.g., `pi_over_k²`, `data.inner.clone()`)
    body = re.sub(r"`[^`]*`", "", body)
    # Remove file paths and line numbers
    body = re.sub(r"[\w/]+\.(rs|py|toml)(:\d+)?", "", body)
    # Remove markdown links
    body = re.sub(r"\[.*?\]\(.*?\)", "", body)
    # Strip punctuation so "energy)" and "energy" become the same token
    body = re.sub(r"[^\w\s]", " ", body)
    # Collapse whitespace
    return re.sub(r"\s+", " ", body).strip().lower()


def _deduplicate(comments: list[ReviewComment], threshold: float = 0.4) -> list[ReviewComment]:
    """Group comments with similar bodies; keep one representative per group.

    Uses word-level Jaccard similarity. Comments above *threshold* similarity
    are grouped together. The representative is the longest body in the group.

    Returns comments with ``group_id`` set. Duplicates are excluded from the
    returned list, but a count of duplicates is noted in the body.
    """
    if not comments:
        return comments

    # Compute word sets for each comment
    word_sets = [set(_normalize_body(c.body).split()) for c in comments]
    n = len(comments)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Also extract backtick-enclosed identifiers for concept-level grouping
    ident_sets = []
    for c in comments:
        idents = set(re.findall(r"`([^`]+)`", c.body))
        # Normalize: strip parens, trim whitespace
        idents = {re.sub(r"[()]+$", "", x).strip().lower() for x in idents}
        ident_sets.append(idents)

    for i in range(n):
        for j in range(i + 1, n):
            if not word_sets[i] or not word_sets[j]:
                continue
            # Pass 1: word-level Jaccard on normalized body
            intersection = len(word_sets[i] & word_sets[j])
            union_size = len(word_sets[i] | word_sets[j])
            word_sim = intersection / union_size if union_size > 0 else 0

            # Pass 2: shared backtick identifiers (concept-level)
            # If two comments share >50% of identifiers, they discuss the same thing
            ident_inter = len(ident_sets[i] & ident_sets[j])
            ident_union = len(ident_sets[i] | ident_sets[j])
            ident_sim = ident_inter / ident_union if ident_union > 0 else 0

            if word_sim >= threshold or ident_sim >= 0.5:
                union(i, j)

    # Group by root
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    result: list[ReviewComment] = []
    for gid, (_, members) in enumerate(sorted(groups.items())):
        # Pick the longest comment as representative
        rep_idx = max(members, key=lambda i: len(comments[i].body))
        rep = comments[rep_idx]
        rep.group_id = gid
        if len(members) > 1:
            others = [comments[i] for i in members if i != rep_idx]
            locs = ", ".join(
                f"PR#{c.pr} {c.path}:{c.line}" if c.line else f"PR#{c.pr} {c.path}"
                for c in others
            )
            rep.body += f"\n\n*({len(members) - 1} duplicate(s) at: {locs})*"
        result.append(rep)

    return result


def print_markdown(
    comments: list[ReviewComment], *, dedup: bool = False
) -> None:
    """Print comments grouped by PR in readable markdown."""
    if dedup:
        comments = _deduplicate(comments)
        deduped_label = " (deduplicated)"
    else:
        deduped_label = ""

    by_pr: dict[int, list[ReviewComment]] = {}
    for c in comments:
        by_pr.setdefault(c.pr, []).append(c)

    for pr, items in sorted(by_pr.items()):
        print(f"\n## PR #{pr} — {len(items)} Copilot comment(s){deduped_label}\n")
        if not items:
            print("No Copilot comments.\n")
            continue
        for i, c in enumerate(items, 1):
            loc = f"`{c.path}:{c.line}`" if c.line else f"`{c.path}`"
            print(f"### {i}. {loc}\n")
            print(c.body)
            print(f"\n[View on GitHub]({c.url})\n")


def print_summary(comments: list[ReviewComment]) -> None:
    """Print one-line summaries."""
    for c in comments:
        loc = f"{c.path}:{c.line}" if c.line else c.path
        # First line of body, truncated
        first_line = c.body.split("\n")[0][:120]
        print(f"PR#{c.pr}  {loc}  {first_line}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Copilot review comments from GitHub PRs"
    )
    parser.add_argument("prs", nargs="+", type=int, help="PR numbers to fetch")
    parser.add_argument(
        "--json", action="store_true", dest="as_json", help="Output as JSON"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="One-line summary per comment",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Deduplicate similar comments (Jaccard similarity >= 0.6)",
    )
    args = parser.parse_args()

    all_comments: list[ReviewComment] = []
    for pr in args.prs:
        all_comments.extend(fetch_copilot_comments(pr))

    if not all_comments:
        print("No Copilot comments found.", file=sys.stderr)
        sys.exit(0)

    if args.as_json:
        out = _deduplicate(all_comments) if args.dedup else all_comments
        print(json.dumps([asdict(c) for c in out], indent=2))
    elif args.summary_only:
        out = _deduplicate(all_comments) if args.dedup else all_comments
        print_summary(out)
    else:
        print_markdown(all_comments, dedup=args.dedup)


if __name__ == "__main__":
    main()
