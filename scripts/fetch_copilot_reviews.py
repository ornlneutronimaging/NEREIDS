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
import subprocess
import sys
from dataclasses import asdict, dataclass


@dataclass
class ReviewComment:
    pr: int
    path: str
    line: int | None
    body: str
    url: str


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


def print_markdown(comments: list[ReviewComment]) -> None:
    """Print comments grouped by PR in readable markdown."""
    by_pr: dict[int, list[ReviewComment]] = {}
    for c in comments:
        by_pr.setdefault(c.pr, []).append(c)

    for pr, items in sorted(by_pr.items()):
        print(f"\n## PR #{pr} — {len(items)} Copilot comment(s)\n")
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
    args = parser.parse_args()

    all_comments: list[ReviewComment] = []
    for pr in args.prs:
        all_comments.extend(fetch_copilot_comments(pr))

    if not all_comments:
        print("No Copilot comments found.", file=sys.stderr)
        sys.exit(0)

    if args.as_json:
        print(json.dumps([asdict(c) for c in all_comments], indent=2))
    elif args.summary_only:
        print_summary(all_comments)
    else:
        print_markdown(all_comments)


if __name__ == "__main__":
    main()
