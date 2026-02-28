#!/usr/bin/env python3
"""Consistent GitHub issue fetching with structured output.

Usage:
    python scripts/gh_issues.py list                     # open issues grouped by label
    python scripts/gh_issues.py list --label perf        # filter by label
    python scripts/gh_issues.py view 86 89 157           # detailed view of specific issues
    python scripts/gh_issues.py epic 80                  # epic + all sub-issues with status
    python scripts/gh_issues.py batch 161 162 163 164    # PR batch status (for merge planning)

Requires: `gh` CLI authenticated with repo access.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass


REPO = "ornlneutronimaging/NEREIDS"


@dataclass
class Issue:
    number: int
    title: str
    state: str
    labels: list[str]
    body: str
    url: str


@dataclass
class PullRequest:
    number: int
    title: str
    state: str
    mergeable: str
    base: str
    head: str
    url: str
    labels: list[str]
    changed_files: int


def _gh(args: list[str]) -> str:
    """Run a gh CLI command and return stdout."""
    result = subprocess.run(
        ["gh"] + args,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error: gh {' '.join(args)}\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def _gh_json(args: list[str]) -> list[dict] | dict:
    """Run a gh CLI command expecting JSON output."""
    raw = _gh(args)
    if not raw:
        return []
    return json.loads(raw)


# --- Commands ---


def cmd_list(args: argparse.Namespace) -> None:
    """List open issues grouped by label."""
    gh_args = ["issue", "list", "-R", REPO, "--state", args.state, "--limit", "200",
               "--json", "number,title,state,labels,url"]
    if args.label:
        gh_args.extend(["--label", args.label])

    issues = _gh_json(gh_args)
    if not issues:
        print("No issues found.")
        return

    # Group by first label (or "unlabeled")
    groups: dict[str, list[dict]] = {}
    for iss in issues:
        label_names = [l["name"] for l in iss.get("labels", [])]
        key = label_names[0] if label_names else "unlabeled"
        groups.setdefault(key, []).append(iss)

    for label, items in sorted(groups.items()):
        print(f"\n## {label} ({len(items)})\n")
        for iss in sorted(items, key=lambda x: x["number"]):
            state_icon = "x" if iss["state"] == "CLOSED" else " "
            all_labels = ", ".join(l["name"] for l in iss.get("labels", []))
            print(f"- [{state_icon}] #{iss['number']}: {iss['title']}  [{all_labels}]")


def cmd_view(args: argparse.Namespace) -> None:
    """View one or more issues with full detail."""
    for num in args.numbers:
        data = _gh_json(["issue", "view", str(num), "-R", REPO,
                         "--json", "number,title,state,labels,body,url,comments"])
        if not data:
            print(f"Issue #{num} not found.", file=sys.stderr)
            continue

        labels = ", ".join(l["name"] for l in data.get("labels", []))
        print(f"\n## #{data['number']}: {data['title']}")
        print(f"**State**: {data['state']}  **Labels**: {labels}")
        print(f"**URL**: {data['url']}")
        if data.get("body"):
            # Truncate very long bodies
            body = data["body"]
            if len(body) > 2000:
                body = body[:2000] + "\n\n... (truncated)"
            print(f"\n{body}")
        comment_count = len(data.get("comments", []))
        if comment_count:
            print(f"\n*{comment_count} comment(s)*")
        print()


def cmd_epic(args: argparse.Namespace) -> None:
    """Show an epic issue and resolve sub-issue status from checkbox lines."""
    data = _gh_json(["issue", "view", str(args.number), "-R", REPO,
                     "--json", "number,title,state,labels,body,url"])
    if not data:
        print(f"Epic #{args.number} not found.", file=sys.stderr)
        return

    print(f"\n## Epic #{data['number']}: {data['title']} [{data['state']}]\n")

    # Extract sub-issue references from body (e.g., "- [ ] #86 — description")
    body = data.get("body", "")
    sub_pattern = re.compile(r"-\s*\[[ x]\]\s*#(\d+)\s*[—–-]?\s*(.*)")
    sub_numbers = []
    for match in sub_pattern.finditer(body):
        sub_numbers.append(int(match.group(1)))

    if not sub_numbers:
        print("No sub-issues found in epic body.")
        print(f"\n{body}")
        return

    # Batch-fetch all sub-issue states
    total = len(sub_numbers)
    closed = 0
    print(f"| # | Title | State |")
    print(f"|---|-------|-------|")
    for num in sub_numbers:
        try:
            sub = _gh_json(["issue", "view", str(num), "-R", REPO,
                            "--json", "number,title,state"])
            state = sub.get("state", "UNKNOWN")
            title = sub.get("title", "")
            icon = "CLOSED" if state == "CLOSED" else "OPEN"
            if state == "CLOSED":
                closed += 1
            print(f"| #{num} | {title} | {icon} |")
        except Exception:
            print(f"| #{num} | (fetch failed) | ? |")

    print(f"\n**Progress**: {closed}/{total} closed")
    if closed == total:
        print("All sub-issues resolved. Epic can be closed.")


def cmd_batch(args: argparse.Namespace) -> None:
    """Show PR batch status for merge planning."""
    print(f"\n## PR Batch Status\n")
    print(f"| PR | Branch | State | Mergeable | Changed Files | Labels |")
    print(f"|----|--------|-------|-----------|---------------|--------|")

    for num in args.numbers:
        try:
            data = _gh_json(["pr", "view", str(num), "-R", REPO,
                             "--json", "number,title,state,mergeable,baseRefName,headRefName,url,labels,changedFiles"])
            labels = ", ".join(l["name"] for l in data.get("labels", []))
            print(f"| #{data['number']} | {data['headRefName']} | {data['state']} | "
                  f"{data.get('mergeable', '?')} | {data.get('changedFiles', '?')} | {labels} |")
        except Exception:
            print(f"| #{num} | ? | (fetch failed) | ? | ? | ? |")

    # File overlap analysis
    print(f"\n### File Overlap\n")
    branch_files: dict[int, set[str]] = {}
    for num in args.numbers:
        try:
            diff_output = _gh(["pr", "diff", str(num), "-R", REPO, "--name-only"])
            files = set(diff_output.strip().split("\n")) if diff_output.strip() else set()
            branch_files[num] = files
        except Exception:
            branch_files[num] = set()

    # Find overlaps
    nums = list(branch_files.keys())
    has_overlap = False
    for i, a in enumerate(nums):
        for b in nums[i + 1:]:
            overlap = branch_files[a] & branch_files[b]
            if overlap:
                has_overlap = True
                print(f"- **#{a} & #{b}**: {', '.join(sorted(overlap))}")

    if not has_overlap:
        print("No file overlaps — PRs can merge in any order.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consistent GitHub issue/PR fetching for NEREIDS"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = sub.add_parser("list", help="List issues grouped by label")
    p_list.add_argument("--label", help="Filter by label")
    p_list.add_argument("--state", default="open", choices=["open", "closed", "all"])

    # view
    p_view = sub.add_parser("view", help="View issue details")
    p_view.add_argument("numbers", nargs="+", type=int, help="Issue numbers")

    # epic
    p_epic = sub.add_parser("epic", help="Show epic with sub-issue status")
    p_epic.add_argument("number", type=int, help="Epic issue number")

    # batch
    p_batch = sub.add_parser("batch", help="PR batch status for merge planning")
    p_batch.add_argument("numbers", nargs="+", type=int, help="PR numbers")

    args = parser.parse_args()

    handlers = {
        "list": cmd_list,
        "view": cmd_view,
        "epic": cmd_epic,
        "batch": cmd_batch,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
