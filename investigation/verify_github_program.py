#!/usr/bin/env python3
"""Validate the durable IC-remediation GitHub program registry."""

from __future__ import annotations

import csv
import re
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVESTIGATION = ROOT / "investigation"
OWNER_MAP = INVESTIGATION / "github-program-owner-map.csv"
TRACKING = INVESTIGATION / "github-program-tracking.md"

EXPECTED = (
    {f"M{i:02d}" for i in range(1, 62)}
    | {f"X{i:02d}" for i in range(1, 18)}
    | {f"CL-{i:02d}" for i in range(1, 51)}
)
EXPECTED_KIND = {
    **{f"M{i:02d}": "core" for i in range(1, 62)},
    **{f"X{i:02d}": "surface" for i in range(1, 18)},
    **{f"CL-{i:02d}": "cleanup" for i in range(1, 51)},
}
ALLOWED_BOOTSTRAP_PATHS = (
    ".gitignore",
    ".github/",
    ".harness/",
    "investigation/",
)


def fail(message: str) -> None:
    raise SystemExit(f"github program validation FAILED: {message}")


def identifier_sort_key(identifier: str) -> tuple[str, int]:
    prefix, number = re.fullmatch(r"(M|X|CL-)(\d{2})", identifier).groups()
    return prefix, int(number)


with OWNER_MAP.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

if not rows:
    fail("owner map is empty")
if set(rows[0]) != {"id", "kind", "primary_phase", "work_package"}:
    fail(f"unexpected owner-map columns: {sorted(rows[0])}")

ids = [row["id"] for row in rows]
counts = Counter(ids)
duplicates = sorted(
    (identifier for identifier, count in counts.items() if count != 1),
    key=identifier_sort_key,
)
if duplicates:
    fail(f"owner-map IDs do not occur exactly once: {duplicates}")

actual = set(ids)
if actual != EXPECTED:
    fail(
        "owner-map ID mismatch; "
        f"missing={sorted(EXPECTED - actual, key=identifier_sort_key)}, "
        f"extra={sorted(actual - EXPECTED, key=identifier_sort_key)}"
    )

csv_owners: dict[str, str] = {}
for row in rows:
    identifier = row["id"]
    if row["kind"] != EXPECTED_KIND[identifier]:
        fail(f"wrong kind for {identifier}: {row['kind']}")
    if row["primary_phase"] not in {f"G{i}" for i in range(1, 10)}:
        fail(f"invalid implementation owner for {identifier}: {row['primary_phase']}")
    if not row["work_package"].strip():
        fail(f"empty work package for {identifier}")
    csv_owners[identifier] = row["primary_phase"]

support = (INVESTIGATION / "phase-0-support-matrix.md").read_text(encoding="utf-8")
cleanup = (INVESTIGATION / "phase-0-cleanup-ledger.md").read_text(encoding="utf-8")
for identifier in sorted(EXPECTED, key=identifier_sort_key):
    source = cleanup if identifier.startswith("CL-") else support
    if re.search(rf"(?<![A-Z0-9-]){re.escape(identifier)}(?![A-Z0-9-])", source) is None:
        fail(f"{identifier} is absent from its Phase 0 source registry")

tracking = TRACKING.read_text(encoding="utf-8")
try:
    registry = tracking.split("## Lossless primary-owner registry", 1)[1].split(
        "## Deprecation and rewiring policy", 1
    )[0]
except IndexError:
    fail("tracking document lacks the bounded owner-registry section")

tracking_owners: dict[str, str] = {}
phase_blocks = re.findall(
    r"^- (G[1-9]):\s*(.*?)(?=^- G[1-9]:|\Z)",
    registry,
    flags=re.MULTILINE | re.DOTALL,
)
if len(phase_blocks) != 9:
    fail(f"expected nine G1-G9 owner blocks, found {len(phase_blocks)}")
for phase, block in phase_blocks:
    for identifier in re.findall(r"(?<![A-Z0-9-])(M\d{2}|X\d{2}|CL-\d{2})(?![A-Z0-9-])", block):
        if identifier in tracking_owners:
            fail(
                f"{identifier} appears in both {tracking_owners[identifier]} and "
                f"{phase} tracking blocks"
            )
        tracking_owners[identifier] = phase

if tracking_owners != csv_owners:
    mismatches = sorted(
        (
            identifier
            for identifier in EXPECTED
            if tracking_owners.get(identifier) != csv_owners.get(identifier)
        ),
        key=identifier_sort_key,
    )
    fail(f"tracking document and CSV owners disagree for: {mismatches}")

required_phrases = (
    "NEREIDS Development Project #8",
    "Scientific gate",
    "Disposition",
    "Risk",
    "IC Program Table",
    "IC Program Board",
    "IC Program Roadmap",
    "IC Gate Reviews",
    "IC Deprecation and Removal",
    "IC Blocked Evidence",
    "Issue-ready",
    "PR merge",
    "Phase science",
    "Post-merge",
    "Explicit solver/objective/version requests are never",
    "tagged minor release",
    "usage audit",
)
for phrase in required_phrases:
    if phrase not in tracking:
        fail(f"tracking document lacks required phrase: {phrase!r}")

for number in range(688, 706):
    url = f"https://github.com/ornlneutronimaging/NEREIDS/issues/{number}"
    if url not in tracking:
        fail(f"tracking document lacks issue URL #{number}")
for number in (459, 529, 625, 628):
    url = f"https://github.com/ornlneutronimaging/NEREIDS/issues/{number}"
    if url not in tracking:
        fail(f"tracking document does not reconcile existing issue #{number}")
for view in range(4, 10):
    url = f"https://github.com/orgs/ornlneutronimaging/projects/8/views/{view}"
    if url not in tracking:
        fail(f"tracking document lacks Project view {view}")

for path in (ROOT / ".harness/verify.sh", ROOT / ".harness/review"):
    if not path.exists():
        fail(f"required harness artifact is missing: {path.relative_to(ROOT)}")

changed = subprocess.run(
    ["git", "diff", "--name-only", "b01e077..HEAD"],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()
disallowed = [
    path
    for path in changed
    if not any(path == prefix or path.startswith(prefix) for prefix in ALLOWED_BOOTSTRAP_PATHS)
]
if disallowed:
    fail(f"bootstrap commit touches paths outside the allowed boundary: {disallowed}")

phase_counts = Counter(csv_owners.values())
print(
    "github program validation passed: "
    "61 M + 17 X + 50 CL = 128 identifiers; each has exactly one primary owner"
)
print(
    "phase owners: "
    + ", ".join(f"G{i}={phase_counts[f'G{i}']}" for i in range(1, 10))
)
print("GitHub registry: epic #688, phases #689-#698, G0 leaves #699-#705")
print("Project #8 registry: 6 fields and 6 program views")
print(f"bootstrap boundary: {len(changed)} changed paths, no production source")
