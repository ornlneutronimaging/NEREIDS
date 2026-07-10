"""Fail-fast structural and traceability checks for the Phase-0 audit."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
INVESTIGATION = ROOT / "investigation"

REQUIRED = [
    INVESTIGATION / "phase-0-plan.md",
    INVESTIGATION / "phase-0-path-audit.md",
    INVESTIGATION / "phase-0-support-matrix.md",
    INVESTIGATION / "phase-0-disposition.md",
    INVESTIGATION / "phase-0-cleanup-ledger.md",
    INVESTIGATION / "evidence/phase-0/test-results.md",
]

SCRIPTS = [
    "phase0_route_semantics.py",
    "phase0_spatial_flux_probe.py",
    "phase0_counts_ensemble.py",
    "phase0_remaining_routes.py",
    "phase0_ic_fit_probe.py",
    "phase0_real_open_beam.py",
    "phase0_mcp_route_probe.py",
]

ROUTES = [
    *(f"F{i}" for i in range(1, 7)),
    *(f"S{i}" for i in range(1, 7)),
    *(f"C{i}" for i in range(1, 4)),
]

DISPOSITIONS = {
    "keep",
    "keep+improve",
    "deprecate",
    "remove",
    "complete before exposure",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def table_rows(text: str, id_pattern: str) -> list[tuple[str, str]]:
    """Return an ID and full row for Markdown rows matching ``id_pattern``."""

    pattern = re.compile(rf"^\|\s*({id_pattern})\s*\|.*$", re.MULTILINE | re.IGNORECASE)
    return [(match.group(1).upper(), match.group(0)) for match in pattern.finditer(text)]


def numbered_ids(rows: list[tuple[str, str]], prefix: str) -> list[int]:
    return [int(re.search(r"\d+", identifier).group()) for identifier, _ in rows]


def heading_slug(title: str) -> str:
    """Approximate GitHub's ASCII heading slug for the headings used here."""

    slug = re.sub(r"[^a-z0-9 -]", "", title.lower())
    return re.sub(r" +", "-", slug.strip())


def main() -> None:
    for path in REQUIRED:
        require(path.is_file() and path.stat().st_size > 0, f"missing/empty: {path}")

    support = (INVESTIGATION / "phase-0-support-matrix.md").read_text()
    disposition = (INVESTIGATION / "phase-0-disposition.md").read_text()
    audit = (INVESTIGATION / "phase-0-path-audit.md").read_text()
    evidence = (INVESTIGATION / "evidence/phase-0/test-results.md").read_text()
    cleanup = (INVESTIGATION / "phase-0-cleanup-ledger.md").read_text()

    # R13: each route must have its own audit row and direct evidence reference.
    for route in ROUTES:
        rows = table_rows(audit, rf"{re.escape(route)}(?:\s+[^|]+)?")
        require(len(rows) == 1, f"path audit must contain exactly one {route} row")
        require(
            re.search(r"\[E\d{2}\]", rows[0][1]) is not None,
            f"path audit {route} row lacks a direct evidence ID/link",
        )

    evidence_rows = table_rows(evidence, r"E\d{2}")
    evidence_ids = numbered_ids(evidence_rows, "E")
    require(evidence_ids, "evidence registry is empty")
    require(
        evidence_ids == list(range(1, max(evidence_ids) + 1)),
        f"evidence IDs are not unique/contiguous: {evidence_ids}",
    )
    for evidence_id, row in evidence_rows:
        anchor_match = re.search(r"\]\(#([^)]+)\)", row)
        require(anchor_match is not None, f"{evidence_id} lacks an exact-command section link")

        headings = list(re.finditer(r"^##\s+(.+?)\s*$", evidence, re.MULTILINE))
        sections = {
            heading_slug(match.group(1)): evidence[
                match.end() : headings[index + 1].start() if index + 1 < len(headings) else len(evidence)
            ]
            for index, match in enumerate(headings)
        }
        anchor = anchor_match.group(1)
        require(anchor in sections, f"{evidence_id} links to missing section #{anchor}")
        section = sections[anchor]
        require(
            re.search(r"pixi run|cargo test|git fetch|git rev-parse|pytest", section) is not None,
            f"{evidence_id} section has no reproducible command",
        )
        require(
            re.search(r"exit status|passed|failed|skipped|rejected|unavailable|output", section, re.IGNORECASE)
            is not None,
            f"{evidence_id} section has no recorded result/limitation",
        )
    for route in ROUTES:
        require(
            re.search(rf"\|\s*E\d{{2}}\s*\|[^\n|]*\b{route}\b", evidence) is not None,
            f"evidence registry does not name route {route}",
        )

    # R14/R15: normalized core compatibility cells must be one-to-one between
    # support and disposition. Surface reachability (P) and orthogonal wrapper
    # defects (X) are separately complete so they cannot create competing M
    # dispositions for one core tuple.
    support_cells = table_rows(support, r"M-?\d{2}")
    disposition_cells = table_rows(disposition, r"M-?\d{2}")
    support_numbers = numbered_ids(support_cells, "M")
    disposition_numbers = numbered_ids(disposition_cells, "M")
    require(len(support_numbers) >= 20, "support matrix needs normalized compatibility cells")
    require(
        support_numbers == list(range(1, max(support_numbers) + 1)),
        f"support matrix cell IDs are not unique/contiguous: {support_numbers}",
    )
    require(
        disposition_numbers == support_numbers,
        "disposition registry must contain every matrix cell exactly once and no extras",
    )

    support_issues = table_rows(support, r"X-?\d{2}")
    disposition_issues = table_rows(disposition, r"X-?\d{2}")
    support_issue_numbers = numbered_ids(support_issues, "X")
    disposition_issue_numbers = numbered_ids(disposition_issues, "X")
    require(
        support_issue_numbers == list(range(1, max(support_issue_numbers) + 1)),
        f"surface issue IDs are not unique/contiguous: {support_issue_numbers}",
    )
    require(
        disposition_issue_numbers == support_issue_numbers,
        "every orthogonal surface issue needs exactly one disposition",
    )
    surface_rows = table_rows(support, r"P-?\d{2}")
    surface_numbers = numbered_ids(surface_rows, "P")
    require(
        surface_numbers == list(range(1, max(surface_numbers) + 1)),
        f"surface reachability IDs are not unique/contiguous: {surface_numbers}",
    )
    require(surface_numbers == list(range(1, 19)), f"expected P01-P18, got: {surface_numbers}")
    require("lossless" in support.lower(), "support matrix does not define the M x P/X factorization")
    surface_blob = "\n".join(row for _, row in surface_rows)
    for issue_id, _ in support_issues:
        require(
            re.search(rf"\b{re.escape(issue_id)}\b", surface_blob) is not None,
            f"{issue_id} is not attached to any P surface mapping",
        )

    source_pattern = re.compile(
        r"[A-Za-z0-9_./-]+\.(?:rs|py|pyi|md|ipynb):\d+",
        re.IGNORECASE,
    )
    evidence_pattern = re.compile(r"\bE\d{2}\b|static[- ]only|test gap", re.IGNORECASE)
    for cell_id, row in support_cells:
        require(source_pattern.search(row) is not None, f"{cell_id} lacks precise file:line evidence")
        require(
            evidence_pattern.search(row) is not None,
            f"{cell_id} lacks a direct test evidence ID or explicit static-only gap",
        )
    for issue_id, row in support_issues:
        require(source_pattern.search(row) is not None, f"{issue_id} lacks precise file:line evidence")
        require(
            evidence_pattern.search(row) is not None,
            f"{issue_id} lacks direct evidence or an explicit static-only gap",
        )
    for surface_id, row in surface_rows:
        require(source_pattern.search(row) is not None, f"{surface_id} lacks precise file:line evidence")
        require(
            re.search(r"\bM\d{2}\b", row) is not None,
            f"{surface_id} does not map to a core M cell",
        )
    for cell_id, row in disposition_cells:
        fields = [field.strip() for field in row.strip().strip("|").split("|")]
        require(len(fields) >= 4, f"{cell_id} lacks migration/gate columns")
        selected = re.sub(r"[*`]", "", fields[1]).lower().strip()
        require(
            selected in DISPOSITIONS,
            f"{cell_id} must have exactly one allowed disposition, found: {fields[1]}",
        )
        require(all(fields[-2:]), f"{cell_id} has an empty migration or acceptance gate")
    for issue_id, row in disposition_issues:
        fields = [field.strip() for field in row.strip().strip("|").split("|")]
        require(len(fields) >= 4, f"{issue_id} lacks migration/gate columns")
        selected = re.sub(r"[*`]", "", fields[1]).lower().strip()
        require(
            selected in DISPOSITIONS,
            f"{issue_id} must have exactly one allowed disposition, found: {fields[1]}",
        )
        require(all(fields[-2:]), f"{issue_id} has an empty migration or acceptance gate")

    for protocol in range(1, 11):
        require(
            re.search(rf"\*\*Q{protocol}\s+—", disposition) is not None,
            f"missing fixed quantitative protocol Q{protocol}",
        )
    vague_gate = re.compile(
        r"\b(predeclared|declared|documented)\s+"
        r"(tolerance|threshold|envelope|benchmark|oracle|limit|bounds?|uncertainty|coverage|interval)",
        re.IGNORECASE,
    )
    for item_id, row in [*disposition_cells, *disposition_issues]:
        require(vague_gate.search(row) is None, f"{item_id} still defers its pass threshold")

    # R16: every cleanup finding needs source/reachability plus either executed
    # coverage or a named missing acceptance test. A separate coverage registry
    # is accepted, but it must contain every current CL ID one-for-one.
    canonical_heading = re.search(
        r"^## Canonical cleanup disposition registry\s*$",
        cleanup,
        re.MULTILINE | re.IGNORECASE,
    )
    require(canonical_heading is not None, "cleanup ledger lacks a canonical disposition registry")
    coverage_heading = re.search(
        r"^## Test[- ]coverage(?: registry| and explicit gaps)\s*$",
        cleanup,
        re.MULTILINE | re.IGNORECASE,
    )
    require(coverage_heading is not None, "cleanup ledger lacks a test-coverage registry")
    require(canonical_heading.start() < coverage_heading.start(), "cleanup registries are out of order")
    main_cleanup = cleanup[: canonical_heading.start()]
    canonical_cleanup = cleanup[canonical_heading.end() : coverage_heading.start()]
    coverage_cleanup = cleanup[coverage_heading.end() :]
    canonical_rows = table_rows(canonical_cleanup, r"CL-\d{2}")
    coverage_rows = table_rows(coverage_cleanup, r"CL-\d{2}")

    cleanup_rows = table_rows(main_cleanup, r"CL-\d{2}")
    cleanup_ids = numbered_ids(cleanup_rows, "CL")
    coverage_ids = numbered_ids(coverage_rows, "CL")
    expected_cleanup = list(range(1, max(cleanup_ids) + 1))
    canonical_ids = numbered_ids(canonical_rows, "CL")
    require(cleanup_ids == expected_cleanup, f"cleanup IDs not unique/contiguous: {cleanup_ids}")
    require(canonical_ids == expected_cleanup, f"cleanup disposition IDs do not match: {canonical_ids}")
    require(coverage_ids == expected_cleanup, f"coverage IDs do not match: {coverage_ids}")
    for cleanup_id, row in cleanup_rows:
        require(
            re.search(
                r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.(?:rs|py|pyi|md|ipynb)(?::\d+)?"
                r"|\bE\d{2}\b",
                row,
                re.IGNORECASE,
            )
            is not None,
            f"{cleanup_id} lacks concrete source/evidence reachability",
        )
    for cleanup_id, row in coverage_rows:
        require(
            re.search(r"\bE\d{2}\b|static[- ]only|not covered|missing", row, re.IGNORECASE) is not None,
            f"{cleanup_id} lacks executed coverage or an explicit coverage gap",
        )
    for cleanup_id, row in canonical_rows:
        fields = [field.strip() for field in row.strip().strip("|").split("|")]
        require(len(fields) >= 3, f"{cleanup_id} lacks disposition/action fields")
        selected = re.sub(r"[*`]", "", fields[1]).lower().strip()
        require(selected in DISPOSITIONS, f"{cleanup_id} has invalid disposition: {fields[1]}")
        require(bool(fields[2]), f"{cleanup_id} has no assigned action")

    dependency_match = re.search(
        r"^### Cleanup-ledger dependency map\s*$([\s\S]*?)\Z",
        disposition,
        re.MULTILINE | re.IGNORECASE,
    )
    require(dependency_match is not None, "disposition lacks the cleanup dependency map")
    dependency_ids = sorted(
        int(value) for value in re.findall(r"\bCL-(\d{2})\b", dependency_match.group(1))
    )
    require(
        dependency_ids == expected_cleanup,
        f"cleanup dependency map must assign every CL exactly once: {dependency_ids}",
    )

    contradictions = {
        "no independent KL spatial recovery gate": "S2 recovery evidence was run",
        "no F1/S route gate": "F1 IC-as-tabulated evidence was run",
        "runtime signature and stub PSR defaults also disagree": "both PSR defaults are 350 ns",
        "no counts-fit integration covers IC/UDR": "F3 IC-as-tabulated integration was run",
    }
    combined = "\n".join((support, cleanup))
    for phrase, correction in contradictions.items():
        require(phrase.lower() not in combined.lower(), f"known contradiction remains ({correction}): {phrase}")

    # Content checks precede the contract check so a checked box can never make
    # an incomplete artifact pass this gate.
    task = (ROOT / ".harness/TASK.md").read_text()
    for requirement in range(13, 18):
        require(
            re.search(rf"^- \[x\] R{requirement}\b", task, re.MULTILINE) is not None,
            f"R{requirement} is not checked",
        )

    for name in SCRIPTS:
        path = INVESTIGATION / name
        source = path.read_text()
        compile(source, str(path), "exec")

    broken: list[str] = []
    for markdown in REQUIRED:
        text = markdown.read_text()
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://", "#", "/")):
                continue
            relative = target.split("#", 1)[0]
            if relative and not (markdown.parent / relative).resolve().exists():
                broken.append(f"{markdown.relative_to(ROOT)} -> {target}")
    require(not broken, "broken relative links:\n" + "\n".join(broken))

    print(
        "phase0 verification passed: "
        f"{len(REQUIRED)} artifacts, {len(ROUTES)} routes, "
        f"{len(evidence_ids)} evidence records, {len(support_numbers)} core cells, "
        f"{len(support_issue_numbers)} surface issues, {len(surface_numbers)} surface mappings, "
        f"{len(cleanup_ids)} cleanup findings with coverage, "
        f"{len(SCRIPTS)} scripts compiled, no broken relative links"
    )


if __name__ == "__main__":
    main()
