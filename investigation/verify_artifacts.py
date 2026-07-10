"""Mechanical verification for the committed investigation artifacts."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
INVESTIGATION = ROOT / "investigation"
REQUIRED = (
    ROOT / ".harness/TASK.md",
    INVESTIGATION / "report.md",
    INVESTIGATION / "code-audit.md",
    INVESTIGATION / "archive-inventory.md",
    INVESTIGATION / "elimination-ledger.md",
)


def verify_links() -> int:
    pattern = re.compile(r"\[[^]]*\]\(([^)]+)\)")
    checked = 0
    for markdown in sorted(INVESTIGATION.rglob("*.md")):
        for target in pattern.findall(markdown.read_text()):
            if target.startswith(("http://", "https://", "#")):
                continue
            path_text = target.split("#", 1)[0]
            if not path_text:
                continue
            target_path = (markdown.parent / path_text).resolve()
            if not target_path.exists():
                raise FileNotFoundError(f"broken link in {markdown}: {target}")
            checked += 1
    return checked


def main() -> None:
    for path in REQUIRED:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty required artifact: {path}")

    python_files = sorted(INVESTIGATION.rglob("*.py"))
    for path in python_files:
        compile(path.read_text(), str(path), "exec")

    contract = (ROOT / ".harness/TASK.md").read_text()
    requirement_states = re.findall(
        r"^- \[([x ])\] R(\d+)\b", contract, flags=re.MULTILINE
    )
    if not requirement_states:
        raise AssertionError("contract state: no R-numbered requirements found")
    requirement_numbers = [int(number) for _, number in requirement_states]
    expected_numbers = list(range(1, max(requirement_numbers) + 1))
    unchecked_requirements = [
        f"R{number}" for state, number in requirement_states if state != "x"
    ]
    if requirement_numbers != expected_numbers or unchecked_requirements:
        raise AssertionError(
            f"contract state: requirements={requirement_numbers}, "
            f"expected={expected_numbers}, unchecked={unchecked_requirements}"
        )

    inventory = (INVESTIGATION / "archive-inventory.md").read_text()
    inventory_section = inventory.split("## Complete member inventory", 1)[1].split(
        "## Notebook and input/output audit", 1
    )[0]
    inventory_rows = sum(line.startswith("| `") for line in inventory_section.splitlines())
    if inventory_rows != 71:
        raise AssertionError(f"archive inventory rows: expected 71, got {inventory_rows}")

    combined = "\n".join(path.read_text() for path in INVESTIGATION.rglob("*.md"))
    forbidden = (
        "/tmp/nereids-archive-audit.",
        "/var/folders/",
        "investigation/probes/<name>",
    )
    found = [value for value in forbidden if value in combined]
    if found:
        raise AssertionError(f"non-durable command placeholder(s): {found}")

    link_count = verify_links()
    print(f"required_artifacts={len(REQUIRED)}")
    print(f"python_files_compiled={len(python_files)}")
    print(f"contract_checked_requirements={len(requirement_states)}")
    print(f"archive_inventory_rows={inventory_rows}")
    print(f"relative_links_checked={link_count}")
    print("verification=PASS")


if __name__ == "__main__":
    main()
