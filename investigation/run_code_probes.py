"""Compile and run every standalone Rust investigation probe."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent.parent
PROBES = ("grid", "lscale", "tau", "r_threshold", "resolution", "ic")


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def main() -> None:
    build = run(["cargo", "build", "-p", "nereids-physics", "--lib"])
    if build.stdout:
        print(build.stdout, end="")

    dependencies = ROOT / "target/debug/deps"
    libraries = sorted(
        dependencies.glob("libnereids_physics-*.rlib"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not libraries:
        raise FileNotFoundError("cargo build produced no nereids_physics rlib")
    library = libraries[0]
    print(f"nereids_physics_rlib={library.relative_to(ROOT)}")

    output_dir = Path(tempfile.gettempdir()) / "nereids-investigation-probes"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in PROBES:
        source = ROOT / f"investigation/probes/{name}.rs"
        binary = output_dir / name
        run(
            [
                "rustc",
                str(source),
                "--edition=2024",
                "-L",
                f"dependency={dependencies}",
                "--extern",
                f"nereids_physics={library}",
                "-o",
                str(binary),
            ]
        )
        result = run([str(binary)])
        print(f"## {name}")
        print(result.stdout, end="")


if __name__ == "__main__":
    main()
