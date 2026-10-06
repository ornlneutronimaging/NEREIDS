"""Tests for scripts/check_public_surface.py."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_public_surface as surface  # noqa: E402

needs_cargo = pytest.mark.skipif(shutil.which("cargo") is None, reason="needs cargo")

LIB = """\
mod hidden {
    pub fn moved() {}
}
pub use hidden::moved;
pub use m::E as Renamed;

pub mod m {
    #[derive(Clone)]
    pub struct S {
        pub x: f64,
        y: f64,
    }
    impl S {
        pub fn len(&self) -> f64 {
            self.y
        }
    }
    impl S {
        pub fn width(&self) -> f64 {
            self.x
        }
    }
    pub enum E {
        A,
        B { inner: f64 },
    }
    pub trait T {
        fn predict(&self) -> f64;
        fn provided(&self) {}
    }
    impl T for S {
        fn predict(&self) -> f64 {
            0.0
        }
    }
}

pub mod empty {}
"""


def write_crate(root: Path, name: str, lib: str) -> None:
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "Cargo.toml").write_text(f'[package]\nname = "{name}"\nversion = "0.1.0"\nedition = "2021"\n')
    (root / "src" / "lib.rs").write_text(lib)


@needs_cargo
def test_doc_names_read_the_public_surface_from_rustdoc(tmp_path):
    """Redirects, trait implementations and private fields are not public names."""
    write_crate(tmp_path / "probe", "probe", LIB)
    subprocess.run(
        ["cargo", "doc", "--no-deps", "--quiet", "--target-dir", str(tmp_path / "target")],
        cwd=tmp_path / "probe",
        check=True,
    )
    assert surface.doc_names(tmp_path / "target" / "doc", ["probe"]) == {
        "probe",
        "probe::moved",
        "probe::Renamed",
        "probe::empty",
        "probe::m",
        "probe::m::S",
        "probe::m::S::x",
        "probe::m::S::len",
        "probe::m::S::width",
        "probe::m::E",
        "probe::m::E::A",
        "probe::m::E::B",
        "probe::m::E::B.field.inner",
        "probe::m::T",
        "probe::m::T::predict",
        "probe::m::T::provided",
    }


def test_stub_names_include_public_class_methods():
    stub = "def f(): ...\ndef _g(): ...\nclass C:\n    def m(self): ...\n    def _p(self): ...\n"
    assert surface.stub_names(stub) == {"f", "C", "C.m"}


def test_an_added_name_fails_until_the_map_changes():
    assert surface.violations({"c::a"}, {"c::a", "c::b"}, {"crates/c/src/lib.rs"}) == ["c::b"]
    assert surface.violations({"c::a"}, {"c::a", "c::b"}, {"crates/c/src/lib.rs", surface.MAP}) == []
    assert surface.violations({"c::a", "c::b"}, {"c::a"}, set()) == []


@needs_cargo
def test_main_checks_out_the_base_and_restores_the_branch(tmp_path, monkeypatch):
    """Exit 1 for a new name, 0 once the map changes, 2 for a bad base; HEAD is restored."""
    repo = tmp_path / "repo"
    write_crate(repo / "crates" / "c", "c", "pub fn a() {}\n")
    (repo / "Cargo.toml").write_text('[workspace]\nmembers = ["crates/c"]\nresolver = "2"\n')
    for rel, text in ((surface.MAP, "map\n"), (surface.STUB, "def f(): ...\n")):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)

    def git(*args):
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
            cwd=repo, check=True, capture_output=True, text=True,
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    (repo / ".gitignore").write_text("target/\nCargo.lock\n")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    (repo / "crates" / "c" / "src" / "lib.rs").write_text("pub fn a() {}\npub fn b() {}\n")
    git("commit", "-q", "-am", "add b")
    head = git("rev-parse", "HEAD")
    monkeypatch.setattr(surface, "REPO_ROOT", repo)
    monkeypatch.setenv("CARGO_TARGET_DIR", str(tmp_path / "target"))

    assert surface.main(["check", "HEAD~1"]) == 1
    assert git("rev-parse", "HEAD") == head and git("symbolic-ref", "--short", "HEAD") == "main"
    assert surface.main(["check", "no-such-revision"]) == 2
    (repo / surface.MAP).write_text("map, changed\n")
    git("commit", "-q", "-am", "change the map")
    assert surface.main(["check", "HEAD~2"]) == 0
