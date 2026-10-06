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
//! ```
//! pub use std::*;
//! ```
mod hidden {
    pub fn moved() {}
}
pub use hidden::moved;
pub use m::E as Renamed;
pub use globbed::*;

pub mod globbed {
    pub fn g() {}
}

pub mod util {}
pub fn util() {}

pub mod m {
    #[derive(Clone)]
    pub struct S {
        pub x: f64,
        pub fixed: bool,
        y: f64,
    }
    impl S {
        pub fn len(&self) -> f64 {
            self.y
        }
        pub fn fixed(&self) -> bool {
            self.fixed
        }
    }
    impl S {
        pub fn width(&self) -> f64 {
            self.x
        }
    }
    pub enum E {
        A,
        B { inner: f64, fields: u8 },
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
    impl T for f64 {
        fn predict(&self) -> f64 {
            *self
        }
    }
}
"""


def write_crate(root: Path, name: str, lib: str, extra: str = "") -> None:
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "Cargo.toml").write_text(
        f'[package]\nname = "{name}"\nversion = "0.1.0"\nedition = "2021"\n{extra}'
    )
    (root / "src" / "lib.rs").write_text(lib)


@needs_cargo
def test_doc_names_read_the_public_surface_from_rustdoc(tmp_path):
    """Kinds keep same-named items apart; redirects, trait implementations and doc examples are not names."""
    write_crate(tmp_path / "probe", "probe", LIB)
    subprocess.run(
        ["cargo", "doc", "--no-deps", "--quiet", "--target-dir", str(tmp_path / "target")],
        cwd=tmp_path / "probe",
        check=True,
    )
    s, e, t = "probe::m::struct.S", "probe::m::enum.E", "probe::m::trait.T"
    assert surface.doc_names(tmp_path / "target" / "doc", ["probe"]) == {
        "probe",
        "probe::fn.moved",
        "probe::reexport.Renamed",
        "probe::reexport.globbed::*",
        "probe::globbed",
        "probe::globbed::fn.g",
        "probe::util",
        "probe::fn.util",
        "probe::m",
        s,
        f"{s}::structfield.x",
        f"{s}::structfield.fixed",
        f"{s}::method.len",
        f"{s}::method.fixed",
        f"{s}::method.width",
        e,
        f"{e}::variant.A",
        f"{e}::variant.B",
        f"{e}::variant.B.field.inner",
        f"{e}::variant.B.field.fields",
        t,
        f"{t}::tymethod.predict",
        f"{t}::method.provided",
    }


def test_stub_names_include_public_class_methods():
    stub = "def f(): ...\ndef _g(): ...\nclass C:\n    def m(self): ...\n    def _p(self): ...\n"
    assert surface.stub_names(stub) == {"f", "C", "C.m"}


def test_an_added_name_fails_until_the_map_changes():
    assert surface.violations({"c::a"}, {"c::a", "c::b"}, {"crates/c/src/lib.rs"}) == ["c::b"]
    assert surface.violations({"c::a"}, {"c::a", "c::b"}, {"crates/c/src/lib.rs", surface.MAP}) == []
    assert surface.violations({"c::a", "c::b"}, {"c::a"}, set()) == []


@needs_cargo
def test_main_compares_the_base_and_restores_the_checkout(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    lib = repo / "crates" / "c" / "src" / "lib.rs"
    app = repo / "apps" / "app" / "Cargo.toml"
    write_crate(
        repo / "crates" / "c",
        "c",
        'pub fn a() {}\n#[cfg(feature = "extra")]\npub fn extra() {}\n',
        "[features]\nextra = []\n",
    )
    write_crate(repo / "apps" / "app", "app", "", '[dependencies]\nc = { path = "../../crates/c" }\n')
    (repo / "Cargo.toml").write_text('[workspace]\nmembers = ["crates/c", "apps/app"]\nresolver = "2"\n')
    (repo / ".gitignore").write_text("target/\nCargo.lock\n")
    for rel, text in ((surface.MAP, "map\n"), (surface.STUB, "def f(): ...\n")):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)

    def git(*args):
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
            cwd=repo, check=True, capture_output=True, text=True,
        ).stdout.strip()

    def commit(path, old, new):
        path.write_text(path.read_text().replace(old, new))
        git("commit", "-q", "-am", f"edit {path.name}")

    def check_and_restore(base):
        head = git("rev-parse", "HEAD")
        code = surface.main(["check", base])
        assert git("rev-parse", "HEAD") == head and git("symbolic-ref", "--short", "HEAD") == "main"
        return code

    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    monkeypatch.setattr(surface, "REPO_ROOT", repo)
    monkeypatch.setenv("CARGO_TARGET_DIR", str(tmp_path / "target"))

    commit(lib, "pub fn a() {}", "pub fn a() {}\npub fn b() {}")
    assert check_and_restore("HEAD~1") == 1
    commit(lib, "pub fn a() {}", "pub fn a() {\n    let _ = 1;\n}")
    assert check_and_restore("HEAD~1") == 0
    commit(lib, "pub fn b", "/// See [missing].\npub fn b")
    commit(lib, "/// See [missing].\n", "")
    monkeypatch.setenv("RUSTDOCFLAGS", "-D warnings")
    assert check_and_restore("HEAD~1") == 0
    monkeypatch.delenv("RUSTDOCFLAGS")
    commit(app, 'path = "../../crates/c" }', 'path = "../../crates/c", features = ["extra"] }')
    assert check_and_restore("HEAD~1") == 1
    commit(lib, "pub fn b() {}", "pub fn b() {}\npub fn broken( {}")
    commit(lib, "\npub fn broken( {}", "")
    assert check_and_restore("HEAD~1") == 2
    assert check_and_restore("no-such-revision") == 2
    commit(repo / surface.MAP, "map", "map, changed")
    assert check_and_restore("HEAD~3") == 0
    commit(repo / surface.STUB, "def f(): ...", "def f(): ...\ndef g(): ...")
    assert check_and_restore("HEAD~1") == 1
    lib.write_text(lib.read_text() + "// uncommitted\n")
    assert check_and_restore("HEAD~1") == 2
    assert lib.read_text().endswith("// uncommitted\n")
    git("checkout", "--", "crates/c/src/lib.rs")

    commit(lib, "pub fn a()", "pub fn z() {}\npub fn a()")
    base = git("rev-parse", "HEAD~1")
    real = surface.checked_out_names

    def edit_during_base(rustdocflags=None):
        if rustdocflags is not None:
            (repo / surface.MAP).write_text("edited during the run\n")
        return real(rustdocflags)

    monkeypatch.setattr(surface, "checked_out_names", edit_during_base)
    assert surface.main(["check", "HEAD~1"]) == 2
    assert (repo / surface.MAP).read_text() == "edited during the run\n"
    assert git("rev-parse", "HEAD") == base
