"""Tests for scripts/check_public_surface.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_public_surface import MAP, rust_names, stub_names, violations  # noqa: E402

RUST = """\
pub fn free() {}
pub(crate) fn hidden() {}
pub struct S {
    pub x: f64,
    y: f64,
}
pub enum E {
    A,
    B(u32),
    C {
        inner: f64,
    },
}
struct Private {
    pub z: f64,
}
impl S {
    pub fn len(&self) -> usize {
        0
    }
}
impl std::fmt::Display for E {
    fn fmt(&self) {}
}
impl<T: Clone> E {
    pub fn len(&self) -> usize {
        0
    }
}
"""


def test_rust_names_are_qualified_by_crate_and_owner():
    """Methods carry their impl type; restricted and private items are left out."""
    assert rust_names({"crates/c/src/lib.rs": RUST}) == {
        "c::free",
        "c::S",
        "c::S.x",
        "c::S::len",
        "c::E",
        "c::E::A",
        "c::E::B",
        "c::E::C",
        "c::E::len",
    }


def test_stub_names_include_public_class_methods():
    stub = "def f(): ...\ndef _g(): ...\nclass C:\n    def m(self): ...\n    def _p(self): ...\n"
    assert stub_names(stub) == {"f", "C", "C.m"}


def test_an_added_name_fails_until_the_map_changes():
    assert violations({"c::a"}, {"c::a", "c::b"}, {"crates/c/src/lib.rs"}) == ["c::b"]
    assert violations({"c::a"}, {"c::a", "c::b"}, {"crates/c/src/lib.rs", MAP}) == []
    assert violations({"c::a", "c::b"}, {"c::a"}, set()) == []
