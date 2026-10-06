"""Tests for scripts/check_public_surface.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_public_surface import MAP, doc_names, main, stub_names, violations  # noqa: E402

PAGES = {
    "c/index.html": '<span id="reexport.Thing"></span>',
    "c/fn.free.html": "",
    "c/m/struct.S.html": (
        '<section id="structfield.x"></section><div id="implementations-list">'
        '<section id="method.len"></section><section id="method.len-1"></section></div>'
        '<h2 id="trait-implementations"></h2><section id="method.fmt"></section>'
        '<h2 id="blanket-implementations"></h2><section id="method.into"></section>'
    ),
    "c/m/enum.E.html": '<section id="variant.A"></section><section id="variant.B"></section>',
    "c/m/trait.T.html": '<section id="tymethod.predict"></section><section id="method.provided">',
}


def test_doc_names_take_own_members_and_reexports(tmp_path):
    """Trait-impl and blanket methods are left out; a repeated anchor counts once."""
    for rel, html in PAGES.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(html)
    assert doc_names(tmp_path, ["c"]) == {
        "c::Thing",
        "c::free",
        "c::m::S",
        "c::m::S::x",
        "c::m::S::len",
        "c::m::E",
        "c::m::E::A",
        "c::m::E::B",
        "c::m::T",
        "c::m::T::predict",
        "c::m::T::provided",
    }


def test_stub_names_include_public_class_methods():
    stub = "def f(): ...\ndef _g(): ...\nclass C:\n    def m(self): ...\n    def _p(self): ...\n"
    assert stub_names(stub) == {"f", "C", "C.m"}


def test_an_added_name_fails_until_the_map_changes():
    assert violations({"c::a"}, {"c::a", "c::b"}, {"crates/c/src/lib.rs"}) == ["c::b"]
    assert violations({"c::a"}, {"c::a", "c::b"}, {"crates/c/src/lib.rs", MAP}) == []
    assert violations({"c::a", "c::b"}, {"c::a"}, set()) == []


def test_main_exits_0_with_nothing_changed_and_2_on_a_bad_base():
    assert main(["check_public_surface.py", "HEAD"]) == 0
    assert main(["check_public_surface.py", "no-such-revision"]) == 2
