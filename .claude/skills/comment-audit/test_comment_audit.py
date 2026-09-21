"""The comment stripper respects Rust lexical structure and round-trips."""

import importlib.util
import json
import pathlib
import sys

SCRIPT = pathlib.Path(__file__).resolve().parent / "comment_audit.py"
spec = importlib.util.spec_from_file_location("comment_audit", SCRIPT)
ca = importlib.util.module_from_spec(spec)
sys.modules["comment_audit"] = ca
spec.loader.exec_module(ca)


def strip(text):
    comments = ca.group_comments(text)
    stripped, line_map = ca.strip_text(text, comments)
    ca.attach_anchors(comments, stripped, line_map)
    return comments, stripped


def test_comment_markers_inside_strings_and_chars_are_not_comments():
    text = (
        'let s = "// not a comment";\n'
        'let r = r#"a /* b */ c"#;\n'
        "let q = '\"';\n"
        "let b = b\"// bytes\";\n"
        "fn f<'a>(x: &'a str) -> &'a str { x }\n"
    )
    comments, stripped = strip(text)
    assert comments == []
    assert stripped == text


def test_trailing_comment_is_cut_and_anchored_to_its_own_line():
    text = "let q = '\"'; // trailing\nlet z = 1;\n"
    comments, stripped = strip(text)
    assert len(comments) == 1
    c = comments[0]
    assert c.trailing and c.text == ["// trailing"]
    assert c.anchor == {"line": 1, "text": "let q = '\"';"}
    assert stripped == "let q = '\"';\nlet z = 1;\n"


def test_doc_block_groups_and_anchors_to_the_item():
    text = (
        "use std::fmt;\n"
        "\n"
        "/// First.\n"
        "/// Second.\n"
        "#[must_use]\n"
        "pub fn f() {}\n"
        "// alone\n"
        "\n"
        "// after a blank\n"
        "fn g() {}\n"
    )
    comments, stripped = strip(text)
    assert [c.kind for c in comments] == ["doc", "line", "line"]
    doc = comments[0]
    assert (doc.first_line, doc.last_line) == (3, 4)
    assert doc.text == ["/// First.", "/// Second."]
    assert doc.anchor == {"line": 3, "text": "#[must_use]"}
    assert comments[1].anchor == {"line": 6, "text": "fn g() {}"}
    assert stripped.rstrip("\n").split("\n") == [
        "use std::fmt;",
        "",
        "#[must_use]",
        "pub fn f() {}",
        "",
        "fn g() {}",
    ]


def test_nested_block_comment_is_one_comment():
    text = "let a = 1; /* x /* y */ z */ let b = 2;\n"
    comments, stripped = strip(text)
    assert len(comments) == 1 and comments[0].kind == "block"
    assert stripped == "let a = 1;  let b = 2;\n"


def test_safety_comment_is_exempt_and_kept():
    text = "// SAFETY: the pointer is valid.\nunsafe { go() }\n// plain\nlet x = 1;\n"
    comments, stripped = strip(text)
    assert [c.exempt for c in comments] == [True, False]
    assert stripped.rstrip("\n").split("\n") == [
        "// SAFETY: the pointer is valid.",
        "unsafe { go() }",
        "let x = 1;",
    ]


def test_apply_deletes_only_the_named_comments():
    text = "/// doc\nfn f() {} // tail\n// stray\nfn g() {}\n"
    comments = ca.group_comments(text)
    ids = [c.id for c in comments]
    assert ids == ["c1", "c2", "c3"]
    assert ca.apply_deletions(text, ca.group_comments(text), set()) == text
    assert ca.apply_deletions(text, ca.group_comments(text), {"c2"}) == (
        "/// doc\nfn f() {}\n// stray\nfn g() {}\n"
    )
    assert ca.apply_deletions(text, ca.group_comments(text), {"c1", "c3"}) == (
        "fn f() {} // tail\nfn g() {}\n"
    )


def test_cli_strip_and_apply_round_trip(tmp_path):
    src = tmp_path / "a.rs"
    src.write_text("// top\nfn f() {}\n", encoding="utf-8")
    out = tmp_path / "out"
    assert ca.main(["strip", str(src), "--out", str(out), "--root", str(tmp_path)]) == 0
    assert (out / "a.rs.stripped.rs").read_text() == "fn f() {}\n"
    listed = json.loads((out / "a.rs.comments.json").read_text())
    assert [c["id"] for c in listed["comments"]] == ["c1"]
    assert listed["comments"][0]["anchor"] == {"line": 1, "text": "fn f() {}"}
    comments_json = str(out / "a.rs.comments.json")
    assert ca.main(["apply", str(src), "--comments", comments_json, "--delete", "c1"]) == 0
    assert src.read_text() == "fn f() {}\n"
    assert ca.main(["apply", str(src), "--comments", comments_json, "--delete", "c1"]) == 2


def test_scope_is_the_comments_touching_added_lines():
    diff = "@@ -10,0 +11,3 @@ fn f\n+a\n+b\n+c\n@@ -20 +24 @@\n-x\n+y\n@@ -30,2 +33,0 @@\n-p\n-q\n"
    assert ca.added_ranges(diff) == [(11, 13), (24, 24)]
    comments = [
        {"id": "c1", "first_line": 9, "last_line": 10},
        {"id": "c2", "first_line": 12, "last_line": 15},
        {"id": "c3", "first_line": 24, "last_line": 24},
        {"id": "c4", "first_line": 33, "last_line": 34},
    ]
    assert ca.ids_in_ranges(comments, ca.added_ranges(diff)) == ["c2", "c3"]


def test_windows_merge_when_they_touch():
    lines = [f"l{k}" for k in range(1, 201)]
    w = ca.windows_around(lines, [5, 100, 130, 199], reach=20)
    assert [(x["start"], x["end"]) for x in w] == [(1, 25), (80, 150), (179, 200)]
    assert w[0]["lines"][0] == "l1" and w[1]["lines"][-1] == "l150"


def test_verify_isolation_flags_any_tool_but_structured_output(tmp_path):
    clean = tmp_path / "agent-a.jsonl"
    clean.write_text(
        json.dumps({"message": {"content": [{"type": "tool_use", "name": "StructuredOutput"}]}})
        + "\n"
    )
    assert ca.main(["verify-isolation", str(tmp_path)]) == 0
    dirty = tmp_path / "agent-b.jsonl"
    dirty.write_text(
        json.dumps({"message": {"content": [{"type": "tool_use", "name": "Read", "input": {}}]}})
        + "\nnot json\n"
    )
    assert ca.main(["verify-isolation", str(tmp_path)]) == 1


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__]))
