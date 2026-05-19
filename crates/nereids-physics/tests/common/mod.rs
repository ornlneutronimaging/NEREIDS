//! Common helpers for fixture-gated integration tests.

use std::path::PathBuf;

/// Locate the VENUS instrument tabulated resolution fixture
/// (`_fts_bl10_0p5meV_1keV_25pts.txt`, SAMMY USR format) at the
/// workspace root.  The fixture is gitignored per ORNL release
/// policy (.gitignore:48 — "Instrument resolution files (not
/// approved for public release)").
///
/// Returns `Some(path)` when the fixture is present (developer
/// machines with the file at repo root) and `None` otherwise (CI
/// runners, fresh checkouts).  Callers should early-return on `None`
/// so the test becomes a no-op without `#[ignore]` noise in
/// `cargo test` output.
pub fn venus_usr_resolution_path() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()? // crates/
        .parent()? // workspace root
        .join("_fts_bl10_0p5meV_1keV_25pts.txt");
    if p.exists() { Some(p) } else { None }
}
