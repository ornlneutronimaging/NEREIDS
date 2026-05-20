# Integration tests for `nereids-physics`

## VENUS USR resolution tests

The three `venus_usr_*.rs` integration tests exercise the SAMMY USR
(user-supplied tabulated resolution) parser + broadening pipeline that
the SoftwareX paper documents.

Historically these tests loaded an external instrument-characterization
file — `_fts_bl10_0p5meV_1keV_25pts.txt`, the VENUS BL10 (SNS Beam
Line 10) kernel — that ORNL has not approved for public release.  The
file is gitignored (`.gitignore:49`); on CI / fresh checkouts the
tests early-returned silently, leaving the SAMMY-format kernel path
with zero CI coverage (issue #557).

The tests now build a **synthetic** SAMMY USR-format kernel
in-memory via [`common::synthetic_venus_usr_tab`]
(`tests/common/mod.rs`).  The synthetic kernel is parsed through the
same `TabulatedResolution::from_text` entry point the production
fixture used, so every stage of the kernel pipeline (parser → plan
→ apply → CSR compile → matvec) is exercised on every `cargo test`
invocation.

### No-op-regression pre-check

Every test that broadens via `plan.apply` / `compile_to_matrix`
calls `common::assert_kernel_broadens(&plan, &energies)` up front.
This asserts `‖T_kernel − T_none‖_∞ > 1 % · ‖T_none‖_∞` on a sharp
Gaussian-dip probe spectrum, so a future tweak that collapses the
synthetic kernel toward a delta (which would silently turn every
equivalence test into a vacuous identity) fails loudly at the first
test instead.

See PR #544 for the silent-no-op-via-kernel-shrink failure mode
this pre-check guards against.
