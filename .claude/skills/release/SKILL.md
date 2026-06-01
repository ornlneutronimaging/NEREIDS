# Release Pipeline

Cut a new NEREIDS release. The work is mostly bumping versions across the
workspace, rolling the changelog, tagging, and pushing — the actual building /
publishing / homebrew update happens on CI when the tag arrives. This skill is
the wrapper that gets the bumps right and verifies CI succeeded.

## Arguments

- No arguments: prompts for the new version, defaults to incrementing the
  patch component of the current version.
- `<version>`: target version without the leading `v` (e.g. `0.1.8`).
- `--dry-run`: do everything locally, but use the workflow's `dry_run`
  dispatch path so PyPI / crates.io / Homebrew aren't touched.
- `--skip-bump`: skip Step 2 (version bumps) — assumes the user already
  bumped manually and just wants the tag-and-push automation.

## Pre-flight gates (Step 1)

1. **On main, clean tree.** Stash unrelated working-tree items first; the
   release commit must be the *next* commit on main with no other staged
   changes.
2. **`origin/main` up to date.** `git fetch origin && git status` should
   confirm `Your branch is up to date with 'origin/main'`. If not, `git
   pull --ff-only` before proceeding.
3. **Last batch's post-merge gate passed.** The release tag is cut on the
   commit you just verified through `/post-merge`. Don't tag a commit
   that hasn't passed the integration gate.
4. **Tag does not already exist.** `git tag -l v<version>` should be empty
   locally and on origin (`git ls-remote --tags origin v<version>`).

## Step 2: Bump versions

Six files (treat as a single atomic edit; do not commit each separately).
`pixi run bump-version <NEW>` (`scripts/bump-version.sh`) automates all six, plus
the Cargo.lock refresh (Step 3) — so seven files are staged together in Step 5:

- [Cargo.toml](Cargo.toml)
  - `workspace.package.version = "<NEW>"` (around L16)
  - 7 inter-crate path deps in `[workspace.dependencies]` — each line
    `<crate> = { version = "<NEW>", path = "..." }` (around L23-29)
- [pyproject.toml](pyproject.toml)
  - `[project] version = "<NEW>"` (around L3)
  - `[project.optional-dependencies] gui = ["nereids-gui==<NEW>"]`
    (around L40)
- [apps/gui/pyproject.toml](apps/gui/pyproject.toml)
  - `[project] version = "<NEW>"` (around L7)
- [homebrew/nereids.rb](homebrew/nereids.rb)
  - `version "<NEW>"` (around L4) — note this is the in-repo template;
    the CI's `update-homebrew` job substitutes into the *separate*
    `homebrew-nereids` tap repo (`Casks/nereids.rb`). Bumping the in-repo
    template anyway keeps it as a useful reference.
- [CITATION.cff](CITATION.cff)
  - `version: <NEW>` (around L6) and `date-released: <today>` (L7) — both
    stamped by the bump script (the date line is updated in place; it must
    already exist, which it does as of v0.1.8)
- [CHANGELOG.md](CHANGELOG.md)
  - the bump script rolls `## [Unreleased]` into a dated `## [<NEW>]` section
    and updates the `compare/...` link-refs. Curate the release notes *under
    `[Unreleased]`* **before** running the bump so they land in the new section.

Reference: commit `973bdf2` (Release v0.1.7) shows the historical five-file
version-bump diff shape; releases now also touch `CITATION.cff` +
`CHANGELOG.md` (~7 files total).

**Skip in the bump** anything matching `0.1.<old>` in code/docs that isn't
a version declaration (e.g. example notebooks may have hardcoded version
strings — those are reference, not authoritative). If you find such
references, mention them in the release commit message but don't update
unless the user asks.

## Step 3: Refresh Cargo.lock

```
cargo check --workspace --exclude nereids-python
```

This regenerates `Cargo.lock` with the new internal-crate version pins.
The lockfile delta should be a clean version bump for the 7 internal
crates (no drift in third-party deps).

## Step 4: Pre-commit checklist (mandatory)

```
cargo fmt --all
cargo clippy --workspace --exclude nereids-python --all-targets -- -D warnings
cargo test --workspace --exclude nereids-python
cargo check --workspace --exclude nereids-python --examples
pixi run test-python
```

All five must pass before tagging. If any fails, the release commit
should NOT be created.

The `--examples` gate catches drift between mdBook quickstart snippets and
the live crate APIs (the snippets in `docs/guide/src/quickstart-rust.md`
are `{{#include}}`-spliced from `crates/nereids-fitting/examples/quickstart.rs`).

## Step 4a: Docs build + extras smoke test (mandatory)

```
pixi run doc-build
```

`doc-build` chains `doc-guide` (mdBook), `doc-api` (rustdoc), and
`doc-python` (pdoc on the installed wheel). Visually scan:

- `target/book/index.html` — landing page
- `target/book/api/` — rustdoc renders without missing items
- `target/book/python/index.html` — pdoc reference is non-empty and the
  typed APIs (`fit_spectrum_typed`, `spatial_map_typed`, `load_endf`, …)
  appear under `nereids/nereids.html`
- The curated `python-api.md` page reflects any new `__init__.pyi` surface

Then verify both extras install on a clean venv. `pixi run build` is
`maturin develop` and writes the wheel to a temp dir, so an explicit
`maturin build --out target/wheels` is needed first; the smoke test then
installs the newest matching wheel:

```
rm -rf target/wheels
pixi run -- maturin build --release \
  --manifest-path bindings/python/Cargo.toml --out target/wheels
WHEEL="$(ls -t target/wheels/nereids-*.whl | head -1)"
python -m venv /tmp/nereids-extras && source /tmp/nereids-extras/bin/activate
pip install "${WHEEL}"
pip install "${WHEEL}[gui]"   # exercises the gui extra
pip install "${WHEEL}[mcp]"   # exercises the mcp extra
python -c "import nereids; help(nereids.fit_spectrum_typed)" | head -5
deactivate && rm -rf /tmp/nereids-extras
```

All three `pip install` lines must succeed on the target Python version
declared in `pyproject.toml`. The `ls -t | head -1` is defensive against
multiple wheels accumulating in `target/wheels/` across release-rehearsal
runs — `rm -rf target/wheels` at the top is the cleaner reset path.

## Step 5: Commit + tag

```
git add Cargo.toml Cargo.lock pyproject.toml apps/gui/pyproject.toml homebrew/nereids.rb CITATION.cff CHANGELOG.md
git commit -S -m "Release v<NEW>"
git tag -s v<NEW> -m "Release v<NEW>"
```

**Always sign the tag** (`-s`) and the commit (`-S`) — the project's
publish workflow doesn't enforce signatures, but the project CLAUDE.md
mandates GPG-signed commits and the same standard applies to tags. Use
the same key configured for `git commit -S`.

The tag *message* is what `softprops/action-gh-release@v2 +
generate_release_notes: true` will surface alongside the auto-generated
PR list. Keep it short ("Release v<NEW>") — the body comes from the
auto-generated notes, not the tag message.

## Step 6: Push (this is the trigger)

```
git push origin main
git push origin v<NEW>
```

The `publish.yml` workflow fires on the tag push (matches `tags: ['v*']`).
**Push the branch first** so the tagged commit is reachable on origin/main
before CI starts pulling it.

For `--dry-run`: skip the tag push entirely. Instead use
`gh workflow run publish.yml -f dry_run=true` to exercise the build matrix
without publishing.

## Step 7: Monitor the pipeline

The pipeline has 9 jobs and runs ~20-30 min end-to-end. Stream:

```
gh run watch $(gh run list --workflow=publish.yml --limit 1 --json databaseId --jq '.[0].databaseId')
```

Or check status without blocking:

```
gh run list --workflow=publish.yml --limit 3
```

Jobs to expect (from [`.github/workflows/publish.yml`](.github/workflows/publish.yml)):

1. `build-wheels` — Python library wheels for Linux x86_64, macOS arm64,
   Windows. Windows is `continue-on-error: true` (HDF5 static-link can
   fail there); a Windows failure is *not* a blocker.
2. `build-sdist` — source distribution.
3. `build-gui` — GUI wheels for Linux + macOS.
4. `build-macos-app` — macOS `.app` bundle + DMG via `cargo bundle`.
5. `publish-pypi` — uploads `nereids` to PyPI (env: `pypi`).
6. `publish-gui-pypi` — uploads `nereids-gui` to PyPI (env: `pypi-gui`).
7. `publish-crates` — publishes 7 crates to crates.io in dependency order
   with 60s sleeps between, gracefully skipping already-published versions.
8. `github-release` — creates the GitHub Release with all artifacts and
   `generate_release_notes: true` (auto-fills from merged PRs since the
   previous tag).
9. `update-homebrew` — bumps version + sha256 in the tap repo's
   `Casks/nereids.rb`. Skipped for prereleases (`rc`/`alpha`/`beta`).

## Step 8: Verify artifacts landed

Run all four checks; report failures explicitly.

```
# PyPI library
curl -sf "https://pypi.org/pypi/nereids/<NEW>/json" >/dev/null && echo OK || echo MISSING

# PyPI GUI
curl -sf "https://pypi.org/pypi/nereids-gui/<NEW>/json" >/dev/null && echo OK || echo MISSING

# crates.io (sample one — others publish in dep order)
curl -sf "https://crates.io/api/v1/crates/nereids-core/<NEW>" >/dev/null && echo OK || echo MISSING

# GitHub Release
gh release view v<NEW> >/dev/null 2>&1 && echo OK || echo MISSING
```

## Step 9: Post-release housekeeping

1. **Visual check** of the GitHub Release page — confirm the auto-generated
   notes captured the expected PRs and the release-asset list contains
   wheels (Linux+macOS+Win), sdist, GUI wheels (Linux+macOS), and the
   macOS DMG.
2. **Docs site rebuilt.** The `docs.yml` workflow re-runs on every push to
   `main`, so the release commit (Step 6) auto-triggers a Pages rebuild.
   Confirm at `https://ornlneutronimaging.github.io/NEREIDS/`:
   - mdBook landing page loads (`/`)
   - rustdoc renders (`/api/`)
   - pdoc Python reference renders (`/python/`) and reflects any new
     `__init__.pyi` surface from this release
3. **Homebrew tap** — visit `https://github.com/ornlneutronimaging/homebrew-nereids`
   and confirm the latest commit there bumped `Casks/nereids.rb`. (The
   workflow's `update-homebrew` job needs `HOMEBREW_TAP_TOKEN`; if absent
   it errors and the tap will be stale.)
4. **Memory**: only update memory if the *release process itself* surfaced
   a non-obvious lesson (e.g. a new file location to bump, a CI flake
   pattern). Routine release outcomes are not memory-worthy.

## Step 10: Report

Present a concise summary table to the user:

```markdown
### Release v<NEW> — published

| Artifact | Status | URL |
|----------|--------|-----|
| GitHub Release | ✓ | https://github.com/ornlneutronimaging/NEREIDS/releases/tag/v<NEW> |
| PyPI nereids | ✓ | https://pypi.org/project/nereids/<NEW>/ |
| PyPI nereids-gui | ✓ | https://pypi.org/project/nereids-gui/<NEW>/ |
| crates.io | ✓ | https://crates.io/crates/nereids-core/<NEW> |
| Homebrew tap | ✓ | https://github.com/ornlneutronimaging/homebrew-nereids |

Pipeline: [run #N](https://github.com/ornlneutronimaging/NEREIDS/actions/runs/<id>)
Release notes auto-generated from PRs since v<PREV>.
```

If a Windows wheel build failed (`continue-on-error: true`), note it
explicitly so the user can decide whether to investigate or accept.

## Failure modes & remediation

- **Tag exists locally but not on origin (or vice versa).** Resolve with
  `git tag -d v<NEW>` (local) or `git push --delete origin v<NEW>`
  (remote). Re-tag and push fresh.
- **`publish-pypi` succeeded but `publish-gui-pypi` failed.** PyPI
  rejects re-uploads of the same filename, so deleting the partial
  release and re-running won't help. Fix: bump to `v<NEW>.1` or `v<NEW>+1`
  and re-release. (`publish-pypi` uses `skip-existing: true` so a partial
  retry is safe for the library wheel matrix.)
- **`publish-crates` halfway through.** The script skips already-published
  crates (`already (uploaded|exists)` substring match), so re-running
  the workflow on the same tag is safe. If a *non*-already-published
  failure happens (network, sleep race), `gh workflow run publish.yml
  -f dry_run=false` re-runs the workflow on the same tag.
- **macOS DMG download timeout in `update-homebrew`.** The job retries
  for 5 minutes; if all 30 attempts fail, the GitHub Release exists but
  Homebrew is stale. Manual fix: clone the tap, edit Casks/nereids.rb
  with the new version + sha256, push.
- **Pipeline doesn't start after tag push.** Verify the tag matches
  `v*` (not just any tag) and check the Actions tab. If the workflow is
  disabled, re-enable it.
