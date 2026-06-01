#!/usr/bin/env bash
# bump-version.sh — Update the version string across the entire workspace.
#
# Usage:
#   ./scripts/bump-version.sh 0.2.0
#   ./scripts/bump-version.sh 0.2.0 --dry-run
#
# Locations updated:
#   1. Cargo.toml  [workspace.package] version
#   2. Cargo.toml  [workspace.dependencies] — all internal crate version fields
#   3. pyproject.toml  (Python bindings)
#   4. apps/gui/pyproject.toml  (GUI wheel)
#   5. homebrew/nereids.rb  (local template)
#   6. pyproject.toml  — nereids-gui optional-dependency pin
#   7. CITATION.cff  — version + date-released
#   8. CHANGELOG.md  — roll [Unreleased] → new dated section + link-refs
#   9. Cargo.lock  (via cargo update --workspace)
#
# The script does NOT touch:
#   - Test fixtures with hardcoded versions (those are test data)
#   - Runtime code (uses env!("CARGO_PKG_VERSION") from Cargo.toml)
#   - The tap repo (ornlneutronimaging/homebrew-nereids) — CI handles that

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_URL="https://github.com/ornlneutronimaging/NEREIDS"
RELEASE_DATE="$(date +%F)" # YYYY-MM-DD; portable on BSD + GNU date

# --- Parse arguments ---
NEW_VERSION=""
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --) ;; # ignore -- separator from pixi
        -*) echo "Unknown flag: $arg" >&2; exit 1 ;;
        *)
            if [ -n "$NEW_VERSION" ]; then
                echo "Error: multiple version arguments" >&2; exit 1
            fi
            NEW_VERSION="$arg"
            ;;
    esac
done

if [ -z "$NEW_VERSION" ]; then
    echo "Usage: $0 <new-version> [--dry-run]" >&2
    echo "Example: $0 0.2.0" >&2
    exit 1
fi

# Validate semver format (strict: MAJOR.MINOR.PATCH with optional pre-release)
if ! echo "$NEW_VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$'; then
    echo "Error: '$NEW_VERSION' is not valid semver (expected X.Y.Z or X.Y.Z-pre)" >&2
    exit 1
fi

# --- Read current version from workspace Cargo.toml ---
CURRENT_VERSION=$(grep -m1 '^version = ' "$REPO_ROOT/Cargo.toml" | sed 's/version = "\(.*\)"/\1/')
if [ -z "$CURRENT_VERSION" ]; then
    echo "Error: could not read current version from Cargo.toml" >&2
    exit 1
fi

# Regex-safe form of the current version for sed/grep PATTERN contexts: dots are
# regex any-char, so escape them (e.g. a pre-release like 0.2.0-rc.1 then matches
# literally, never an unintended 0x2x0-rcx1). NEW_VERSION needs no such form — it
# only ever appears on sed replacement sides or as literal awk string concatenation.
CURRENT_RE="${CURRENT_VERSION//./\\.}"

if [ "$CURRENT_VERSION" = "$NEW_VERSION" ]; then
    echo "Already at version $NEW_VERSION — nothing to do."
    exit 0
fi

echo "Bumping version: $CURRENT_VERSION → $NEW_VERSION"
if $DRY_RUN; then
    echo "(dry run — no files will be modified)"
fi

# --- Helper: apply sed in-place (macOS + Linux compatible) ---
# Reports "unchanged" when the pattern matched nothing, rather than a misleading
# "updated" — a silent no-op on release metadata (a drifted version-line format)
# would otherwise look successful. Not a hard error: some calls (CITATION.cff
# date-released when absent) are intentional no-ops.
apply_sed() {
    local file="$1"
    local pattern="$2"
    if $DRY_RUN; then
        echo "  would update: $file"
        return
    fi
    local before
    before="$(cat "$file")"
    if [[ "$OSTYPE" == darwin* ]]; then
        sed -i '' "$pattern" "$file"
    else
        sed -i "$pattern" "$file"
    fi
    if [ "$before" = "$(cat "$file")" ]; then
        echo "  unchanged: $file (pattern not matched)"
    else
        echo "  updated: $file"
    fi
}

# --- Helper: roll the CHANGELOG [Unreleased] section to the new version ---
# Freezes the curated `## [Unreleased]` notes into a dated `## [NEW]` section
# (Keep a Changelog workflow) and fixes the link-reference definitions.
# Guarded: only touches a well-formed file, so it can never corrupt a
# hand-maintained CHANGELOG. awk (not sed) because inserting lines portably
# is awkward in sed — `\n` in a replacement is a GNU-only extension.
roll_changelog() {
    local file="$1"
    if [ ! -f "$file" ]; then
        echo "  skipped: $file (not found)"
        return
    fi
    # Require EXACTLY one '## [Unreleased]' heading and one '[Unreleased]:' link,
    # and that the single link matches the current version. awk rewrites *every*
    # '## [Unreleased]' heading and *every* '[Unreleased]: ' line, so the guard
    # must count the same broad sets it acts on (n_link_any) — not just the
    # version-matching subset (n_link_ok) — or a file with one correct link plus
    # a stale duplicate would pass yet get double-rewritten.
    local n_head n_link_any n_link_ok
    n_head=$(grep -cE '^## \[Unreleased\]$' "$file" || true)
    n_link_any=$(grep -cE '^\[Unreleased\]: ' "$file" || true)
    n_link_ok=$(grep -cE "^\[Unreleased\]: .*compare/v${CURRENT_RE}\.\.\.HEAD$" "$file" || true)
    if [ "${n_head:-0}" -ne 1 ] || [ "${n_link_any:-0}" -ne 1 ] || [ "${n_link_ok:-0}" -ne 1 ]; then
        echo "  skipped: $file (need exactly one '## [Unreleased]' heading and one matching '[Unreleased]:' link — roll manually)"
        return
    fi
    if $DRY_RUN; then
        echo "  would update: $file (roll [Unreleased] → $NEW_VERSION)"
        return
    fi
    local tmp
    tmp="$(mktemp)"
    if awk -v cur="$CURRENT_VERSION" -v new="$NEW_VERSION" -v date="$RELEASE_DATE" -v url="$REPO_URL" '
        /^## \[Unreleased\]$/ {
            print
            print ""
            print "## [" new "] - " date
            next
        }
        /^\[Unreleased\]: / {
            print "[Unreleased]: " url "/compare/v" new "...HEAD"
            print "[" new "]: " url "/compare/v" cur "...v" new
            next
        }
        { print }
    ' "$file" >"$tmp"; then
        mv "$tmp" "$file"
        echo "  updated: $file (rolled [Unreleased] → $NEW_VERSION)"
    else
        rm -f "$tmp"
        echo "  ERROR: failed to roll $file" >&2
        return 1
    fi
}

# 1. Cargo.toml — workspace.package version
#    This is the only bare `version = "X.Y.Z"` line (deps have `, path =` after)
apply_sed "$REPO_ROOT/Cargo.toml" \
    "s/^version = \"$CURRENT_RE\"$/version = \"$NEW_VERSION\"/"

# 2. Cargo.toml — workspace.dependencies internal crate versions
#    These lines look like: endf-mat = { version = "0.1.0", path = "..." }
apply_sed "$REPO_ROOT/Cargo.toml" \
    "s/version = \"$CURRENT_RE\", path =/version = \"$NEW_VERSION\", path =/g"

# 3. pyproject.toml (Python bindings)
apply_sed "$REPO_ROOT/pyproject.toml" \
    "s/^version = \"$CURRENT_RE\"/version = \"$NEW_VERSION\"/"

# 4. apps/gui/pyproject.toml (GUI wheel)
apply_sed "$REPO_ROOT/apps/gui/pyproject.toml" \
    "s/^version = \"$CURRENT_RE\"/version = \"$NEW_VERSION\"/"

# 5. homebrew/nereids.rb (local template)
apply_sed "$REPO_ROOT/homebrew/nereids.rb" \
    "s/version \"$CURRENT_RE\"/version \"$NEW_VERSION\"/"

# 6. pyproject.toml — gui optional dependency version
apply_sed "$REPO_ROOT/pyproject.toml" \
    "s/nereids-gui==$CURRENT_RE/nereids-gui==$NEW_VERSION/"

# 7. CITATION.cff — version + date-released
#    `^version:` is anchored so it never matches the `cff-version:` line.
#    The date-released line is updated in place; harmless no-op if absent.
apply_sed "$REPO_ROOT/CITATION.cff" \
    "s/^version: $CURRENT_RE$/version: $NEW_VERSION/"
apply_sed "$REPO_ROOT/CITATION.cff" \
    "s/^date-released:.*/date-released: $RELEASE_DATE/"

# 8. CHANGELOG.md — roll [Unreleased] to the new dated version + link-refs
roll_changelog "$REPO_ROOT/CHANGELOG.md"

# 9. Cargo.lock — regenerate from updated Cargo.toml
if ! $DRY_RUN; then
    echo "  updating Cargo.lock..."
    (cd "$REPO_ROOT" && cargo update --workspace 2>/dev/null)
    echo "  updated: Cargo.lock"
fi

# --- Summary ---
echo ""
if $DRY_RUN; then
    echo "Dry run complete. Run without --dry-run to apply."
else
    echo "Done. Version is now $NEW_VERSION across the workspace."
    echo ""
    echo "Next steps:"
    echo "  1. Review changes: git diff"
    echo "  2. Commit: git commit -S -am 'Bump version to $NEW_VERSION'"
    echo "  3. Tag: git tag -s v$NEW_VERSION -m 'Release v$NEW_VERSION'"
    echo "  4. Push: git push origin main --tags"
fi
