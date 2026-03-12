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
#   6. Cargo.lock  (via cargo update --workspace)
#
# The script does NOT touch:
#   - Test fixtures with hardcoded versions (those are test data)
#   - Runtime code (uses env!("CARGO_PKG_VERSION") from Cargo.toml)
#   - The tap repo (ornlneutronimaging/homebrew-nereids) — CI handles that

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

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

if [ "$CURRENT_VERSION" = "$NEW_VERSION" ]; then
    echo "Already at version $NEW_VERSION — nothing to do."
    exit 0
fi

echo "Bumping version: $CURRENT_VERSION → $NEW_VERSION"
if $DRY_RUN; then
    echo "(dry run — no files will be modified)"
fi

# --- Helper: apply sed in-place (macOS + Linux compatible) ---
apply_sed() {
    local file="$1"
    local pattern="$2"
    if $DRY_RUN; then
        echo "  would update: $file"
        return
    fi
    if [[ "$OSTYPE" == darwin* ]]; then
        sed -i '' "$pattern" "$file"
    else
        sed -i "$pattern" "$file"
    fi
    echo "  updated: $file"
}

# 1. Cargo.toml — workspace.package version (first occurrence)
apply_sed "$REPO_ROOT/Cargo.toml" \
    "0,/^version = \"$CURRENT_VERSION\"/{s/^version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/}"

# 2. Cargo.toml — workspace.dependencies internal crate versions
#    These lines look like: endf-mat = { version = "0.1.0", path = "..." }
apply_sed "$REPO_ROOT/Cargo.toml" \
    "s/version = \"$CURRENT_VERSION\", path =/version = \"$NEW_VERSION\", path =/g"

# 3. pyproject.toml (Python bindings)
apply_sed "$REPO_ROOT/pyproject.toml" \
    "s/^version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/"

# 4. apps/gui/pyproject.toml (GUI wheel)
apply_sed "$REPO_ROOT/apps/gui/pyproject.toml" \
    "s/^version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/"

# 5. homebrew/nereids.rb (local template)
apply_sed "$REPO_ROOT/homebrew/nereids.rb" \
    "s/version \"$CURRENT_VERSION\"/version \"$NEW_VERSION\"/"

# 6. pyproject.toml — gui optional dependency version
apply_sed "$REPO_ROOT/pyproject.toml" \
    "s/nereids-gui==$CURRENT_VERSION/nereids-gui==$NEW_VERSION/"

# 7. Cargo.lock — regenerate from updated Cargo.toml
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
