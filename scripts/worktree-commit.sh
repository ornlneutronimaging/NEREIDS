#!/usr/bin/env bash
# Worktree commit helper — validates directory and commits with GPG signing.
#
# Usage:
#   scripts/worktree-commit.sh <worktree-name> "<commit message>" [files...]
#
# Examples:
#   scripts/worktree-commit.sh perf-flat-jacobian "Perf: flat Jacobian storage" crates/nereids-fitting/src/lm.rs
#   scripts/worktree-commit.sh fix-physics-p2 "Fix: DopplerParams panic→Result"
#
# If no files are specified, stages all tracked modified files (git add -u).
# Always GPG-signs commits (-S).
#
# Designed for use by Claude subagents — eliminates wrong-directory commits.

set -euo pipefail

WORKTREE_BASE=".claude/worktrees"
REPO_ROOT="$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"

usage() {
    echo "Usage: $0 <worktree-name> \"<commit message>\" [files...]" >&2
    exit 1
}

if [ $# -lt 2 ]; then
    usage
fi

WORKTREE_NAME="$1"
COMMIT_MSG="$2"
shift 2

WORKTREE_PATH="$REPO_ROOT/$WORKTREE_BASE/$WORKTREE_NAME"

# Validate worktree exists
if [ ! -d "$WORKTREE_PATH" ]; then
    echo "Error: worktree '$WORKTREE_NAME' not found at $WORKTREE_PATH" >&2
    echo "Available worktrees:" >&2
    ls "$REPO_ROOT/$WORKTREE_BASE/" 2>/dev/null || echo "  (none)" >&2
    exit 1
fi

# Validate it's a git worktree
if ! git -C "$WORKTREE_PATH" rev-parse --is-inside-work-tree &>/dev/null; then
    echo "Error: $WORKTREE_PATH is not a git worktree" >&2
    exit 1
fi

# Show current branch for confirmation
BRANCH=$(git -C "$WORKTREE_PATH" branch --show-current)
echo "Worktree: $WORKTREE_NAME"
echo "Branch:   $BRANCH"
echo "Path:     $WORKTREE_PATH"
echo ""

# Stage files
if [ $# -gt 0 ]; then
    echo "Staging specified files:"
    for f in "$@"; do
        echo "  $f"
        git -C "$WORKTREE_PATH" add "$f"
    done
else
    echo "Staging all tracked modified files (git add -u)"
    git -C "$WORKTREE_PATH" add -u
fi

# Check there's something to commit
if git -C "$WORKTREE_PATH" diff --cached --quiet; then
    echo ""
    echo "Warning: nothing staged to commit. Working tree status:"
    git -C "$WORKTREE_PATH" status --short
    exit 1
fi

echo ""
echo "Staged changes:"
git -C "$WORKTREE_PATH" diff --cached --stat
echo ""

# Commit with GPG signing
git -C "$WORKTREE_PATH" commit -S -m "$COMMIT_MSG"

echo ""
echo "Committed on branch $BRANCH:"
git -C "$WORKTREE_PATH" log --oneline -1
