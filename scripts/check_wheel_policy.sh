#!/usr/bin/env bash
# check_wheel_policy.sh — Linux wheel compatibility policy.
#
# SINGLE SOURCE OF TRUTH for the manylinux ceiling. ORNL's neutron-imaging
# analysis fleet runs RHEL 8 (glibc 2.28), so every Linux wheel we publish
# must be manylinux_2_28-compatible. Raise MAX_GLIBC_MINOR only when ORNL
# retires RHEL 8.
#
# Checks, per Linux wheel:
#   (a) filename platform tag is manylinux_2_N with N <= MAX_GLIBC_MINOR
#       (bare linux_* tags are rejected — PyPI refuses them anyway);
#   (b) `auditwheel show` grades the wheel at or below the ceiling;
#   (c) the wheel vendors NO shared libraries (a "<dist>.libs/" directory
#       is the tripwire that a system C library entered the link graph —
#       exactly how rfd's gtk3 backend broke the 0.2.0/0.2.1 releases:
#       GTK forced the build off the 2_28 image AND grafted a fragile
#       64-library GTK stack into the wheel);
#   (d) no ELF inside the wheel references a GLIBC_2.N symbol version
#       above the ceiling.
#
# Runs on an ubuntu GitHub runner (needs: unzip, file, objdump — all
# preinstalled — plus auditwheel via pipx on first use).
#
# Usage: scripts/check_wheel_policy.sh dist/*.whl
#        (non-Linux wheels are skipped; at least one Linux wheel required)
set -euo pipefail

MAX_GLIBC_MINOR=28 # ORNL RHEL 8 ceiling — see header before changing.
CEILING="manylinux_2_${MAX_GLIBC_MINOR}"

# Wheel-extraction temp dirs are removed inline on the success path,
# but set -e can abort between mktemp and that rm (unzip/objdump
# failure) — the EXIT trap guarantees no leak on the runner either way.
TMPDIRS=()
cleanup() {
    if [ "${#TMPDIRS[@]}" -gt 0 ]; then
        rm -rf "${TMPDIRS[@]}"
    fi
}
trap cleanup EXIT

if ! command -v auditwheel >/dev/null 2>&1; then
    pipx install auditwheel >/dev/null
    # pipx's bin dir is normally on PATH on GitHub runners; hash again.
    command -v auditwheel >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"
fi

fail=0
seen_linux=0
for whl in "$@"; do
    case "$(basename "$whl")" in
    *linux*) ;;
    *)
        echo "SKIP (not a Linux wheel): $whl"
        continue
        ;;
    esac
    seen_linux=1
    whl_fail=0

    # (a) filename platform tag — a multi-tagged wheel (e.g.
    # manylinux_2_17.manylinux_2_34) is installable via its LOWEST tag,
    # so the minimum governs fleet installability.
    tag=$(basename "$whl" | grep -oE 'manylinux_2_[0-9]+' | sort -t_ -k3 -n | head -1 || true)
    if [ -z "$tag" ]; then
        echo "FAIL: $whl has no manylinux_2_N platform tag"
        fail=1
        continue
    fi
    minor=${tag##manylinux_2_}
    if [ "$minor" -gt "$MAX_GLIBC_MINOR" ]; then
        echo "FAIL: $whl filename tag $tag exceeds the $CEILING ceiling"
        whl_fail=1
    fi

    # (b) auditwheel grade (its "is consistent with the tag" verdict is
    # not enough — extract the policy it grades the wheel at)
    show_output=$(auditwheel show "$whl")
    echo "---- auditwheel show $(basename "$whl") ----"
    echo "$show_output"
    # auditwheel line-wraps its verdict, so flatten before matching.
    grade=$(echo "$show_output" | tr '\n' ' ' |
        grep -oE 'is consistent with the following platform tag: "manylinux_2_[0-9]+' |
        grep -oE 'manylinux_2_[0-9]+' | head -1 || true)
    if [ -z "$grade" ]; then
        echo "FAIL: could not extract an auditwheel manylinux grade for $whl"
        whl_fail=1
    else
        gminor=${grade##manylinux_2_}
        if [ "$gminor" -gt "$MAX_GLIBC_MINOR" ]; then
            echo "FAIL: auditwheel grades $whl as $grade (> $CEILING)"
            whl_fail=1
        fi
    fi

    # (c) no vendored shared-library directory
    if unzip -l "$whl" | grep -E '\.libs/'; then
        echo "FAIL: $whl vendors shared libraries (*.libs/) — a system C" \
            "dependency entered the link graph"
        whl_fail=1
    fi

    # (d) versioned GLIBC symbols in every contained ELF
    tmp=$(mktemp -d)
    TMPDIRS+=("$tmp")
    unzip -q "$whl" -d "$tmp"
    while IFS= read -r elf; do
        max=$(objdump -T "$elf" 2>/dev/null | grep -oE 'GLIBC_2\.[0-9]+' | sort -uV | tail -1 || true)
        [ -z "$max" ] && continue
        max_minor=${max##GLIBC_2.}
        if [ "$max_minor" -gt "$MAX_GLIBC_MINOR" ]; then
            echo "FAIL: ${elf#"$tmp"/} requires $max (> GLIBC_2.$MAX_GLIBC_MINOR)"
            whl_fail=1
        fi
    done < <(find "$tmp" -type f -exec sh -c 'file -b "$1" | grep -q "^ELF" && echo "$1"' _ {} \;)
    rm -rf "$tmp"

    if [ "$whl_fail" -eq 0 ]; then
        echo "OK: $whl satisfies the $CEILING policy"
    else
        fail=1
    fi
done

if [ "$seen_linux" -eq 0 ]; then
    echo "FAIL: no Linux wheel among: $*"
    fail=1
fi
exit "$fail"
