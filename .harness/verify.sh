#!/usr/bin/env bash
# Local completion gate for NEREIDS development.
#
# This mirrors the repository's CI/build entry points while keeping the local
# gate bounded: all non-fitting workspace tests run, all non-calibration
# fitting unit tests run, and the three Phase-0 calibration families each have
# a direct route gate. The two multi-minute IC closed-loop tests remain CI/
# phase-gate evidence rather than making every local completion check exceed
# the harden workflow's approximately five-minute target.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

START_SECONDS=$SECONDS
PROBE_OUTPUT=""

run() {
    printf '\n>>>'
    printf ' %q' "$@"
    printf '\n'
    "$@"
}

run_probe() {
    local label=$1
    shift
    printf '\n>>> %s\n' "$label"
    printf '   command:'
    printf ' %q' "$@"
    printf '\n'
    PROBE_OUTPUT=$("$@")
    printf '%s\n' "$PROBE_OUTPUT"
}

require_marker() {
    local marker=$1
    if [[ "$PROBE_OUTPUT" != *"$marker"* ]]; then
        printf 'ERROR: probe output lacks required mechanism marker: %s\n' "$marker" >&2
        return 1
    fi
}

require_awk() {
    local description=$1
    local program=$2
    if ! awk -F= "$program" <<<"$PROBE_OUTPUT"; then
        printf 'ERROR: probe invariant failed: %s\n' "$description" >&2
        return 1
    fi
}

printf 'NEREIDS completion gate\n'
printf 'root=%s\n' "$ROOT"

# Fast, dependency-light failures first.
run cargo fmt --all -- --check
run pixi run python scripts/check_python_api_drift.py
run pixi run python investigation/verify_artifacts.py
run pixi run python investigation/verify_phase0.py
run pixi run python investigation/verify_github_program.py

# Compile and lint the same workspace surfaces used by CI.
run cargo check --workspace --exclude nereids-python
run cargo check --workspace --exclude nereids-python --examples
run cargo clippy --workspace --exclude nereids-python --all-targets -- -D warnings

# Broad Rust behavior gate, with the known long-running calibrator isolated
# into one representative matched test per resolution family.
run cargo test --workspace --exclude nereids-python --exclude nereids-fitting
run cargo test -p nereids-fitting --lib -- --skip resolution_calib::tests
run cargo test -p nereids-fitting \
    resolution_calib::tests::gaussian_recovers_known_width -- --nocapture
run cargo test -p nereids-fitting \
    resolution_calib::tests::udr_corr_recovers_known_width_scale_and_exponent \
    -- --nocapture
run cargo test -p nereids-fitting \
    resolution_calib::tests::gaussian_and_ic_families_run_and_converge \
    -- --nocapture

# This task rebuilds the native extension before running the complete public
# Python suite, so the probes below cannot accidentally exercise a stale wheel.
run pixi run test-python

# Phase-0 mechanism anchors. The investigation scripts intentionally print
# observations; assertions here turn those observations into a red/green gate
# instead of treating mere script exit as evidence.
run_probe "Phase-0 real-route semantics" \
    pixi run python investigation/phase0_route_semantics.py
require_marker "kl_transmission_reweight_delta=0"
require_marker "kl_negative_transmission_observation=accepted converged=True"
require_marker "lm_raw_counts_converged=False"
require_marker "lm_prescaled_ob_converged=True"

run_probe "Phase-0 spatial flux handling" \
    pixi run python investigation/phase0_spatial_flux_probe.py
require_marker "from_counts_density_map=[[0.000629"
require_marker "from_counts_converged_map=[[True, True], [True, True]]"
require_marker "paired_nuisance_converged_map=[[True, True], [True, True]]"
require_awk "paired open beam must recover matched truth" '
    $1 == "paired_max_relative_error" { found = 1; error = $2 + 0 }
    END { exit !(found && error < 1e-6) }
'

run_probe "Phase-0 stochastic counts ensemble" \
    pixi run python investigation/phase0_counts_ensemble.py
require_marker "route=joint_poisson exposure=25 total=50 converged=50"
require_marker "route=counts_lm_fallback exposure=25 total=50 converged=50"
require_awk "low-count joint-Poisson bias must remain below the LM fallback bias" '
    /^route=joint_poisson exposure=25 / { route = "joint"; next }
    /^route=counts_lm_fallback exposure=25 / { route = "fallback"; next }
    /relative_bias=/ && route != "" {
        match($0, /relative_bias=[^ ]+/)
        value = substr($0, RSTART + 14, RLENGTH - 14) + 0
        if (route == "joint") joint = value
        if (route == "fallback") fallback = value
        route = ""
    }
    END { exit !(joint > 0 && joint < 0.10 && fallback > 0.40 && fallback > joint) }
'

run_probe "Phase-0 remaining public routes" \
    pixi run python investigation/phase0_remaining_routes.py
require_marker "spatial_transmission_kl_negative_observation=converged=True"
require_marker "single_nonzero_detector_background=rejected type=ValueError"
require_marker "single_counts_lm_fit_alpha_1=accepted converged=True"
require_marker "alpha_1=None alpha_2=None"
require_marker "spatial_nonzero_detector_background=accepted n_failed=1 converged=False density=nan"
require_marker "spatial_counts_nuisance_lm=rejected type=ValueError"

run_probe "Phase-0 IC ordinary-fit handoff" \
    pixi run python investigation/phase0_ic_fit_probe.py
require_marker "transmission_lm_converged=True"
require_marker "counts_joint_poisson_converged=True"
require_awk "IC-as-tabulated matched fits must recover density" '
    $1 ~ /^transmission_lm_converged/ {
        transmission = $3 + 0; have_t = 1
    }
    $1 ~ /^counts_joint_poisson_converged/ {
        counts = $3 + 0; have_c = 1
    }
    END {
        truth = 8e-5
        exit !(have_t && have_c &&
               (transmission - truth < 1e-10) && (truth - transmission < 1e-10) &&
               (counts - truth < 1e-10) && (truth - counts < 1e-10))
    }
'

run_probe "Phase-0 real open-beam diagnostic" \
    pixi run python investigation/phase0_real_open_beam.py
require_marker "pixels_total=65536"
require_marker "pixels_zero_total=24"
require_awk "measured spatial variation must exceed the Poisson-only expectation" '
    $1 == "pixel_total_nonzero_cv" { spatial = $2 + 0; have_spatial = 1 }
    $1 == "poisson_cv_at_mean_total" { poisson = $2 + 0; have_poisson = 1 }
    END { exit !(have_spatial && have_poisson && spatial > 0.04 && spatial > 3 * poisson) }
'

run_probe "Phase-0 MCP route dispatch" \
    pixi run python investigation/phase0_mcp_route_probe.py
require_marker "no-fit-block ('transmission', '<python-default>') True"
require_marker "kl-default-domain ('counts', 'kl') True"
require_marker "lm-explicit-counts ('counts', 'lm') True"
require_marker "kl-domain-typo ('transmission', 'kl') True"

ELAPSED=$((SECONDS - START_SECONDS))
printf '\nPASS: NEREIDS completion gate (%dm%02ds)\n' \
    "$((ELAPSED / 60))" "$((ELAPSED % 60))"
