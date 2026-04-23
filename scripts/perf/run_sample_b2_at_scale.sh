#!/usr/bin/env bash
# Profile driver for `profile_b2_kl_grouped_at_scale.py`.  Passes
# CROP_SIZE (default 32) to the Python driver and samples for a
# window proportional to the expected wall.
set -euo pipefail
cd "$(dirname "$0")/../.."

CROP=${1:-32}
# Expected wall ~ (CROP/4)^2 × 0.2 s = CROP²/80 seconds (scales with n_pixels).
# Sample for 1.5× that to cover the full fit plus the sampler's startup
# handshake.  Minimum 10 s, maximum 90 s.
EXPECTED=$(( (CROP * CROP / 80) * 3 / 2 ))
SAMPLE_WINDOW=$(( EXPECTED > 90 ? 90 : (EXPECTED < 10 ? 10 : EXPECTED) ))

PY_BIN=$(pixi info --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['environments_info'][0]['prefix'])")/bin/python

rm -f /tmp/b2_at_scale_ready
"$PY_BIN" scripts/perf/profile_b2_kl_grouped_at_scale.py "$CROP" &
PID=$!

for _ in $(seq 1 1200); do
    if [ -f /tmp/b2_at_scale_ready ]; then break; fi
    sleep 0.05
done
sleep 0.2

# Tolerate sampler races: `/usr/bin/sample` can race the Python driver
# to completion, making its exit non-zero.  That's a benign failure —
# the Python driver still produced its output.  But the Python driver
# itself must NOT be masked: if it hits an import error, missing
# fixture, or runtime crash, the wrapper should exit non-zero so
# callers don't silently treat a failed profile as a successful run.
/usr/bin/sample "$PID" "$SAMPLE_WINDOW" 1 -file /tmp/b2_at_scale.sample.txt -fullPaths >/dev/null 2>&1 || true

set +e
wait "$PID"
PY_STATUS=$?
set -e

if [ -f /tmp/b2_at_scale.sample.txt ]; then
    echo "profile: /tmp/b2_at_scale.sample.txt ($(wc -l </tmp/b2_at_scale.sample.txt) lines)"
else
    echo "profile: /tmp/b2_at_scale.sample.txt was not created"
fi

# Propagate the Python driver's status (signal 15 / SIGTERM from the
# sampler race is also benign and produces status 143 — accept both 0
# and 143; any other non-zero indicates a real driver failure).
if [ "$PY_STATUS" -ne 0 ] && [ "$PY_STATUS" -ne 143 ]; then
    echo "driver failed (exit $PY_STATUS)" >&2
    exit "$PY_STATUS"
fi
