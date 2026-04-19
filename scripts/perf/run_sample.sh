#!/usr/bin/env bash
# Run scripts/perf/profile_a1_lm_grouped.py under /usr/bin/sample (macOS).
# Produces /tmp/a1.sample.txt with symbolicated stacks.
set -euo pipefail

cd "$(dirname "$0")/../.."

PY_BIN=$(pixi info --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['environments_info'][0]['prefix'])")/bin/python

# Launch the driver in the background
rm -f /tmp/a1_ready
"$PY_BIN" scripts/perf/profile_a1_lm_grouped.py &
PID=$!

# Wait for the driver to signal it has finished the load/warmup phase.
for _ in $(seq 1 600); do
    if [ -f /tmp/a1_ready ]; then break; fi
    sleep 0.05
done
# Small slack so the warm-up LM iters settle into the hot loop.
sleep 0.2

# Sample for 7s at 1ms intervals. The LM fit runs for ~5.4s total.
/usr/bin/sample "$PID" 7 1 -file /tmp/a1.sample.txt -fullPaths >/dev/null 2>&1 || true

# Let Python finish (already mostly done by now)
wait "$PID" 2>/dev/null || true

echo "profile written: /tmp/a1.sample.txt"
echo "size: $(wc -l </tmp/a1.sample.txt) lines"
