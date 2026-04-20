#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PY_BIN=$(pixi info --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['environments_info'][0]['prefix'])")/bin/python

rm -f /tmp/b3_ready
"$PY_BIN" scripts/perf/profile_b3_lm_periso_tzero.py &
PID=$!

for _ in $(seq 1 1200); do
    if [ -f /tmp/b3_ready ]; then break; fi
    sleep 0.05
done
sleep 0.2

# B.3 4×4 LM+per-iso+TZERO at max_iter=200 typically runs 40-80 s.
# Sample for 30 s at 1 ms intervals for a solid profile window.
/usr/bin/sample "$PID" 30 1 -file /tmp/b3.sample.txt -fullPaths >/dev/null 2>&1 || true

wait "$PID" 2>/dev/null || true

echo "profile: /tmp/b3.sample.txt ($(wc -l </tmp/b3.sample.txt) lines)"
