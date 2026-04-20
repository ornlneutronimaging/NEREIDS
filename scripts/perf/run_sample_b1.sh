#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PY_BIN=$(pixi info --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['environments_info'][0]['prefix'])")/bin/python

rm -f /tmp/b1_ready
"$PY_BIN" scripts/perf/profile_b1_lm_tzero_grouped.py &
PID=$!

for _ in $(seq 1 1200); do
    if [ -f /tmp/b1_ready ]; then break; fi
    sleep 0.05
done
sleep 0.2

# Profile for a decent window — 4-pixel LM+TZERO at max_iter=50 takes ~60-120 s
# depending on convergence path; sample for 100 s at 1 ms intervals.
/usr/bin/sample "$PID" 100 1 -file /tmp/b1.sample.txt -fullPaths >/dev/null 2>&1 || true

wait "$PID" 2>/dev/null || true

echo "profile: /tmp/b1.sample.txt ($(wc -l </tmp/b1.sample.txt) lines)"
