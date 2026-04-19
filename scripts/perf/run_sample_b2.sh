#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PY_BIN=$(pixi info --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['environments_info'][0]['prefix'])")/bin/python

rm -f /tmp/a1_ready
"$PY_BIN" scripts/perf/profile_b2_kl_grouped.py &
PID=$!

for _ in $(seq 1 1200); do
    if [ -f /tmp/a1_ready ]; then break; fi
    sleep 0.05
done
sleep 0.2

/usr/bin/sample "$PID" 6 1 -file /tmp/b2.sample.txt -fullPaths >/dev/null 2>&1 || true

wait "$PID" 2>/dev/null || true

echo "profile: /tmp/b2.sample.txt ($(wc -l </tmp/b2.sample.txt) lines)"
