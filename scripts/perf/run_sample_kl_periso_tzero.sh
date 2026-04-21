#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PY_BIN=$(pixi info --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['environments_info'][0]['prefix'])")/bin/python

rm -f /tmp/kl_periso_tzero_ready
"$PY_BIN" scripts/perf/profile_kl_periso_tzero.py &
PID=$!

for _ in $(seq 1 1200); do
    if [ -f /tmp/kl_periso_tzero_ready ]; then break; fi
    sleep 0.05
done
sleep 0.2

/usr/bin/sample "$PID" 15 1 -file /tmp/kl_periso_tzero.sample.txt -fullPaths >/dev/null 2>&1 || true

wait "$PID" 2>/dev/null || true

if [ -f /tmp/kl_periso_tzero.sample.txt ]; then
    echo "profile: /tmp/kl_periso_tzero.sample.txt ($(wc -l </tmp/kl_periso_tzero.sample.txt) lines)"
else
    echo "profile: /tmp/kl_periso_tzero.sample.txt was not created (sampling unavailable or target exited early)"
fi
