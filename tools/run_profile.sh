#!/usr/bin/env bash
set -euo pipefail
python tools/profiler.py --module orchestrator.app.main:iso_now --out results/profile.prof || true
python - <<'PY'
import pstats
p=pstats.Stats("results/profile.prof")
p.sort_stats("cumtime").print_stats(20)
PY
echo "Profile written to results/profile.prof"
