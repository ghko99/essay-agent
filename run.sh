#!/usr/bin/env bash
# Launch the Kanana AES Agent STEP 1 prototype.
set -euo pipefail

cd "$(dirname "$0")"

export KANANA_BASE="${KANANA_BASE:-/home/khko/models/kanana}"
export KANANA_ADAPTER="${KANANA_ADAPTER:-$(pwd)/adapter}"
export PORT="${PORT:-8000}"
export HOST="${HOST:-0.0.0.0}"

echo "[run.sh] Base model : $KANANA_BASE"
echo "[run.sh] LoRA adapter: $KANANA_ADAPTER"
echo "[run.sh] Listening   : http://$HOST:$PORT"

exec uvicorn backend.main:app --host "$HOST" --port "$PORT" --workers 1
