#!/usr/bin/env bash
# Start the full ASR stack on the Jitsi box so the Whisper/Moonshine toggle works:
#
#   jigasi ──ws──▶ router :2700 ─┬─▶ Moonshine :2710
#                                └─▶ Whisper   :2701
#   control: POST http://<this-host>:2770/set_engine  (the orchestrator calls this)
#
# jigasi already connects to ws://<this-host>:2700, which is now the ROUTER, so no
# jigasi config change is needed. Requires faster-whisper in the venv for Whisper.
#
#   ./run-all-asr.sh                       # all three local (Whisper on this box)
#   SKIP_WHISPER=1 ASR_BACKEND_WHISPER=ws://<windows-ip>:2701 ./run-all-asr.sh
#                                          # Whisper runs elsewhere (e.g. Windows GPU);
#                                          # this box runs only Moonshine + router
#   Ctrl-C                                 # stops what it started
set -euo pipefail
cd "$(dirname "$0")"

PY="./.venv/bin/python"
[ -x "$PY" ] || PY="python3"
mkdir -p logs
PIDS=()

echo "[asr] starting Moonshine on :2710"
ASR_PORT=2710 "$PY" asr_ws_server.py > logs/moonshine.log 2>&1 &
PIDS+=($!)

if [ "${SKIP_WHISPER:-0}" = "1" ]; then
  echo "[asr] SKIP_WHISPER=1 -> Whisper runs remotely at ${ASR_BACKEND_WHISPER:-<unset!>}"
else
  echo "[asr] starting Whisper on :2701"
  WHISPER_ASR_PORT=2701 "$PY" whisper_ws_server.py > logs/whisper.log 2>&1 &
  PIDS+=($!)
fi

echo "[asr] starting router on :2700 (control :2770); whisper backend = ${ASR_BACKEND_WHISPER:-ws://127.0.0.1:2701}"
"$PY" asr_router.py > logs/router.log 2>&1 &
PIDS+=($!)

trap 'echo; echo "[asr] stopping..."; kill "${PIDS[@]}" 2>/dev/null || true' INT TERM
echo "[asr] started (${#PIDS[@]} processes). logs in ./logs/. Ctrl-C to stop."
wait
