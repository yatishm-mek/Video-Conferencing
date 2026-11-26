#!/usr/bin/env bash
#
# Push this project's source to a server over SSH with rsync.
# Copies source + inputs; skips the venv, caches, runtime output and .env
# (see deploy/rsync-exclude.txt).
#
# Usage:
#   ./deploy/push_to_server.sh user@host:/remote/path/interview-bot [ssh_port]
#
# Examples:
#   ./deploy/push_to_server.sh deploy@10.0.0.5:/home/deploy/interview-bot
#   ./deploy/push_to_server.sh deploy@server.lan:/srv/interview-bot 2222
#
# Add --delete after you trust it, to remove files on the server that no longer
# exist locally (off by default so a first push can't delete anything).

set -euo pipefail

DEST="${1:?Usage: push_to_server.sh user@host:/remote/path [ssh_port]}"
PORT="${2:-22}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Pushing ${HERE}/ -> ${DEST}  (ssh port ${PORT})"
rsync -avz --human-readable \
  -e "ssh -p ${PORT}" \
  --exclude-from="${HERE}/deploy/rsync-exclude.txt" \
  "${HERE}/" "${DEST%/}/"

cat <<'NEXT'

Pushed. Next, on the server:
  cd <remote path>
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  cp .env.example .env         # then set: HOST=0.0.0.0, LIVE_LLM_MODEL=qwen3.5:9b
  ollama pull qwen3.5:9b       # if not already pulled
  python -m interview_bot.server
Remember: the candidate's browser needs HTTPS (or an SSH tunnel) for the mic.
NEXT
