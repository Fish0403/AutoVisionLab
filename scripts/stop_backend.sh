#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORCE_STOP=0

if [[ "${1:-}" == "--force" ]]; then
  FORCE_STOP=1
fi

find_backend_pids() {
  ps -eo pid=,args= | awk -v root_dir="$ROOT_DIR" '
    index($0, root_dir) > 0 && index($0, "uvicorn app.main:app") > 0 && index($0, "--app-dir backend") > 0 {
      print $1
    }
  '
}

backend_pids="$(find_backend_pids)"
if [[ -z "$backend_pids" ]]; then
  echo "No AutoVisionLab backend process found."
  exit 0
fi

echo "Stopping backend PIDs: $(echo "$backend_pids" | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
echo "$backend_pids" | xargs kill

sleep 1

remaining_pids="$(find_backend_pids)"
if [[ -z "$remaining_pids" ]]; then
  echo "Backend stopped."
  exit 0
fi

if [[ "$FORCE_STOP" -eq 1 ]]; then
  echo "Force stopping remaining backend PIDs: $(echo "$remaining_pids" | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
  echo "$remaining_pids" | xargs kill -9
  echo "Backend force stopped."
  exit 0
fi

echo "Backend is still running: $(echo "$remaining_pids" | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
echo "Run ./scripts/stop_backend.sh --force to send SIGKILL."
exit 1
