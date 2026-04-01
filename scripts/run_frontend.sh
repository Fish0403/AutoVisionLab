#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_DIR="$ROOT_DIR/frontend/web"

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required to run the React frontend."
  exit 1
fi

if [[ ! -d "$WEB_DIR" ]]; then
  echo "React frontend directory not found at $WEB_DIR"
  exit 1
fi

cd "$WEB_DIR"

if [[ ! -d node_modules ]]; then
  echo "Installing frontend dependencies..."
  npm install
fi

npm run dev -- --host 127.0.0.1 --port 5173
