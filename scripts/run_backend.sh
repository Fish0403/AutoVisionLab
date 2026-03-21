#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Virtual environment not found at $VENV_DIR"
  echo "Create it first: python3 -m venv .venv"
  exit 1
fi

source "$VENV_DIR/bin/activate"
cd "$ROOT_DIR"
uvicorn app.main:app --reload --app-dir backend
