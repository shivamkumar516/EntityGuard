#!/usr/bin/env bash
# EntityGuard launcher for macOS / Linux:  bash run_app.sh
# First run: creates .venv and installs requirements.txt (a few minutes). Later runs start in seconds.
set -e
cd "$(dirname "$0")"

PY=""
for c in python3.13 python3.12 python3.11 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
    PY="$c"; break
  fi
done
if [ ! -x .venv/bin/python ]; then
  if [ -z "$PY" ]; then
    echo "EntityGuard needs Python 3.11 or newer: https://www.python.org/downloads/" >&2
    exit 1
  fi
  echo "Creating the virtual environment in .venv ..."
  "$PY" -m venv .venv
fi
if ! cmp -s requirements.txt .venv/requirements.installed; then
  echo "Installing packages from requirements.txt ..."
  .venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
  cp requirements.txt .venv/requirements.installed
fi
echo "Starting EntityGuard (Ctrl+C to stop) ..."
exec .venv/bin/python app.py --open "$@"
