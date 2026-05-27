#!/bin/sh
# Bootstrap the piano-sampler toolchain on a fresh pod spawn.
#
# The pod's $HOME is ephemeral but /data/workspaces persists, so we install
# Python + venv + the piano-sampler package under
# /data/workspaces/C0B62K2A3NX/.tools/. Idempotent — fast on re-runs.

set -eu

TOOLS=/data/workspaces/C0B62K2A3NX/.tools
PY="$TOOLS/python/bin/python3"
VENV="$TOOLS/venv"
PYBSURL="https://github.com/astral-sh/python-build-standalone/releases/download/20260510/cpython-3.12.13%2B20260510-x86_64-unknown-linux-musl-install_only_stripped.tar.gz"

mkdir -p "$TOOLS"

if [ ! -x "$PY" ]; then
  echo "[bootstrap] downloading Python..."
  (cd "$TOOLS" && curl -fsSL -o py.tar.gz "$PYBSURL" && tar xzf py.tar.gz && rm py.tar.gz)
fi

if [ ! -x "$VENV/bin/python3" ]; then
  echo "[bootstrap] creating venv..."
  "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
fi

if ! "$VENV/bin/python3" -c 'import piano_sampler' 2>/dev/null; then
  echo "[bootstrap] installing piano-sampler..."
  cd /data/workspaces/C0B62K2A3NX/audio-plugins/piano-sampler
  "$VENV/bin/pip" install --quiet numpy click
  "$VENV/bin/pip" install --quiet -e .
fi

echo "[bootstrap] ready. Activate with: . $VENV/bin/activate"
