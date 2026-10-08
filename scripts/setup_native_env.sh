#!/usr/bin/env bash
set -euo pipefail

VENV_ROOT="${GPU_VENV_ROOT:-$HOME/.cache/gpu-nvidia/native}"
mkdir -p "$(dirname "$VENV_ROOT")"

if [ ! -x "$VENV_ROOT/bin/python" ]; then
  python3 -m venv "$VENV_ROOT"
fi

"$VENV_ROOT/bin/python" -m pip install -q --upgrade pip
"$VENV_ROOT/bin/python" -m pip install -q -r "$GITHUB_WORKSPACE/requirements-dev.txt"
"$VENV_ROOT/bin/python" -m pip install -q "cupy-cuda12x"
"$VENV_ROOT/bin/python" -m pip install -q "nvidia-cuda-runtime-cu12==12.4.*"

echo "NATIVE_VENV=$VENV_ROOT" >> "$GITHUB_ENV"
echo "CUPY_CACHE_DIR=$HOME/.cache/gpu-nvidia/cupy" >> "$GITHUB_ENV"
mkdir -p "$HOME/.cache/gpu-nvidia/cupy"
