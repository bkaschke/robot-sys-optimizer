#!/usr/bin/env bash
# Launch the Robot Cell Optimizer on Linux (equivalent of scripts/run_app.bat).
# Uses a bundled environment in ../env when present (unpacked bundle), otherwise the conda /
# micromamba / miniforge environment "rso".
# Usage: scripts/run_app.sh [--dark] [--port 8080]
set -eu
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"
env_dir="$root/../env"
py=""
if [ -x "$env_dir/bin/python" ]; then
  py="$env_dir/bin/python"
  if [ ! -f "$env_dir/.unpacked" ]; then
    echo "First start: preparing the bundled Python environment (once)..."
    if [ -x "$env_dir/bin/conda-unpack" ]; then
      "$env_dir/bin/conda-unpack" || { echo "conda-unpack failed. Unpack the bundle again into a writable folder."; exit 1; }
    fi
    echo done > "$env_dir/.unpacked"
  fi
else
  for c in "$HOME/miniforge3" "$HOME/micromamba" "$HOME/.local/share/mamba" "$HOME/mambaforge" "$HOME/miniconda3" "$HOME/anaconda3" "${CONDA_PREFIX:-}"; do
    [ -n "$c" ] || continue
    if [ -x "$c/envs/rso/bin/python" ]; then py="$c/envs/rso/bin/python"; break; fi
    if [ "$(basename "$c")" = rso ] && [ -x "$c/bin/python" ]; then py="$c/bin/python"; break; fi
  done
fi
if [ -z "$py" ]; then
  echo "No Python environment found: neither $env_dir nor a conda environment named rso." >&2
  echo "See docs/USER_GUIDE.md, section \"Set up on a new computer\" (conda env create -f environment.yml)." >&2
  exit 1
fi
echo "Starting... then open the http://localhost address shown below in your browser (default 8080; Ctrl+C here to stop)."
exec "$py" "$root/app.py" "$@"
