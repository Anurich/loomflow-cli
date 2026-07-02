#!/usr/bin/env bash
# Build a loom-code wheel from the current working tree into ./dist,
# for the Terminal-Bench adapter to copy into task containers.
#
# Rebuild this whenever you change loom-code and want the benchmark to
# pick up the change — the adapter installs the NEWEST wheel in dist/.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$here/../.." && pwd)"
python_bin="${PYTHON:-$repo_root/.venv/bin/python}"
command -v "$python_bin" >/dev/null 2>&1 || python_bin="python3"

echo "[build] using $python_bin"
"$python_bin" -m pip install --quiet build
# Clean old wheels so the adapter never picks a stale one.
rm -f "$here/dist"/loom_code-*.whl
"$python_bin" -m build --wheel --outdir "$here/dist" "$repo_root"
echo "[build] done:"
ls -la "$here/dist"/loom_code-*.whl
