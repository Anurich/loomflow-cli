#!/usr/bin/env bash
# Install loom-code inside a Terminal-Bench task container.
#
# Harbor copies this in and runs it once before the agent commands. The
# container is Debian/Ubuntu-based; assume nothing beyond that. Keep it
# idempotent and fail loudly — a silent half-install would score as a
# task failure with no clue why.
set -euo pipefail

echo "[loom-code setup] installing prerequisites..."

# Python 3.11+, pip, and git. The TB base images vary — some ship
# python3+pip but NOT git (which pip needs to clone a git+https spec),
# so check each tool independently rather than assuming one implies the
# others. Only apt-get update once, and only if something's missing.
missing=""
command -v python3 >/dev/null 2>&1 || missing="$missing python3 python3-venv"
command -v pip3    >/dev/null 2>&1 || missing="$missing python3-pip"
command -v git     >/dev/null 2>&1 || missing="$missing git"

if [ -n "$missing" ]; then
  echo "[loom-code setup] installing:$missing"
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    # shellcheck disable=SC2086
    apt-get install -y -qq $missing
  elif command -v apk >/dev/null 2>&1; then
    # Alpine base — package names differ slightly.
    apk add --no-cache python3 py3-pip git
  else
    echo "[loom-code setup] ERROR: no apt-get/apk to install:$missing" >&2
    exit 1
  fi
fi

python3 --version

# Install loom-code from the wheel the adapter copied in (the repo is
# PRIVATE — cloning would hang on a credential prompt). The wheel's deps
# (loomflow, graphifyy, jedi, ...) are public and pulled from PyPI.
LOOM_WHEEL_DIR="${LOOM_WHEEL_DIR:-/installed-agent/loom-wheels}"
wheel=$(ls -t "${LOOM_WHEEL_DIR}"/loom_code-*.whl 2>/dev/null | head -1 || true)
if [ -z "${wheel}" ]; then
  echo "[loom-code setup] ERROR: no wheel in ${LOOM_WHEEL_DIR}" >&2
  exit 1
fi

echo "[loom-code setup] pip install ${wheel} ..."
# --break-system-packages: TB containers are ephemeral; a global install
# is fine and avoids venv-activation plumbing in the run command.
pip3 install --no-cache-dir --break-system-packages "${wheel}"

# Sanity: the console script must be on PATH for _run_agent_commands.
if ! command -v loom-code >/dev/null 2>&1; then
  echo "[loom-code setup] ERROR: loom-code not on PATH after install" >&2
  exit 1
fi

loom-code --help >/dev/null
echo "[loom-code setup] done — loom-code installed and runnable."
