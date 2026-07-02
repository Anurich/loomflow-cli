#!/usr/bin/env bash
# Reproducible Terminal-Bench runner for loom-code.
#
# Handles the fiddly parts so every run is the same: loads the provider
# key from loom-code's saved credentials, rebuilds the wheel from the
# current tree (so the container runs THIS code), and invokes tb with a
# consistent set of flags. Override anything via env or flags.
#
# Usage:
#   ./run_bench.sh                      # default slice, free Nemotron
#   ./run_bench.sh -t hello-world       # one task
#   TASKS="a b c" ./run_bench.sh        # explicit task list
#   MODEL=gpt-4.1 ./run_bench.sh -n 8   # different model + concurrency
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$here/../.." && pwd)"

# --- config (env-overridable) ------------------------------------------
MODEL="${MODEL:-litellm/nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2}"
DATASET="${DATASET:-terminal-bench-core==0.1.1}"
N_CONCURRENT="${N_CONCURRENT:-2}"
# A fixed 15-task slice spanning difficulties — a representative first
# signal, not a cherry-picked easy set. Override with TASKS="..." or -t.
DEFAULT_TASKS="hello-world fibonacci-server fix-permissions csv-to-parquet \
count-dataset-tokens configure-git-webserver create-bucket chess-best-move \
crack-7z-hash.easy raman-fitting.easy fix-pandas-version extract-safely \
conda-env-conflict-resolution cron-broken-network get-bitcoin-nodes"
TASKS="${TASKS:-$DEFAULT_TASKS}"

# Pass-through extra args (e.g. -t single-task, -n N) to tb.
extra_tb_args=()
while [ $# -gt 0 ]; do extra_tb_args+=("$1"); shift; done

# --- key: pull from loom-code credentials if not already exported -------
creds="$HOME/.loom-code/credentials"
for var in NVIDIA_NIM_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY GROQ_API_KEY; do
  if [ -z "${!var:-}" ] && [ -f "$creds" ]; then
    # `|| true` so a grep miss under `set -e`/`pipefail` doesn't abort
    # the whole script — a missing key is fine (that provider's just
    # unavailable), not a fatal error.
    val=$(grep "^${var}=" "$creds" 2>/dev/null | head -1 | cut -d= -f2- | tr -d "\"' " || true)
    [ -n "$val" ] && export "$var=$val" || true
  fi
done

# --- fresh wheel so the container runs the CURRENT code -----------------
echo "[run_bench] building wheel from working tree..."
"$here/build_wheel.sh" >/dev/null
echo "[run_bench] model:   $MODEL"
echo "[run_bench] dataset: $DATASET"
echo "[run_bench] tasks:   $(echo $TASKS | wc -w | tr -d ' ') task(s)"

# --- build -t flags from the task list ---------------------------------
task_flags=()
if [ ${#extra_tb_args[@]} -eq 0 ]; then
  for t in $TASKS; do task_flags+=(-t "$t"); done
fi

cd "$repo_root"
# Expand arrays safely under `set -u`: an empty array with "${arr[@]}"
# trips "unbound variable" on some bash versions, so guard with :+.
set -x
tb run \
  --agent-import-path benchmarks.terminal_bench.loom_code_agent:LoomCodeAgent \
  --model "$MODEL" \
  -d "$DATASET" \
  --n-concurrent "$N_CONCURRENT" \
  --no-livestream \
  ${task_flags[@]:+"${task_flags[@]}"} \
  ${extra_tb_args[@]:+"${extra_tb_args[@]}"}
