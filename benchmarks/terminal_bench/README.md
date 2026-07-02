# Running loom-code on Terminal-Bench

This directory plugs loom-code into the [Terminal-Bench](https://www.tbench.ai)
(Harbor) harness so we can get a real, leaderboard-comparable score —
and, using NVIDIA's free Nemotron tier, do it at **zero API cost**.

- `loom_code_agent.py` — the `AbstractInstalledAgent` adapter.
- `setup.sh` — installs loom-code inside each task container.

## How it works

Terminal-Bench runs every task in a Docker container: an instruction +
hidden tests + a time limit. The harness installs loom-code in the
container (`setup.sh`), runs `loom-code --yes --model <model> "<task>"`,
then runs the hidden tests to score pass/fail. Because loom-code runs as
the real shipped CLI, the number reflects the actual tool.

## Prerequisites (one-time)

1. **Docker** — installed and *running*. Every task is a container.
   - macOS: install Docker Desktop, launch it, confirm `docker ps` works.
2. **The `tb` CLI**:
   ```bash
   uv tool install terminal-bench      # or: pipx install terminal-bench
   ```
   (Install `uv` first if needed: `brew install uv`.)
3. **A model key** exported on the host. For the free path:
   ```bash
   export NVIDIA_NIM_API_KEY=nvapi-...   # from build.nvidia.com
   ```
   The adapter forwards whichever provider key is set into the container.

## Smoke test first (free, ~a few tasks)

Prove the plumbing end-to-end on ONE task before spending time on a full
run. From the repo root:

```bash
tb run \
  --agent-import-path benchmarks.terminal_bench.loom_code_agent:LoomCodeAgent \
  --model litellm/nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2 \
  --dataset-name terminal-bench-core --dataset-version 0.1.1 \
  --task-id hello-world \
  --n-concurrent 1
```

Watch for: `setup.sh` installs loom-code cleanly, loom-code runs
headless (never blocks on an approval prompt — `--yes` handles that),
and the harness records a result. If that works, the adapter is good.

## Fuller runs

A small slice for a first real signal + failure transcripts:

```bash
tb run \
  --agent-import-path benchmarks.terminal_bench.loom_code_agent:LoomCodeAgent \
  --model litellm/nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2 \
  --dataset-name terminal-bench-core --dataset-version 0.1.1 \
  --n-concurrent 4
```

Drop to a stronger model (`--model claude-opus-...` / `gpt-...`) once
Nemotron's ceiling is the bottleneck rather than the harness.

## Known caveats

- **Per-container install is heavy.** loom-code pulls loomflow + graphifyy
  (tree-sitter grammars) + jedi — tens of seconds and hundreds of MB per
  container. Fine for a smoke test; for a full run, pre-bake a base image
  with loom-code already installed and skip `setup.sh`'s pip step.
- **`setup.sh` installs from GitHub** by default (loom-code isn't on PyPI
  yet). Override the source with `LOOM_CODE_SPEC`, e.g. a branch you're
  testing: `LOOM_CODE_SPEC="git+https://github.com/Anurich/loomflow-cli@my-branch"`.
- **Nemotron is a small model.** Expect a modest score — the point of the
  free run is to validate the harness + surface loom-code's failure modes
  cheaply, not to top the leaderboard. Re-run on a frontier model for a
  headline number.
