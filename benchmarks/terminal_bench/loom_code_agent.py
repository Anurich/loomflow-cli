"""Terminal-Bench adapter for loom-code.

Plugs loom-code into the Terminal-Bench (Harbor) harness as an
``AbstractInstalledAgent``: the harness installs loom-code *inside* each
task's Docker container and drives it exactly as a real user would —
``loom-code --yes "<task>"`` — so the score reflects the shipped CLI, not
a wrapper.

Run a smoke test (a handful of tasks, local Docker) with a FREE NVIDIA
Nemotron model — no API spend:

    export NVIDIA_NIM_API_KEY=nvapi-...
    tb run \\
      --agent-import-path benchmarks.terminal_bench.loom_code_agent:LoomCodeAgent \\
      --model litellm/nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2 \\
      --dataset-name terminal-bench-core --dataset-version 0.1.1 \\
      --task-id hello-world --n-concurrent 1

Drop ``--task-id`` (and raise ``--n-concurrent``) for the full run once
the smoke test passes. The ``--model`` value is forwarded to loom-code's
``--model`` flag verbatim, so any string loom-code routes works here
(NVIDIA/Groq/OpenAI/Anthropic/Ollama).

Interface verified against harbor-framework/terminal-bench @ main:
``AbstractInstalledAgent`` requires ``_env``, ``_install_agent_script_path``,
and ``_run_agent_commands(instruction) -> list[TerminalCommand]``.
"""

from __future__ import annotations

import glob
import os
import re
import shlex
from pathlib import Path

from terminal_bench.agents.installed_agents.abstract_installed_agent import (
    AbstractInstalledAgent,
)
from terminal_bench.terminal.models import TerminalCommand

# The repo is PRIVATE, so cloning it inside a container would prompt for
# GitHub credentials and hang a headless run. Instead we build a wheel
# locally (see build_wheel.sh / the README) and copy it in — no network,
# no auth. This is the path to that wheel; ``dist/`` next to this file.
_DIST_DIR = Path(__file__).parent / "dist"
_CONTAINER_WHEEL_DIR = "/installed-agent/loom-wheels"


def _local_wheel() -> Path | None:
    """Newest loom_code wheel in ``dist/`` (built from the working tree),
    or None if none exists — setup.sh then errors with a clear hint."""
    matches = sorted(
        glob.glob(str(_DIST_DIR / "loom_code-*.whl")),
        key=os.path.getmtime,
    )
    return Path(matches[-1]) if matches else None


# loom-code prints ``LOOM_USAGE turns=.. tokens_in=.. cached_in=..
# tokens_out=.. cost_usd=..`` at the end of a one-shot run. Parse the
# LAST occurrence (the final summary) from the captured terminal.
_USAGE_RE = re.compile(
    r"LOOM_USAGE\s+turns=(\d+)\s+tokens_in=(\d+)\s+cached_in=(\d+)"
    r"\s+tokens_out=(\d+)\s+cost_usd=([0-9.]+)"
)


def _parse_loom_usage(pane_text: str) -> dict[str, int] | None:
    """Extract the final LOOM_USAGE marker from terminal output, or None
    if absent. Ints for tokens; cost is left to downstream tooling."""
    matches = list(_USAGE_RE.finditer(pane_text))
    if not matches:
        return None
    turns, tin, cached, tout, _cost = matches[-1].groups()
    return {
        "turns": int(turns),
        "tokens_in": int(tin),
        "cached_in": int(cached),
        "tokens_out": int(tout),
    }


class LoomCodeAgent(AbstractInstalledAgent):
    """Installs + drives loom-code inside each task container."""

    @staticmethod
    def name() -> str:
        return "loom-code"

    def __init__(self, model_name: str | None = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Harbor passes --model through as ``model_name``. Default to the
        # free NVIDIA Nemotron so a bare run costs nothing.
        self._model_name = (
            model_name
            or "litellm/nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2"
        )

    @property
    def _env(self) -> dict[str, str]:
        """Keys forwarded into the container. Only whichever provider
        key is present on the host is passed — loom-code prompts
        interactively otherwise, which a headless run can't answer, so a
        missing key surfaces as a clean failure rather than a hang.
        """
        env: dict[str, str] = {}
        for var in (
            "NVIDIA_NIM_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GROQ_API_KEY",
            "TOGETHERAI_API_KEY",
            "GEMINI_API_KEY",
            "DEEPSEEK_API_KEY",
            "MISTRAL_API_KEY",
        ):
            if os.environ.get(var):
                env[var] = os.environ[var]
        # Tell setup.sh where the copied-in wheel dir lives.
        env["LOOM_WHEEL_DIR"] = _CONTAINER_WHEEL_DIR
        return env

    @property
    def _install_agent_script_path(self) -> Path:
        """Script Harbor copies in + runs to install loom-code."""
        return Path(__file__).parent / "setup.sh"

    def perform_task(self, instruction, session, logging_dir=None):
        """Copy the locally-built wheel into the container BEFORE the
        base class runs setup.sh, so the install is offline (the repo is
        private — cloning it would hang on a credential prompt).

        Raises early with an actionable message if no wheel exists, so a
        forgotten build step fails loudly instead of as a mysterious
        task failure.
        """
        wheel = _local_wheel()
        if wheel is None:
            raise FileNotFoundError(
                "No loom-code wheel in "
                f"{_DIST_DIR}. Build one first:\n"
                "  ./benchmarks/terminal_bench/build_wheel.sh\n"
                "(or: python -m build --wheel --outdir "
                "benchmarks/terminal_bench/dist)"
            )
        session.copy_to_container(
            wheel,
            container_dir=_CONTAINER_WHEEL_DIR,
            container_filename=wheel.name,
        )
        result = super().perform_task(instruction, session, logging_dir)

        # The base class hardcodes tokens to 0 for installed agents. Read
        # loom-code's own machine-parseable ``LOOM_USAGE`` line from the
        # terminal instead, so runs report real token counts (and a paid
        # run's cost can be estimated). Best-effort: if the marker isn't
        # found — task errored before the summary, output scrolled off —
        # leave the base result untouched rather than fail the trial.
        try:
            pane = session.capture_pane()
            usage = _parse_loom_usage(pane)
            if usage is not None:
                result.total_input_tokens = usage["tokens_in"] + usage[
                    "cached_in"
                ]
                result.total_output_tokens = usage["tokens_out"]
        except Exception:
            pass
        return result

    def _run_agent_commands(self, instruction: str) -> list[TerminalCommand]:
        """One blocking command: loom-code one-shot, auto-approving.

        ``--yes`` swaps the interactive approval gate for auto-approve
        (no TTY in the harness). ``--model`` forwards the chosen model.
        The instruction is shell-quoted so task prompts with quotes /
        newlines survive intact.
        """
        task = shlex.quote(instruction)
        model = shlex.quote(self._model_name)
        return [
            TerminalCommand(
                command=f"loom-code --yes --model {model} {task}",
                min_timeout_sec=0.0,
                max_timeout_sec=float("inf"),
                block=True,
                append_enter=True,
            ),
        ]
