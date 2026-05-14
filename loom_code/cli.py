"""loom-code CLI entry point.

Two modes:

* ``loom-code "do X"`` — one-shot. Detects the project, builds the
  loomflow Agent, streams one run, exits.
* ``loom-code`` (no args) — interactive REPL. Chat, code, approve,
  repeat — with conversation continuity across turns.

Both go through the same project detection + agent build; the REPL
just loops and adds slash commands.
"""

from __future__ import annotations

import argparse
import sys

import anyio

from .agent import DEFAULT_MODEL, build_agent
from .approval import ApprovalGate, auto_approve
from .project import detect_project
from .render import StreamRenderer, banner, console
from .repl import run_repl


async def _run_once(prompt: str, model: str, yes: bool) -> int:
    """Detect project, build agent, stream one run. Returns an
    exit code.

    ``yes`` swaps the interactive approval gate for the
    auto-approve handler — for unattended runs (CI, scripted use)
    where there's no TTY to answer prompts.

    Consumed via :func:`anyio.run` (NOT ``asyncio.run``) — the
    streaming generator from ``Agent.stream`` opens internal
    ``anyio`` task groups, and exiting those cleanly requires the
    anyio event loop. loomflow's rule: anyio everywhere.
    """
    project = detect_project()
    banner(model, str(project.root), project.is_git)
    if project.context_file:
        console.print(
            f"  [dim]loaded context: "
            f"{project.context_file.name}[/dim]\n"
        )

    handler = auto_approve if yes else ApprovalGate().handler
    agent, workspace = build_agent(
        project, model=model, approval_handler=handler
    )
    renderer = StreamRenderer()
    try:
        async for event in agent.stream(prompt, user_id="loom-code"):
            renderer.handle(event)
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/yellow]")
        return 130
    except Exception as exc:  # noqa: BLE001 — top-level guard
        console.print(f"\n[bold red]fatal: {exc}[/bold red]")
        return 1

    # Self-improvement loop: a clean one-shot completion is treated
    # as success — credit the notes the agent read so future runs
    # rank them higher. (The REPL is more nuanced: it waits for a
    # /good / /bad signal or "moved on" before attributing.)
    result = renderer.last_result
    if result and not result.get("interrupted"):
        slugs = result.get("cited_slugs") or []
        if slugs:
            try:
                n = await workspace.attribute_outcome(
                    success=True, slugs=slugs, user_id="loom-code"
                )
                if n:
                    console.print(
                        f"  [dim]credited {n} note(s) — future "
                        f"runs will rank them higher[/dim]"
                    )
            except Exception:  # noqa: BLE001 — best-effort
                pass
    return 0


def main() -> None:
    """``loom-code`` console-script entry point."""
    parser = argparse.ArgumentParser(
        prog="loom-code",
        description=(
            "loom-code — a loomflow-native terminal coding agent"
        ),
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help=(
            "The task. Omit it to drop into the interactive REPL."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help=(
            "Auto-approve all destructive tool calls — no prompts. "
            "For unattended / scripted runs on a disposable tree. "
            "One-shot mode only."
        ),
    )
    args = parser.parse_args()

    if not args.prompt:
        # No task → interactive REPL. --yes is meaningless here
        # (the REPL is interactive by definition); ignore it.
        project = detect_project()
        exit_code = anyio.run(run_repl, project, args.model)
        sys.exit(exit_code)

    # One-shot mode.
    prompt = " ".join(args.prompt)
    exit_code = anyio.run(_run_once, prompt, args.model, args.yes)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
