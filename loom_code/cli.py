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
import subprocess
import sys

import anyio

from .agent import DEFAULT_MODEL, build_agent
from .approval import ApprovalGate, auto_approve
from .project import Project, detect_project
from .render import StreamRenderer, banner, console
from .repl import run_repl

# Substrings we treat as "this bash call was a verify step." Hit
# anywhere in the lowercased command. Detected, not declared by the
# agent — we sniff the tool stream so the summary is honest about
# what was actually run rather than what the agent claims.
_VERIFY_PATTERNS = (
    "pytest", "py.test", "jest", "vitest", "mocha",
    "cargo test", "go test", "mvn test", "gradle test",
    "rake test", "phpunit", "python -m unittest", "tox",
    "make test", "make check", "npm test", "yarn test",
    "pnpm test", "bundle exec",
)


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

    # End-of-run summary — what changed on disk, what was verified,
    # what the agent captured. Detected from the event stream + git;
    # the agent doesn't have to report any of it itself.
    _print_run_summary(project, renderer)
    return 0


def _is_verify_command(cmd: str) -> bool:
    """True for bash commands that look like a project's test /
    build runner — what the agent's VERIFY step should produce."""
    head = cmd.lstrip().lower()
    return any(p in head for p in _VERIFY_PATTERNS)


def _git_changes(project: Project) -> list[str]:
    """Lines from ``git status --short`` for the user's repo, with
    loom-code's own ``.loom/`` state filtered out (it's runtime
    chatter, not part of what the agent changed for the user)."""
    res = subprocess.run(
        ["git", "-C", str(project.root), "status", "--short"],
        capture_output=True, text=True,
    )
    out: list[str] = []
    for raw in res.stdout.splitlines():
        if not raw.strip():
            continue
        # Status lines are ``XY path`` — split off the path so we
        # can filter on it.
        parts = raw.split(maxsplit=1)
        if len(parts) < 2:
            continue
        path = parts[1].strip().strip('"')
        if path == ".loom" or path == ".loom/" or path.startswith(".loom/"):
            continue
        out.append(raw)
    return out


def _print_run_summary(
    project: Project, renderer: StreamRenderer
) -> None:
    """Append a structured summary after the cost line."""
    if project.is_git:
        changes = _git_changes(project)
        if changes:
            console.print()
            console.print("  [bold]Files changed:[/bold]")
            for line in changes[:20]:  # cap noise on huge changesets
                console.print(f"    [dim]{line}[/dim]")
            if len(changes) > 20:
                console.print(
                    f"    [dim]... (+{len(changes) - 20} more)[/dim]"
                )

    verify = [c for c in renderer.bash_commands if _is_verify_command(c)]
    if verify:
        console.print()
        console.print("  [bold]What was verified:[/bold]")
        for cmd in verify[:5]:
            short = cmd if len(cmd) <= 80 else cmd[:80] + "…"
            console.print(f"    [dim]$[/dim] {short}")

    if renderer.notes_written:
        console.print()
        console.print("  [bold]Notes captured:[/bold]")
        for kind, title in renderer.notes_written[:5]:
            tag = f"[dim]{kind}:[/dim] " if kind else ""
            short = title if len(title) <= 80 else title[:80] + "…"
            console.print(f"    • {tag}{short}")


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
