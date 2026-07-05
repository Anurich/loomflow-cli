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
import functools
import os
import subprocess
import sys

# Silence huggingface_hub's unauthenticated-request notice (pulled in
# transitively via graphifyy). It writes to raw stdout, which would
# otherwise leak into the full-screen TUI's layout. Set before any
# import that might load huggingface_hub.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import anyio

from .agent import DEFAULT_MODEL, build_agent
from .approval import ApprovalGate
from .credentials import ensure_key_for_model, load_credentials
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


async def _run_once(
    prompt: str,
    model: str,
    yes: bool,
    *,
    sandbox: bool = False,
    sandbox_allow_network: bool = False,
    output_format: str = "text",
) -> int:
    """Detect project, build agent, stream one run. Returns an
    exit code.

    ``yes`` swaps the interactive approval gate for the
    auto-approve handler — for unattended runs (CI, scripted use)
    where there's no TTY to answer prompts.

    ``output_format="json"`` silences the Rich UI and prints ONE
    machine-readable envelope on stdout when the run ends — for
    CI / scripting / benchmark harnesses. Usually paired with
    ``--yes`` (there's no visible approval prompt in quiet mode).

    Consumed via :func:`anyio.run` (NOT ``asyncio.run``) — the
    streaming generator from ``Agent.stream`` opens internal
    ``anyio`` task groups, and exiting those cleanly requires the
    anyio event loop. loomflow's rule: anyio everywhere.
    """
    json_mode = output_format == "json"
    if json_mode:
        # Silence every Rich print (banner, stream, summary) — the
        # envelope at the end is the only stdout. ``console`` is the
        # shared module singleton, so this covers render.py too.
        console.quiet = True

    project = detect_project()
    banner(
        model,
        str(project.root),
        project.is_git,
        sandbox=sandbox,
        sandbox_allow_network=sandbox_allow_network,
    )
    if project.context_file:
        console.print(
            f"  [dim]loaded context: "
            f"{project.context_file.name}[/dim]\n"
        )

    # --yes routes through the gate in YOLO mode rather than the raw
    # auto_approve — so settings.toml ``deny`` rules (e.g.
    # deny "edit(*.env)") and the irreversible-danger gate STILL fire
    # in unattended runs, instead of a blanket allow-everything.
    from pathlib import Path as _Path

    from .permissions import Mode, load_rules

    rules = load_rules(
        [_Path.home() / ".loom-code", project.root / ".loom"]
    )
    gate = ApprovalGate(
        rules=rules,
        mode=Mode.YOLO if yes else Mode.DEFAULT,
        project_root=project.root,
    )
    handler = gate.handler
    agent, workspace = build_agent(
        project,
        model=model,
        approval_handler=handler,
        sandbox=sandbox,
        sandbox_allow_network=sandbox_allow_network,
    )
    renderer = StreamRenderer(sandbox=sandbox)
    # "thinking..." spinner until the first user-visible event —
    # same pattern as the REPL so one-shot mode has the same
    # responsiveness feedback. ``started`` is internal framing;
    # the spinner drops on the first non-started event.
    status = console.status(
        "[dim]thinking...[/dim]", spinner="dots"
    )
    status.start()
    spinner_dropped = False

    def _drop_spinner() -> None:
        nonlocal spinner_dropped
        if not spinner_dropped:
            status.stop()
            spinner_dropped = True

    try:
        async for event in agent.stream(prompt, user_id="loom-code"):
            if str(event.kind) != "started":
                _drop_spinner()
            renderer.handle(event)
    except KeyboardInterrupt:
        _drop_spinner()
        console.print("\n[yellow]interrupted[/yellow]")
        if json_mode:
            _emit_json_envelope(
                project, renderer, exit_code=130, error="interrupted"
            )
        return 130
    except Exception as exc:  # noqa: BLE001 — top-level guard
        _drop_spinner()
        # anyio task groups surface as an opaque ExceptionGroup —
        # flatten to the real cause(s) and add an actionable hint,
        # same presentation as the REPL's turn errors.
        from .repl import _flatten_exception_group, friendly_error_hint

        inners = (
            _flatten_exception_group(exc)
            if isinstance(exc, BaseExceptionGroup)
            else [exc]
        )
        for inner in inners:
            console.print(
                f"\n[bold red]fatal: "
                f"{type(inner).__name__}: {inner}[/bold red]"
            )
            hint = friendly_error_hint(inner)
            if hint:
                console.print(f"  [yellow]→ {hint}[/yellow]")
        if json_mode:
            first = inners[0] if inners else exc
            _emit_json_envelope(
                project,
                renderer,
                exit_code=1,
                error=f"{type(first).__name__}: {first}",
            )
        return 1
    finally:
        _drop_spinner()

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
    if json_mode:
        _emit_json_envelope(project, renderer, exit_code=0)
    return 0


def _emit_json_envelope(
    project: Project,
    renderer: StreamRenderer,
    *,
    exit_code: int,
    error: str | None = None,
) -> None:
    """The single machine-readable line ``--output-format json``
    promises: printed with plain ``print`` (the Rich console is quiet
    in json mode), one JSON object, always the last stdout of the run.
    Append-only contract — add fields, never rename them, so CI
    parsers don't break."""
    import json

    result = renderer.last_result or {}
    envelope = {
        "is_error": exit_code != 0,
        "exit_code": exit_code,
        "error": error,
        "output": str(result.get("output") or ""),
        "turns": int(result.get("turns", 0)),
        "tokens_in": int(result.get("tokens_in", 0)),
        "cached_tokens_in": int(result.get("cached_tokens_in", 0)),
        "tokens_out": int(result.get("tokens_out", 0)),
        "cost_usd": float(result.get("cost_usd", 0.0)),
        "files_changed": (
            _git_changes(project) if project.is_git else []
        ),
        "verify_commands": [
            c
            for c in renderer.bash_commands
            if _is_verify_command(c)
        ],
        "notes_written": [
            {"kind": k, "title": t}
            for k, t in renderer.notes_written
        ],
    }
    print(json.dumps(envelope))


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
    """Append a structured summary after the agent finishes —
    cost line (one-shot owns this since there's no REPL pre-prompt
    status to show it), then files changed / verified / notes."""
    result = renderer.last_result or {}
    turns = result.get("turns", 0)
    cost = float(result.get("cost_usd", 0.0))
    tin = int(result.get("tokens_in", 0))
    cached = int(result.get("cached_tokens_in", 0))
    tout = int(result.get("tokens_out", 0))
    if turns or cost or tin or tout:
        from rich.text import Text  # local — keeps cli.py top tidy
        console.print(
            Text.assemble(
                ("  ", ""),
                (f"{turns} turns", "dim"),
                ("  ·  ", "dim"),
                (f"{tin:,}+{cached:,} in / {tout:,} out", "dim"),
                ("  ·  ", "dim"),
                (f"${cost:.4f}", "dim green"),
            )
        )
        # Stable, machine-parseable usage marker on its own line. The
        # pretty line above is for humans (Rich markup, commas); this one
        # is for tooling — a benchmark harness / script greps LOOM_USAGE
        # and reads plain ints, no formatting to unpick. Kept minimal and
        # append-only so parsers don't break when fields are added.
        console.print(
            f"LOOM_USAGE turns={turns} "
            f"tokens_in={tin} cached_in={cached} "
            f"tokens_out={tout} cost_usd={cost:.6f}"
        )

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
        default=None,
        help=(
            f"Model to use (default: last-used, else {DEFAULT_MODEL}). "
            "Accepts any "
            "string loomflow's resolver routes: Anthropic names "
            "(claude-sonnet-4-6, claude-opus-4-7, ...), OpenAI "
            "names (gpt-4.1-mini, gpt-4.1, o4-mini, ...), local "
            "Ollama (ollama/llama3, ollama/qwen2.5-coder), or "
            "force LiteLLM with litellm/<anything>."
        ),
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
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help=(
            "Run the coder's bash inside an OS sandbox (macOS Seatbelt "
            "/ Linux bwrap): commands may read anywhere but only WRITE "
            "under the project root, with no network. edit/write keep "
            "the approval gate. Off by default."
        ),
    )
    parser.add_argument(
        "--sandbox-allow-network",
        action="store_true",
        help="With --sandbox, permit network access from bash commands.",
    )
    parser.add_argument(
        "--output-format",
        choices=("text", "json"),
        default="text",
        help=(
            "One-shot output format. 'json' silences the UI and "
            "prints one machine-readable envelope on stdout (output, "
            "tokens, cost, files changed, exit_code) — for CI and "
            "scripting; usually paired with --yes. Ignored in the "
            "interactive REPL."
        ),
    )
    parser.add_argument(
        "--continue",
        dest="continue_",
        action="store_true",
        help=(
            "Start the REPL resumed on this project's most recent "
            "session (same as typing /resume first)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Start the REPL with a picker of recent sessions to "
            "resume (same as typing /resume pick first)."
        ),
    )
    parser.add_argument(
        "--classic",
        action="store_true",
        help=(
            "Use the classic inline input box instead of the "
            "full-screen chat UI (fixed bottom box + scrolling "
            "conversation). Auto-selected on a non-TTY."
        ),
    )
    args = parser.parse_args()

    # 1. Read ~/.loom-code/credentials so any keys saved on a
    #    previous run are available without the user having to
    #    `export` again.
    # 2. If the chosen model still needs a key, prompt for it
    #    inline (hidden input), save it, and continue.
    # Both happen BEFORE any Agent is constructed — loomflow's
    # model adapter would crash at init on a missing key otherwise.
    load_credentials()
    # Resolve the startup model: an explicit --model wins; else the model
    # the user last chose (persisted via /model or /set_model); else the
    # built-in default. This makes the chosen model STICK across launches.
    if args.model is None:
        from .credentials import load_preferred_model

        args.model = load_preferred_model() or DEFAULT_MODEL
    # Expand friendly provider aliases (e.g. ``nvidia/nemotron-...`` →
    # ``litellm/nvidia_nim/nvidia/nemotron-...``) so the rest of the
    # pipeline — key prompt, resolver, persistence — sees the canonical
    # form loomflow understands.
    from .credentials import normalize_model, quiet_litellm_model_warnings

    args.model = normalize_model(args.model)
    # litellm-routed models trigger loomflow "unknown model" warnings
    # (context window + pricing) that loom-code already handles; hush
    # them so the startup output stays clean. Native models are
    # untouched, so a real misconfig there still surfaces.
    quiet_litellm_model_warnings(args.model)
    if not ensure_key_for_model(args.model, console):
        sys.exit(1)
    # Remember it so the next launch starts here too.
    from .credentials import save_preferred_model

    save_preferred_model(args.model)

    if not args.prompt:
        # No task → interactive REPL. --yes is meaningless here
        # (the REPL is interactive by definition); ignore it.
        project = detect_project()
        resume = (
            "pick" if args.resume else "last" if args.continue_ else None
        )
        # Full-screen chat UI by default; --classic (or a non-TTY,
        # where a full-screen app can't run) uses the inline box.
        classic = args.classic or not sys.stdout.isatty()
        exit_code = anyio.run(
            functools.partial(
                run_repl,
                project,
                args.model,
                sandbox=args.sandbox,
                sandbox_allow_network=args.sandbox_allow_network,
                resume=resume,
                classic=classic,
            )
        )
        sys.exit(exit_code)

    # One-shot mode.
    prompt = " ".join(args.prompt)
    exit_code = anyio.run(
        functools.partial(
            _run_once,
            prompt,
            args.model,
            args.yes,
            sandbox=args.sandbox,
            sandbox_allow_network=args.sandbox_allow_network,
            output_format=args.output_format,
        )
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
