"""Builds the loomflow team that powers loom-code.

This is the one place loom-code wires loomflow primitives
together. Everything here is configuration — no agent-loop logic,
no tool implementations, no memory logic. If this file ever grows
real behaviour, that behaviour belongs in loomflow.

loom-code is a single ``Team.supervisor`` whose coordinator holds
the full coding kernel AND a ``delegate`` tool. It decides inline —
as a tool choice, after it has begun reading — whether to do
focused work itself or to delegate multi-file / parallel work to a
roster of worker Agents (``coder``, ``explorer``, ``auditor``,
``reviewer`` — see :mod:`loom_code.workers`). ``Team.supervisor``
returns a plain ``Agent``, so the rest of loom-code (REPL, CLI,
renderer) treats it exactly like any agent.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from importlib.resources import files as _pkg_files
from pathlib import Path

from loomflow import Agent, StandardPermissions
from loomflow.team import Team
from loomflow.tools import bash_tool, find_tool, ls_tool, read_tool, write_tool
from loomflow.workspace import LocalDiskWorkspace

from .edit_tool import multi_edit_tool
from .edit_tool import verifying_edit_tool as edit_tool
from .extensions import Extensions, safe_role_name
from .grep_tool import enhanced_grep_tool as grep_tool
from .hooks import attach_tool_hooks
from .project import Project
from .prompts import build_unified_coordinator_instructions
from .trust import discover_trusted
from .web_fetch import web_fetch_tool
from .workers import (
    BUILTIN_WORKER_NAMES,
    build_custom_worker,
    build_workers,
)


# Bundled skills shipped with loom-code. Each entry is a directory
# under ``loom_code/skills/`` with a ``SKILL.md`` + optional
# ``tools.py``. The framework's SkillRegistry discovers them on
# Agent construction; the agent sees a 50-token (name + description)
# entry for each, and calls ``load_skill(name)`` to materialise the
# body + tools when relevant. Cheap baseline — no LLM cost unless
# the agent actually loads a skill.
def _bundled_skill_paths() -> list[Path]:
    """Return absolute Paths to every shipped skill directory.
    Uses ``importlib.resources`` so the lookup works whether
    loom-code is installed editable, as a wheel, or zipped."""
    root = _pkg_files("loom_code.skills")
    out: list[Path] = []
    for entry in root.iterdir():  # type: ignore[attr-defined]
        if entry.is_dir() and (entry / "SKILL.md").is_file():
            out.append(Path(str(entry)))
    return out

# loom-code keeps its per-project state under <root>/.loom/ —
# the workspace notebook and the sqlite memory db both live here.
# Mirrors how Claude Code uses .claude/ and Pi uses .pi/.
LOOM_DIR = ".loom"

# Default model. Overridable via --model / the /model command.
DEFAULT_MODEL = "gpt-4.1-mini"


def build_agent(
    project: Project,
    *,
    model: str = DEFAULT_MODEL,
    approval_handler: Callable[..., Awaitable[bool]] | None = None,
    max_turns: int = 100,
    web_backend: str | None = None,
    max_stop_hook_iterations: int = 0,
    snip_window: int = 8,
    auto_compact: bool = True,
    tool_result_summarizer: str | None = None,
    extensions: Extensions | None = None,
    effort: str | None = None,
) -> tuple[Agent, LocalDiskWorkspace]:
    """Wire the loom-code agent for a given project.

    Returns ``(coordinator, workspace)`` — the coordinator is a
    single ``Team.supervisor`` Agent; the caller needs the workspace
    handle to drive the self-improvement loop
    (``attribute_outcome`` after a run, ``prune`` for retention).

    The coordinator holds the FULL coding kernel itself
    (read/write/edit/multi_edit/grep/find/ls/bash/web) *and* a
    ``delegate`` tool. The model decides inline — as a tool choice,
    after it has begun reading — whether to do focused / single-file
    work itself or to ``delegate`` multi-file / parallel work to the
    worker roster. This mirrors how Claude Code / Cursor / Copilot
    dispatch (one adaptive loop + delegation) and rides loomflow's
    ``delegate`` → ``SubagentInvocation`` path, which streams worker
    events to the parent and rolls up their token cost.

    The whole brain in one builder call:

    * **workers** — the delegate roster (:func:`build_workers`):
      ``coder`` (full file-and-shell kernel), plus read-only
      ``explorer`` / ``auditor`` / ``reviewer``. Custom
      ``.loom/agents/*.md`` join as additional delegate workers.
    * **coordinator** — ``Team.supervisor`` with the coding kernel
      on ``tools=``; owns the living plan; delegates when work is
      genuinely parallel / multi-file, otherwise does it itself.
    * **living_plan** — on the coordinator; mirrors to the
      workspace so plans persist across sessions.
    * **workspace** — ``<root>/.loom/notebook`` — shared notebook,
      wired onto the coordinator AND every worker.
    * **memory** — ``sqlite:<root>/.loom/memory.db`` — episodes +
      auto-extracted facts, persisted across sessions.

    Because the coordinator now executes destructive tools itself,
    it carries the permission gate + ``approval_handler`` (the
    workers do too); tool hooks attach to the coordinator AND every
    worker. Prompt caching, persistent tool transcripts, snip
    window and auto-compaction are all on, threaded through
    ``Team.supervisor``'s forwarded Agent kwargs.
    """
    loom_dir = project.root / LOOM_DIR
    loom_dir.mkdir(exist_ok=True)

    workspace = LocalDiskWorkspace(str(loom_dir / "notebook"))
    memory_url = f"sqlite:{loom_dir / 'memory.db'}"

    # Bundled skills (graphify today, more later) computed before the
    # workers + coordinator so the SAME list lands on every agent.
    # Without skills on a worker, the coordinator delegating "build
    # the graph" to coder fails: coder's tool host lacks the skill's
    # tools. ``skill_paths`` append AFTER bundled so last-source-wins
    # gives project > user > bundled.
    bundled_skills = _bundled_skill_paths()
    # User + project extensions (the ``.loom`` folder — skills,
    # subagents, hooks). When NOT supplied (desktop sidecar, scripts,
    # tests) we self-discover with a deny-by-default trust gate so an
    # untrusted project's hooks aren't auto-run.
    if extensions is None:
        extensions = discover_trusted(project.root)
    all_skills = bundled_skills + extensions.skill_paths

    # Auto-compact threshold — 80% of the model's context window,
    # computed before the workers so it lands on EVERY worker too
    # (a long delegation otherwise grows past the window and 400s
    # with context_length_exceeded). ``None`` disables compaction.
    auto_compact_at_tokens: int | None = None
    if auto_compact:
        from loomflow.agent.auto_compact import context_window_for
        window = context_window_for(model)
        auto_compact_at_tokens = int(window * 0.8)

    workers = build_workers(
        project,
        model=model,
        approval_handler=approval_handler,
        web_backend=web_backend,
        skills=all_skills,
        auto_compact_at_tokens=auto_compact_at_tokens,
        snip_window=snip_window,
        effort=effort,
    )

    # Custom .loom subagents join as delegate WORKERS. The coordinator
    # reaches them through `delegate`, which keeps streaming + cost
    # rollup (a raw agent-as-tool would lose both). A custom agent
    # whose name collides with a builtin role is skipped — never let a
    # dropped-in spec shadow the known roster, above all ``coder``.
    for spec in extensions.agent_specs:
        # loomflow worker names must be Python identifiers; Claude-
        # Code-style names use hyphens (security-auditor -> ...).
        role = safe_role_name(spec.name)
        if role in BUILTIN_WORKER_NAMES or role in workers:
            continue
        workers[role] = build_custom_worker(
            project,
            spec,
            model=model,
            approval_handler=approval_handler,
            skills=all_skills,
            auto_compact_at_tokens=auto_compact_at_tokens,
            snip_window=snip_window,
            effort=effort,
        )

    # The coordinator's OWN tool kernel — the full writer surface, so
    # it can do single-file work itself instead of always delegating.
    root = project.root
    coordinator_tools: list[object] = [
        read_tool(root),
        write_tool(root),
        edit_tool(root),
        multi_edit_tool(root),
        grep_tool(root),
        find_tool(root),
        ls_tool(root),
        bash_tool(root, timeout=300.0),
        web_fetch_tool(),
    ]
    if web_backend is not None:
        from loomflow.tools import web_tool
        coordinator_tools.append(web_tool(backend=web_backend))

    # ``max_stop_hook_iterations`` bounds the framework Ralph loop.
    # Default 0 (2026-05): hooks NEVER re-prompt — when the model
    # emits a final answer the run is OVER (Claude-Code-shaped). The
    # living plan is a tracking aid, not a contract forcing
    # continuation. ``/set_continue_cap`` exposes the knob.
    coordinator = Team.supervisor(
        workers=workers,
        instructions=build_unified_coordinator_instructions(project),
        tools=coordinator_tools,
        model=model,
        memory=memory_url,
        workspace=workspace,
        living_plan=True,
        skills=all_skills,
        max_turns=max_turns,
        max_stop_hook_iterations=max_stop_hook_iterations,
        prompt_caching=True,
        tool_result_summarizer=tool_result_summarizer,
        snip_window=snip_window,
        auto_compact_at_tokens=auto_compact_at_tokens,
        effort=effort,
        persist_tool_transcripts=True,
        # The coordinator executes destructive tools itself, so it
        # needs the permission gate + approval bridge.
        permissions=StandardPermissions(),
        approval_handler=approval_handler,
    )

    # Tool hooks attach to every agent that executes tools against the
    # codebase — the coordinator (it does work itself) plus all
    # workers. No-op (fast-hooks path intact) when none are declared.
    for tool_agent in (coordinator, *workers.values()):
        attach_tool_hooks(
            tool_agent, extensions.hook_specs, cwd=project.root
        )

    return coordinator, workspace


def loom_dir_for(root: Path) -> Path:
    """Return (and create) the ``.loom`` dir for a project root."""
    d = root / LOOM_DIR
    d.mkdir(exist_ok=True)
    return d
