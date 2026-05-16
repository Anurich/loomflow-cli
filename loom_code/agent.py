"""Builds the loomflow team that powers loom-code.

This is the one place loom-code wires loomflow primitives
together. Everything here is configuration — no agent-loop logic,
no tool implementations, no memory logic. If this file ever grows
real behaviour, that behaviour belongs in loomflow.

loom-code is a ``Team.supervisor``: a coordinator Agent (the tech
lead) that delegates to a roster of worker Agents — ``coder``,
``explorer``, ``auditor``, ``reviewer`` (see :mod:`loom_code.workers`).
``Team.supervisor`` returns a plain ``Agent``, so the rest of
loom-code (REPL, CLI, renderer) treats it exactly like any agent.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from loomflow import Agent
from loomflow.team import Team
from loomflow.workspace import LocalDiskWorkspace

from .project import Project
from .prompts import build_coordinator_instructions
from .workers import build_workers

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
    max_stop_hook_iterations: int = 15,
) -> tuple[Agent, LocalDiskWorkspace]:
    """Wire the loom-code team for a given project.

    Returns ``(coordinator, workspace)`` — the coordinator is the
    ``Team.supervisor`` Agent; the caller needs the workspace
    handle to drive the self-improvement loop
    (``attribute_outcome`` after a run, ``prune`` for retention).
    The same workspace instance is wired into the team, so
    citations the agents log and outcomes the caller attributes
    hit the same notebook.

    The whole brain in one builder call:

    * **workers** — the delegate roster (:func:`build_workers`):
      ``coder`` (the sole writer — full file-and-shell kernel),
      plus read-only ``explorer`` / ``auditor`` / ``reviewer``.
    * **coordinator** — ``Team.supervisor`` builds the tech-lead
      Agent. It owns the living plan, delegates to workers (the
      read-only ones in parallel, ``coder`` serialised), and
      integrates their results.
    * **living_plan** — on the coordinator; mirrors to the
      workspace so plans persist across sessions.
    * **workspace** — ``<root>/.loom/notebook`` — shared notebook,
      wired onto the coordinator AND every worker (each worker's
      dict key is its author identity in the notebook).
    * **memory** — ``sqlite:<root>/.loom/memory.db`` — episodes +
      auto-extracted facts, persisted across sessions.

    ``approval_handler`` is threaded into the ``coder`` and
    ``reviewer`` workers (they hold the destructive tools); the
    coordinator only delegates, so it needs no permissions policy.

    NOTE: ``Team.supervisor`` does not forward a ``prompt_caching``
    kwarg, so the coordinator runs without prompt caching. The
    workers DO cache (built as plain Agents in
    :mod:`loom_code.workers`). A loomflow gap worth closing later.
    """
    loom_dir = project.root / LOOM_DIR
    loom_dir.mkdir(exist_ok=True)

    workspace = LocalDiskWorkspace(str(loom_dir / "notebook"))
    memory_url = f"sqlite:{loom_dir / 'memory.db'}"

    workers = build_workers(
        project,
        model=model,
        approval_handler=approval_handler,
        web_backend=web_backend,
    )

    coordinator = Team.supervisor(
        workers=workers,
        instructions=build_coordinator_instructions(project),
        model=model,
        memory=memory_url,
        auto_consolidate=True,
        workspace=workspace,
        living_plan=True,
        max_turns=max_turns,
    )
    # Framework Ralph loop — living_plan=True auto-registers a
    # StopHook that re-prompts when any plan step is still
    # `doing`/`todo` after the coordinator emits a final answer.
    # Bound the loop here; /set_continue_cap exposes this knob.
    #
    # Set post-construction because Team.supervisor doesn't
    # forward this kwarg yet (same limitation as prompt_caching).
    # Future loomflow release should accept it directly in
    # Team.supervisor(...). The attribute IS public-shaped on
    # Agent (no leading underscore on the kwarg) — we just don't
    # have a setter through the builder.
    coordinator._max_stop_hook_iterations = max_stop_hook_iterations
    return coordinator, workspace


def loom_dir_for(root: Path) -> Path:
    """Return (and create) the ``.loom`` dir for a project root."""
    d = root / LOOM_DIR
    d.mkdir(exist_ok=True)
    return d
