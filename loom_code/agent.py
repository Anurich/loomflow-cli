"""Builds the loomflow Agent that powers loom-code.

This is the one place loom-code wires loomflow primitives
together. Everything here is configuration — no agent-loop logic,
no tool implementations, no memory logic. If this file ever grows
real behaviour, that behaviour belongs in loomflow.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from loomflow import Agent, StandardPermissions
from loomflow.architecture import ReAct
from loomflow.tools import (
    bash_tool,
    edit_tool,
    find_tool,
    grep_tool,
    ls_tool,
    read_tool,
    write_tool,
)
from loomflow.workspace import LocalDiskWorkspace

from .project import Project
from .prompts import build_system_prompt
from .subagents import build_subagent_tools

# loom-code keeps its per-project state under <root>/.loom/ —
# the workspace notebook and the sqlite memory db both live here.
# Mirrors how Claude Code uses .claude/ and Pi uses .pi/.
LOOM_DIR = ".loom"

# Default model. Opus-class is the right call for a coding agent —
# the research is unambiguous that model strength dominates on
# SWE-style tasks. Overridable via --model / the /model command.
DEFAULT_MODEL = "claude-sonnet-4-6"


def build_agent(
    project: Project,
    *,
    model: str = DEFAULT_MODEL,
    approval_handler: Callable[..., Awaitable[bool]] | None = None,
    max_turns: int = 100,
) -> tuple[Agent, LocalDiskWorkspace]:
    """Wire the loom-code Agent for a given project.

    Returns ``(agent, workspace)`` — the caller needs the
    workspace handle to drive the self-improvement loop
    (``attribute_outcome`` after a run, ``prune`` for retention).
    The same workspace instance is wired into the agent, so
    citations the agent logs and outcomes the caller attributes
    hit the same notebook.

    The whole brain in one constructor call:

    * **system prompt** — coding behaviour + project context
      (from :func:`build_system_prompt`).
    * **architecture** — ``ReAct`` (gather → act → verify loop).
    * **tools** — the 7-tool kernel, all scoped to the project
      root so the agent can't escape it via ``../``, plus two
      specialist sub-agents wrapped as tools: ``explore``
      (read-only investigation) and ``review`` (independent
      verification). See :mod:`loom_code.subagents`.
    * **living_plan** — the task tracker; mirrors to the
      workspace so plans persist across sessions and
      ``recall_past_plans`` works.
    * **workspace** — ``<root>/.loom/notebook`` — the per-project
      notebook + the self-improvement substrate (citation
      tracking, relevance-aware recall).
    * **memory** — ``sqlite:<root>/.loom/memory.db`` — episodes +
      auto-extracted facts, persisted across sessions.
    * **permissions** — ``"default"`` gates destructive tool
      calls through ``approval_handler``.
    * **prompt_caching** — system prompt + tool defs are stable
      across every turn of a session; caching makes that cheap.
    """
    loom_dir = project.root / LOOM_DIR
    loom_dir.mkdir(exist_ok=True)

    root = project.root
    # The 7-tool kernel, every tool workdir-scoped to the project
    # root. bash gets a generous timeout — builds / test suites
    # are slow.
    tools: list[Any] = [
        read_tool(root),
        write_tool(root),
        edit_tool(root),
        grep_tool(root),
        find_tool(root),
        ls_tool(root),
        bash_tool(root, timeout=300.0),
    ]
    # Specialist sub-agents the main loop can call as tools:
    # `explore` (read-only investigation) and `review` (independent
    # verification). They run on the same model; the specialism is
    # in the prompt + tool scoping. See subagents.py for why this
    # shape over a supervisor.
    tools.extend(
        build_subagent_tools(
            project, model=model, approval_handler=approval_handler
        )
    )

    workspace = LocalDiskWorkspace(str(loom_dir / "notebook"))
    memory_url = f"sqlite:{loom_dir / 'memory.db'}"

    # ``Agent.__init__`` wants a ``Permissions`` INSTANCE, not a
    # string — the string-resolver path only runs in
    # ``Agent.from_dict`` / ``from_config``. ``StandardPermissions``
    # in its default mode asks on destructive tool calls (write /
    # edit / bash), which routes through ``approval_handler``.
    agent = Agent(
        build_system_prompt(project),
        model=model,
        architecture=ReAct(),
        tools=tools,
        living_plan=True,
        workspace=workspace,
        memory=memory_url,
        auto_consolidate=True,
        permissions=StandardPermissions(),
        approval_handler=approval_handler,
        prompt_caching=True,
        max_turns=max_turns,
    )
    return agent, workspace


def loom_dir_for(root: Path) -> Path:
    """Return (and create) the ``.loom`` dir for a project root."""
    d = root / LOOM_DIR
    d.mkdir(exist_ok=True)
    return d
