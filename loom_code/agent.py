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
    snip_window: int = 12,
    auto_compact: bool = True,
    tool_result_summarizer: str | None = None,
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

    Prompt caching is on for BOTH the coordinator (via
    loomflow 0.10.12's ``Team.supervisor(prompt_caching=)`` kwarg)
    and every worker (each built as a plain ``Agent`` in
    :mod:`loom_code.workers` with ``prompt_caching=True``). On
    Anthropic models that pins a ``cache_control`` marker on the
    last system block + last tool def, so the system prompt + tool
    schemas hit the cache on every turn after the first; on OpenAI
    models caching is automatic, the flag just enables cache-aware
    token accounting.

    Persistent subagents (loomflow 0.10.10+) is default-on. Each
    worker gets a stable ``worker_<role>_<ULID>`` id + session_id
    so the same researcher/coder/reviewer carries conversation
    memory across delegations and across multiple ``run()`` calls
    inside one REPL session. The ``send_message(to=<worker_id>,
    content=...)`` tool is auto-wired into the coordinator's tool
    surface, letting the model follow up with a specific worker
    instead of always re-delegating from scratch.

    Three context-budget controls (loomflow 0.10.13 framework
    features), all opt-in but defaulted to safe values:

    * ``snip_window=12`` — the coordinator keeps the last 12
      user-anchored turn groups in conversation history; older
      ones drop before each model call. Pure list-slicing; no
      LLM call. Catches very long REPL sessions where the
      coordinator's delegate-result-integrate cycle accumulates
      a lot of intermediate turns.
    * ``auto_compact=True`` — when conversation tokens exceed
      80% of the model's context window mid-run, the older half
      is collapsed into a single summary system message via an
      LLM call. The summariser is the same model (no extra
      provider setup); on Anthropic/OpenAI Opus-class models the
      threshold is ~160k tokens.
    * ``tool_result_summarizer=<model>`` — opt-IN; when provided,
      large tool results (>500 chars by default) get summarised
      via the named model before entering conversation history.
      Trades 1 extra LLM call per oversized result for ~10x
      reduction in subsequent-turn input tokens. The win-loss
      depends on how often tool results live more than one
      additional turn; loom-code defaults this OFF and lets
      operators opt in per-model based on observed usage.
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

    # ``max_stop_hook_iterations`` bounds the framework Ralph loop —
    # ``living_plan=True`` auto-registers a StopHook that re-prompts
    # the coordinator when any plan step is still ``doing``/``todo``
    # after a final answer. ``/set_continue_cap`` exposes the knob.
    # Forwarded directly through ``Team.supervisor`` since
    # loomflow 0.10.10.
    #
    # Auto-compact threshold — 80% of the model's context window.
    # The framework helper ``context_window_for`` does substring
    # lookup against known model families; unknown models fall
    # back to the conservative 8k cap (which would fire compaction
    # aggressively — user can override by passing
    # ``auto_compact=False`` at construction).
    auto_compact_at_tokens: int | None = None
    if auto_compact:
        from loomflow.agent.auto_compact import context_window_for
        window = context_window_for(model)
        auto_compact_at_tokens = int(window * 0.8)

    # ``tool_result_summarizer`` forwards cleanly through
    # ``Team.supervisor`` since loomflow 0.10.13. ``snip_window``
    # and ``auto_compact_at_tokens`` are NOT yet forwarded
    # through the Team.* builders (the same papercut pattern that
    # ``max_stop_hook_iterations`` had before 0.10.10) — we set
    # them post-construction on the returned coordinator until a
    # loomflow release adds the Team forwarding.
    coordinator = Team.supervisor(
        workers=workers,
        instructions=build_coordinator_instructions(project),
        model=model,
        memory=memory_url,
        auto_consolidate=True,
        workspace=workspace,
        living_plan=True,
        max_turns=max_turns,
        max_stop_hook_iterations=max_stop_hook_iterations,
        prompt_caching=True,
        tool_result_summarizer=tool_result_summarizer,
    )
    # Post-construction stamping for the two knobs Team.supervisor
    # doesn't accept yet. The attributes ARE Agent's runtime state —
    # snip_window > 0 flips ``fast_snip`` False at the next deps
    # build; auto_compact_at_tokens > 0 + a summariser triggers
    # the Ralph-loop compactor.
    coordinator._snip_window = snip_window
    coordinator._auto_compact_at_tokens = auto_compact_at_tokens
    coordinator._auto_compact_summariser = (
        coordinator._model if auto_compact_at_tokens is not None else None
    )
    return coordinator, workspace


def loom_dir_for(root: Path) -> Path:
    """Return (and create) the ``.loom`` dir for a project root."""
    d = root / LOOM_DIR
    d.mkdir(exist_ok=True)
    return d
