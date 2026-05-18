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
from loomflow.architecture.router import RouterRoute
from loomflow.mcp import MCPRegistry
from loomflow.team import Team
from loomflow.workspace import LocalDiskWorkspace

from .project import Project
from .prompts import build_coordinator_instructions
from .workers import build_simple_coder, build_workers

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
    mcp_registry: MCPRegistry | None = None,
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
    features), all defaulted to ON:

    * ``snip_window=8`` — the coordinator keeps the last 8
      user-anchored turn groups in conversation history; older
      ones drop before each model call. Pure list-slicing; no
      LLM call. Catches very long REPL sessions where the
      coordinator's delegate-result-integrate cycle accumulates
      a lot of intermediate turns. (Lowered from 12 → 8 in
      2026-05 after profiling showed the older turn groups
      rarely informed later decisions once the plan had moved
      on; bigger value just paid input-token cost without
      improving decisions.)
    * ``auto_compact=True`` — when conversation tokens exceed
      80% of the model's context window mid-run, the older half
      is collapsed into a single summary system message via an
      LLM call. The summariser is the same model (no extra
      provider setup); on Anthropic/OpenAI Opus-class models the
      threshold is ~160k tokens.
    * ``tool_result_summarizer=<model>`` — opt-IN (default
      ``None``). When set to a model name, large tool results
      (>500 chars by default) get summarised via the named model
      before entering conversation history. Trades 1 extra LLM
      call per oversized result for ~10x reduction in
      subsequent-turn input tokens. Default is ``None`` because
      hardcoding a real model (e.g. ``"gpt-4.1-mini"``) at
      construction time would force every loom-code user to have
      an ``OPENAI_API_KEY`` even for tests with
      ``model="echo"``. STRONGLY recommended to enable in
      production REPL sessions — pass
      ``tool_result_summarizer="gpt-4.1-mini"`` (or your
      preferred cheap-and-fast model) to ``build_agent`` or use
      the ``/set_tool_summarizer`` REPL command. With persistent
      tool transcripts on workers (see ``persist_tool_transcripts``
      in workers.py), workers don't re-read files, but the
      coordinator still sees worker delegation outputs verbatim
      — summarising those is the biggest single win for the
      coordinator's context budget.

    Persistent tool-transcripts (loomflow 0.10.15+) live on the
    workers, not the coordinator: ``Team.supervisor`` doesn't
    forward the kwarg yet, and the coordinator's tool calls
    (delegate / forward_message / send_message) are lightweight
    anyway. See ``loom_code.workers._build_*`` for the per-worker
    wiring — it's what stops the coder from re-reading the same
    file on every delegation.
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

    # ``max_stop_hook_iterations`` bounds the framework Ralph loop.
    # ``living_plan=True`` auto-registers a StopHook that returns a
    # ``StopHookResult(inject_message=...)`` whenever any plan step
    # is still ``doing``/``todo`` after the architecture exits; each
    # firing re-runs the full ``architecture.run()`` with the
    # injected prompt. ``/set_continue_cap`` exposes the knob.
    # Forwarded directly through ``Team.supervisor`` since
    # loomflow 0.10.10.
    #
    # Default = 0 (2026-05, after observing the runaway pattern even
    # at 2). 0 means: hooks NEVER re-prompt. When the model emits a
    # final answer, the run is OVER — same shape as Claude Code's
    # loop. The plan becomes a tracking aid (TodoWrite-style)
    # instead of a contract that forces continuation.
    #
    # Why 0 and not 1 or 2: every iteration is a FULL extra
    # ``architecture.run()`` — could be 5-15 turns of new tool
    # calls. Once the model says done, any "are you sure?" prompt
    # is the framework second-guessing the model, and in practice
    # the model invents redundant work to "drain the plan." The
    # genuine "model forgot to mark a step done but is actually
    # finished" case is much rarer than the "model says done and
    # IS done, but auto-StopHook says otherwise" case — the user
    # types "continue" if they want more.
    #
    # History: 15 (framework default, original) → 2 (2026-05 after
    # the first runaway diagnosis) → 0 (after observing the
    # malformed plan_write reset → loop pattern even at 2).
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

    # All token-optimisation knobs forward cleanly through
    # ``Team.supervisor`` (tool_result_summarizer since 0.10.13;
    # snip_window + auto_compact_* since 0.10.14;
    # persist_tool_transcripts + tool_transcript_max_bytes since
    # 0.10.16's Team-kwarg-forwarding sweep). No more post-
    # construction monkey-patching.
    # MCP tool surface for the coordinator. When ``mcp_registry``
    # is provided (typically a graphify stdio server — see the
    # README quickstart), the registry's tools surface in the
    # coordinator's tool list alongside ``delegate`` /
    # ``forward_message`` / ``send_message``. The supervisor
    # architecture wraps whatever ``tools=`` we pass with an
    # ``ExtendedToolHost`` that adds its own delegate-family
    # tools — passing MCPRegistry as the base means both layers
    # coexist (MCP tools + delegate-family) without surgery.
    #
    # Caller owns the registry lifecycle (``async with
    # mcp_registry:``) — we just wire the reference. v1 limitation:
    # MCP tools only reach the COMPLEX route (supervisor); the
    # SIMPLE coder doesn't have MCP yet (would need
    # ExtendedToolHost composition of local kernel + MCP, follow-
    # up phase).
    supervisor = Team.supervisor(
        workers=workers,
        instructions=build_coordinator_instructions(project),
        model=model,
        memory=memory_url,
        auto_consolidate=True,
        workspace=workspace,
        living_plan=True,
        tools=mcp_registry,
        max_turns=max_turns,
        max_stop_hook_iterations=max_stop_hook_iterations,
        prompt_caching=True,
        tool_result_summarizer=tool_result_summarizer,
        snip_window=snip_window,
        auto_compact_at_tokens=auto_compact_at_tokens,
        # Persistent tool transcripts on the coordinator
        # (loomflow 0.10.16+ — Team.* now forwards the kwarg).
        # Workers already opt in via ``persist_tool_transcripts=True``
        # in workers.py; turning it on for the coordinator too
        # means the supervisor remembers prior delegations
        # (delegate / forward_message / send_message tool calls
        # and their results) across ``Agent.run()`` invocations
        # within a REPL session. Coordinator tool calls are
        # lightweight so the storage cost is bounded; the payoff
        # is the supervisor knowing "I delegated X to coder and
        # got Y back" when the user says "now fix Z" two prompts
        # later instead of restarting cold.
        persist_tool_transcripts=True,
    )

    # SIMPLE mode — a single coder Agent that talks directly to
    # the user. No team apparatus, no plan, no delegation, no
    # notebook. Same memory backend as the supervisor so the
    # router can pick either path on consecutive turns without
    # losing conversation continuity.
    simple_coder = build_simple_coder(
        project,
        model=model,
        approval_handler=approval_handler,
        memory_url=memory_url,
        web_backend=web_backend,
    )

    # ROUTER — the actual entrypoint loom-code returns. One
    # LLM classification per user message picks SIMPLE or
    # COMPLEX, then dispatches the user's prompt to the chosen
    # agent which runs to completion. Uses the SAME model the
    # user picked (no hardcoded classifier model) — typically
    # cheaper than the worker calls it routes to, but consistent
    # with what the user provisioned.
    #
    # Why the descriptions matter: they're injected verbatim
    # into the classifier's prompt. The discriminator is "does
    # this need parallel investigation / multi-step planning /
    # cross-file work" vs "single focused change / question
    # / lookup." Keep them tight so the classifier picks
    # reliably; vague descriptions = mis-routes.
    coordinator = Team.router(
        routes=[
            RouterRoute(
                name="simple",
                agent=simple_coder,
                description=(
                    "Use SIMPLE for one-shot tasks the user can describe "
                    "in a sentence and a single coding agent can handle "
                    "in a handful of tool calls. Examples: 'create a "
                    "hello-world file', 'fix this typo on line 12', 'add "
                    "a docstring to the foo function', 'rename bar to "
                    "baz in this file', 'what does this function do?', "
                    "'read this URL and tell me about X'. Single file, "
                    "single concern, no investigation needed across the "
                    "codebase, no multi-step plan required."
                ),
            ),
            RouterRoute(
                name="complex",
                agent=supervisor,
                description=(
                    "Use COMPLEX for tasks that benefit from a team — "
                    "parallel investigation, multi-step planning, cross-"
                    "file refactors, code review, dedicated test pass, "
                    "architecture changes. Examples: 'add OAuth to this "
                    "app', 'refactor the data layer to use Postgres', "
                    "'review my PR and write a test plan', 'investigate "
                    "why X is slow then fix it'. Multiple files, "
                    "multiple concerns, real planning + verification "
                    "loops needed, parallel research pays off."
                ),
            ),
        ],
        model=model,
        memory=memory_url,
        workspace=workspace,
        prompt_caching=True,
        max_turns=max_turns,
        snip_window=snip_window,
        auto_compact_at_tokens=auto_compact_at_tokens,
        tool_result_summarizer=tool_result_summarizer,
        persist_tool_transcripts=True,
        # Single classification call per user message; the routed
        # agent then runs to completion. The router's StopHook
        # behavior is moot since classification doesn't have a
        # plan — but pass it anyway for symmetry with supervisor.
        max_stop_hook_iterations=max_stop_hook_iterations,
    )
    return coordinator, workspace


def loom_dir_for(root: Path) -> Path:
    """Return (and create) the ``.loom`` dir for a project root."""
    d = root / LOOM_DIR
    d.mkdir(exist_ok=True)
    return d
