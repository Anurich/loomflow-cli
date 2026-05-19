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
from importlib.resources import files as _pkg_files
from pathlib import Path
from typing import Any

from loomflow import Agent
from loomflow.architecture.router import RouterRoute
from loomflow.team import Team
from loomflow.workspace import LocalDiskWorkspace

from .project import Project
from .prompts import build_coordinator_instructions
from .workers import build_simple_coder, build_workers


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
    loom_retrieval: str = "agentic",
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

    # Bundled skills — graphify today, more shipped here later.
    # Computed here (before workers/supervisor are built) so the
    # SAME list lands on the coordinator AND every worker AND
    # the simple-mode coder. Without skills on workers, the
    # coordinator delegating "build the graph" to coder fails:
    # coder's tool host doesn't have ``graphify__build`` and the
    # model falls back to ``bash graphify__build`` (no such
    # executable). Skills on the worker = tool actually callable
    # wherever execution lands.
    bundled_skills = _bundled_skill_paths()

    workers = build_workers(
        project,
        model=model,
        approval_handler=approval_handler,
        web_backend=web_backend,
        skills=bundled_skills,
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
    #
    if loom_retrieval not in ("bm25", "agentic"):
        raise ValueError(
            "loom_retrieval must be 'bm25' or 'agentic', "
            f"got {loom_retrieval!r}"
        )

    # Agentic LOOM.md retrieval: wire the ``read_loom_section``
    # tool into both the coordinator and the simple coder. The
    # TOC injection (which tells the model the slugs) happens via
    # the REPL's per-turn LoomRetriever; the tool fetches a
    # specific section body on demand.
    coordinator_extra_tools: list[Any] = []
    simple_coder_extra_tools: list[Any] = []
    if loom_retrieval == "agentic":
        from .loom_section_tool import read_loom_section_tool
        loom_tool = read_loom_section_tool(project.root)
        coordinator_extra_tools.append(loom_tool)
        simple_coder_extra_tools.append(loom_tool)

    supervisor = Team.supervisor(
        workers=workers,
        instructions=build_coordinator_instructions(project),
        model=model,
        memory=memory_url,
        auto_consolidate=True,
        workspace=workspace,
        living_plan=True,
        tools=coordinator_extra_tools or None,
        skills=bundled_skills,
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
        skills=bundled_skills,
        extra_tools=simple_coder_extra_tools or None,
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
                    "Use SIMPLE for any task that lives in ONE FILE or "
                    "addresses ONE CONCERN — regardless of how many "
                    "steps inside. A sequential checklist on a single "
                    "file (e.g. 'fix all 12 issues in observer.py', "
                    "'add docstrings to every function in foo.py', "
                    "'clean up linter warnings here') is SIMPLE — "
                    "the steps are sequential, not parallel, and a "
                    "single coder working through them is faster + "
                    "more accurate than a team. Also SIMPLE: one-shot "
                    "tasks ('fix this typo', 'rename bar to baz in "
                    "this file', 'what does this function do?', "
                    "'read this URL and tell me about X'), single-"
                    "file questions, and any prompt that doesn't "
                    "benefit from parallel investigation across "
                    "files. When unsure between SIMPLE and COMPLEX, "
                    "PREFER SIMPLE — a competent single coder rarely "
                    "loses to team overhead on file-local work."
                ),
            ),
            RouterRoute(
                name="complex",
                agent=supervisor,
                description=(
                    "Use COMPLEX ONLY when the task genuinely "
                    "benefits from PARALLEL work across MULTIPLE "
                    "files or MULTIPLE concerns. Trigger shapes: "
                    "(1) cross-file refactors touching N modules in "
                    "ways that need coordination ('refactor the data "
                    "layer to use Postgres', 'add OAuth — touches "
                    "auth + middleware + tests'); (2) work that "
                    "splits naturally into independent sub-tasks an "
                    "explorer + auditor + reviewer can do in "
                    "parallel ('investigate why X is slow then fix "
                    "it', 'review my PR end-to-end'); (3) "
                    "architecture changes affecting the whole "
                    "system. DO NOT pick COMPLEX for single-file "
                    "work, no matter how many issues that file has "
                    "— a checklist of 20 fixes in one file is still "
                    "SIMPLE because the steps are sequential, not "
                    "parallel. The team's overhead (planning, "
                    "delegation, review-of-review) only pays off "
                    "when there's real parallelism to exploit."
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
        # ``conversation_scope='shared'`` (loomflow 0.10.18+): every
        # turn — whether routed to SIMPLE or COMPLEX — runs under
        # the PARENT REPL session_id instead of the default per-
        # route ``{parent}__route_{name}`` derivation. Without this,
        # ``what is this code about?`` (turn 1 → COMPLEX) followed
        # by ``can you check what is this code about?`` (turn 2 →
        # SIMPLE) lost continuity: the SIMPLE coder woke up under
        # a fresh per-route session_id with zero prior messages.
        # In shared mode the routed agent rehydrates from the
        # parent session and sees the WHOLE conversation, no
        # matter which route handled each turn. Tradeoff: routes
        # see each other's tool calls in history (the SIMPLE coder
        # sees ``delegate``/``send_message`` from prior COMPLEX
        # turns) — empirically harmless since the model treats
        # them as context, not actionable history.
        conversation_scope="shared",
        # Single classification call per user message; the routed
        # agent then runs to completion. The router's StopHook
        # behavior is moot since classification doesn't have a
        # plan — but pass it anyway for symmetry with supervisor.
        max_stop_hook_iterations=max_stop_hook_iterations,
    )
    # Stamp the retrieval mode on the coordinator so the REPL's
    # per-turn LoomRetriever build can read it back without
    # plumbing yet another arg through every call site. The REPL
    # checks ``getattr(self.agent, '_loom_retrieval_mode', 'bm25')``
    # when instantiating LoomRetriever and the two stay in sync.
    coordinator._loom_retrieval_mode = loom_retrieval  # type: ignore[attr-defined]
    return coordinator, workspace


def loom_dir_for(root: Path) -> Path:
    """Return (and create) the ``.loom`` dir for a project root."""
    d = root / LOOM_DIR
    d.mkdir(exist_ok=True)
    return d
