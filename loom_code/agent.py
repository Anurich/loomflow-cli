"""Builds the loomflow team that powers loom-code.

This is the one place loom-code wires loomflow primitives
together. Everything here is configuration — no agent-loop logic,
no tool implementations, no memory logic. If this file ever grows
real behaviour, that behaviour belongs in loomflow.

loom-code is a single ``Team.supervisor``. The coordinator is a
READ-ONLY tech lead: it has ``read``/``grep``/``ls``/``find``/
``web_fetch`` to understand the code and answer questions, plus a
``delegate`` tool — but NO writer/exec tools. So it plans, tracks,
and manages, and hands every change (writes/edits) to ``coder`` and
every test-run to ``reviewer``, with ``explorer``/``auditor`` for
investigation (see :mod:`loom_code.workers`). Removing the writer
kernel from the coordinator is deliberate: with it, the model just
grinds edits itself and leaves the workers idle. ``Team.supervisor``
returns a plain ``Agent``, so the rest of loom-code (REPL, CLI,
renderer) treats it exactly like any agent.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from importlib.resources import files as _pkg_files
from pathlib import Path
from typing import Any

from loomflow import Agent
from loomflow.team import Team
from loomflow.tools import find_tool, ls_tool, read_tool
from loomflow.workspace import LocalDiskWorkspace

from .code_index import codebase_search_tool
from .credentials import patient_retry_policy_for
from .extensions import Extensions, safe_role_name
from .file_tools import loom_read_tool
from .grep_tool import enhanced_grep_tool as grep_tool
from .hooks import attach_tool_hooks
from .lsp_tools import lsp_tools
from .project import Project
from .prompts import build_unified_coordinator_instructions
from .rules import remember_rule_tool
from .trust import discover_trusted
from .web_fetch import web_fetch_tool
from .workers import (
    BUILTIN_WORKER_NAMES,
    SUMMARY_THRESHOLD_CHARS,
    _build_coder,
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


def _is_openai_model(model: str) -> bool:
    """True when ``model`` is served by OpenAI (so its embeddings are
    paid for by the same key). Used to pin the memory embedder to the
    selected chat model's provider — OpenAI models get OpenAI
    embeddings, everything else (Claude, Gemini, local) uses the
    zero-key ``hash`` embedder so recall never makes a cross-provider
    OpenAI call. Anthropic / Gemini / Ollama have no embeddings API we
    use, so the test is simply 'is this an OpenAI chat model'."""
    m = model.lower()
    # OpenAI chat models: gpt-*, the o-series (o1/o3/o4...), and the
    # ``openai/`` litellm prefix. Anthropic/Gemini/etc. never match.
    return (
        m.startswith(("gpt-", "gpt", "o1", "o3", "o4", "openai/", "chatgpt"))
        or m in {"o1", "o3", "o4"}
    )


def build_agent(
    project: Project,
    *,
    model: str = DEFAULT_MODEL,
    approval_handler: Callable[..., Awaitable[bool]] | None = None,
    max_turns: int = 100,
    web_backend: str | None = None,
    max_stop_hook_iterations: int = 2,
    snip_window: int = 8,
    auto_compact: bool = True,
    tool_result_summarizer: str | None = None,
    extensions: Extensions | None = None,
    effort: str | None = None,
    sandbox: bool = False,
    sandbox_allow_network: bool = False,
    operator: bool = False,
    run_until: str | dict[str, Any] | None = None,
) -> tuple[Agent, LocalDiskWorkspace]:
    """Wire the loom-code agent for a given project.

    Returns ``(coordinator, workspace)`` — the coordinator is a
    single ``Team.supervisor`` Agent; the caller needs the workspace
    handle to drive the self-improvement loop
    (``attribute_outcome`` after a run, ``prune`` for retention).

    The coordinator is READ-ONLY: it holds ``read``/``grep``/``ls``/
    ``find``/``web_fetch`` (+ ``delegate``) to understand the code
    and answer questions, but has NO ``write``/``edit``/``bash`` —
    so it CANNOT make changes itself and MUST delegate every
    mutation to a worker. It rides loomflow's ``delegate`` →
    ``SubagentInvocation`` path, which streams worker events to the
    parent and rolls up their token cost. (Giving the coordinator
    the writer kernel was tried and reverted: the model just ground
    edits itself and never delegated, leaving the roster idle.)

    The whole brain in one builder call:

    * **workers** — the delegate roster (:func:`build_workers`):
      ``coder`` (the ONLY writer — full file-and-shell kernel), plus
      read-only ``explorer`` / ``auditor`` / ``reviewer``. Custom
      ``.loom/agents/*.md`` join as additional delegate workers.
    * **coordinator** — ``Team.supervisor`` with read-only ``tools=``;
      owns the living plan; plans, tracks, and delegates all writes
      to ``coder`` and verification to ``reviewer``.
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
    # Embedder follows the SELECTED chat model's provider. An OpenAI
    # model uses OpenAI embeddings (the key is already present + funded
    # for OpenAI users); every other provider (Claude, Gemini, local)
    # uses the zero-key ``hash`` embedder so memory recall NEVER makes a
    # cross-provider OpenAI call. Without this, loomflow's default
    # embedder auto-picks OpenAI whenever OPENAI_API_KEY happens to be
    # set — which crashed Claude-only runs with an OpenAI 429 during
    # fact recall. Passing memory as a dict (not the ``sqlite:`` string)
    # is what lets us pin the embedder.
    embedder = "openai" if _is_openai_model(model) else "hash"
    memory_cfg: dict[str, str] = {
        "backend": "sqlite",
        "path": str(loom_dir / "memory.db"),
        "embedder": embedder,
    }

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

        from .credentials import context_window_override
        # Prefer a known window for litellm-routed models (NVIDIA
        # Nemotron, Groq Llama, ...) that context_window_for doesn't
        # recognise — otherwise it returns a conservative 8192 and
        # compaction fires far too early.
        window = context_window_override(model) or context_window_for(model)
        auto_compact_at_tokens = int(window * 0.8)

    # Cheap same-provider sibling for low-stakes utility LLM calls —
    # auto-compact summaries and per-result tool-output compression.
    # Running those on the main coding model (Opus / GPT-4-class)
    # wastes real money on summarisation; Haiku / gpt-4.1-mini do
    # the job. ``None`` (no usable cheap sibling) falls back to the
    # main model inside the framework.
    from .credentials import cheap_model_for
    cheap_model = cheap_model_for(model)
    # Per-result tool-output compression for the read-only WORKERS
    # only. The framework replaces the result IN-TURN (the agent sees
    # the digest, never the verbatim output), so this is a last-resort
    # bound against a single huge dump 400-ing a worker's run — snip
    # is turn-count-based and auto-compact never fires inside a
    # worker's single run. Excluded on purpose:
    # * the CODER — needs verbatim ``read`` output to build
    #   exact-match ``edit`` old_strings;
    # * the COORDINATOR — its ``delegate`` results ARE the worker
    #   briefings (digesting them loses the findings), and in
    #   operator mode it holds writer tools, hitting the same
    #   exact-match problem as the coder.
    worker_summarizer = (
        tool_result_summarizer
        if tool_result_summarizer is not None
        else cheap_model
    )
    summary_threshold = SUMMARY_THRESHOLD_CHARS

    # MCP servers (trust-gated above, so only servers from a trusted
    # repo or the user's own config survive). Built into one registry
    # and handed to the coder — the sole writer/executor — so its tools
    # join the coder's kernel. The registry connects lazily (on first
    # tool use), so an unreachable server costs nothing until called.
    # Stashed on the coordinator (``_mcp_registry``) so the REPL/sidecar
    # can ``await registry.aclose()`` on exit.
    mcp_registry: Any | None = None
    if extensions.mcp_specs:
        try:
            from loomflow.mcp import MCPRegistry

            mcp_registry = MCPRegistry(
                [entry.spec for entry in extensions.mcp_specs]
            )
        except ImportError:
            # ``mcp`` extra not installed — skip MCP rather than fail the
            # build. (Discovery already degrades, but a user could pass
            # pre-built Extensions; belt-and-suspenders.)
            mcp_registry = None

    workers = build_workers(
        project,
        model=model,
        approval_handler=approval_handler,
        web_backend=web_backend,
        skills=all_skills,
        auto_compact_at_tokens=auto_compact_at_tokens,
        snip_window=snip_window,
        tool_result_summarizer=worker_summarizer,
        effort=effort,
        mcp_registry=mcp_registry,
        sandbox=sandbox,
        sandbox_allow_network=sandbox_allow_network,
        # Same embedder name memory uses (resolved above) so every
        # worker's codebase_search hits the one shared index; the
        # workspace handle fuses learned notes into results (Phase 1b).
        embedder=embedder,
        workspace=workspace,
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

    # The coordinator is READ-ONLY: it reads to understand + answers
    # questions, but has NO writer/exec tools — so it CANNOT grind
    # edits itself and MUST delegate every change to ``coder`` (and
    # test-runs to ``reviewer``). Removing the writer kernel is what
    # stops the coordinator doing everything itself and leaving the
    # worker roster idle — a prompt nudge alone didn't hold.
    root = project.root
    coordinator_tools: list[object] = [
        # Policy-bounded read — reaches user-referenced files outside
        # the project (grep/find/ls stay project-scoped).
        loom_read_tool(root),
        grep_tool(root),
        # Semantic search — finds code by MEANING where grep needs the
        # literal string. Embeds in the SAME space (``embedder``) as
        # memory, so the index and the note store fuse (Phase 1b): the
        # ``workspace`` handle makes every search blend code symbols
        # with what we've LEARNED about them. The coordinator gets it
        # to locate the right subsystem before delegating.
        codebase_search_tool(root, embedder, workspace=workspace),
        # LSP navigation (jedi) — go_to_definition / find_references /
        # hover resolve symbols through imports + scope like an IDE,
        # where grep only matches strings. Read-only; Python only.
        *lsp_tools(root),
        find_tool(root),
        ls_tool(root),
        web_fetch_tool(),
        # Lets the coordinator persist a durable, user-stated rule to
        # AGENTS.md (always-in-prompt next session) instead of trusting
        # probabilistic recall. See loom_code.rules.
        remember_rule_tool(root),
    ]
    if web_backend is not None:
        from loomflow.tools import web_tool
        coordinator_tools.append(web_tool(backend=web_backend))

    # COMPUTER OPERATOR mode (/computer): the coordinator is the agent the
    # user talks to, so in operator mode it must hold the ACTION tools
    # DIRECTLY — not delegate to a worker (the bug that made browser tools
    # unreachable). Add write/edit/bash (act on files + system) + native
    # media/app tools + loom-code's OWN browser engine (page_open/observe/
    # act/check — stable data-loom-id handles + overlay-safe acting +
    # vision verify, replacing the Playwright MCP browser).
    coordinator_instructions = build_unified_coordinator_instructions(project)
    coordinator_tool_host: Any = coordinator_tools
    if operator:
        from pathlib import Path as _Path

        from loomflow.tools import bash_tool, edit_tool, write_tool

        from .browse import browse_tools
        from .operator import build_operator_prompt, media_app_tools

        # OPERATOR mode = "operate my whole computer". The default coding
        # tools are rooted at the PROJECT and reject paths outside it
        # (".. escapes workdir"), which breaks "create a file in
        # Downloads" / "read my Documents". So in operator mode, root the
        # file + shell tools at HOME — the user's actual machine, like a
        # human at the keyboard. The approval gate still confirms every
        # write/destructive action. Coding mode stays project-scoped.
        home = str(_Path.home())
        coordinator_tools.extend(
            [
                read_tool(home),
                ls_tool(home),
                find_tool(home),
                write_tool(home),
                edit_tool(home),
                bash_tool(home, timeout=300.0),
                *media_app_tools(),
                *browse_tools(model=model),
            ]
        )
        # build_operator_prompt() injects today's date so relative dates
        # ("tomorrow") resolve correctly.
        coordinator_instructions = build_operator_prompt()
    # Compose the coordinator's tools with the MCP registry so the
    # coordinator itself can call browser_* (and any other MCP) tools.
    # In operator mode this is what makes browser control reachable by
    # the agent the user talks to.
    if mcp_registry is not None:
        from loomflow.tools.registry import InProcessToolHost

        from .mcp_host import McpAugmentedHost

        coordinator_tool_host = McpAugmentedHost(
            InProcessToolHost(coordinator_tools), mcp_registry
        )

    # ``max_stop_hook_iterations`` bounds the framework Ralph loop:
    # while the LivingPlan still has todo/doing steps after the model
    # stops, the StopHook re-prompts it to continue — up to this many
    # times. The cap only bites when the agent is STUCK (a converging
    # task drains its plan before hitting it), so keep it LOW: a high
    # value just re-prompts a confused model into a re-planning spin
    # (observed: 8 → the model re-plans + re-asks for input it already
    # has, 650k tokens). Default 2 — one or two continuation nudges,
    # then stop. In-turn persistence (the "own the run" prompt rule)
    # is the primary mechanism; this is just a small safety net.
    # ``/set_continue_cap`` tunes it per session.
    # In operator mode the coordinator runs destructive/real-world tools
    # itself (write/edit/bash/browser), so it needs the approval gate +
    # permissions — exactly as the coder does in coding mode. In coding
    # mode the coordinator stays read-only (gate lives on the workers).
    _coord_extra: dict[str, Any] = {}
    if operator:
        from loomflow import StandardPermissions

        _coord_extra["permissions"] = StandardPermissions()
        _coord_extra["approval_handler"] = approval_handler
    # /goal run-until-done loop. Passed ONLY when armed: ``run_until=``
    # needs a loomflow newer than any released 0.10.x, and passing the
    # kwarg unconditionally (even as None) would TypeError at startup
    # on a PyPI install — /goal degrades to an error on old framework
    # versions instead of bricking the whole CLI.
    if run_until is not None:
        _coord_extra["run_until"] = run_until
    coordinator = Team.supervisor(
        workers=workers,
        instructions=coordinator_instructions,
        tools=coordinator_tool_host,
        model=model,
        memory=memory_cfg,
        workspace=workspace,
        living_plan=True,
        skills=all_skills,
        max_turns=max_turns,
        max_stop_hook_iterations=max_stop_hook_iterations,
        prompt_caching=True,
        tool_result_summarizer=tool_result_summarizer,
        tool_result_summary_threshold=summary_threshold,
        snip_window=snip_window,
        auto_compact_at_tokens=auto_compact_at_tokens,
        # Auto-compact summaries are low-stakes — run them on the
        # cheap same-provider sibling instead of the coding model.
        # ``None`` falls back to the main model inside the framework.
        auto_compact_summariser=cheap_model,
        effort=effort,
        persist_tool_transcripts=True,
        # Patient retry schedule on free-tier/litellm providers —
        # None elsewhere keeps loomflow's default 3 attempts.
        retry_policy=patient_retry_policy_for(model),
        **_coord_extra,
    )

    # Tool hooks attach to every agent that touches the codebase. The
    # coordinator only reads, but a user PreToolUse hook can match
    # ``read``/``grep`` too, so keep it in the set; the workers (which
    # write + run bash) are the main target. No-op when none declared.
    # The loop guard (doom-loop + missing-binary steering) rides the
    # same registries — it's a native post hook, not a shell spec.
    from . import loop_guard

    for tool_agent in (coordinator, *workers.values()):
        attach_tool_hooks(
            tool_agent, extensions.hook_specs, cwd=project.root
        )
        loop_guard.attach(tool_agent)

    # Stash the MCP registry on the coordinator so the REPL / sidecar can
    # tear it down (``await coordinator._mcp_registry.aclose()``) on exit
    # — mirrors how the worker registry is carried on the coordinator.
    # ``None`` when no MCP servers were discovered (the common case).
    coordinator._mcp_registry = mcp_registry  # type: ignore[attr-defined]

    return coordinator, workspace


def build_solo_agent(
    project: Project,
    *,
    model: str = DEFAULT_MODEL,
    approval_handler: Callable[..., Awaitable[bool]] | None = None,
    web_backend: str | None = None,
    snip_window: int = 8,
    auto_compact: bool = True,
    effort: str | None = None,
    sandbox: bool = False,
    sandbox_allow_network: bool = False,
    extensions: Extensions | None = None,
) -> Agent:
    """The trivial-task FAST PATH: one coder-kernel agent, no team.

    The supervisor topology earns its keep on multi-file features and
    verification-worthy work — but it taxes a one-line fix with a full
    delegation round-trip (coordinator reads → delegates → coder
    re-reads → coordinator integrates): 2-3x the turns and model calls
    of just doing it. The REPL routes obviously-small tasks here
    instead (see ``Repl._route_turn``); everything real still goes
    through the team.

    Context is CONTINUOUS across routes: the solo agent shares the
    team's memory (same ``.loom/memory.db``, same embedder pivot) and
    notebook workspace, and the REPL runs it under the same
    ``session_id`` — so a solo fix shows up in the team's history next
    turn and vice versa. Approval gate + tool hooks apply exactly as
    they do for the team's coder; permissions are identical. MCP
    servers are NOT attached (an external-integration task isn't a
    trivial fix — the router sends those to the team).
    """
    loom_dir = project.root / LOOM_DIR
    loom_dir.mkdir(exist_ok=True)
    workspace = LocalDiskWorkspace(str(loom_dir / "notebook"))
    embedder = "openai" if _is_openai_model(model) else "hash"
    memory_cfg: dict[str, str] = {
        "backend": "sqlite",
        "path": str(loom_dir / "memory.db"),
        "embedder": embedder,
    }
    if extensions is None:
        extensions = discover_trusted(project.root)
    all_skills = _bundled_skill_paths() + extensions.skill_paths

    auto_compact_at_tokens: int | None = None
    if auto_compact:
        from loomflow.agent.auto_compact import context_window_for

        from .credentials import context_window_override
        window = context_window_override(model) or context_window_for(model)
        auto_compact_at_tokens = int(window * 0.8)

    agent = _build_coder(
        project,
        model=model,
        approval_handler=approval_handler,
        has_web=web_backend is not None,
        skills=all_skills,
        auto_compact_at_tokens=auto_compact_at_tokens,
        snip_window=snip_window,
        effort=effort,
        sandbox=sandbox,
        sandbox_allow_network=sandbox_allow_network,
        embedder=embedder,
        workspace=workspace,
        # Standalone — no parent to inherit memory/workspace from.
        memory=memory_cfg,
        attach_workspace=True,
        # Shares the REPL session_id with the read-only coordinator —
        # persisting writer transcripts would make the coordinator
        # rehydrate history of "itself" editing (the grind failure).
        persist_tool_transcripts=False,
    )
    if web_backend is not None:
        from loomflow.tools import web_tool
        agent.add_tool(web_tool(backend=web_backend))  # type: ignore[arg-type]
    attach_tool_hooks(agent, extensions.hook_specs, cwd=project.root)
    from . import loop_guard

    loop_guard.attach(agent)
    return agent


def loom_dir_for(root: Path) -> Path:
    """Return (and create) the ``.loom`` dir for a project root."""
    d = root / LOOM_DIR
    d.mkdir(exist_ok=True)
    return d
