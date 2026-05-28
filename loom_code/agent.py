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

from loomflow import Agent
from loomflow.team import Team
from loomflow.tools import find_tool, ls_tool, read_tool
from loomflow.workspace import LocalDiskWorkspace

from .extensions import Extensions, safe_role_name
from .grep_tool import enhanced_grep_tool as grep_tool
from .hooks import attach_tool_hooks
from .project import Project
from .prompts import build_unified_coordinator_instructions
from .rules import remember_rule_tool
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

    # The coordinator is READ-ONLY: it reads to understand + answers
    # questions, but has NO writer/exec tools — so it CANNOT grind
    # edits itself and MUST delegate every change to ``coder`` (and
    # test-runs to ``reviewer``). Removing the writer kernel is what
    # stops the coordinator doing everything itself and leaving the
    # worker roster idle — a prompt nudge alone didn't hold.
    root = project.root
    coordinator_tools: list[object] = [
        read_tool(root),
        grep_tool(root),
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
    coordinator = Team.supervisor(
        workers=workers,
        instructions=build_unified_coordinator_instructions(project),
        tools=coordinator_tools,
        model=model,
        memory=memory_cfg,
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
        # No permissions/approval on the coordinator — its tools are
        # read-only. The destructive-tool gate + approval bridge live
        # on the workers (coder/reviewer), where edits and bash run.
    )

    # Tool hooks attach to every agent that touches the codebase. The
    # coordinator only reads, but a user PreToolUse hook can match
    # ``read``/``grep`` too, so keep it in the set; the workers (which
    # write + run bash) are the main target. No-op when none declared.
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
