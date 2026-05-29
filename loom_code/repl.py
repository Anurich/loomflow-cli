"""The loom-code interactive REPL.

``loom-code`` with no args drops here. You chat, it codes, it asks
before destructive changes, you keep going — the Claude-Code / Pi
loop. Conversation continuity comes free: every turn reuses the
same ``session_id``, so loomflow rehydrates prior turns as real
chat history.

The self-improvement loop (Phase 3)
-----------------------------------

Every turn, the agent READS notes from the project notebook —
past plans, past findings (``recall_past_plans``, ``search_notes``,
``read_note``). loomflow records those reads as *citations* on
``RunResult.cited_slugs``. When a turn is judged successful, we
call ``workspace.attribute_outcome(success=True, slugs=...)`` — the
cited notes' ``cited_count`` / ``success_count`` climb, and future
``search_notes(boost_relevance=True)`` ranks them higher.

How "success" is judged — the **moved-on heuristic**:

* We DON'T attribute immediately. We hold the last turn's
  ``cited_slugs`` as ``pending``.
* If you give loom-code another task without complaint, the
  previous turn must have been fine → attribute the pending as
  ``success=True``.
* ``/bad`` attributes pending as ``success=False`` (it broke
  something / wasn't useful).
* ``/good`` attributes pending as ``success=True`` immediately.
* On ``/exit``, any pending is attributed ``success=True`` — you
  left satisfied.

That matches how a developer actually signals: silence + moving
on means "worked", an explicit "no" means "didn't".

Slash commands (handled here, never sent to the agent):
  /help            this list
  /plan            show the current living plan
  /cost            cumulative cost + token totals
  /good            mark the last turn useful (credit cited notes)
  /bad             mark the last turn unhelpful
  /model <name>    switch model (rebuilds the agent, keeps the repo)
  /clear           start a fresh conversation (new session_id)
  /exit, /quit     leave (Ctrl-D works too)
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from loomflow import new_id
from prompt_toolkit import HTML, PromptSession
from prompt_toolkit.completion import (
    CompleteEvent,
    Completer,
    Completion,
)
from prompt_toolkit.document import Document
from rich.text import Text

from . import worktree
from .agent import LOOM_DIR, build_agent
from .approval import ApprovalGate
from .compact import Compactor, default_compact_threshold
from .credentials import (
    ensure_key_for_model,
    save_credential,
)
from .extensions import Extensions, HookSpec
from .extensions import discover as discover_extensions
from .hooks import run_repl_hooks
from .paste import (
    build_paste_keybindings,
    expand_pastes,
    reset_paste_stash,
)
from .project import Project, detect_project
from .render import StreamRenderer, banner, console
from .trust import filter_trusted_hooks

# Provider defaults for /set_model — picking a provider switches
# to that provider's commonly-used model.
_OPENAI_DEFAULT_MODEL = "gpt-4.1-mini"
_ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-6"

_SLASH_HELP = """\
[bold]loom-code commands[/bold]
  [cyan]/help[/cyan]            this list
  [cyan]/plan[/cyan] [<task>]   show the current plan — or start one
  [cyan]/cost[/cyan]            session cost + token totals
  [cyan]/good[/cyan]            mark the last turn useful (credits notes)
  [cyan]/bad[/cyan]             mark the last turn unhelpful
  [cyan]/model[/cyan] <name>    switch to a specific model by name
  [cyan]/set_model[/cyan]       pick OpenAI / Anthropic + save API key
  [cyan]/set_web[/cyan]         enable web search (Serper / DDG / off)
  [cyan]/resume[/cyan]          resume the last session — rehydrates prior
                   turns from loomflow memory so you pick up where
                   you left off
  [cyan]/set_continue_cap[/cyan] [N]
                   show / set the auto-continue cap. When the plan
                   has non-done steps after the agent emits a final
                   answer, the REPL auto-runs up to N more
                   iterations. Default 15. ``/set_continue_cap 0``
                   disables auto-continue.
  [cyan]/clear[/cyan]           fresh conversation (new session)
  [cyan]/compress_token_length[/cyan] [N|auto|off]
                   show / set / disable the auto-compact threshold
                   (default: 80% of the model's context window)
  [cyan]/exit[/cyan]            leave (Ctrl-D also works)

Anything else is a task — loom-code plans, codes, and verifies it.
Long sessions auto-compact: when cumulative tokens cross the
threshold, a compactor agent writes a dense summary to memory and
the conversation continues with that summary as its only history.
"""

_USER_ID = "loom-code"

# DEFAULT cap on auto-continue iterations per turn. This is the
# Ralph-loop / Cursor-judge-agent pattern: the model's "I'm done"
# judgement is unreliable on multi-step plans, so the REPL
# overrules it as long as the plan explicitly disagrees.
#
# Bumped from 5 → 15 after empirical observation: real scaffold
# tasks the user threw at us had 6-12 plan steps, and 5 left them
# stuck mid-stream. 15 gives headroom; stall detection still kicks
# in early on genuinely-runaway loops so the higher cap doesn't
# inflate worst-case cost. Per-session overridable via
# ``/set_continue_cap N`` for power users who want more or less.
_AUTO_CONTINUE_LIMIT_DEFAULT = 15



def _flatten_exception_group(
    eg: BaseExceptionGroup,
) -> list[BaseException]:
    """Recursively unwrap nested ``BaseExceptionGroup`` into a flat
    list of the underlying exceptions.

    anyio task groups raise an ``ExceptionGroup`` whose default
    ``str()`` is "unhandled errors in a TaskGroup (N sub-exception)"
    — useless for the user. Flatten to surface what ACTUALLY went
    wrong (the wrapper might nest more wrappers if multiple groups
    were involved)."""
    out: list[BaseException] = []
    for inner in eg.exceptions:
        if isinstance(inner, BaseExceptionGroup):
            out.extend(_flatten_exception_group(inner))
        else:
            out.append(inner)
    return out

# The single source of truth for slash commands the REPL accepts.
# The autocomplete menu (popped the moment the user types '/')
# reads off this list, so adding a new command here is enough —
# no need to also update the autocomplete separately.
_COMMAND_DEFS: list[tuple[str, str]] = [
    ("/help", "show all commands"),
    ("/init-loom", "create a starter AGENTS.md rules file"),
    ("/plan", "show the current plan, or start one"),
    ("/cost", "session cost + token totals"),
    ("/good", "mark the last turn useful (credit notes)"),
    ("/bad", "mark the last turn unhelpful"),
    ("/model", "switch to a specific model by name"),
    ("/effort", "reasoning effort: low | medium | high | off"),
    ("/isolate", "run this session in its own git worktree"),
    ("/review", "show the isolated session's diff vs base"),
    ("/merge", "merge the isolated session's edits into base"),
    ("/discard", "discard the isolated session's edits"),
    ("/set_model", "pick OpenAI or Anthropic + save API key"),
    ("/set_web", "enable web search (Serper / DuckDuckGo / off)"),
    ("/mcp", "list connected MCP servers + their tools"),
    ("/resume", "resume the last session (rehydrate prior turns)"),
    ("/set_continue_cap", "set auto-continue cap (current=default 15)"),
    ("/clear", "fresh conversation (new session)"),
    (
        "/compress_token_length",
        "auto-compact threshold: <N> | auto | off",
    ),
    ("/exit", "leave (Ctrl-D also works)"),
]


class _SlashCompleter(Completer):
    """Pop the slash-command menu the moment the user types '/'.

    Only fires when the line starts with '/' — typing a normal
    task message stays clean, no popup. Filters as the user types
    more characters, so '/co' narrows to /cost +
    /compress_token_length.
    """

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        for cmd, desc in _COMMAND_DEFS:
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display_meta=desc,
                )


class Repl:
    """One interactive loom-code session over one project."""

    def __init__(self, project: Project, model: str) -> None:
        self.project = project
        self.model = model
        # Per-turn spinner controls. The ApprovalGate must pause the
        # spinner while it prompts (its Live refresh otherwise mangles
        # the keystroke). ``_turn`` points these at the live closures;
        # the gate calls them through the stable wrapper methods.
        self._active_pause_spinner: Any = None
        self._active_resume_spinner: Any = None
        # ApprovalGate persists across turns so 'allow all' sticks
        # for the whole session.
        self._gate = ApprovalGate(
            pause_spinner=self._pause_active_spinner,
            resume_spinner=self._resume_active_spinner,
        )
        self._auto_continue_limit = _AUTO_CONTINUE_LIMIT_DEFAULT
        # Reasoning effort (None | "low" | "medium" | "high"). None =
        # provider default. Set via /effort; threaded into build_agent
        # → every work agent. Inert on non-reasoning models.
        self._effort: str | None = None
        # Session worktree isolation (/isolate). When set, this session
        # edits in its own git worktree on ``_worktree.branch`` and the
        # agent is rebuilt rooted there (_isolated_project); /merge or
        # /discard restores the main tree.
        self._worktree: worktree.WorktreeInfo | None = None
        self._isolated_project: Project | None = None
        # User + project extensions (the ``.loom`` folder). Discovered
        # once here so the SAME bundle drives both build_agent (skills,
        # subagents, tool hooks) and the REPL-lifecycle hooks fired
        # below (SessionStart / UserPromptSubmit / SessionEnd). The
        # REPL owns discovery because it also runs the trust prompt for
        # project hooks (see _consume_trusted_extensions).
        self._extensions = self._consume_trusted_extensions(
            discover_extensions(project.root)
        )
        # Graphify and other bundled skills are auto-registered
        # by build_agent (see _bundled_skill_paths). No per-session
        # toggle needed — the agent decides when to load skills.
        self.agent, self.workspace = build_agent(
            project,
            model=model,
            approval_handler=self._gate.handler,
            max_stop_hook_iterations=self._auto_continue_limit,
            extensions=self._extensions,
            effort=self._effort,
        )
        # One session_id for the whole REPL → loomflow rehydrates
        # prior turns so the agent has real conversation memory.
        self.session_id = new_id()
        # Session accumulators. ``total_in`` is *combined* input
        # tokens (uncached + cached); ``total_cached_in`` is the
        # cached subset, tracked separately so the status line can
        # show the same split (``uncached+cached in``) that the
        # end-of-turn summary uses.
        self.total_cost = 0.0
        self.total_in = 0
        self.total_cached_in = 0
        # ``total_cache_write`` is Anthropic-only — the cache CREATION
        # tokens (1.25x base price on 5m TTL, 2x on 1h). Tracked
        # separately from cached_in (which is the cache READ — cheap)
        # so /cost can surface both directions of the cache
        # accounting. OpenAI returns 0 here (no separate billing for
        # cache writes).
        self.total_cache_write = 0
        self.total_out = 0
        # Framework-event counters (loomflow 0.10.13+):
        # ``total_summaries`` ticks each time
        # ``tool_result_summarized`` fires (per-tool-result LLM
        # compression — only when ``tool_result_summarizer=`` is
        # wired). ``total_compacts`` ticks each
        # ``auto_compacted`` event (mid-Ralph-loop conversation
        # summarisation when tokens cross the budget threshold).
        # ``total_snips`` ticks each ``messages_snipped`` event
        # (free list-slicing trim of older user-anchored turn
        # groups). All three surface in ``/cost`` so the user can
        # see the token-optimisation tiers actually firing.
        self.total_summaries = 0
        self.total_compacts = 0
        self.total_snips = 0
        self.turns = 0
        self.last_plan: str | None = None
        # Self-improvement: cited slugs from the last turn, awaiting
        # a success/failure judgement (the moved-on heuristic).
        self._pending_slugs: list[str] = []
        # Automatic compaction state. ``_compact_threshold = -1``
        # means "auto, recompute from model"; ``0`` means "off";
        # any positive int is an explicit user override. The
        # exchange list is what the compactor sees on trigger; the
        # cumulative-tokens counter is what fires the trigger.
        self._compactor = Compactor(model=model)
        self._compact_threshold = -1  # auto
        self._compact_tokens = 0
        self._compact_exchanges: list[tuple[str, str]] = []
        # Web-search backend: ``"serper"``, ``"duckduckgo"``, or
        # ``None`` (off — default). Toggled via /set_web. Rebuilding
        # the agent picks the new backend up by adding (or not
        # adding) a ``web_tool`` to coder + explorer.
        self._web_backend: str | None = None
        # ``self._auto_continue_limit`` is initialised earlier in
        # __init__ (before build_agent is called) so the framework
        # gets the right ``max_stop_hook_iterations`` on construction.
        # See the build_agent call above.
        # prompt_toolkit drives the input line. complete_while_typing
        # opens the autocomplete menu the moment the user types '/'
        # without any extra keystroke (Tab also still works for
        # explicit completion). History gives free up-arrow recall
        # within the session. The paste keybindings collapse large
        # pastes into `[paste-N: <lines>, <chars>]` placeholders so
        # the visible prompt stays readable; expand_pastes() restores
        # the full content before the line goes to the agent.
        self._prompt_session: PromptSession[str] = PromptSession(
            completer=_SlashCompleter(),
            complete_while_typing=True,
            key_bindings=build_paste_keybindings(),
        )

    async def run(self) -> int:
        """The REPL loop. Returns an exit code.

        Skills (graphify and friends) are wired in at agent
        construction time via :func:`build_agent` — no per-session
        spawning, no subprocess lifecycle to manage here.

        The ``finally`` DOES tear down the MCP registry: ``build_agent``
        stashes any connected MCP servers on ``agent._mcp_registry``,
        and those hold live subprocess / HTTP sessions (stdio servers
        are child processes). Closing them on every exit path — normal
        quit, Ctrl-C, or an exception — avoids leaking processes.
        """
        try:
            return await self._run_inner()
        finally:
            await self._aclose_mcp()

    async def _aclose_mcp(self) -> None:
        """Best-effort teardown of the MCP registry's sessions. Never
        raises — shutdown must not turn a clean exit into an error."""
        registry = getattr(self.agent, "_mcp_registry", None)
        if registry is None:
            return
        try:
            await registry.aclose()
        except Exception:  # noqa: BLE001 — teardown must not fail exit
            pass

    async def _run_inner(self) -> int:
        """The REPL loop body. Held as a separate method in case a
        future feature wants to wrap it in a context manager again
        — keeps the wrapping point obvious."""
        banner(self.model, str(self.project.root), self.project.is_git)
        if self.project.context_file:
            console.print(
                f"  [dim]loaded context: "
                f"{self.project.context_file.name}[/dim]"
            )
        # Brief getting-started hints right after the banner. Surfaces
        # provider/web setup AND the bare /model command so users who
        # already have an API key but want a different model name
        # don't go hunting through /help. Ordering: most-common first.
        console.print(
            "  [dim]▸ type a task, or [cyan]/help[/cyan] "
            "for the command menu[/dim]"
        )
        console.print(
            "  [dim]▸ [cyan]/model <name>[/cyan]  switch to a "
            "specific model by name (e.g. gpt-4.1, claude-opus-4-7)[/dim]"
        )
        console.print(
            "  [dim]▸ [cyan]/set_model[/cyan]     pick "
            "OpenAI or Anthropic and save your API key[/dim]"
        )
        console.print(
            "  [dim]▸ [cyan]/set_web[/cyan]       enable web "
            "search (Serper / DuckDuckGo)[/dim]"
        )
        # Show the resume hint ONLY when a prior session pointer
        # exists — no point telling first-time users about a
        # feature they can't use yet.
        if self._load_session_pointer() is not None:
            console.print(
                "  [dim]▸ [cyan]/resume[/cyan]        pick up "
                "the last session for this project (rehydrates "
                "prior turns)[/dim]"
            )
        self._print_extensions_banner()
        console.print()

        # SessionStart hooks fire once, after the banner and before the
        # first prompt — for side effects (env setup, logging). Their
        # added context is surfaced as a dim note rather than injected,
        # since there's no user turn to attach it to yet.
        start_result = await self._fire_repl_hooks("SessionStart")
        if start_result.added_context:
            console.print(
                f"  [dim]{start_result.added_context}[/dim]"
            )

        while True:
            # Persistent session status — printed before every prompt
            # so the user always sees where they stand on cost/tokens,
            # not just when they ask via /cost. Same split format as
            # the end-of-turn summary for visual consistency.
            self._print_status_line()
            try:
                line = await self._read_line()
            except (EOFError, KeyboardInterrupt):
                # Leaving satisfied — credit any pending turn.
                await self._attribute_pending(success=True, quiet=True)
                await self._fire_repl_hooks("SessionEnd")
                console.print("\n[dim]bye[/dim]")
                return 0

            line = line.strip()
            if not line:
                continue

            # Expand any [paste-N: ...] placeholders to the full
            # stashed content BEFORE dispatch — slash commands
            # generally won't contain pastes, but expanding here
            # keeps a single canonical "what the user really said"
            # point of truth and matches how Claude Code does it.
            line = expand_pastes(line)

            if line.startswith("/"):
                should_continue = await self._handle_slash(line)
                if not should_continue:
                    await self._attribute_pending(
                        success=True, quiet=True
                    )
                    await self._fire_repl_hooks("SessionEnd")
                    console.print("[dim]bye[/dim]")
                    return 0
                continue

            # UserPromptSubmit hooks see the prompt before the agent
            # does. A hook may BLOCK the turn (exit 2) — e.g. a policy
            # gate — or return additionalContext we fold into the
            # prompt (e.g. inject the current ticket / branch).
            submit = await self._fire_repl_hooks(
                "UserPromptSubmit", prompt=line
            )
            if submit.blocked:
                console.print(
                    f"  [red]⊘ blocked by hook[/red]: "
                    f"{submit.reason or '(no reason given)'}"
                )
                continue
            if submit.added_context:
                line = f"{line}\n\n[context from hook]\n{submit.added_context}"

            # A new task with no prior complaint → the previous
            # turn must have been fine. Credit it, then run.
            await self._attribute_pending(success=True, quiet=False)
            # Per-turn repo-map injection — populates the
            # ``loom_index`` working block with the deterministic repo
            # map. Loomflow auto-injects working blocks into the next
            # system prompt.
            await self._inject_loom_context(line)
            await self._turn(line)

    # ---- input ----------------------------------------------------------

    async def _read_line(self) -> str:
        """Read one line from the user via prompt_toolkit.

        Async because ``PromptSession.prompt_async`` integrates with
        the asyncio event loop directly — no thread hop needed.
        The slash-command autocomplete + history come from the
        ``PromptSession`` configured in ``__init__``.
        """
        return await self._prompt_session.prompt_async(
            HTML("<ansigreen><b>loom</b></ansigreen>: ")
        )

    # ---- a task turn ----------------------------------------------------

    async def _inject_loom_context(self, prompt: str) -> None:
        """Update the ``loom_index`` working block with a deterministic
        repo map — the most structurally-important symbols (signatures +
        locations) — which loomflow folds into the next system prompt.

        Built from the structural index (AST walk, no model calls), so
        it needs no ``/loominit`` and is fresh-by-construction: the
        cached builder re-walks only when the tree changed. ``prompt``
        is unused (the map is a stable global overview, which keeps the
        system prompt cache-stable across turns).

        Failures are swallowed (never let memory I/O kill a turn).
        """
        del prompt  # map is global, not prompt-ranked
        try:
            from .loominit.repomap import repo_map_for_root_cached

            # Deterministic repo map (top symbols by structural
            # importance) built from the structural index — no LLM, no
            # LOOM.md/loominit needed, and fresh-by-construction
            # (re-walked only when the tree changed). Replaces the old
            # BM25-over-LLM-narrative retrieval that drifted as the
            # agent edited code.
            body = repo_map_for_root_cached(self.project.root)
            if body:
                await self.agent.memory.update_block(
                    "loom_index", body, user_id=_USER_ID
                )
            # Auto-reload the project rules file (AGENTS.md): re-read it
            # FRESH each turn into the ``project_rules`` working block, so
            # a mid-session edit applies on the next turn without a
            # restart. The coordinator's static prompt no longer bakes
            # the rules file (see build_unified_coordinator_instructions).
            from .rules import project_rules_block

            await self.agent.memory.update_block(
                "project_rules",
                project_rules_block(self.project.root),
                user_id=_USER_ID,
            )
        except Exception:  # noqa: BLE001 — injection is best-effort
            pass

    async def _consume_agent_stream(
        self,
        agent: Any,
        prompt: str,
        renderer: StreamRenderer,
        pause_status: Any,
    ) -> bool:
        """Stream one agent run into ``renderer`` + tick the token-
        optimisation counters. Returns False (caller should abort
        the turn) if the stream raised; True on clean completion.

        Extracted so the escalation path can re-run a SECOND agent
        (the supervisor) through the identical consume + error-
        handling logic without duplicating it."""
        try:
            async for event in agent.stream(
                prompt,
                user_id=_USER_ID,
                session_id=self.session_id,
            ):
                renderer.handle(event)
                # Tick the token-optimisation counters (loomflow
                # 0.10.13+). The renderer doesn't surface
                # architecture_events to the user — we inspect them
                # here to drive the /cost display.
                kind = getattr(event, "kind", None)
                payload = getattr(event, "payload", None)
                if (
                    payload is not None
                    and kind is not None
                    and str(kind).endswith("architecture_event")
                ):
                    name = payload.get("name")
                    if name == "tool_result_summarized":
                        self.total_summaries += 1
                    elif name == "auto_compacted":
                        self.total_compacts += 1
                    elif name == "messages_snipped":
                        self.total_snips += 1
        except KeyboardInterrupt:
            pause_status()
            console.print(
                "\n[yellow]interrupted — turn abandoned[/yellow]"
            )
            return False
        except BaseExceptionGroup as eg:
            # anyio's task groups raise ``ExceptionGroup`` when any
            # child task fails. Unwrap to surface the REAL cause(s)
            # instead of the opaque wrapper message.
            pause_status()
            for inner in _flatten_exception_group(eg):
                console.print(
                    f"\n[bold red]error: "
                    f"{type(inner).__name__}: {inner}[/bold red]"
                )
            return False
        except Exception as exc:  # noqa: BLE001 — REPL must survive
            pause_status()
            console.print(
                f"\n[bold red]error: "
                f"{type(exc).__name__}: {exc}[/bold red]"
            )
            return False
        return True

    # ---- .loom extensions: trust gate + REPL-lifecycle hooks --------

    def _consume_trusted_extensions(
        self, extensions: Extensions
    ) -> Extensions:
        """Apply the project-hook trust gate to a discovered bundle.

        User hooks, skills, and subagents pass through untouched;
        project hooks survive only if already trusted or approved at
        the prompt below. Called once from ``__init__``."""
        return filter_trusted_hooks(
            extensions,
            project_root=self.project.root,
            prompt=self._prompt_trust_project_hooks,
        )

    def _prompt_trust_project_hooks(self, specs: list[HookSpec]) -> bool:
        """Show a project's hook commands and ask whether to trust them.

        Safe default is NO: a non-tty session never auto-trusts, and at
        the prompt only an explicit ``y`` approves — we don't run a
        cloned repo's shell commands without consent."""
        from .approval import _read_single_key

        console.print()
        console.print(
            "  [bold yellow]⚠ this project defines hooks[/bold yellow] "
            "(.loom/settings.toml) that run shell commands "
            "automatically:"
        )
        for s in specs:
            tag = f" [{s.matcher}]" if s.matcher not in ("", "*") else ""
            console.print(
                f"    [cyan]{s.event}[/cyan]{tag}  →  "
                f"[dim]{s.command}[/dim]"
            )
        if not sys.stdin.isatty():
            console.print(
                "  [dim](non-interactive — skipping project hooks)[/dim]"
            )
            return False
        console.print(
            "  [bold]trust and run these hooks?[/bold] "
            "[dim](press y to trust, any other key to skip)[/dim] ",
            end="",
        )
        trusted = _read_single_key() in ("y", "Y")
        console.print(
            "[green]trusted[/green]" if trusted else "[dim]skipped[/dim]"
        )
        return trusted

    def _print_extensions_banner(self) -> None:
        """Show what got picked up from ``.loom`` so the user can
        confirm their skills / subagents / hooks loaded (and which
        project hooks the trust gate let through)."""
        ext = self._extensions
        bits: list[str] = []
        if ext.skill_paths:
            bits.append(f"{len(ext.skill_paths)} skill(s)")
        if ext.agent_specs:
            names = ", ".join(s.name for s in ext.agent_specs)
            bits.append(f"{len(ext.agent_specs)} subagent(s) ({names})")
        if ext.hook_specs:
            bits.append(f"{len(ext.hook_specs)} hook(s)")
        if bits:
            console.print(
                f"  [dim]▸ .loom extensions: {' · '.join(bits)}[/dim]"
            )

    async def _fire_repl_hooks(
        self, event: str, *, prompt: str | None = None
    ) -> Any:
        """Run every REPL-lifecycle hook registered for ``event``.

        Returns the ``ReplHookResult`` so ``UserPromptSubmit`` can act
        on a block / injected context; ``SessionStart`` / ``SessionEnd``
        callers ignore it (those hooks run for their side effects)."""
        return await run_repl_hooks(
            self._extensions.hook_specs,
            event,
            cwd=self.project.root,
            prompt=prompt,
        )

    def _pause_active_spinner(self) -> None:
        """Stable hook the ApprovalGate calls to stop the current
        turn's spinner before prompting. No-op between turns."""
        cb = self._active_pause_spinner
        if cb is not None:
            cb()

    def _resume_active_spinner(self) -> None:
        """Stable hook the ApprovalGate calls after the prompt to
        bring the spinner back."""
        cb = self._active_resume_spinner
        if cb is not None:
            cb()

    async def _turn(self, prompt: str) -> None:
        """Stream one agent run for ``prompt``, reusing the
        session so conversation history carries forward.

        Spinner UX: Rich's ``console.status`` runs continuously for
        the whole turn. The renderer drives its label via two
        callbacks — ``set_status(label)`` updates the text,
        ``pause_status()`` stops it (used while assistant prose is
        streaming, since the spinner shares the cursor line). Labels
        come from the in-flight event: "delegating to coder...",
        "running: pytest -q", "searching: openpyxl write_only", or
        a generic "thinking..." between events. The point is to
        avoid the long blank stretches the old "drop on first event"
        scheme produced in Supervisor mode."""
        status = console.status(
            "[dim]loomflowing...[/dim]", spinner="dots"
        )
        status.start()
        status_running = True

        def set_status(label: str) -> None:
            """Update the spinner label, restarting it if it was
            paused for a prose burst."""
            nonlocal status_running
            if not status_running:
                status.start()
                status_running = True
            status.update(f"[dim]{label}[/dim]")

        def pause_status() -> None:
            """Stop the spinner so streamed text can use the cursor
            line cleanly. ``set_status`` restarts it later."""
            nonlocal status_running
            if status_running:
                status.stop()
                status_running = False

        # Point the ApprovalGate's spinner hooks at THIS turn's
        # closures. Resume re-labels to a neutral "thinking..." since
        # the gate has no event to name.
        self._active_pause_spinner = pause_status
        self._active_resume_spinner = lambda: set_status("thinking...")

        renderer = StreamRenderer(
            set_status=set_status, pause_status=pause_status
        )

        # The Ralph loop now lives in loomflow itself (>=0.10.8) via
        # the StopHook protocol — Agent(living_plan=True) auto-
        # registers a hook that re-prompts when any plan step is
        # still `doing`/`todo` after the architecture exits. We just
        # consume the agent's stream; the framework handles
        # continuation, bounded by ``max_stop_hook_iterations``.
        ok = await self._consume_agent_stream(
            self.agent, prompt, renderer, pause_status
        )
        if not ok:
            return

        if renderer.last_plan:
            self.last_plan = renderer.last_plan
        result = renderer.last_result
        agent_output = ""
        if result:
            self.total_cost += float(result.get("cost_usd", 0.0))
            tin = int(result.get("tokens_in", 0))
            cached_in = int(result.get("cached_tokens_in", 0))
            cache_write = int(result.get("cache_write_tokens", 0))
            tout = int(result.get("tokens_out", 0))
            self.total_in += tin + cached_in
            self.total_cached_in += cached_in
            self.total_cache_write += cache_write
            self.total_out += tout
            self.turns += int(result.get("turns", 0))
            self._pending_slugs = list(
                result.get("cited_slugs") or []
            )
            self._compact_tokens += tin + cached_in + tout
            agent_output = str(result.get("output") or "")

            # Surface framework-level stop-hook exhaustion so the
            # user knows the cap was hit (and can raise it with
            # /set_continue_cap N).
            if result.get("interrupted") and (
                result.get("interruption_reason")
                == "stop_hook_iterations_exhausted"
            ):
                console.print(
                    "\n  [yellow]plan still had work but the agent "
                    f"hit the auto-continue cap "
                    f"({self._auto_continue_limit}) — type "
                    "'continue' to push further, raise the cap "
                    "with /set_continue_cap N, or accept the "
                    "partial result[/yellow]"
                )

        pause_status()
        self._compact_exchanges.append((prompt, agent_output))

        # Anti-poison gate: if the turn made ZERO tool calls AND the
        # output is a bare completion claim ("all issues fixed"),
        # the episode loomflow just persisted is a hallucination
        # with no grounding — and a self-reinforcing one (recall
        # surfaces it → next turn parrots it → new episode → doom
        # loop). Delete it so it can't poison future recall. We
        # only nuke the no-tool-call completion-claim case;
        # legitimate no-tool answers ("what does X mean?") don't
        # match the completion-claim pattern and are kept.
        n_tool_calls = len(renderer._call_names)
        if n_tool_calls == 0 and _looks_like_completion_claim(
            agent_output
        ):
            deleted = _delete_last_episode(
                self.project.root / LOOM_DIR / "memory.db",
                session_id=self.session_id,
                user_id=_USER_ID,
            )
            if deleted:
                console.print(
                    "  [dim](skipped persisting an unverified "
                    "'done' claim — no tool calls backed it)[/dim]"
                )

        # Persist the current session_id to disk so /resume on the
        # next REPL launch knows what to rehydrate. Done after EVERY
        # turn (not just on /exit) so a crash doesn't lose the
        # session pointer. Cheap — one short write to a small file.
        self._save_session_pointer()

        if self._pending_slugs:
            console.print(
                "  [dim]if that worked, just continue — or "
                "/bad if it didn't[/dim]"
            )
        console.print()

        # Maybe compact. Done AFTER the turn renders + the
        # pending-slugs hint prints so the user sees the natural
        # turn boundary before any compaction status appears.
        await self._maybe_compact()

    # ---- self-improvement attribution -----------------------------------

    async def _attribute_pending(
        self, *, success: bool, quiet: bool
    ) -> None:
        """Flush the pending turn's citations to the workspace,
        crediting (or debiting) the notes the agent read.

        ``quiet`` suppresses the confirmation line — used for the
        implicit 'moved-on = success' path so the REPL doesn't
        chatter on every turn."""
        if not self._pending_slugs:
            return
        slugs = self._pending_slugs
        self._pending_slugs = []
        try:
            n = await self.workspace.attribute_outcome(
                success=success, slugs=slugs, user_id=_USER_ID
            )
        except Exception:  # noqa: BLE001 — best-effort, never fatal
            return
        if n and not quiet:
            verb = "credited" if success else "debited"
            console.print(
                f"  [dim]{verb} {n} note(s) from the last "
                f"turn[/dim]"
            )

    # ---- slash commands -------------------------------------------------

    async def _handle_slash(self, line: str) -> bool:
        """Dispatch a /command. Returns False to exit the REPL."""
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit"):
            return False
        if cmd == "/help":
            console.print(_SLASH_HELP)
        elif cmd == "/init-loom":
            from .rules import init_agents_md

            path, created = init_agents_md(self.project.root)
            if created:
                console.print(
                    f"[green]created {path.name}[/green] — a starter "
                    "rules file loom-code reads every session. Edit it, "
                    'or just state rules in chat (e.g. "never edit X") '
                    "and loom-code will save them here."
                )
            else:
                console.print(
                    f"[dim]{path.name} already exists — loom-code "
                    "already reads it. Edit it directly, or state rules "
                    "in chat.[/dim]"
                )
        elif cmd == "/plan":
            if arg:
                # "/plan <task>" reads as "plan and do <task>" — run
                # it as a normal task. loom-code plans every task
                # anyway (living_plan=True), so the plan shows up
                # mid-stream and `/plan` with no arg replays it.
                await self._attribute_pending(
                    success=True, quiet=False
                )
                await self._inject_loom_context(arg)
                await self._turn(arg)
            elif self.last_plan:
                console.print(Text(self.last_plan, style="dim"))
            else:
                console.print(
                    "[dim]no plan yet — give loom-code a task, or "
                    "`/plan <task>` to start one[/dim]"
                )
        elif cmd == "/cost":
            uncached = self.total_in - self.total_cached_in
            # Cache-hit ratio over total input tokens. The ratio
            # tells the user whether their prompt-caching investment
            # is actually paying off — a low ratio means the system
            # prompt is changing turn-to-turn (cache-bust) or the
            # provider doesn't expose cache reads.
            hit_pct = (
                (self.total_cached_in / self.total_in * 100.0)
                if self.total_in > 0
                else 0.0
            )
            console.print(
                Text.assemble(
                    ("  session: ", "dim"),
                    (f"{self.turns} turns", ""),
                    ("  ·  ", "dim"),
                    (
                        f"{uncached:,}+{self.total_cached_in:,} in / "
                        f"{self.total_out:,} out",
                        "",
                    ),
                    ("  ·  ", "dim"),
                    (f"${self.total_cost:.4f}", "green"),
                )
            )
            # Second line: cache breakdown. Only render when there's
            # something to report — keeps the empty-session output
            # uncluttered. ``cache_write`` only fires on Anthropic
            # (5m TTL = +25%, 1h = +100%); on OpenAI it stays 0.
            if self.total_cached_in > 0 or self.total_cache_write > 0:
                cache_color = (
                    "green" if hit_pct >= 50 else "yellow"
                    if hit_pct >= 20 else "dim"
                )
                segments: list[tuple[str, str]] = [
                    ("  cache:   ", "dim"),
                    (f"{hit_pct:.1f}% hit", cache_color),
                ]
                if self.total_cache_write > 0:
                    segments.extend(
                        [
                            ("  ·  ", "dim"),
                            (
                                f"{self.total_cache_write:,} write",
                                "dim",
                            ),
                        ]
                    )
                console.print(Text.assemble(*segments))
            # Third line: token-optimisation tier counters. Each
            # entry only renders when its counter is non-zero —
            # opted-out features stay invisible. The three counters
            # map 1:1 to the three opt-in framework knobs in
            # build_agent (snip_window, auto_compact_at_tokens,
            # tool_result_summarizer) — seeing zeros across the
            # board means "the conversation never got large enough
            # to need any of them," which is a useful diagnostic
            # signal on its own.
            opt_segments: list[tuple[str, str]] = []
            if self.total_snips > 0:
                opt_segments.append(
                    (f"{self.total_snips} snip", "dim")
                )
            if self.total_compacts > 0:
                if opt_segments:
                    opt_segments.append(("  ·  ", "dim"))
                opt_segments.append(
                    (f"{self.total_compacts} compact", "dim")
                )
            if self.total_summaries > 0:
                if opt_segments:
                    opt_segments.append(("  ·  ", "dim"))
                opt_segments.append(
                    (f"{self.total_summaries} tool-summary", "dim")
                )
            if opt_segments:
                console.print(
                    Text.assemble(("  optim:   ", "dim"), *opt_segments)
                )
        elif cmd == "/good":
            if self._pending_slugs:
                await self._attribute_pending(
                    success=True, quiet=False
                )
            else:
                console.print(
                    "  [dim]nothing pending to rate[/dim]"
                )
        elif cmd == "/bad":
            if self._pending_slugs:
                await self._attribute_pending(
                    success=False, quiet=False
                )
            else:
                console.print(
                    "  [dim]nothing pending to rate[/dim]"
                )
        elif cmd == "/model":
            if not arg:
                console.print(
                    f"  [dim]current model: {self.model}[/dim]"
                )
            else:
                self._switch_model(arg)
        elif cmd == "/clear":
            self.session_id = new_id()
            self.last_plan = None
            self._compact_tokens = 0
            self._compact_exchanges.clear()
            reset_paste_stash()
            # Move the on-disk pointer to the NEW session so a
            # later /resume doesn't rewind into the conversation
            # the user just told us to forget. /clear means "I
            # want a fresh start," and that should survive a
            # quit + relaunch.
            self._save_session_pointer()
            console.print(
                "  [dim]fresh conversation — prior turns "
                "forgotten[/dim]"
            )
        elif cmd == "/compress_token_length":
            self._handle_compress_command(arg)
        elif cmd == "/set_model":
            await self._handle_set_model()
        elif cmd == "/set_web":
            await self._handle_set_web()
        elif cmd == "/resume":
            await self._handle_resume()
        elif cmd == "/set_continue_cap":
            self._handle_set_continue_cap(arg)
        elif cmd == "/effort":
            self._handle_effort(arg)
        elif cmd == "/isolate":
            self._handle_isolate()
        elif cmd == "/review":
            self._handle_review()
        elif cmd == "/merge":
            self._handle_merge()
        elif cmd == "/discard":
            self._handle_discard()
        elif cmd == "/mcp":
            await self._handle_mcp()
        else:
            console.print(
                f"  [yellow]unknown command {cmd}[/yellow] — "
                "/help for the list"
            )
        return True

    async def _handle_mcp(self) -> None:
        """List the connected MCP servers + their tools.

        Reads the registry stashed on the coordinator by ``build_agent``.
        Connecting is lazy, so this is the first thing that actually
        opens the sessions — surfaces a misconfigured server here rather
        than mid-task."""
        registry = getattr(self.agent, "_mcp_registry", None)
        if registry is None:
            console.print(
                "  [dim]No MCP servers configured. Add an [[mcp]] block "
                "to .loom/settings.toml (or ~/.loom-code/settings.toml) "
                "and restart.[/dim]"
            )
            return
        names = registry.server_names
        console.print(
            f"  [cyan]MCP servers[/cyan] ({len(names)}): "
            f"{', '.join(names) if names else '—'}"
        )
        try:
            tools = await registry.list_tools()  # lazily connects
        except Exception as exc:  # noqa: BLE001 — surface, don't crash
            console.print(
                f"  [red]failed to list MCP tools:[/red] {exc}"
            )
            return
        if not tools:
            console.print("  [dim]no tools exposed yet.[/dim]")
            return
        console.print(f"  [cyan]tools[/cyan] ({len(tools)}):")
        for t in tools:
            desc = (t.description or "").strip().splitlines()
            first = desc[0] if desc else ""
            console.print(f"    [green]{t.name}[/green]  [dim]{first}[/dim]")

    def _switch_model(self, model: str) -> None:
        """Rebuild the agent on a new model. Keeps the project +
        approval gate; starts a fresh conversation since the new
        model has no history of the old one. The compactor uses
        the new model too; ``_compact_threshold`` stays as-is so a
        user override survives a model switch (auto = -1 just
        recomputes against the new model on the next check)."""
        # Ensure we have a key for the NEW model before
        # constructing — otherwise build_agent crashes inside the
        # provider SDK on a missing key. ensure_key_for_model
        # prompts inline + saves so the switch just works.
        if not ensure_key_for_model(model, console):
            console.print(
                "  [yellow]model switch cancelled — staying on "
                f"{self.model}[/yellow]"
            )
            return
        self.model = model
        self._rebuild_agent()
        console.print(
            f"  [dim]switched to {model} — fresh conversation[/dim]"
        )

    def _handle_effort(self, arg: str) -> None:
        """``/effort [low|medium|high|off]`` — set the reasoning-effort
        dial + rebuild. No arg shows the current value. ``off`` (or
        ``none``/``default``) clears it back to the provider default.
        Effort only affects reasoning-capable models (Claude extended
        thinking, OpenAI o-series); it's inert on gpt-4.1/4o."""
        choice = arg.strip().lower()
        if not choice:
            console.print(
                f"  [dim]current effort: "
                f"{self._effort or 'default'}[/dim] "
                "[dim](usage: /effort low|medium|high|off)[/dim]"
            )
            return
        if choice in ("off", "none", "default"):
            new_effort: str | None = None
        elif choice in ("low", "medium", "high"):
            new_effort = choice
        else:
            console.print(
                f"  [yellow]unknown effort {choice!r}[/yellow] — "
                "use low | medium | high | off"
            )
            return
        if new_effort == self._effort:
            console.print(
                f"  [dim]effort already {new_effort or 'default'}[/dim]"
            )
            return
        self._effort = new_effort
        self._rebuild_agent()
        console.print(
            f"  [dim]reasoning effort → {new_effort or 'default'} "
            "— fresh conversation[/dim]"
        )

    # ---- session worktree isolation -----------------------------------

    def _handle_isolate(self) -> None:
        """``/isolate`` — run this session in its own git worktree so
        its edits can't collide with another loom-code session on the
        same repo (e.g. a second terminal). Rebuilds the agent rooted
        in the worktree; /merge or /discard finishes."""
        if self._worktree is not None:
            console.print(
                f"  [dim]already isolated on "
                f"{self._worktree.branch}[/dim]"
            )
            return
        if not worktree.is_git_repo(self.project.root):
            console.print("  [yellow]/isolate needs a git repo[/yellow]")
            return
        info, err = worktree.create(self.project.root, self.session_id)
        if info is None:
            console.print(f"  [red]isolate failed:[/red] {err}")
            return
        self._worktree = info
        self._isolated_project = detect_project(info.path)
        self._rebuild_agent()
        console.print(
            f"  [dim]isolated → worktree on [cyan]{info.branch}[/cyan] "
            f"(base {info.base}). Edits stay here until "
            "/merge or /discard.[/dim]"
        )

    def _handle_review(self) -> None:
        """``/review`` — show this isolated session's diff vs its base
        branch (read-only)."""
        if self._worktree is None:
            console.print("  [dim]not isolated — /isolate first[/dim]")
            return
        text, err = worktree.diff(self._worktree)
        if err:
            console.print(f"  [red]diff failed:[/red] {err}")
            return
        if not text.strip():
            console.print("  [dim]no changes in this session yet[/dim]")
            return
        self._print_diff(text)

    def _handle_merge(self) -> None:
        """``/merge`` — review the session's diff, then commit + merge
        its branch into base and return to the main tree."""
        if self._worktree is None:
            console.print("  [dim]not isolated — nothing to merge[/dim]")
            return
        text, _ = worktree.diff(self._worktree)
        if text.strip():
            self._print_diff(text)
        else:
            console.print("  [dim](no changes to merge)[/dim]")
        info = self._worktree
        ok, err = worktree.merge(self.project.root, info)
        if not ok:
            console.print(f"  [red]merge failed:[/red] {err}")
            return
        worktree.remove(self.project.root, info)
        self._worktree = None
        self._isolated_project = None
        self._rebuild_agent()
        console.print(
            f"  [dim]merged [cyan]{info.branch}[/cyan] → {info.base} "
            "+ cleaned up — back on the main tree[/dim]"
        )

    def _handle_discard(self) -> None:
        """``/discard`` — drop this isolated session's edits + remove
        the worktree, returning to the main tree."""
        if self._worktree is None:
            console.print("  [dim]not isolated — nothing to discard[/dim]")
            return
        info = self._worktree
        worktree.remove(self.project.root, info)
        self._worktree = None
        self._isolated_project = None
        self._rebuild_agent()
        console.print(
            f"  [dim]discarded [cyan]{info.branch}[/cyan] — back on "
            "the main tree[/dim]"
        )

    def _print_diff(self, text: str) -> None:
        """Print a unified diff with green/red/hunk colours — same
        vocabulary as the desktop's review modal + edit cards."""
        for raw in text.splitlines():
            if raw.startswith(("+++", "---")):
                style = "dim"
            elif raw.startswith("@@"):
                style = "cyan"
            elif raw.startswith("diff --git") or raw.startswith("index "):
                style = "bold dim"
            elif raw.startswith("+"):
                style = "green"
            elif raw.startswith("-"):
                style = "red"
            else:
                style = "default"
            console.print(Text(raw or " ", style=style))

    def _rebuild_agent(self) -> None:
        """Reconstruct the supervisor + workers using the current
        ``self.model`` and ``self._web_backend``. Used by
        ``/model`` (model change) and ``/set_web`` (backend change).
        Bundled skills (graphify et al.) are auto-registered
        inside ``build_agent`` so we don't pass them explicitly
        here."""
        # When isolated, build rooted at the worktree (its own working
        # copy + .loom). Extensions stay ``self._extensions`` — they're
        # the MAIN project's .loom config, which the worktree (being
        # gitignored) doesn't have a copy of, so an isolated session
        # would otherwise lose its skills/subagents/hooks.
        build_project = self._isolated_project or self.project
        self.agent, self.workspace = build_agent(
            build_project,
            model=self.model,
            approval_handler=self._gate.handler,
            web_backend=self._web_backend,
            max_stop_hook_iterations=self._auto_continue_limit,
            extensions=self._extensions,
            effort=self._effort,
        )
        self._compactor = Compactor(model=self.model)
        self._compact_tokens = 0
        self._compact_exchanges.clear()
        self.session_id = new_id()

    # ---- persistent status line ----------------------------------------

    def _print_status_line(self) -> None:
        """One dim line printed before every prompt so cost/tokens
        are always visible. Format mirrors the end-of-turn summary
        (``uncached+cached in / out · $cost``) for consistency —
        but represents cumulative SESSION totals here, not per-run."""
        uncached = self.total_in - self.total_cached_in
        console.print(
            Text.assemble(
                ("  ", ""),
                (f"{self.turns} turns", "dim"),
                ("  ·  ", "dim"),
                (
                    f"{uncached:,}+{self.total_cached_in:,} in / "
                    f"{self.total_out:,} out",
                    "dim",
                ),
                ("  ·  ", "dim"),
                (f"${self.total_cost:.4f}", "dim green"),
            )
        )

    # ---- automatic compaction ------------------------------------------

    def _active_threshold(self) -> int:
        """Resolve the live threshold:

        * positive int  → explicit user override (set via
          ``/compress_token_length N``)
        * 0             → user disabled compaction (``... off``)
        * -1 (sentinel) → recompute from the active model
        """
        if self._compact_threshold >= 0:
            return self._compact_threshold
        return default_compact_threshold(self.model)

    async def _maybe_compact(self) -> None:
        """If cumulative tokens have crossed the active threshold,
        run the compactor, write its summary to the agent's memory
        as a working block (auto-injected into every subsequent
        system prompt), and reset the conversation thread."""
        threshold = self._active_threshold()
        if threshold == 0:
            return  # explicitly disabled
        if self._compact_tokens < threshold:
            return
        if not self._compact_exchanges:
            return

        before_tokens = self._compact_tokens
        console.print(
            f"  [dim]compacting {before_tokens:,} tokens of "
            f"history (threshold {threshold:,})...[/dim]"
        )
        try:
            summary = await self._compactor.compact(
                self._compact_exchanges
            )
        except Exception as exc:  # noqa: BLE001 — never fatal
            console.print(
                f"  [yellow]compaction failed: {exc} — continuing "
                "without it (use /clear if you hit context "
                "limits)[/yellow]"
            )
            return

        if not summary:
            return

        # Land the summary as a working block. loomflow auto-
        # injects working blocks into every subsequent system
        # prompt, so the next turn starts on a fresh session_id
        # but immediately "remembers" the session via this block.
        try:
            await self.agent.memory.update_block(
                "session_summary", summary, user_id=_USER_ID
            )
        except Exception as exc:  # noqa: BLE001 — never fatal
            console.print(
                f"  [yellow]could not write summary to memory: "
                f"{exc}[/yellow]"
            )
            return

        self.session_id = new_id()
        self._compact_tokens = 0
        self._compact_exchanges.clear()
        console.print(
            f"  [dim]compacted into {len(summary)}-char summary "
            f"in memory; new conversation thread.[/dim]"
        )

    def _handle_set_continue_cap(self, arg: str) -> None:
        """``/set_continue_cap [N]`` — view or set the auto-continue cap.

        Bare ``/set_continue_cap`` shows the current value. ``N=0``
        disables auto-continue entirely (turns become single-shot
        again — useful when debugging a model's behaviour and you
        want to see exactly what it does on its own). Otherwise N
        is the new cap; we clamp at 100 to prevent typos like
        ``/set_continue_cap 1000`` from costing the user real money.
        """
        arg = arg.strip()
        if not arg:
            console.print(
                f"  [dim]auto-continue cap: "
                f"[b]{self._auto_continue_limit}[/b]  "
                f"(default {_AUTO_CONTINUE_LIMIT_DEFAULT}, "
                "0 disables)[/dim]"
            )
            return
        try:
            n = int(arg)
        except ValueError:
            console.print(
                "  [yellow]usage: /set_continue_cap <N> — N is "
                "an integer ≥ 0 (0 disables)[/yellow]"
            )
            return
        if n < 0:
            console.print(
                "  [yellow]cap must be non-negative (use 0 to "
                "disable auto-continue)[/yellow]"
            )
            return
        if n > 100:
            console.print(
                "  [yellow]cap clamped to 100 to prevent runaway "
                "cost on a typo. Use /set_continue_cap 100 if you "
                "really meant that.[/yellow]"
            )
            n = 100
        old = self._auto_continue_limit
        self._auto_continue_limit = n
        # The cap is a construction-time kwarg on loomflow's Agent
        # (max_stop_hook_iterations). Rebuild so the new value
        # takes effect; this also resets the conversation, which
        # matches the rebuild semantics of /model and /set_web.
        self._rebuild_agent()
        if n == 0:
            console.print(
                f"  [dim]auto-continue [b red]disabled[/b red]  "
                f"(was {old}). Multi-step plans now stop after "
                "their first ReAct exit; type 'continue' to nudge "
                "manually.[/dim]"
            )
        else:
            console.print(
                f"  [dim]auto-continue cap: [b]{old}[/b] → "
                f"[b green]{n}[/b green][/dim]"
            )

    def _handle_compress_command(self, arg: str) -> None:
        """Dispatch ``/compress_token_length`` — view, set, auto, off."""
        arg = arg.strip().lower()
        if not arg:
            current = self._active_threshold()
            mode = (
                "off (disabled)"
                if self._compact_threshold == 0
                else (
                    f"user-set ({current:,})"
                    if self._compact_threshold > 0
                    else f"auto ({current:,}, "
                    f"80% of {self.model}'s context window)"
                )
            )
            console.print(
                f"  [dim]compaction threshold: {mode}[/dim]\n"
                f"  [dim]used this session so far: "
                f"{self._compact_tokens:,} tokens[/dim]"
            )
            return
        if arg == "auto":
            self._compact_threshold = -1
            console.print(
                f"  [dim]threshold: auto "
                f"({self._active_threshold():,})[/dim]"
            )
            return
        if arg == "off":
            self._compact_threshold = 0
            console.print(
                "  [dim]auto-compaction disabled[/dim]"
            )
            return
        try:
            n = int(arg.replace(",", "").replace("_", ""))
        except ValueError:
            console.print(
                "  [yellow]usage: /compress_token_length <N> | "
                "auto | off[/yellow]"
            )
            return
        if n <= 0:
            console.print(
                "  [yellow]threshold must be positive (use 'off' "
                "to disable)[/yellow]"
            )
            return
        self._compact_threshold = n
        console.print(
            f"  [dim]threshold set to {n:,} tokens[/dim]"
        )

    # ---- /set_model + /set_web (interactive provider setup) ----------

    async def _prompt_line(self, message: str) -> str | None:
        """Read one line from the user with a fresh PromptSession.

        We deliberately do NOT reuse ``self._prompt_session`` here.
        prompt_toolkit's PromptSession holds state on its instance
        (``is_password``, completers, key bindings) and even though
        ``prompt_async`` is supposed to save/restore per-call,
        empirically the redact-mode leaked into the next main-loop
        prompt after the secret prompt returned. A throwaway
        session per inline question keeps the main REPL's session
        pristine.

        Returns ``None`` on EOF / Ctrl-C so callers can treat the
        cancel path uniformly."""
        try:
            return (
                await PromptSession().prompt_async(message)
            ).strip()
        except (EOFError, KeyboardInterrupt):
            return None

    async def _prompt_secret(self, message: str) -> str | None:
        """Same as ``_prompt_line`` but hides the input —
        ``is_password=True`` makes prompt_toolkit redact keystrokes
        (no terminal echo, no shell history). Same fresh-session
        rationale as ``_prompt_line`` — and ESPECIALLY important
        here, because this is the prompt whose state was leaking
        back into the main REPL."""
        try:
            return (
                await PromptSession().prompt_async(
                    message, is_password=True
                )
            ).strip()
        except (EOFError, KeyboardInterrupt):
            return None

    async def _handle_set_model(self) -> None:
        """``/set_model`` — pick OpenAI / Anthropic, prompt for the
        API key if not already set, save it to credentials, switch
        to that provider's default model. Convenient for the
        "first run on a new machine" flow."""
        console.print()
        console.print("  [bold]Pick a model provider:[/bold]")
        console.print(
            "    [cyan]1[/cyan]. OpenAI     "
            "(gpt-4.1-mini, gpt-4.1, ...)"
        )
        console.print(
            "    [cyan]2[/cyan]. Anthropic  "
            "(claude-sonnet-4-6, claude-opus-4-7, ...)"
        )
        choice = await self._prompt_line("  Enter 1 or 2: ")
        if choice is None:
            console.print("  [dim]cancelled[/dim]")
            return
        if choice == "1":
            env_name = "OPENAI_API_KEY"
            target_model = _OPENAI_DEFAULT_MODEL
            label = "OpenAI"
        elif choice == "2":
            env_name = "ANTHROPIC_API_KEY"
            target_model = _ANTHROPIC_DEFAULT_MODEL
            label = "Anthropic"
        else:
            console.print(
                f"  [yellow]invalid choice {choice!r} — "
                "enter 1 or 2[/yellow]"
            )
            return

        if not os.environ.get(env_name):
            console.print(
                f"  [dim]No {env_name} set yet — paste one to "
                f"save it for future sessions too.[/dim]"
            )
            key = await self._prompt_secret(
                f"  Paste your {env_name}: "
            )
            if not key:
                console.print(
                    "  [yellow]no key entered — aborting[/yellow]"
                )
                return
            save_credential(env_name, key)
            os.environ[env_name] = key
            console.print(
                f"  [green]✓[/green] saved {env_name} "
                "(future sessions pick it up automatically)"
            )
        else:
            console.print(
                f"  [dim]{env_name} already set — using it[/dim]"
            )
        console.print(
            f"  [dim]switching to {label}'s default model "
            f"({target_model})[/dim]"
        )
        self._switch_model(target_model)

    async def _handle_set_web(self) -> None:
        """``/set_web`` — pick a web-search backend (or disable).
        Serper prompts for the API key on first use; DuckDuckGo
        needs nothing. Rebuilds the agent so the new tool wiring
        takes effect on the next turn."""
        console.print()
        console.print("  [bold]Web search backend:[/bold]")
        console.print(
            "    [cyan]1[/cyan]. Serper      "
            "(Google, best quality, needs API key)"
        )
        console.print(
            "    [cyan]2[/cyan]. DuckDuckGo  "
            "(free, no key, lower quality)"
        )
        console.print(
            "    [cyan]3[/cyan]. Off         (disable web search)"
        )
        choice = await self._prompt_line("  Enter 1, 2, or 3: ")
        if choice is None:
            console.print("  [dim]cancelled[/dim]")
            return

        if choice == "1":
            # Serper needs SERPER_API_KEY. Prompt if missing,
            # save it so future sessions pick it up.
            if not os.environ.get("SERPER_API_KEY"):
                console.print(
                    "  [dim]Get a key at "
                    "https://serper.dev "
                    "(2,500 lifetime free searches).[/dim]"
                )
                key = await self._prompt_secret(
                    "  Paste your SERPER_API_KEY: "
                )
                if not key:
                    console.print(
                        "  [yellow]no key entered — "
                        "aborting[/yellow]"
                    )
                    return
                save_credential("SERPER_API_KEY", key)
                os.environ["SERPER_API_KEY"] = key
                console.print(
                    "  [green]✓[/green] saved SERPER_API_KEY"
                )
            self._web_backend = "serper"
        elif choice == "2":
            self._web_backend = "duckduckgo"
        elif choice == "3":
            self._web_backend = None
        else:
            console.print(
                f"  [yellow]invalid choice {choice!r} — "
                "enter 1, 2, or 3[/yellow]"
            )
            return

        self._rebuild_agent()
        state = self._web_backend or "off"
        console.print(
            f"  [dim]web search: {state} — "
            "fresh conversation[/dim]"
        )

    # ---- /resume --------------------------------------------------------

    def _session_pointer_path(self) -> Path:
        """Where we stash the last-used session_id for this project.

        Lives under ``.loom/`` (same dir loom-code already uses for
        per-project state — notebook, memory db, repo map).
        One file per project, single line: the session_id ULID.
        """
        return self.project.root / ".loom" / "last_session.txt"

    def _save_session_pointer(self) -> None:
        """Write the current ``session_id`` to the project's
        ``.loom/last_session.txt``. Best-effort — a write failure
        is logged once but never blocks a turn (the file is a
        convenience; the agent's actual memory keys off
        ``session_id`` in loomflow's Memory which we don't touch
        here)."""
        try:
            p = self._session_pointer_path()
            p.parent.mkdir(exist_ok=True)
            p.write_text(self.session_id + "\n", encoding="utf-8")
        except OSError:
            # Silent failure: a read-only filesystem or perms
            # issue would otherwise spam the chat with the same
            # warning every turn.
            pass

    def _load_session_pointer(self) -> str | None:
        """Read the last saved session_id for this project, or
        ``None`` if no prior session has been recorded yet (first
        run on this project)."""
        try:
            p = self._session_pointer_path()
            if not p.exists():
                return None
            value = p.read_text(encoding="utf-8").strip()
            return value or None
        except OSError:
            return None

    async def _handle_resume(self) -> None:
        """``/resume`` — point the REPL at the LAST session_id used
        on this project.

        loomflow's Memory keys episodes by ``(user_id, session_id)``;
        when the agent's next ``run()`` reuses the same session_id,
        loomflow rehydrates the prior turns into the conversation
        context for free. We don't need to do any rehydration here
        — just swap the id and let loomflow do its thing.

        Edge case: the saved session_id might be from a /clear
        boundary (i.e. the user explicitly told us to forget) or
        from a different model. We don't try to guard against
        either — /resume is a deliberate gesture the user owns.
        """
        prior = self._load_session_pointer()
        if prior is None:
            console.print(
                "  [yellow]no prior session recorded for this "
                "project — nothing to resume.[/yellow]"
            )
            console.print(
                "  [dim](sessions are saved per project after each "
                "turn — your first task here starts a fresh one.)"
                "[/dim]"
            )
            return
        if prior == self.session_id:
            console.print(
                "  [dim]you're already on the latest session "
                f"({prior[:8]}…) — nothing to resume.[/dim]"
            )
            return
        # Swap. Reset the compaction state so we don't blend the
        # newly-resumed session with whatever happened in this
        # REPL launch before /resume was called.
        old = self.session_id
        self.session_id = prior
        self._compact_tokens = 0
        self._compact_exchanges.clear()

        # Legacy data migration — loom-code pre-0.10.18 ran the
        # Router in ``per_route`` mode, so episodes were stored
        # under ``{prior}__route_simple`` / ``{prior}__route_complex``,
        # NOT under ``prior`` itself. Post-upgrade we run
        # ``conversation_scope='shared'`` which keys rehydration on
        # ``prior`` — so a /resume to a pre-upgrade session loses
        # all context unless we migrate.
        #
        # One-shot UPDATE in the sqlite db (loom-code hardcodes the
        # sqlite backend). Idempotent — a post-upgrade session has
        # nothing under the derived names. Episode_tool_transcripts
        # cascades via episode_id, so no separate migration needed.
        migrated = _migrate_legacy_per_route_episodes(
            self.project.root / LOOM_DIR / "memory.db", prior
        )
        if migrated:
            console.print(
                f"  [dim]migrated {migrated} legacy per-route "
                "episode(s) into the shared session for "
                "rehydration[/dim]"
            )

        console.print(
            f"  [green]✓[/green] resumed session [cyan]{prior[:8]}…"
            f"[/cyan] (was on {old[:8]}…)"
        )
        console.print(
            "  [dim]loomflow will rehydrate prior turns from "
            "memory on your next task.[/dim]"
        )

        # Surface the last N turns of the resumed session so the
        # user has visual context of WHAT they're resuming. Without
        # this, /resume is invisible — user has no way to confirm
        # the rehydration actually picked up real content vs an
        # empty session id, and no way to catch a wrong-session
        # mistake before they type the next prompt.
        await self._render_resumed_history_preview(prior)

    async def _render_resumed_history_preview(
        self, session_id: str
    ) -> None:
        """Fetch + render the last 5 turn groups from the resumed
        session so the user sees what they're inheriting. Silently
        no-ops when the memory backend doesn't expose
        ``session_messages`` (some custom backends don't) or the
        session is empty."""
        try:
            messages = await self.agent._memory.session_messages(
                session_id, user_id=_USER_ID, limit=100
            )
        except (AttributeError, TypeError):
            return
        if not messages:
            return
        turn_groups = _group_messages_into_turns(messages)
        if not turn_groups:
            return
        raw_count = len(turn_groups)
        # Collapse consecutive identical (user, assistant) pairs
        # into one row with a repeat count — without this, runs of
        # "user typed the same thing twice" or "stop-hook re-fired
        # the same prompt" produce visual noise in the preview.
        collapsed = _collapse_consecutive_duplicate_turns(
            turn_groups
        )
        recent = collapsed[-5:]
        skipped = raw_count - sum(r[3] for r in recent)
        console.print()
        title = (
            f"history (last {len(recent)} of {raw_count} "
            "turns — agent sees the full set)"
        )
        rule = "─" * max(0, 64 - len(title) - 4)
        console.print(f"  [dim]── {title} {rule}[/dim]")
        for user_prompt, assistant_text, n_tool_calls, repeats in recent:
            console.print()
            u = _truncate_one_line(user_prompt, 140)
            repeat_tag = f" [dim](×{repeats})[/dim]" if repeats > 1 else ""
            console.print(
                f"  [bold]user:[/bold] {u}{repeat_tag}"
            )
            a = _truncate_one_line(assistant_text, 200)
            if a:
                console.print(f"  [dim]loom:[/dim] {a}")
            else:
                console.print(
                    "  [dim]loom: (no text response)[/dim]"
                )
            if n_tool_calls:
                console.print(
                    f"        [dim]({n_tool_calls} tool call"
                    f"{'s' if n_tool_calls != 1 else ''})[/dim]"
                )
        console.print(f"  [dim]{'─' * 68}[/dim]")
        if skipped > 0:
            console.print(
                f"  [dim]+ {skipped} earlier turn(s) recovered "
                "(visible to the agent, not shown here)[/dim]"
            )


def _truncate_one_line(text: str, max_chars: int) -> str:
    """Collapse to one line + cap length. For the /resume history
    preview where multi-line messages would blow the layout."""
    if not text:
        return ""
    first = text.replace("\r", " ").strip()
    # Collapse all whitespace runs to a single space so multi-line
    # responses fit on one line cleanly.
    first = " ".join(first.split())
    if len(first) <= max_chars:
        return first
    return first[: max_chars - 1].rstrip() + "…"


def _collapse_consecutive_duplicate_turns(
    groups: list[tuple[str, str, int]],
) -> list[tuple[str, str, int, int]]:
    """Collapse runs of consecutive identical
    ``(user_prompt, assistant_text)`` turn groups into one entry
    annotated with a repeat count.

    Used by the /resume history preview to dedupe the visual when
    the user (or a prior framework version's stop-hook re-prompt)
    persisted the same exchange multiple times in a row. Three
    consecutive identical groups collapse to one ``(user, asst,
    n_tool, repeats=3)`` row; non-consecutive duplicates are kept
    as separate rows (different points in the conversation should
    show separately even if identical).

    ``n_tool`` from the FIRST occurrence is preserved — the
    assumption being that all collapsed copies had the same
    tool-call shape (they had identical assistant text, so
    almost certainly identical tools).
    """
    if not groups:
        return []
    out: list[tuple[str, str, int, int]] = []
    cur_user, cur_asst, cur_tools = groups[0]
    repeats = 1
    for user, asst, tools in groups[1:]:
        if user == cur_user and asst == cur_asst:
            repeats += 1
        else:
            out.append((cur_user, cur_asst, cur_tools, repeats))
            cur_user, cur_asst, cur_tools = user, asst, tools
            repeats = 1
    out.append((cur_user, cur_asst, cur_tools, repeats))
    return out


def _group_messages_into_turns(
    messages: list[Any],
) -> list[tuple[str, str, int]]:
    """Walk a rehydrated message list and group it into the
    natural ``(user_prompt, assistant_text, n_tool_calls)`` shape
    used by the /resume preview.

    Each USER message starts a new turn group; ASSISTANT messages
    contribute their text content + tool_call count to the
    currently-open group; TOOL result messages are folded into the
    current group's tool-call count too (they're the other half of
    a tool_call pair). SYSTEM messages are ignored — they're
    framework context, not conversation.

    Returns groups in source order (oldest first). Empty list for
    a message stream with no USER turns.
    """
    groups: list[tuple[str, str, int]] = []
    cur_user: str | None = None
    cur_assistant: list[str] = []
    cur_tool_calls = 0
    for m in messages:
        role = getattr(m, "role", None)
        # Role enum values are lowercase strings: 'user', 'assistant',
        # 'tool', 'system'. Some custom backends may pass plain strings.
        role_s = str(role).lower().split(".")[-1]
        content = str(getattr(m, "content", "") or "")
        if role_s == "user":
            # Close the previous group if any.
            if cur_user is not None:
                groups.append((
                    cur_user,
                    " ".join(cur_assistant).strip(),
                    cur_tool_calls,
                ))
            cur_user = content
            cur_assistant = []
            cur_tool_calls = 0
        elif role_s == "assistant":
            if content:
                cur_assistant.append(content)
            tool_calls = getattr(m, "tool_calls", None) or ()
            cur_tool_calls += len(tool_calls)
        elif role_s == "tool":
            # Tool result — counts as part of the open group's
            # tool activity. We don't double-count vs the
            # assistant's tool_calls list (which counted CALLS);
            # the tool message is the RESULT of one of those.
            # Skipping it avoids 2x-ing the displayed count.
            pass
        # SYSTEM messages: drop, not user-facing.
    # Close the final group.
    if cur_user is not None:
        groups.append((
            cur_user,
            " ".join(cur_assistant).strip(),
            cur_tool_calls,
        ))
    return groups


# Phrases a hallucinated "I'm done" turn uses. Matched against the
# agent's output when the turn made ZERO tool calls. Deliberately
# narrow — we want completion CLAIMS, not legitimate no-tool
# answers ("here's what X means"). Each pattern is "verb of
# completion + object of work".
_COMPLETION_CLAIM_RE = re.compile(
    r"\b("
    r"all (the )?(detected |previously )?(issues|problems|"
    r"bugs|fixes)\b.{0,40}\b(fixed|addressed|resolved|done)"
    r"|already been fixed"
    r"|have been fixed"
    r"|were fixed"
    r"|no (remaining |outstanding )?(issues|problems|blockers)"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)


def _looks_like_completion_claim(text: str) -> bool:
    """True if ``text`` reads like "I finished the work" — used to
    detect hallucinated completion claims on zero-tool-call turns.
    Narrow on purpose: a normal answer that happens to say 'fixed'
    once shouldn't trip it, but 'all the detected issues have been
    fixed' should."""
    if not text:
        return False
    return _COMPLETION_CLAIM_RE.search(text) is not None


def _delete_last_episode(
    db_path: Path, *, session_id: str, user_id: str
) -> bool:
    """Delete the most-recently-persisted episode for
    ``(user_id, session_id)``. Used by the anti-poison gate to
    remove a just-written no-tool-call completion claim before it
    pollutes recall.

    Direct sqlite (loom-code hardcodes the sqlite backend) because
    the Memory protocol's ``forget`` is coarse (by user/session/
    time, not 'the single most-recent row'). Returns True if a row
    was deleted. Best-effort — swallows errors so a gate failure
    never breaks the turn.
    """
    if not db_path.is_file():
        return False
    try:
        import sqlite3
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.cursor()
            # Find the most-recent episode id for this scope, then
            # delete by id (episode_tool_transcripts cascades via
            # the episode_id FK).
            cur.execute(
                "SELECT id FROM episodes "
                "WHERE user_id = ? AND session_id = ? "
                "ORDER BY occurred_at DESC LIMIT 1",
                (user_id, session_id),
            )
            row = cur.fetchone()
            if row is None:
                return False
            cur.execute(
                "DELETE FROM episodes WHERE id = ?", (row[0],)
            )
            conn.commit()
            return (cur.rowcount or 0) > 0
    except (sqlite3.Error, OSError):
        return False


def _migrate_legacy_per_route_episodes(
    db_path: Path, parent_session_id: str
) -> int:
    """Re-key any legacy per-route episodes into the parent
    session_id so ``conversation_scope='shared'`` rehydration sees
    them.

    Pre-0.10.18 loom-code ran the Router in default ``per_route``
    mode, persisting episodes under ``{parent}__route_simple`` and
    ``{parent}__route_complex``. The new shared-mode lookup keys on
    ``parent`` alone, so /resume'd pre-upgrade sessions had no
    visible history. This UPDATE rewrites the session_id column for
    any matching legacy rows. Idempotent — re-running on a
    post-upgrade session is a no-op.

    Returns the number of rows migrated. Silently no-ops when the
    db file is absent or unreadable — failure here must NEVER
    block /resume.

    Why direct sqlite (not via the Memory protocol): the Memory
    protocol exposes ``remember(Episode)`` and ``session_messages``
    but no primitive for ``rekey-session``. Adding one to the
    framework just to satisfy this one-shot loom-code migration
    isn't worth the surface. We know the backend is sqlite (the
    REPL hardcodes it) and the column name is stable.
    """
    if not db_path.is_file():
        return 0
    legacy_simple = f"{parent_session_id}__route_simple"
    legacy_complex = f"{parent_session_id}__route_complex"
    try:
        import sqlite3
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE episodes SET session_id = ? "
                "WHERE session_id IN (?, ?)",
                (parent_session_id, legacy_simple, legacy_complex),
            )
            migrated = cur.rowcount or 0
            conn.commit()
            return int(migrated)
    except (sqlite3.Error, OSError):
        return 0


async def run_repl(project: Project, model: str) -> int:
    """Entry point for the interactive REPL — construct the Repl and
    run its loop until the user exits."""
    repl = Repl(project, model)
    return await repl.run()
