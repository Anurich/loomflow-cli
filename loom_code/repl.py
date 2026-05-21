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

from .agent import LOOM_DIR, build_agent
from .approval import ApprovalGate
from .compact import Compactor, default_compact_threshold
from .credentials import (
    ensure_key_for_model,
    save_credential,
)
from .paste import (
    build_paste_keybindings,
    expand_pastes,
    reset_paste_stash,
)
from .project import Project
from .render import StreamRenderer, banner, console

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
  [cyan]/loominit[/cyan]        index this codebase → LOOM.md + a
                   bundled knowledge graph (graphify) so the agent
                   can skip re-reading source AND has cross-file
                   structural queries available from turn one
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
    ("/plan", "show the current plan, or start one"),
    ("/cost", "session cost + token totals"),
    ("/good", "mark the last turn useful (credit notes)"),
    ("/bad", "mark the last turn unhelpful"),
    ("/model", "switch to a specific model by name"),
    ("/set_model", "pick OpenAI or Anthropic + save API key"),
    ("/set_web", "enable web search (Serper / DuckDuckGo / off)"),
    ("/loominit", "index this codebase → LOOM.md + knowledge graph"),
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
        # ApprovalGate persists across turns so 'allow all' sticks
        # for the whole session.
        self._gate = ApprovalGate()
        self._auto_continue_limit = _AUTO_CONTINUE_LIMIT_DEFAULT
        # Graphify and other bundled skills are auto-registered
        # by build_agent (see _bundled_skill_paths). No per-session
        # toggle needed — the agent decides when to load skills.
        self.agent, self.workspace = build_agent(
            project,
            model=model,
            approval_handler=self._gate.handler,
            max_stop_hook_iterations=self._auto_continue_limit,
        )
        # LOOM.md retrieval. ``LoomRetriever.from_repo_root`` returns
        # ``None`` when LOOM.md is missing or empty — we treat that as
        # "no per-turn injection, fall back to whatever the static
        # context block in project.context_text already wired." When
        # the retriever loads, the per-turn ``_inject_loom_context``
        # call before ``_turn`` updates the ``loom_index`` working
        # block.
        #
        # ``mode`` is pulled from the coordinator
        # (``_loom_retrieval_mode``, stamped in ``build_agent``) so
        # the retriever's strategy stays in sync with whether the
        # agent has the ``read_loom_section`` tool. ``"agentic"``
        # injects a stable TOC (good for cache hits); ``"bm25"``
        # (default) keeps the per-turn keyword-ranked section dump.
        from .loominit.injection import LoomRetriever
        mode = getattr(self.agent, "_loom_retrieval_mode", "bm25")
        self._loom_retriever = LoomRetriever.from_repo_root(
            project.root, mode=mode
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
        spawning, no subprocess lifecycle to manage here. The
        run-method's ``finally`` block used to ``aclose()`` an MCP
        subprocess from the prior MCP-based graphify integration;
        that's no longer needed now that graphify is a bundled
        skill (Mode B Python tools, in-process).
        """
        return await self._run_inner()

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
        console.print(
            "  [dim]▸ [cyan]/loominit[/cyan]      index this "
            "codebase + build a knowledge graph (so future turns "
            "skip re-reading source AND get structural queries)"
            "[/dim]"
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
        console.print()

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
                    console.print("[dim]bye[/dim]")
                    return 0
                continue

            # A new task with no prior complaint → the previous
            # turn must have been fine. Credit it, then run.
            await self._attribute_pending(success=True, quiet=False)
            # Per-turn LOOM.md retrieval — populates the
            # ``loom_index`` working block with sections BM25-relevant
            # to ``line``. Loomflow auto-injects working blocks into
            # the next system prompt.
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
        """Update the ``loom_index`` working block with the top-N
        LOOM.md sections most relevant to ``prompt``.

        No-op when there's no retriever (LOOM.md absent) or when BM25
        scores no overlap with the prompt — in both cases we leave
        the prior block untouched (loomflow keeps the last written
        value), so a follow-up turn like ``"why?"`` still has the
        previous turn's context to lean on.

        Failures are swallowed (never let memory I/O kill a turn) —
        the same defensive pattern the session-summary writer uses.
        """
        if self._loom_retriever is None:
            return
        body = self._loom_retriever.relevant(prompt)
        if not body:
            return
        try:
            await self.agent.memory.update_block(
                "loom_index", body, user_id=_USER_ID
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

        # Escalation: the SIMPLE coder called ``escalate_to_team``.
        # Re-dispatch the same prompt through the supervisor — which
        # inherits SIMPLE's partial conversation via the shared
        # session_id (conversation_scope="shared"), so the work
        # SIMPLE already did becomes the team's context, not waste.
        # The supervisor has no escalate tool, so this can't loop.
        if renderer.escalation_requested:
            sup = getattr(self.agent, "_complex_agent", None)
            pause_status()
            reason = renderer.escalation_reason or "(no reason given)"
            console.print(
                f"\n  [yellow]→ escalating to the team:[/yellow] "
                f"{reason}"
            )
            if sup is not None:
                team_renderer = StreamRenderer(
                    set_status=set_status, pause_status=pause_status
                )
                ok = await self._consume_agent_stream(
                    sup, prompt, team_renderer, pause_status
                )
                if not ok:
                    return
                # The team's run is now the authoritative result for
                # cost accounting + output + the rest of the turn.
                renderer = team_renderer
            else:
                console.print(
                    "  [dim](no team agent wired — staying in "
                    "simple mode)[/dim]"
                )

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
        elif cmd == "/loominit":
            await self._handle_loominit()
        elif cmd == "/resume":
            await self._handle_resume()
        elif cmd == "/set_continue_cap":
            self._handle_set_continue_cap(arg)
        else:
            console.print(
                f"  [yellow]unknown command {cmd}[/yellow] — "
                "/help for the list"
            )
        return True

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

    def _rebuild_agent(self) -> None:
        """Reconstruct the supervisor + workers using the current
        ``self.model`` and ``self._web_backend``. Used by
        ``/model`` (model change) and ``/set_web`` (backend change).
        Bundled skills (graphify et al.) are auto-registered
        inside ``build_agent`` so we don't pass them explicitly
        here."""
        self.agent, self.workspace = build_agent(
            self.project,
            model=self.model,
            approval_handler=self._gate.handler,
            web_backend=self._web_backend,
            max_stop_hook_iterations=self._auto_continue_limit,
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
        per-project state — notebook, memory db, LOOM.md index).
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

    # ---- /loominit ------------------------------------------------------

    async def _handle_loominit(self) -> None:
        """Run the full codebase-indexing pipeline.

        Phase 1 (structural, sub-second): walk the repo, AST-parse
        every .py, build the symbol + import graph, score by
        PageRank, mine entry points, cluster by path. Writes
        ``.loom/index.json``.

        Phase 2 (LLM annotation, parallel): for the project as a
        whole + each cluster, spawn a loomflow ``Agent`` (no tools,
        ``output_schema``-validated JSON) that produces the
        narrative chunk. Stitched into ``LOOM.md`` at the repo root.

        Costs one model run per cluster + one for the overview —
        bounded by ``annotator.DEFAULT_CONCURRENCY``. Re-run via
        ``/loominit`` any time; surgical refresh / staleness
        tracking lands in later slices.
        """
        # Imports kept local so a slim ``loom-code`` startup doesn't
        # pull in the AST walker + pydantic models for users who
        # never run /loominit.
        from .loominit.annotator import annotate
        from .loominit.extractor import build_index
        from .loominit.persistence import (
            markdown_path,
            save_index,
            write_markdown,
        )
        from .skills.graphify.tools import (
            GraphifyBuildResult,
            graphify_build_impl,
        )

        console.print()
        console.print(
            "  [dim]indexing codebase (this may take a few "
            "seconds for the structural pass + 5-30s for the "
            "LLM annotation)…[/dim]"
        )
        status = console.status(
            "[dim]building structural index…[/dim]", spinner="dots"
        )
        status.start()
        # Graph build runs BEFORE annotation so its summary can be
        # embedded into LOOM.md by the assembler. Wrapped in its own
        # try/except so a graphify failure (e.g. tree-sitter missing
        # for an exotic file mix) doesn't kill the loominit pass —
        # LOOM.md still gets written, just without the
        # ``## Knowledge Graph`` section.
        graphify_result: GraphifyBuildResult | None = None
        graphify_error: str | None = None
        try:
            index = build_index(self.project.root)
            if not index.files:
                status.stop()
                console.print(
                    "  [yellow]no indexable source files found — "
                    "nothing to do[/yellow]"
                )
                return
            save_index(self.project.root, index)

            status.update(
                "[dim]building knowledge graph "
                "(graphify — tree-sitter + Leiden)…[/dim]"
            )
            try:
                graphify_result = await graphify_build_impl(
                    self.project.root
                )
            except Exception as gx:  # noqa: BLE001
                # Capture for a status line; do NOT abort loominit.
                graphify_error = f"{type(gx).__name__}: {gx}"

            status.update(
                f"[dim]annotating {len(index.clusters)} cluster(s) "
                "via the model — this is the LLM-cost step…[/dim]"
            )
            metadata = _read_pyproject_metadata(self.project.root)
            graphify_section: str | None = None
            if (
                graphify_result is not None
                and graphify_result.skipped_reason is None
            ):
                graphify_section = _render_graphify_section(
                    graph_rel_path=str(
                        graphify_result.graph_path.relative_to(
                            graphify_result.project_root
                        )
                    ),
                    n_nodes=graphify_result.n_nodes,
                    n_edges=graphify_result.n_edges,
                    n_communities=graphify_result.n_communities,
                    source=graphify_result.source,
                )
            body = await annotate(
                index,
                model=self.model,
                project_metadata=metadata,
                graphify_section=graphify_section,
            )
            write_markdown(self.project.root, body)
        except Exception as exc:  # noqa: BLE001 — REPL must survive
            status.stop()
            console.print(
                f"  [bold red]/loominit failed: {exc}[/bold red]"
            )
            return
        finally:
            status.stop()

        n_symbols = len(index.symbols)
        n_clusters = len(index.clusters)
        n_files = len(index.files)
        out_path = markdown_path(self.project.root)
        console.print(
            f"  [green]✓[/green] wrote [cyan]{out_path.name}[/cyan] "
            f"— {n_files} files, {n_symbols} symbols, "
            f"{n_clusters} cluster(s)"
        )
        # Separate status line for graphify — three branches: ✓
        # (built), ⚠ (skipped — no extractable files), ✗ (error).
        if (
            graphify_result is not None
            and graphify_result.skipped_reason is None
        ):
            rel = graphify_result.graph_path.relative_to(
                graphify_result.project_root
            )
            console.print(
                f"  [green]✓[/green] wrote [cyan]{rel}[/cyan] "
                f"— {graphify_result.n_nodes} nodes, "
                f"{graphify_result.n_edges} edges, "
                f"{graphify_result.n_communities} communities "
                f"(via {graphify_result.source})"
            )
        elif graphify_result is not None and graphify_result.skipped_reason:
            console.print(
                f"  [yellow]graphify: skipped — "
                f"{graphify_result.skipped_reason}[/yellow]"
            )
        elif graphify_error is not None:
            console.print(
                f"  [yellow]graphify: failed — {graphify_error}"
                "[/yellow] (LOOM.md still written)"
            )
        console.print(
            "  [dim]future turns will reference this file. Re-run "
            "[cyan]/loominit[/cyan] after major refactors.[/dim]"
        )
        # Install the debounced post-commit hook so the structural
        # index refreshes itself every N commits. ``LOOM.md`` stays
        # whatever the annotator wrote; the structural refresh
        # updates ``.loom/index.json`` so per-turn injection picks
        # up file changes + the inline ``(stale: path:line)``
        # markers in LOOM.md surface when annotated claims drift.
        # Idempotent + git-aware (silent skip on non-git).
        from .git_hook import install as install_hook
        hook_status = install_hook(self.project.root)
        if hook_status in ("installed", "updated"):
            console.print(
                f"  [dim]post-commit hook {hook_status} "
                "— structural index refreshes every 5 commits[/dim]"
            )

        # Rebuild the LoomRetriever so the next turn picks up the
        # freshly-written LOOM.md. Without this, ``/loominit`` on a
        # fresh repo writes the file but ``self._loom_retriever``
        # stays at whatever ``__init__`` set it to (typically
        # ``None`` on a first-time launch), and the agentic TOC
        # injection never fires until the REPL is restarted. Mode
        # is pulled from the coordinator so it stays in sync with
        # ``build_agent(loom_retrieval=...)`` — same logic as the
        # initial build in ``__init__``.
        from .loominit.injection import LoomRetriever
        mode = getattr(self.agent, "_loom_retrieval_mode", "bm25")
        self._loom_retriever = LoomRetriever.from_repo_root(
            self.project.root, mode=mode
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


def _render_graphify_section(
    *,
    graph_rel_path: str,
    n_nodes: int,
    n_edges: int,
    n_communities: int,
    source: str,
) -> str:
    """Format graphify build stats into the body of LOOM.md's
    ``## Knowledge Graph`` section. Returned text gets the heading
    added by ``_assemble_markdown``; we just supply the body.

    Why this lives in the REPL (not the annotator): the annotator
    deliberately doesn't import ``skills/graphify/`` so it stays a
    standalone module. The REPL owns the cross-package wiring.
    Takes plain primitives — the caller unpacks the
    ``GraphifyBuildResult`` so this function has no coupling to the
    graphify skill's types.

    The body is structured as load-bearing context for the agent:
    artifact path + counts up top (so the agent knows the graph
    exists and how big it is) followed by per-tool usage hints (so
    the agent picks the right call without needing to
    ``load_skill('graphify')`` first to read the docstrings). That's
    the whole efficiency win — graph tools become discoverable from
    the always-injected LOOM.md, not only from the skill listing.
    """
    return (
        f"Pre-built knowledge graph at `{graph_rel_path}` "
        f"({n_nodes} nodes, {n_edges} edges, "
        f"{n_communities} communities — generated by "
        f"graphify, AST-only, deterministic, no LLM cost). "
        f"Source files discovered via `{source}`.\n\n"
        "**When to query the graph instead of grepping**:\n"
        "- `graphify__query(question)` — BFS from nodes matching "
        "question keywords. Use for *structural* questions like "
        "\"what's involved in auth\" or \"which symbols touch the "
        "config loader\" — graph returns the neighbourhood, grep "
        "returns string hits.\n"
        "- `graphify__path(a, b)` — shortest path between two named "
        "concepts. The one graph query grep genuinely can't do: "
        "\"how does A reach B in this codebase?\".\n"
        "- `graphify__explain(node)` — one-symbol report (source "
        "file/line, neighbours, community). Faster than reading the "
        "file when the user asks \"what is X\".\n"
        "- `graphify__build(path)` — rebuild from scratch. Already "
        "run by `/loominit`; only call manually after a large "
        "refactor (the post-commit hook auto-refreshes every 5 "
        "commits).\n\n"
        "**When NOT to use it**: single-file content questions (use "
        "`read` / `grep`), or asking for raw source text (the graph "
        "stores *structure*, not bodies). For one-file changes, "
        "skip the graph entirely — it costs context without adding "
        "anything."
    )


def _read_pyproject_metadata(repo_root) -> dict[str, str]:
    """Best-effort read of pyproject.toml ``[project]`` metadata —
    feeds the annotator's overview prompt. Quiet failure mode:
    returns ``{}`` on any read/parse error; the annotator handles
    missing fields gracefully."""
    import tomllib

    pyproject = repo_root / "pyproject.toml"
    if not pyproject.exists():
        return {}
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError):
        return {}
    project = data.get("project", {})
    if not isinstance(project, dict):
        return {}
    out: dict[str, str] = {}
    for key in ("name", "description"):
        val = project.get(key)
        if isinstance(val, str):
            out[key] = val
    req = project.get("requires-python")
    if isinstance(req, str):
        out["requires_python"] = req
    return out


async def run_repl(project: Project, model: str) -> int:
    """Entry point for the interactive REPL. Graphify is opt-in
    via ``/graphify on`` from within the session."""
    repl = Repl(project, model)
    return await repl.run()
