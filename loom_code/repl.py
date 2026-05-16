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
from datetime import UTC
from pathlib import Path

from loomflow import new_id
from prompt_toolkit import HTML, PromptSession
from prompt_toolkit.completion import (
    CompleteEvent,
    Completer,
    Completion,
)
from prompt_toolkit.document import Document
from rich.text import Text

from .agent import build_agent
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
  [cyan]/loominit[/cyan]        index this codebase → LOOM.md so the
                   agent can skip re-reading source on every task
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

# Prompt fed in on each auto-continue iteration. Matches the
# coordinator's own "LOOP" workflow step (step 5 in prompts.py)
# so it lands as a natural next-instruction, not a directive
# from outside its mental model.
_AUTO_CONTINUE_PROMPT = (
    "[auto-continue] Your living plan still has non-done steps. "
    "Continue with the next `todo` or `doing` step right now. "
    "Do NOT respond to the user yet — only respond when every "
    "plan step is `done`, `skipped`, or `blocked`."
)


def _count_plan_remaining(
    plan_steps: list[dict] | None = None,
    plan_text: str | None = None,
) -> int:
    """How many plan steps are NOT yet done/skipped/blocked?

    Two input shapes, in preference order:

    1. ``plan_steps`` — structured list captured from the
       ``plan_write`` tool_call args. Loomflow's stable API; no
       parsing brittleness. Always prefer this when available.
    2. ``plan_text`` — fallback. Regex-parses the rendered
       markdown's ``**Progress:** X/Y done`` line. Used when the
       renderer didn't observe a structured ``plan_write`` event
       this run (e.g. the agent updated an existing plan via a
       different path).

    Returns 0 if neither is available, neither parses, or the
    plan is fully drained. Used by the auto-continue logic in
    ``_turn`` to decide whether to iterate.
    """
    if plan_steps is not None:
        # Structured path — exact, no regex. Steps in todo/doing
        # state count as "remaining work"; done/skipped/blocked
        # don't.
        return sum(
            1
            for s in plan_steps
            if isinstance(s, dict)
            and s.get("status") in ("todo", "doing")
        )
    if not plan_text:
        return 0
    import re

    match = re.search(
        r"Progress:?\s*\*?\*?\s*(\d+)\s*/\s*(\d+)\s+done",
        plan_text,
    )
    if not match:
        return 0
    done = int(match.group(1))
    total = int(match.group(2))
    return max(0, total - done)


def should_auto_continue(
    *,
    remaining: int,
    previous_remaining: int | None,
    iterations_used: int,
    limit: int,
) -> tuple[bool, str]:
    """Decide whether to fire another auto-continue iteration.

    Returns ``(continue?, reason)``. When ``continue=False`` the
    reason is one of ``"plan_drained"`` / ``"cap_reached"`` /
    ``"stalled"`` — useful for both UI (tell the user WHY we
    stopped) and the telemetry log.

    Stall detection: if the plan's remaining count did NOT
    decrease between iterations, the model is talking but not
    making progress. Bailing early saves cost — the next 3
    iterations would almost certainly be the same.
    """
    if remaining <= 0:
        return False, "plan_drained"
    if iterations_used >= limit:
        return False, "cap_reached"
    if (
        previous_remaining is not None
        and remaining >= previous_remaining
        and iterations_used > 0
    ):
        return False, "stalled"
    return True, ""


def _log_auto_continue(
    *,
    root: Path,
    session_id: str,
    iteration: int,
    limit: int,
    remaining: int,
    previous_remaining: int | None,
    decision: str,
) -> None:
    """Append one telemetry line to ``.loom/auto_continue.log``.

    Used to tune the cap + diagnose stuck plans. One JSON object
    per line so the file is append-only-safe and easy to grep /
    parse later (``jq`` / ``cat`` both work). Best-effort: any
    IO failure is silently swallowed — telemetry must never block
    the agent."""
    import json
    from datetime import datetime

    try:
        target = root / ".loom" / "auto_continue.log"
        target.parent.mkdir(exist_ok=True)
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "session_id": session_id,
            "iter": iteration,
            "limit": limit,
            "remaining": remaining,
            "previous_remaining": previous_remaining,
            "decision": decision,
        }
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        pass


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
    ("/loominit", "index this codebase → LOOM.md"),
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
        self.agent, self.workspace = build_agent(
            project, model=model, approval_handler=self._gate.handler
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
        self.total_out = 0
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
        # Auto-continue cap. Mutable per session via
        # /set_continue_cap so users can dial it up for very large
        # scaffolds or down to disable. Stall detection still kicks
        # in regardless of this value, so a higher cap doesn't
        # mean "burn money on a stuck plan" — it means "give a
        # progressing plan more headroom before we bail."
        self._auto_continue_limit = _AUTO_CONTINUE_LIMIT_DEFAULT
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
        """The REPL loop. Returns an exit code."""
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
            "codebase so future turns skip re-reading source[/dim]"
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

        # The Ralph-loop wrapper: ReAct exits when the model emits
        # text without a tool call, even mid-plan. We check after
        # each stream completes — if the living plan still has
        # todo/doing steps, we feed an auto-continue prompt and
        # iterate. Bounded by _AUTO_CONTINUE_LIMIT, and we bail
        # early if the plan didn't progress between iterations
        # (model is talking but not making bookkeeping moves).
        current_prompt = prompt
        auto_continues_used = 0
        agent_output_last = ""
        previous_remaining: int | None = None

        while True:
            try:
                async for event in self.agent.stream(
                    current_prompt,
                    user_id=_USER_ID,
                    session_id=self.session_id,
                ):
                    renderer.handle(event)
            except KeyboardInterrupt:
                pause_status()
                console.print(
                    "\n[yellow]interrupted — turn abandoned[/yellow]"
                )
                return
            except BaseExceptionGroup as eg:
                # anyio's task groups raise ``ExceptionGroup`` when
                # any child task fails. The default str() reads
                # "unhandled errors in a TaskGroup (1 sub-exception)"
                # — uninformative. Unwrap to surface the REAL
                # cause(s) so the user can act on them.
                pause_status()
                for inner in _flatten_exception_group(eg):
                    console.print(
                        f"\n[bold red]error: "
                        f"{type(inner).__name__}: {inner}[/bold red]"
                    )
                return
            except Exception as exc:  # noqa: BLE001 — REPL must survive
                pause_status()
                console.print(
                    f"\n[bold red]error: "
                    f"{type(exc).__name__}: {exc}[/bold red]"
                )
                return

            if renderer.last_plan:
                self.last_plan = renderer.last_plan
            result = renderer.last_result
            agent_output = ""
            if result:
                self.total_cost += float(result.get("cost_usd", 0.0))
                tin = int(result.get("tokens_in", 0))
                cached_in = int(result.get("cached_tokens_in", 0))
                tout = int(result.get("tokens_out", 0))
                self.total_in += tin + cached_in
                self.total_cached_in += cached_in
                self.total_out += tout
                self.turns += int(result.get("turns", 0))
                # Hold this turn's citations for the moved-on
                # heuristic. Updated on each iteration so the
                # final pending_slugs reflect the LAST sub-turn.
                self._pending_slugs = list(
                    result.get("cited_slugs") or []
                )
                self._compact_tokens += tin + cached_in + tout
                agent_output = str(result.get("output") or "")
            agent_output_last = agent_output

            # Compute remaining from structured args FIRST (stable
            # loomflow API), fall back to markdown parsing for
            # cases where the renderer didn't observe a structured
            # plan_write this iteration.
            remaining = _count_plan_remaining(
                plan_steps=renderer.last_plan_steps,
                plan_text=self.last_plan,
            )

            # Skip auto-continue entirely if the run crashed
            # silently (no result). Retrying would just crash again.
            if result is None:
                break

            decision_continue, reason = should_auto_continue(
                remaining=remaining,
                previous_remaining=previous_remaining,
                iterations_used=auto_continues_used,
                limit=self._auto_continue_limit,
            )
            _log_auto_continue(
                root=self.project.root,
                session_id=self.session_id,
                iteration=auto_continues_used,
                limit=self._auto_continue_limit,
                remaining=remaining,
                previous_remaining=previous_remaining,
                decision=("continue" if decision_continue else reason),
            )

            if decision_continue:
                auto_continues_used += 1
                previous_remaining = remaining
                pause_status()
                console.print(
                    f"\n  [dim magenta]▸ plan has {remaining} step(s) "
                    f"remaining — auto-continuing "
                    f"({auto_continues_used}/"
                    f"{self._auto_continue_limit})[/dim magenta]"
                )
                current_prompt = _AUTO_CONTINUE_PROMPT
                set_status("loomflowing...")
                continue

            # Stopped — tell the user WHY when the reason is
            # actionable (cap hit / stalled). plan_drained =
            # everything's fine; no message needed.
            if reason == "cap_reached":
                console.print(
                    f"\n  [yellow]plan still has {remaining} "
                    f"step(s) but hit auto-continue cap "
                    f"({self._auto_continue_limit}) — type "
                    "'continue' to push further, raise the cap "
                    "with /set_continue_cap N, or accept the "
                    "partial result[/yellow]"
                )
            elif reason == "stalled":
                console.print(
                    f"\n  [yellow]plan didn't progress between "
                    f"iterations (still {remaining} step(s) "
                    "remaining) — stopping. Type 'continue' to "
                    "nudge the agent if you think it can recover, "
                    "or /bad to flag the run.[/yellow]"
                )
            break

        pause_status()
        self._compact_exchanges.append((prompt, agent_output_last))

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
        ``self.model`` and ``self._web_backend``. Used by both
        ``/model`` (changes the model) and ``/set_web`` (changes
        the web backend) — same rebuild semantics, single source
        of truth for what a "rebuild" entails."""
        self.agent, self.workspace = build_agent(
            self.project,
            model=self.model,
            approval_handler=self._gate.handler,
            web_backend=self._web_backend,
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
        console.print(
            f"  [green]✓[/green] resumed session [cyan]{prior[:8]}…"
            f"[/cyan] (was on {old[:8]}…)"
        )
        console.print(
            "  [dim]loomflow will rehydrate prior turns from "
            "memory on your next task.[/dim]"
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
                f"[dim]annotating {len(index.clusters)} cluster(s) "
                "via the model — this is the LLM-cost step…[/dim]"
            )
            metadata = _read_pyproject_metadata(self.project.root)
            body = await annotate(
                index,
                model=self.model,
                project_metadata=metadata,
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
        console.print(
            "  [dim]future turns will reference this file. Re-run "
            "[cyan]/loominit[/cyan] after major refactors.[/dim]"
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
    """Entry point for the interactive REPL."""
    repl = Repl(project, model)
    return await repl.run()
