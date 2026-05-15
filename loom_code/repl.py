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
from .project import Project
from .render import StreamRenderer, banner, console

_SLASH_HELP = """\
[bold]loom-code commands[/bold]
  [cyan]/help[/cyan]            this list
  [cyan]/plan[/cyan] [<task>]   show the current plan — or start one
  [cyan]/cost[/cyan]            session cost + token totals
  [cyan]/good[/cyan]            mark the last turn useful (credits notes)
  [cyan]/bad[/cyan]             mark the last turn unhelpful
  [cyan]/model[/cyan] <name>    switch model (rebuilds the agent)
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
    ("/model", "switch model (rebuilds the agent)"),
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
        # prompt_toolkit drives the input line. complete_while_typing
        # opens the autocomplete menu the moment the user types '/'
        # without any extra keystroke (Tab also still works for
        # explicit completion). History gives free up-arrow recall
        # within the session.
        self._prompt_session: PromptSession[str] = PromptSession(
            completer=_SlashCompleter(),
            complete_while_typing=True,
        )

    async def run(self) -> int:
        """The REPL loop. Returns an exit code."""
        banner(self.model, str(self.project.root), self.project.is_git)
        if self.project.context_file:
            console.print(
                f"  [dim]loaded context: "
                f"{self.project.context_file.name}[/dim]"
            )
        console.print(
            "  [dim]type a task, or /help for commands[/dim]\n"
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
                console.print("\n[dim]bye[/dim]")
                return 0

            line = line.strip()
            if not line:
                continue

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
        session so conversation history carries forward."""
        renderer = StreamRenderer()
        try:
            async for event in self.agent.stream(
                prompt,
                user_id=_USER_ID,
                session_id=self.session_id,
            ):
                renderer.handle(event)
        except KeyboardInterrupt:
            console.print("\n[yellow]interrupted — turn abandoned[/yellow]")
            return
        except Exception as exc:  # noqa: BLE001 — REPL must survive
            console.print(f"\n[bold red]error: {exc}[/bold red]")
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
            # Hold this turn's citations for the moved-on heuristic.
            self._pending_slugs = list(
                result.get("cited_slugs") or []
            )
            # Feed the auto-compactor: count tokens toward the
            # threshold and stash the exchange in case we trigger.
            self._compact_tokens += tin + cached_in + tout
            agent_output = str(result.get("output") or "")
        self._compact_exchanges.append((prompt, agent_output))

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
            console.print(
                "  [dim]fresh conversation — prior turns "
                "forgotten[/dim]"
            )
        elif cmd == "/compress_token_length":
            self._handle_compress_command(arg)
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
        self.model = model
        self.agent, self.workspace = build_agent(
            self.project,
            model=model,
            approval_handler=self._gate.handler,
        )
        self._compactor = Compactor(model=model)
        self._compact_tokens = 0
        self._compact_exchanges.clear()
        self.session_id = new_id()
        console.print(
            f"  [dim]switched to {model} — fresh conversation[/dim]"
        )

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


async def run_repl(project: Project, model: str) -> int:
    """Entry point for the interactive REPL."""
    repl = Repl(project, model)
    return await repl.run()
