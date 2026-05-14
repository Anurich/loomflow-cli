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

import anyio
from loomflow import new_id
from rich.prompt import Prompt
from rich.text import Text

from .agent import build_agent
from .approval import ApprovalGate
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
  [cyan]/exit[/cyan]            leave (Ctrl-D also works)

Anything else is a task — loom-code plans, codes, and verifies it.
Notes the agent reads get credited when a turn goes well, so it
gets sharper at THIS repo over time.
"""

_USER_ID = "loom-code"


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
        # Session accumulators.
        self.total_cost = 0.0
        self.total_in = 0
        self.total_out = 0
        self.turns = 0
        self.last_plan: str | None = None
        # Self-improvement: cited slugs from the last turn, awaiting
        # a success/failure judgement (the moved-on heuristic).
        self._pending_slugs: list[str] = []

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
            try:
                line = await anyio.to_thread.run_sync(self._read_line)
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

    def _read_line(self) -> str:
        """Blocking prompt — runs on a worker thread so the anyio
        loop stays free."""
        return Prompt.ask("[bold green]loom[/bold green]")

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
        if result:
            self.total_cost += float(result.get("cost_usd", 0.0))
            self.total_in += int(result.get("tokens_in", 0))
            self.total_in += int(result.get("cached_tokens_in", 0))
            self.total_out += int(result.get("tokens_out", 0))
            self.turns += int(result.get("turns", 0))
            # Hold this turn's citations for the moved-on heuristic.
            self._pending_slugs = list(
                result.get("cited_slugs") or []
            )
        if self._pending_slugs:
            console.print(
                "  [dim]if that worked, just continue — or "
                "/bad if it didn't[/dim]"
            )
        console.print()

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
            console.print(
                Text.assemble(
                    ("  session: ", "dim"),
                    (f"{self.turns} turns", ""),
                    ("  ·  ", "dim"),
                    (
                        f"{self.total_in:,} in / "
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
            console.print(
                "  [dim]fresh conversation — prior turns "
                "forgotten[/dim]"
            )
        else:
            console.print(
                f"  [yellow]unknown command {cmd}[/yellow] — "
                "/help for the list"
            )
        return True

    def _switch_model(self, model: str) -> None:
        """Rebuild the agent on a new model. Keeps the project +
        approval gate; starts a fresh conversation since the new
        model has no history of the old one."""
        self.model = model
        self.agent, self.workspace = build_agent(
            self.project,
            model=model,
            approval_handler=self._gate.handler,
        )
        self.session_id = new_id()
        console.print(
            f"  [dim]switched to {model} — fresh conversation[/dim]"
        )


async def run_repl(project: Project, model: str) -> int:
    """Entry point for the interactive REPL."""
    repl = Repl(project, model)
    return await repl.run()
