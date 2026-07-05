"""Full-screen chat TUI — a fixed bottom input box with a scrolling
conversation pane above it (Claude-Code / chatbot layout).

The design that avoids rewriting the render layer AND the REPL loop:

* **Render.** loom-code prints everything through ONE Rich ``console``
  singleton (``render.console``). We point that console at a StringIO
  and, after each print, append the produced ANSI to a prompt_toolkit
  scroll buffer. Every existing print — markdown, ``● loom`` labels,
  cost rules, streaming chunks, tool lines, diffs — lands in the pane
  unchanged.

* **Loop.** The REPL keeps its linear ``while`` loop. The full-screen
  ``Application`` runs for the WHOLE session on a background task; the
  loop's ``_read_line`` just awaits a queue that the app's Enter
  handler feeds. So the app is continuously alive (streaming renders
  live, box stays pinned) while the loop stays linear — no rewrite of
  the 20+ downstream turn/command handlers.

The REPL owns integration:

    tui = ChatTUI()
    render.console = tui.console          # redirect all output
    async with tui.session():             # app runs in the background
        line = await tui.read_line()      # awaits the next submit
        ... run the turn (prints flow to the pane) ...
        tui.flush()                       # push buffered output live
"""

from __future__ import annotations

import asyncio
import io
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.widgets import Frame, TextArea
from rich.console import Console


class ChatTUI:
    """A full-screen chat application. One instance per REPL session."""

    def __init__(self) -> None:
        self._ansi = ""  # accumulated conversation, ANSI-encoded
        self._sink = io.StringIO()  # Rich renders here; flush() drains
        self.console = Console(
            file=self._sink,
            force_terminal=True,
            color_system="truecolor",
            width=self._term_width(),
        )
        # Status line (spinner replacement) — driven by the renderer's
        # set_status/pause_status; empty string hides it.
        self._status = ""

        self._pane = Window(
            FormattedTextControl(
                text=lambda: ANSI(self._ansi), focusable=False
            ),
            wrap_lines=True,
            always_hide_cursor=True,
        )
        self._status_win = Window(
            FormattedTextControl(text=self._status_text),
            height=D(max=1),
        )
        self._input = TextArea(
            prompt=HTML("<ansigreen><b>› </b></ansigreen>"),
            multiline=True,
            wrap_lines=True,
            height=D(min=1, max=8),
            dont_extend_height=True,
        )
        # Submitted lines flow through this queue to the REPL loop.
        self._submissions: asyncio.Queue[str | None] = asyncio.Queue()
        self._app = self._build_app()

    # ---- console bridge -------------------------------------------------

    @staticmethod
    def _term_width() -> int:
        import shutil

        return max(40, shutil.get_terminal_size((100, 24)).columns)

    def flush(self) -> None:
        """Move buffered Rich output into the pane + repaint. The REPL
        calls this after prints; the renderer calls it during streaming
        so tokens appear live."""
        text = self._sink.getvalue()
        if text:
            self._ansi += text
            self._sink.seek(0)
            self._sink.truncate(0)
        if self._app.is_running:
            self._app.invalidate()

    def set_status(self, label: str) -> None:
        self._status = label
        if self._app.is_running:
            self._app.invalidate()

    def clear_status(self) -> None:
        self.set_status("")

    def _status_text(self) -> Any:
        if not self._status:
            return ""
        return HTML(f"  <ansibrightblack>{self._status}</ansibrightblack>")

    # ---- layout ---------------------------------------------------------

    def _build_app(self) -> Application[Any]:
        kb = KeyBindings()

        @kb.add("enter")
        def _submit(event: Any) -> None:
            text = self._input.text
            self._input.text = ""
            self._submissions.put_nowait(text)

        @kb.add("escape", "enter")  # Alt+Enter → newline
        def _newline(event: Any) -> None:
            self._input.buffer.insert_text("\n")

        @kb.add("c-c")
        @kb.add("c-d")
        def _abort(event: Any) -> None:
            if self._input.text:
                self._input.text = ""
                self._app.invalidate()
            else:
                # Signal EOF to the REPL loop; the loop tears down.
                self._submissions.put_nowait(None)

        root = HSplit(
            [
                self._pane,  # scrolls, fills available height
                self._status_win,  # one-line status (spinner slot)
                Frame(self._input),  # fixed at the bottom
            ]
        )
        return Application(
            layout=Layout(root, focused_element=self._input),
            key_bindings=kb,
            full_screen=True,
            mouse_support=False,
        )

    # ---- drive ----------------------------------------------------------

    @asynccontextmanager
    async def session(self) -> AsyncIterator[ChatTUI]:
        """Run the full-screen app in the background for the whole
        session. The REPL body runs inside this context; the app
        renders the pane + box the entire time."""
        task = asyncio.create_task(self._app.run_async())
        # Give the app a tick to take the screen before the first read.
        await asyncio.sleep(0)
        try:
            yield self
        finally:
            if self._app.is_running:
                self._app.exit()
            try:
                await task
            except Exception:  # noqa: BLE001 — teardown must not raise
                pass

    async def read_line(self) -> str:
        """Await the next submitted line. Raises EOFError on an empty
        Ctrl-D (mirrors the plain prompt's contract)."""
        text = await self._submissions.get()
        if text is None:
            raise EOFError
        return text

    @property
    def app(self) -> Application[Any]:
        return self._app
