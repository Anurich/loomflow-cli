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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (
    ConditionalContainer,
    Float,
    FloatContainer,
    HSplit,
    Layout,
    Window,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.widgets import Frame, TextArea
from rich.console import Console


class _PaneConsole(Console):
    """A Rich console that pushes into the TUI pane after every print,
    so output appears live without the REPL calling ``flush()``."""

    def __init__(self, tui: ChatTUI, **kw: Any) -> None:
        super().__init__(**kw)
        self.__tui = tui

    def print(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        super().print(*args, **kwargs)
        self.__tui.flush()


class ChatTUI:
    """A full-screen chat application. One instance per REPL session."""

    def __init__(self) -> None:
        # Selector state: when active, its bindings win and the input
        # box is ignored. Driven by :meth:`select`; a filter gates
        # which key-binding group is live, so we never reassign
        # ``app.key_bindings`` on a running app (ptk caches the
        # processor — a swap doesn't reliably take effect).
        self._sel_active = False
        self._sel_options: list[tuple[str, str]] = []
        self._sel_idx = 0
        self._sel_future: asyncio.Future[str | None] | None = None
        self._ansi = ""  # accumulated conversation, ANSI-encoded
        self._sink = io.StringIO()  # Rich renders here; flush() drains
        # A console whose every print immediately lands in the pane —
        # so the REPL keeps calling console.print(...) unchanged and the
        # output appears live, no scattered flush() calls. Streaming
        # chunks (console.print(end="")) flush too.
        self.console = _PaneConsole(
            self,
            file=self._sink,
            force_terminal=True,
            color_system="truecolor",
            width=self._term_width(),
        )
        # Status line (spinner replacement) — driven by the renderer's
        # set_status/pause_status; empty string hides it.
        self._status = ""

        # The conversation pane. It renders the TAIL of the ANSI that
        # fits the window, so the newest output is always pinned just
        # above the input box (chat-style). The tail slicing happens in
        # ``_pane_text`` using the last known window height — simple and
        # reliable, where a get_vertical_scroll hook gets clamped by
        # ptk's own scroll logic.
        self._pane_height = 20  # updated from render_info each frame
        self._pane = Window(
            FormattedTextControl(text=self._pane_text, focusable=False),
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
            completer=None,  # set by attach_completer() from the REPL
            complete_while_typing=True,
        )
        # Submitted lines flow through this queue to the REPL loop.
        self._submissions: asyncio.Queue[str | None] = asyncio.Queue()
        self._app = self._build_app()

    # ---- console bridge -------------------------------------------------

    @staticmethod
    def _term_width() -> int:
        import shutil

        return max(40, shutil.get_terminal_size((100, 24)).columns)

    def _pane_text(self) -> Any:
        """Render the TAIL of the conversation that fits the pane, so
        the newest output is pinned just above the input box. Track the
        window height from the previous frame's render_info; on the
        first paint use a sane default."""
        info = self._pane.render_info
        if info is not None:
            try:
                self._pane_height = max(1, info.window_height)
            except Exception:  # noqa: BLE001
                pass
        # Slice to the last N logical lines. Wrapping means a logical
        # line can occupy several rows, so keep a little extra headroom
        # (×1.0 is usually fine since most lines don't wrap); ptk clamps
        # if we overshoot.
        lines = self._ansi.split("\n")
        tail = lines[-self._pane_height:]
        return ANSI("\n".join(tail))

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
        # While a selector is active, this region shows the menu.
        if self._sel_active:
            lines = []
            for i, (_key, label) in enumerate(self._sel_options):
                if i == self._sel_idx:
                    lines.append(
                        f"  <ansicyan><b>❯ {i + 1}. {label}</b>"
                        "</ansicyan>"
                    )
                else:
                    lines.append(
                        f"  <ansibrightblack>  {i + 1}. {label}"
                        "</ansibrightblack>"
                    )
            return HTML("\n".join(lines))
        if not self._status:
            return ""
        return HTML(f"  <ansibrightblack>{self._status}</ansibrightblack>")

    def attach_completer(self, completer: Any) -> None:
        """Wire the REPL's slash/@-mention completer onto the input
        box so typing ``/`` still pops the command menu (parity with
        the inline prompt). Called once by the REPL after construction."""
        self._input.completer = completer
        self._input.control.completer = completer

    # ---- layout ---------------------------------------------------------

    def _build_app(self) -> Application[Any]:
        from prompt_toolkit.filters import Condition

        input_active = Condition(lambda: not self._sel_active)
        sel_active = Condition(lambda: self._sel_active)

        kb = KeyBindings()

        # ---- input-box bindings (live only when NOT selecting) ----
        # Enter with the completion menu open should ACCEPT the
        # completion, not submit — matches the old prompt's feel where
        # picking a slash-command from the menu doesn't fire the turn.
        @kb.add("enter", filter=input_active)
        def _submit(event: Any) -> None:
            buf = self._input.buffer
            if buf.complete_state:
                buf.apply_completion(
                    buf.complete_state.current_completion
                    or buf.complete_state.completions[0]
                )
                return
            text = self._input.text
            self._input.text = ""
            # Echo the submitted message into the pane (chat-style) so
            # the user sees what they sent above the response — the box
            # clears, so without this the message would vanish.
            if text.strip():
                shown = text.strip().replace("\n", "\n  ")
                self._ansi += f"\n\x1b[32m› {shown}\x1b[0m\n"
            self._submissions.put_nowait(text)

        @kb.add("escape", "enter", filter=input_active)  # Alt+Enter
        def _newline(event: Any) -> None:
            self._input.buffer.insert_text("\n")

        @kb.add("c-c", filter=input_active)
        @kb.add("c-d", filter=input_active)
        def _abort(event: Any) -> None:
            if self._input.text:
                self._input.text = ""
                self._app.invalidate()
            else:
                self._submissions.put_nowait(None)

        # ---- selector bindings (live only WHILE selecting) ----
        @kb.add("up", filter=sel_active)
        def _sel_up(event: Any) -> None:
            self._sel_idx = (self._sel_idx - 1) % len(self._sel_options)
            self._app.invalidate()

        @kb.add("down", filter=sel_active)
        def _sel_down(event: Any) -> None:
            self._sel_idx = (self._sel_idx + 1) % len(self._sel_options)
            self._app.invalidate()

        @kb.add("enter", filter=sel_active)
        def _sel_pick(event: Any) -> None:
            self._resolve_select(self._sel_options[self._sel_idx][0])

        @kb.add("escape", filter=sel_active)
        @kb.add("c-c", filter=sel_active)
        def _sel_cancel(event: Any) -> None:
            self._resolve_select(None)

        # number + first-letter hotkeys, gated on the selector filter
        for n in range(1, 10):
            @kb.add(str(n), filter=sel_active)
            def _sel_num(event: Any, _n: int = n) -> None:
                if _n <= len(self._sel_options):
                    self._resolve_select(
                        self._sel_options[_n - 1][0]
                    )

        for letter in "abcdefghijklmnopqrstuvwxyz":
            @kb.add(letter, filter=sel_active)
            def _sel_letter(event: Any, _l: str = letter) -> None:
                for key, _label in self._sel_options:
                    if key and key[0].lower() == _l:
                        self._resolve_select(key)
                        return

        body = HSplit(
            [
                self._pane,  # scrolls, fills available height
                self._status_win,  # status line / selector overlay
                Frame(self._input),  # fixed at the bottom
            ]
        )
        # The completion menu (for ``/`` commands + ``@`` files) floats
        # above the input — this is what makes the popup appear at all;
        # a TextArea's completer is inert without a CompletionsMenu in
        # the layout. Gated so it never shows while a selector is up.
        root = FloatContainer(
            content=body,
            floats=[
                Float(
                    xcursor=True,
                    ycursor=True,
                    content=ConditionalContainer(
                        CompletionsMenu(max_height=12, scroll_offset=1),
                        filter=input_active,
                    ),
                ),
            ],
        )
        return Application(
            layout=Layout(root, focused_element=self._input),
            key_bindings=kb,
            full_screen=True,
            mouse_support=False,
        )

    def _resolve_select(self, value: str | None) -> None:
        fut = self._sel_future
        self._sel_active = False
        self._sel_future = None
        self._status_win.height = D(max=1)
        if fut is not None and not fut.done():
            fut.set_result(value)
        self._app.invalidate()

    # ---- drive ----------------------------------------------------------

    @asynccontextmanager
    async def session(self) -> AsyncIterator[ChatTUI]:
        """Run the full-screen app in the background for the whole
        session. The REPL body runs inside this context; the app
        renders the pane + box the entire time.

        ``patch_stdout`` captures raw stdout/stderr writes from
        libraries (e.g. huggingface's HF_TOKEN notice via graphifyy)
        that bypass the redirected Rich console — without it those
        writes corrupt the full-screen layout (the notice showed up
        INSIDE the input box)."""
        from prompt_toolkit.patch_stdout import patch_stdout

        with patch_stdout(raw=True):
            task = asyncio.create_task(self._app.run_async())
            # Give the app a tick to take the screen before first read.
            await asyncio.sleep(0)
            try:
                yield self
            finally:
                if self._app.is_running:
                    self._app.exit()
                try:
                    await task
                except Exception:  # noqa: BLE001 — teardown safe
                    pass

    async def read_line(self) -> str:
        """Await the next submitted line. Raises EOFError on an empty
        Ctrl-D (mirrors the plain prompt's contract)."""
        text = await self._submissions.get()
        if text is None:
            raise EOFError
        return text

    async def select(
        self,
        title: str,
        options: list[tuple[str, str]],
        *,
        default: int = 0,
    ) -> str | None:
        """An in-app option selector — the approval gate + slash-menu
        picker route through this while the full-screen app owns the
        terminal (a raw-termios selector would fight the app for input).

        Sets the selector state; the app's ``sel_active``-filtered
        bindings drive ↑/↓ + Enter + number/hotkey selection and the
        status region renders the menu. Returns the chosen key, or None
        on cancel (Esc). ``options`` is ``[(key, label), …]``.

        No overlapping selects — the gate/menus are strictly one at a
        time, which matches how they were before."""
        if title:
            self.console.print(f"  [bold]{title}[/bold]")
            self.flush()
        self._sel_options = list(options)
        self._sel_idx = max(0, min(default, len(options) - 1))
        self._sel_future = asyncio.get_event_loop().create_future()
        self._status_win.height = D(max=len(options))
        self._sel_active = True
        self._app.invalidate()
        try:
            return await self._sel_future
        finally:
            # _resolve_select already cleared state; this is the
            # belt-and-suspenders path if the future was cancelled.
            self._sel_active = False
            self._sel_options = []
            self._status_win.height = D(max=1)
            self._app.invalidate()

    @property
    def app(self) -> Application[Any]:
        return self._app
