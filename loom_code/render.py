"""Render loomflow ``Event``s to the terminal with ``rich``.

``Agent.stream()`` yields ``Event`` objects — ``model_chunk``,
``tool_call``, ``tool_result``, ``permission_ask``, ``completed``,
``error``, etc. This module turns that event stream into the
live terminal UI. It is PURELY presentation — no agent logic.

Event payloads are plain dicts; we ``.get()`` everything
defensively so a payload-shape change in loomflow degrades to a
slightly-uglier line instead of a crash.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text

console = Console()

# Tools whose results are worth showing in full-ish; others get a
# one-line summary so the terminal doesn't flood. We cap BOTH char
# and line count — a single long line (jq output, minified JSON,
# a big SQL row) blows past the char cap with no newlines, and a
# multi-line directory listing exceeds the line cap before chars.
# Truncate on whichever hits first; the trailer says how much was
# elided in BOTH dimensions so the user knows the scale.
_VERBOSE_RESULT_TOOLS = {"read", "grep", "ls", "find"}
_RESULT_PREVIEW_CHARS = 300
_RESULT_PREVIEW_LINES = 8
# Verbose tools (read/grep/ls/find) get this multiplier — they
# legitimately produce more useful long output.
_VERBOSE_MULTIPLIER = 3


def _truncate_preview(
    text: str, *, char_cap: int, line_cap: int
) -> str:
    """Cap a tool-result preview at BOTH a character count AND a
    line count — whichever hits first. Returns the truncated text
    with a trailer naming what was elided.

    Char cap alone is wrong: a one-line minified JSON blob hides
    far less terminal real estate than 20 lines of grep output at
    the same character count. Line cap alone is wrong: a single
    unwrapped line can be thousands of characters. Use both."""
    if not text:
        return ""
    lines = text.splitlines()
    n_lines = len(lines)
    n_chars = len(text)
    # Decide which cap is more restrictive for this output.
    line_truncated = n_lines > line_cap
    char_truncated = n_chars > char_cap
    if not line_truncated and not char_truncated:
        return text
    # Take the smaller of (line_cap lines, char_cap chars).
    by_lines = "\n".join(lines[:line_cap])
    capped = by_lines[:char_cap] if len(by_lines) > char_cap else by_lines
    elided_lines = n_lines - capped.count("\n") - 1
    elided_chars = n_chars - len(capped)
    parts = []
    if elided_lines > 0:
        parts.append(f"+{elided_lines} lines")
    if elided_chars > 0:
        parts.append(f"+{elided_chars} chars")
    trailer = ", ".join(parts) if parts else "truncated"
    return f"{capped}\n    … ({trailer})"


def _status_label_for(tool: str, args: dict[str, Any]) -> str:
    """Pick a human spinner label for an in-flight tool call.

    delegate gets the worker name (loom-code's main workflow shape
    so it's worth a custom label); bash gets a short clip of the
    actual command; web_search shows the query; everything else
    falls back to ``running <tool>...``."""
    if tool == "delegate":
        target = str(args.get("target") or args.get("agent") or "?")
        return f"delegating to {target}..."
    if tool == "bash":
        cmd = str(args.get("command") or "").strip().splitlines()[0:1]
        first = cmd[0] if cmd else ""
        clip = first[:40] + ("…" if len(first) > 40 else "")
        return f"running: {clip}" if clip else "running bash..."
    if tool == "web_search":
        q = str(args.get("query") or "").strip()
        return f"searching: {q[:40]}..." if q else "searching the web..."
    return f"running {tool}..."


class StreamRenderer:
    """Stateful renderer for one ``Agent.stream`` run.

    Tracks whether we're mid-model-chunk so tool calls don't
    interleave awkwardly with streamed assistant text. Also
    captures the last living-plan render + the final result dict
    so the REPL can power ``/plan`` and ``/cost`` without
    re-parsing the stream.
    """

    def __init__(
        self,
        *,
        set_status: Callable[[str], None] | None = None,
        pause_status: Callable[[], None] | None = None,
    ) -> None:
        """``set_status(label)`` / ``pause_status()`` are optional
        callbacks the REPL wires to a Rich ``console.status``
        spinner. The renderer pauses the spinner while assistant
        prose is streaming (it would corrupt the same line) and
        sets a new label on each tool boundary so the user has
        something to read between events. Both default to no-op so
        non-REPL callers (one-shot CLI) keep working unchanged."""
        self._in_text = False
        # Whether ANY assistant prose has been shown this run — drives
        # the _on_completed fallback (print result["output"] only if
        # nothing streamed, so we never double-print).
        self._any_text = False
        # Buffer for one prose burst — chunks accumulate here, then
        # `_end_text` renders the whole thing as Markdown. We don't
        # stream char-by-char because Markdown can't be rendered
        # incrementally (headings / code-fences / list items need
        # the surrounding context to format correctly). The spinner
        # gives feedback during the wait.
        self._text_buffer: list[str] = []
        # call_id -> tool name. loomflow's `tool_result` event only
        # carries `call_id` (ToolResult has no `tool` field) — the
        # tool NAME is only on the `tool_call` event. We bridge the
        # two so result rendering can tell which tool ran.
        self._call_names: dict[str, str] = {}
        # Spinner callbacks (see __init__ docstring). Default to
        # no-ops so all the event handlers can call them unconditionally.
        self._set_status: Callable[[str], None] = (
            set_status if set_status is not None else (lambda _t: None)
        )
        self._pause_status: Callable[[], None] = (
            pause_status if pause_status is not None else (lambda: None)
        )
        # REPL-readable after the stream drains:
        self.last_plan: str | None = None
        self.last_result: dict[str, Any] | None = None
        # Captured during the run so cli.py can build the end-of-run
        # summary (files changed / verified / notes captured) without
        # the agent having to report any of it itself.
        self.bash_commands: list[str] = []
        self.notes_written: list[tuple[str, str]] = []  # (kind, title)

    def handle(self, event: Any) -> None:
        """Render a single ``Event``. ``kind`` is a string enum;
        compare against the lowercase values loomflow uses."""
        kind = str(event.kind)
        payload = event.payload or {}
        method = getattr(self, f"_on_{kind}", None)
        if method is not None:
            method(payload)
        # Unknown event kinds are silently ignored — forward-compat
        # with new loomflow event types.

    # ---- text streaming -------------------------------------------------

    def _on_model_chunk(self, payload: dict[str, Any]) -> None:
        # loomflow shape: payload = {"chunk": ModelChunk.model_dump()}
        # where ModelChunk is discriminated by ``kind`` — only the
        # "text" kind carries assistant prose; "tool_call"/"finish"
        # chunks have text=None and are surfaced by other handlers.
        #
        # We BUFFER chunks rather than printing incrementally — once
        # the prose burst ends we render the whole thing as
        # Markdown. Streaming raw text would mean the user sees
        # half-formed markdown (`#`, `**`, etc.) flash by before
        # the renderer kicks in, which is worse UX than the spinner
        # holding for a few seconds while the response completes.
        chunk = payload.get("chunk") or {}
        if chunk.get("kind") != "text":
            return
        text = chunk.get("text") or ""
        if not text:
            return
        if not self._in_text:
            self._in_text = True
        self._any_text = True
        self._text_buffer.append(text)

    def _end_text(self) -> None:
        if not self._in_text:
            return
        # Drain the buffer, render once. The spinner was running
        # the whole time; pause it now so the markdown render
        # doesn't fight for the cursor line.
        full = "".join(self._text_buffer).strip()
        self._text_buffer.clear()
        self._in_text = False
        if full:
            self._pause_status()
            console.print()
            try:
                console.print(Markdown(full))
            except Exception:  # noqa: BLE001 — must survive odd markdown
                # Markdown rendering can throw on pathological input
                # (unclosed fences, weird escapes). Fall back to
                # plain print so we never lose the response.
                console.print(full, markup=False, highlight=False)
        # Prose burst over — bring the spinner back; the next
        # tool_call will overwrite this label with something
        # more specific.
        self._set_status("thinking...")

    # ---- tools ----------------------------------------------------------

    def _on_tool_call(self, payload: dict[str, Any]) -> None:
        self._end_text()
        call = payload.get("call") or {}
        tool = call.get("tool", "?")
        # Remember id -> name so _on_tool_result can name the tool.
        call_id = call.get("id")
        if call_id:
            self._call_names[str(call_id)] = str(tool)
        args = call.get("args") or {}
        # Capture bash commands + note writes for the end-of-run
        # summary cli.py builds. Done here (on the call event) so
        # we have the args; tool_result only carries call_id.
        if tool == "bash":
            cmd = str(args.get("command") or "").strip()
            if cmd:
                self.bash_commands.append(cmd)
        elif tool == "note":
            title = str(args.get("title") or "").strip()
            if title:
                kind = str(args.get("kind") or "").strip()
                self.notes_written.append((kind, title))
        arg_str = _summarise_args(args)
        console.print(
            Text.assemble(
                ("  → ", "bold cyan"),
                (tool, "bold cyan"),
                (f"  {arg_str}", "dim"),
            )
        )
        # Update the spinner label so the user has something to
        # read while this tool runs. delegate is loom-code's
        # workhorse — name the worker; bash gets a clipped command;
        # everything else just shows the tool name.
        self._set_status(_status_label_for(tool, args))

    def _on_tool_result(self, payload: dict[str, Any]) -> None:
        # Tool came back — model is now picking the next move.
        # Reset to the generic label until the next tool_call (or
        # text stream) overrides it.
        self._set_status("thinking...")
        result = payload.get("result") or {}
        # ToolResult has no `tool` field — only `call_id`. Resolve
        # the tool name via the id->name map built from tool_call
        # events; fall back to the raw call_id if we somehow missed
        # the pairing.
        call_id = str(result.get("call_id", ""))
        tool = self._call_names.get(call_id, call_id)
        output = result.get("output")
        ok = result.get("ok", True)
        error = result.get("error")
        if not ok and error:
            console.print(Text(f"    ✗ {error}", style="red"))
            return
        text = str(output) if output is not None else ""
        # Capture the latest living-plan render so the REPL's
        # /plan command can show it after the stream drains.
        is_plan = "plan" in str(tool) and "**GOAL:**" in text
        if is_plan:
            self.last_plan = text
        if not text:
            console.print(Text("    ✓ (no output)", style="dim green"))
            return
        # The living plan is load-bearing context — show it in full,
        # never truncated. Verbose tools (read/grep/ls/find) get a
        # fuller preview; everything else gets a tight one.
        if is_plan:
            indented = "\n".join(
                "    " + ln for ln in text.splitlines()
            )
            console.print(Text(indented, style="dim"))
            return
        is_verbose = any(
            v in str(tool) for v in _VERBOSE_RESULT_TOOLS
        )
        mult = _VERBOSE_MULTIPLIER if is_verbose else 1
        char_cap = _RESULT_PREVIEW_CHARS * mult
        line_cap = _RESULT_PREVIEW_LINES * mult
        preview = _truncate_preview(
            text, char_cap=char_cap, line_cap=line_cap
        )
        indented = "\n".join(
            "    " + ln for ln in preview.splitlines()
        )
        console.print(Text(indented, style="dim"))

    # ---- plan -----------------------------------------------------------

    def _on_architecture_event(self, payload: dict[str, Any]) -> None:
        # ReAct / living-plan emit free-form architecture events.
        # Surface only the ones worth seeing.
        name = payload.get("event") or payload.get("name") or ""
        if "plan" in str(name).lower():
            self._end_text()
            console.print(
                Text(f"  ▸ {name}", style="dim magenta")
            )

    # ---- permission gate ------------------------------------------------

    def _on_permission_ask(self, payload: dict[str, Any]) -> None:
        self._end_text()
        call = payload.get("call") or {}
        console.print(
            Text(
                f"  ⚠ permission requested for "
                f"{call.get('tool', '?')}",
                style="yellow",
            )
        )

    def _on_permission_decision(self, payload: dict[str, Any]) -> None:
        decision = payload.get("decision") or {}
        allowed = decision.get("allow", decision.get("allowed"))
        if allowed is False:
            console.print(Text("  ⚠ denied", style="red"))

    # ---- lifecycle ------------------------------------------------------

    def _on_error(self, payload: dict[str, Any]) -> None:
        self._end_text()
        msg = payload.get("error") or payload.get("message") or "?"
        console.print(Text(f"\n✗ error: {msg}", style="bold red"))

    def _on_completed(self, payload: dict[str, Any]) -> None:
        self._end_text()
        result = payload.get("result") or payload
        self.last_result = result
        # Fallback: if the run produced a final answer but nothing
        # streamed (buffered .run-style emission, or a turn that
        # ended in pure text the chunk handler somehow missed),
        # print the output so the user is never left staring at a
        # blank turn. Guarded by _any_text so streamed prose is
        # never double-printed.
        output = str(result.get("output") or "").strip()
        if output and not self._any_text:
            console.print()
            console.print(output, markup=False, highlight=False)
        # Visual separator between turns. The cost / token numbers
        # used to print here too — they're now owned by the REPL's
        # pre-prompt status line (cumulative) and the CLI's one-
        # shot run summary, so we don't duplicate them here.
        console.print()
        console.rule(style="dim")


def _summarise_args(args: dict[str, Any]) -> str:
    """One-line, length-capped arg summary for a tool-call line."""
    parts: list[str] = []
    for k, v in args.items():
        sv = str(v).replace("\n", " ")
        if len(sv) > 60:
            sv = sv[:60] + "…"
        parts.append(f"{k}={sv}")
    return ", ".join(parts)


def banner(model: str, root: str, is_git: bool) -> None:
    """Print the loom-code startup banner."""
    git_tag = "git" if is_git else "no-git"
    console.print()
    console.print(
        Text.assemble(
            ("loom-code", "bold"),
            ("  ", ""),
            (f"{model}", "cyan"),
            ("  ·  ", "dim"),
            (f"{root}", "dim"),
            ("  ·  ", "dim"),
            (git_tag, "dim"),
        )
    )
    console.print(
        Text("  loomflow-native coding agent", style="dim italic")
    )
    console.print()


def print_code(text: str, lexer: str = "python") -> None:
    """Render a code block with syntax highlighting (used by the
    diff-approval UI in Phase 2)."""
    console.print(Syntax(text, lexer, theme="ansi_dark"))
