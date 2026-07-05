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
from rich.syntax import Syntax
from rich.text import Text


class _ConsoleProxy:
    """A stand-in for the Rich ``console`` that forwards every
    attribute to the *current* target console.

    Why: dozens of modules do ``from .render import console`` at import
    time, binding the name by-reference. When the full-screen TUI turns
    on it needs ALL that output to flow into its scroll pane — but
    reassigning ``render.console`` wouldn't reach those already-bound
    names. A proxy solves it in one place: swap ``proxy._target`` and
    every ``console.print`` in the app follows, no per-module rewiring.
    The default target is a normal stdout console (classic behaviour).
    """

    def __init__(self, target: Console) -> None:
        object.__setattr__(self, "_target", target)

    def _set_target(self, target: Console) -> None:
        object.__setattr__(self, "_target", target)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_target"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_target"), name, value)


_stdout_console = Console()
console = _ConsoleProxy(_stdout_console)


def set_console_target(target: Console) -> None:
    """Point the shared ``console`` proxy at ``target`` (the TUI's
    pane-backed console). Everything printed app-wide now lands there."""
    console._set_target(target)


def reset_console_target() -> None:
    """Restore the default stdout console (TUI teardown / --classic)."""
    console._set_target(_stdout_console)

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
        sandbox: bool = False,
        gate_active: Callable[[], bool] | None = None,
    ) -> None:
        """``set_status(label)`` / ``pause_status()`` are optional
        callbacks the REPL wires to a Rich ``console.status``
        spinner. The renderer pauses the spinner while assistant
        prose is streaming (it would corrupt the same line) and
        sets a new label on each tool boundary so the user has
        something to read between events. Both default to no-op so
        non-REPL callers (one-shot CLI) keep working unchanged.

        ``sandbox`` adds a 🔒 marker to each ``bash`` tool line so the
        user can see — per command, not just at launch — that the
        coder's shell is kernel-isolated. Off by default."""
        self._sandbox = sandbox
        # While this returns True (approval prompt on screen), events
        # queue in _deferred instead of printing — see handle().
        self._gate_active = gate_active
        self._deferred: list[Any] = []
        self._in_text = False
        # Mid-thinking-burst flag (reasoning chunks stream dim).
        self._in_thinking = False
        # Whether ANY assistant prose has been shown this run — drives
        # the _on_completed fallback (print result["output"] only if
        # nothing streamed, so we never double-print).
        self._any_text = False
        # Buffer for one prose burst — chunks accumulate here while
        # they're also streamed inline as plain text (see
        # ``_on_model_chunk`` for why we DON'T use a Rich Live
        # widget: it recurses against the REPL's status spinner).
        # The buffer lets ``_on_completed`` know whether anything
        # streamed (so it doesn't double-print result["output"]).
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
        # Structured plan steps captured from the most recent
        # ``plan_write`` tool_call args. Used by the auto-continue
        # logic in :mod:`repl` — a list of {description, status,
        # finding} dicts is more reliable than regex-parsing the
        # rendered markdown for progress (the rendering can change;
        # the tool args shape is loomflow's stable API).
        self.last_plan_steps: list[dict[str, Any]] | None = None
        self.last_result: dict[str, Any] | None = None
        # Captured during the run so cli.py can build the end-of-run
        # summary (files changed / verified / notes captured) without
        # the agent having to report any of it itself.
        self.bash_commands: list[str] = []
        self.notes_written: list[tuple[str, str]] = []  # (kind, title)
        # Files this run WROTE to — captured from edit/multi_edit/write
        # tool calls (all use the ``path`` arg). Feeds file-touch
        # history (loom_code.file_history): "last time you touched X,
        # the change was marked bad." Read-only tools (read/grep) are
        # NOT touches — only mutations land here.
        self.files_touched: list[str] = []

    def handle(self, event: Any) -> None:
        """Render a single ``Event``. ``kind`` is a string enum;
        compare against the lowercase values loomflow uses.

        While the approval gate is prompting (``gate_active``), events
        are DEFERRED instead of rendered: the selector redraws itself
        in place with cursor-up escapes, so any concurrent print
        displaces its geometry and the menu visibly duplicates
        (observed live). Deferred events flush in order the moment the
        gate closes."""
        if self._gate_active is not None and self._gate_active():
            self._deferred.append(event)
            return
        if self._deferred:
            pending, self._deferred = self._deferred, []
            for ev in pending:
                self._dispatch(ev)
        self._dispatch(event)

    def _dispatch(self, event: Any) -> None:
        kind = str(event.kind)
        payload = event.payload or {}
        method = getattr(self, f"_on_{kind}", None)
        if method is not None:
            method(payload)
        # Unknown event kinds are silently ignored — forward-compat
        # with new loomflow event types.

    def flush_deferred(self) -> None:
        """Render anything still queued from a gate window — called by
        the REPL after the stream ends so no event is ever lost."""
        pending, self._deferred = self._deferred, []
        for ev in pending:
            self._dispatch(ev)

    # ---- text streaming -------------------------------------------------

    def _on_model_chunk(self, payload: dict[str, Any]) -> None:
        # loomflow shape: payload = {"chunk": ModelChunk.model_dump()}
        # where ModelChunk is discriminated by ``kind`` — only the
        # "text" kind carries assistant prose; "tool_call"/"finish"
        # chunks have text=None and are surfaced by other handlers.
        #
        # We stream the chunks as PLAIN TEXT inline, NOT through a
        # Rich ``Live`` widget. Hard-won reason: a Live nested
        # inside the REPL's ``console.status()`` spinner (which is
        # ALSO a Live) makes Rich's refresh recurse on
        # ``console._live_stack[0].refresh()`` — observed blowing
        # the stack (RecursionError, ~992 frames) AND corrupting
        # the approval prompt's stdin. Two Live contexts on one
        # console don't compose. Plain ``console.print(..., end="")``
        # gives the same token-by-token streaming feel with zero
        # Live machinery, so it can't deadlock or recurse.
        #
        # Tradeoff: streamed prose isn't markdown-rendered (you see
        # literal ``**bold**`` / code fences). Acceptable — a crash
        # that blocks the agent mid-write is infinitely worse than
        # un-prettified streaming.
        chunk = payload.get("chunk") or {}
        kind = chunk.get("kind")
        if kind == "thinking":
            # Reasoning stream (Claude extended thinking / o-series
            # summaries when /effort is set). Shown dim so the user
            # sees the model IS working — previously these chunks
            # were silently dropped and high-effort turns looked
            # stalled. Not buffered: thinking is not the answer.
            text = chunk.get("text") or ""
            if not text:
                return
            if not self._in_thinking:
                self._pause_status()
                console.print()
                console.print(
                    "  ✻ thinking… ", end="", style="dim italic"
                )
                self._in_thinking = True
            console.print(
                text,
                end="",
                markup=False,
                highlight=False,
                style="dim",
            )
            return
        if kind != "text":
            return
        text = chunk.get("text") or ""
        if not text:
            return
        self._end_thinking()
        if not self._in_text:
            # First chunk of the message — keep the spinner up
            # ("responding…") and BUFFER instead of printing raw. We
            # render the whole thing as clean Markdown on completion
            # (Claude-Code style), which is why we don't stream the raw
            # tokens: showing raw ``###``/``**`` then replacing them
            # would need fragile cursor-erase math. The spinner gives
            # the "working" feedback in the meantime.
            self._set_status("responding…")
            self._in_text = True
        self._any_text = True
        self._text_buffer.append(text)

    def _end_thinking(self) -> None:
        """Close an open thinking burst (newline + spinner back)."""
        if not self._in_thinking:
            return
        self._in_thinking = False
        console.print()
        self._set_status("thinking...")

    def _end_text(self) -> None:
        self._end_thinking()
        if not self._in_text:
            return
        # Message complete. It was BUFFERED (not streamed raw), so now
        # print it ONCE — as rendered Markdown when it looks markdown-y
        # (headings, bold, code fences → Claude-Code look), else as
        # plain text. The spinner was paused by the caller/_end path.
        full = "".join(self._text_buffer)
        self._text_buffer.clear()
        self._in_text = False
        self._pause_status()
        console.print()  # blank line before the answer
        if full.strip():
            # A labelled marker before the response, so the model's
            # output is visually attributed + separated from the tool
            # activity above it (Claude-Code ``● Assistant`` style).
            console.print(Text("● loom", style="bold cyan"))
        rendered = self._render_markdown(full)
        if rendered is not None:
            console.print(rendered)
        elif full.strip():
            # Plain text (no markdown to gain) — markup/highlight OFF so
            # a stray ``[`` in the model's text isn't parsed as a tag.
            console.print(full, markup=False, highlight=False)
        # Deliberately DON'T restart the spinner here. A real next event
        # (tool_call / architecture line) sets its own status; the turn
        # end pauses it. Restarting to "thinking..." on every prose
        # burst rendered a spinner line that Rich then cleared, leaving
        # the dead gap between the answer and the turn-summary rule.

    @staticmethod
    def _render_markdown(text: str) -> Any:
        """Render ``text`` as Rich Markdown, or None to print it plain.
        Skips empty output and text with no markdown to gain, and never
        raises — a parse failure falls back to the plain print."""
        if not text.strip():
            return None
        if not any(
            m in text for m in ("#", "```", "**", "- ", "* ", "|", "`")
        ):
            return None
        try:
            from rich.markdown import Markdown

            return Markdown(text, code_theme="ansi_dark")
        except Exception:  # noqa: BLE001 — render must never crash
            return None

    # ---- architecture events --------------------------------------------

    def _on_architecture_event(self, payload: dict[str, Any]) -> None:
        """Surface the bits the user actually cares about. Most
        architecture events are framework-internal progress signals
        the renderer correctly hides — the exceptions:

        * ``router.dispatched`` — WHICH route the classifier picked
          ('simple' vs 'complex'), fixing the observability asymmetry
          where COMPLEX turns showed delegate+worker activity but
          SIMPLE turns showed nothing but a spinner.
        * ``auto_compacted`` — the conversation was just compacted;
          the user should know why the model "forgot" verbatim detail.
        * ``stop_hook.fired`` — an auto-continue iteration started
          (the plan still has open steps), so a long turn visibly
          progresses instead of looking stuck.

        NOTE: keep this the ONLY definition of this method — a
        previous refactor left two copies and the second silently
        shadowed the first, hiding every one of these lines.
        """
        name = payload.get("name") or ""
        if name == "router.dispatched":
            route = payload.get("route")
            if not route:
                return
            # End any in-flight prose burst first so the route line
            # doesn't land mid-Live-render. Then print a single line.
            self._end_text()
            self._pause_status()
            console.print(
                f"  [dim]→ routed to[/dim] [cyan]{route}[/cyan]"
            )
            # Re-set the status spinner so the user has feedback while
            # the specialist starts up. Specific tool labels will
            # overwrite it as the route's agent does its work.
            self._set_status(f"{route} working...")
        elif name == "auto_compacted":
            self._end_text()
            dropped = payload.get("messages_dropped")
            extra = f" ({dropped} messages summarised)" if dropped else ""
            console.print(
                Text(f"  ✦ context compacted{extra}", style="dim magenta")
            )
        elif name == "stop_hook.fired":
            self._end_text()
            iteration = payload.get("iteration")
            tag = f" ({iteration})" if iteration else ""
            console.print(
                Text(
                    f"  ▸ auto-continue{tag} — plan has open steps",
                    style="dim",
                )
            )
        else:
            # Generic living-plan / architecture progress
            # (``plan.updated``, ``self_refine.critique``, …). The
            # previous handler surfaced any 'plan'-ish event as a dim
            # ▸ line; keep that so a long multi-step turn shows
            # movement instead of a silent spinner. Some architectures
            # key the label under ``event`` rather than ``name``.
            label = str(name or payload.get("event") or "")
            if "plan" in label.lower():
                self._end_text()
                console.print(Text(f"  ▸ {label}", style="dim magenta"))

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
        elif tool in ("edit", "multi_edit", "write"):
            # All three mutation tools take the file path as ``path``.
            # Record it as a touch so file-touch history knows what
            # this turn changed (outcome attached later by the repl).
            p = str(args.get("path") or "").strip()
            if p and p not in self.files_touched:
                self.files_touched.append(p)
        elif tool == "note":
            title = str(args.get("title") or "").strip()
            if title:
                kind = str(args.get("kind") or "").strip()
                self.notes_written.append((kind, title))
        elif tool == "plan_write":
            # Capture the structured plan steps so the REPL's
            # auto-continue logic can read progress from a stable
            # API instead of regex-parsing the rendered markdown.
            # Lenient coercion: args["steps"] may be a JSON-string
            # in some adapters; loomflow normalises on the tool
            # side but the event we observe predates that. Fall
            # back to the regex parser if shape is unexpected.
            steps = _coerce_plan_steps(args.get("steps"))
            if steps is not None:
                self.last_plan_steps = steps
        arg_str = _summarise_args(args)
        # Per-command sandbox marker: when --sandbox is on, a 🔒 on the
        # bash line confirms the shell is kernel-isolated for THIS call
        # — the launch banner alone left users unsure it still applied.
        lock = " 🔒" if (self._sandbox and tool == "bash") else ""
        console.print(
            Text.assemble(
                ("  → ", "bold cyan"),
                (tool, "bold cyan"),
                (lock, "bold green"),
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
        # The living plan is load-bearing — show it in full. Prefer
        # the compact glyph view (built from the structured steps
        # captured on the tool_call); fall back to the raw markdown
        # table only if we somehow didn't capture structured steps.
        if is_plan:
            if self.last_plan_steps:
                console.print(
                    _render_plan_glyphs(
                        self.last_plan_steps,
                        goal=_extract_plan_goal(text),
                    )
                )
            else:
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
        msg = str(payload.get("error") or payload.get("message") or "?")
        # loomflow BOTH emits the error event AND re-raises the same
        # exception, so when it re-raises the consumer (repl/cli)
        # prints the flattened, classified form right after this
        # handler — printing the opaque anyio wrapper here too would be
        # a duplicate. Suppress ONLY the BARE wrapper ("unhandled
        # errors in a TaskGroup (N sub-exception)") with nothing else
        # of substance: if the message ALSO carries a real cause (a
        # worker error the run RECOVERS from and does not re-raise —
        # the consumer never sees it), we must still show it, or the
        # user gets a silently degraded answer.
        low = msg.lower()
        # The bare wrapper is exactly "unhandled errors in a TaskGroup
        # (N sub-exception[s])" — it ends at the paren note with no
        # real cause appended. A message that carries a cause has more
        # after the "sub-exception)" — keep those.
        tail = low.split("sub-exception", 1)[-1]
        bare_wrapper = (
            "unhandled errors in a taskgroup" in low
            and tail.strip(" s)") == ""
        )
        if bare_wrapper:
            return
        console.print(Text(f"\n✗ error: {msg}", style="bold red"))

    def _on_completed(self, payload: dict[str, Any]) -> None:
        self._end_text()
        # _end_text restarts the spinner ("thinking...") to bridge to
        # the NEXT event — but this is the LAST event, so kill it now.
        # Left running, it renders a dangling spinner line that Rich
        # then clears, leaving the dead gap between the answer and the
        # turn-summary rule the user saw.
        self._pause_status()
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
            console.print(Text("● loom", style="bold cyan"))
            rendered = self._render_markdown(output)
            if rendered is not None:
                console.print(rendered)
            else:
                console.print(output, markup=False, highlight=False)
        # No trailing blank here — the REPL's end-of-turn summary rule
        # (_print_turn_summary) closes the turn. A blank here just added
        # dead space between the answer and that rule.


# Status → (glyph, Rich style) for the compact plan view. ■ done /
# ▸ doing / □ todo / ⊘ skipped / ✗ blocked — scannable at a glance,
# the way Claude Code / modern TODO panels render task state.
_PLAN_GLYPHS: dict[str, tuple[str, str]] = {
    "done": ("■", "green"),
    "doing": ("▸", "bold yellow"),
    "todo": ("□", "dim"),
    "skipped": ("⊘", "dim"),
    "blocked": ("✗", "red"),
}


def _extract_plan_goal(markdown: str) -> str:
    """Pull the goal line out of loomflow's rendered plan markdown
    (``**GOAL:** ...``) so the glyph header can show it."""
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("**GOAL:**"):
            return stripped.replace("**GOAL:**", "").strip()
    return ""


def _render_plan_glyphs(
    steps: list[dict[str, Any]], *, goal: str = ""
) -> Text:
    """Render a living plan as a compact glyph list instead of a
    markdown table. Far faster to scan mid-run ("what's left?")
    than the bordered table.

    Layout::

        ◆ <goal>  ·  1/3 done  ·  1 blocked
          ■ Fix shell injection        — rewrote without shell=True
          ▸ Fix eval usage  (doing)
          □ Fix path traversal
          ⊘ Update changelog           › skipped: outside ask

    Done steps show their ``finding`` inline (dim); skipped/blocked
    steps show the reason right-flagged with ``›``. Built with Rich
    ``Text.append`` (not markup) so step descriptions containing
    ``[`` don't get mis-parsed as style tags.
    """
    t = Text()
    total = len(steps)
    done = sum(1 for s in steps if str(s.get("status")) == "done")
    blocked = sum(
        1 for s in steps if str(s.get("status")) == "blocked"
    )
    t.append("  ◆ ", style="bold cyan")
    t.append(goal or "Plan", style="bold")
    t.append(f"  ·  {done}/{total} done", style="dim")
    if blocked:
        t.append(f"  ·  {blocked} blocked", style="dim red")
    t.append("\n")
    for s in steps:
        status = str(s.get("status", "todo"))
        glyph, gstyle = _PLAN_GLYPHS.get(status, ("□", "dim"))
        desc = str(s.get("description", "")).strip()
        finding = (
            str(s.get("finding", "")).replace("\n", " ").strip()
        )
        t.append(f"    {glyph} ", style=gstyle)
        if status == "done":
            t.append(desc, style="dim")
        elif status == "doing":
            t.append(desc, style="bold")
            t.append("  (doing)", style="dim yellow")
        else:
            t.append(desc)
        if finding:
            if status in ("skipped", "blocked"):
                t.append(f"   › {finding}", style="dim")
            elif status == "done":
                # Cap the inline finding so a verbose one doesn't
                # blow the line width.
                clip = finding if len(finding) <= 60 else finding[:59] + "…"
                t.append(f"   — {clip}", style="dim")
        t.append("\n")
    return t


def _coerce_plan_steps(raw: Any) -> list[dict[str, Any]] | None:
    """Normalise the ``plan_write`` ``steps`` arg into a list of dicts.

    loomflow's plan tool accepts several shapes (native list, JSON
    string, ``{"steps":[…]}`` wrapper) on the tool side, but the
    ``tool_call`` event we observe carries whatever the model
    emitted — usually a list of dicts. Be lenient about the shape;
    return ``None`` for anything we can't interpret so the auto-
    continue logic falls back to markdown parsing.
    """
    import json

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if isinstance(raw, dict) and "steps" in raw:
        raw = raw["steps"]
    if not isinstance(raw, list):
        return None
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        # Normalise the status field — lower-case, default todo.
        status = item.get("status") or "todo"
        if not isinstance(status, str):
            status = "todo"
        status = status.lower().strip()
        if status not in {"todo", "doing", "done", "blocked", "skipped"}:
            status = "todo"
        out.append(
            {
                "description": str(item.get("description", "")).strip(),
                "status": status,
                "finding": str(item.get("finding") or "").strip() or None,
            }
        )
    return out


def _summarise_args(args: dict[str, Any]) -> str:
    """One-line, length-capped arg summary for a tool-call line."""
    parts: list[str] = []
    for k, v in args.items():
        sv = str(v).replace("\n", " ")
        if len(sv) > 60:
            sv = sv[:60] + "…"
        parts.append(f"{k}={sv}")
    return ", ".join(parts)


def banner(
    model: str,
    root: str,
    is_git: bool,
    *,
    sandbox: bool = False,
    sandbox_allow_network: bool = False,
) -> None:
    """Print the loom-code startup banner.

    ``sandbox`` shows a visible badge in the header (``🔒 sandboxed``)
    plus a one-line explanation, so the user can tell at a glance that
    the coder's bash is kernel-isolated — otherwise the protection is
    invisible (it only surfaces when a command tries to escape)."""
    git_tag = "git" if is_git else "no-git"
    parts = [
        ("loom-code", "bold"),
        ("  ", ""),
        (f"{model}", "cyan"),
        ("  ·  ", "dim"),
        (f"{root}", "dim"),
        ("  ·  ", "dim"),
        (git_tag, "dim"),
    ]
    if sandbox:
        parts += [("  ·  ", "dim"), ("🔒 sandboxed", "bold green")]
    console.print()
    console.print(Text.assemble(*parts))
    console.print(
        Text("  loomflow-native coding agent", style="dim italic")
    )
    if sandbox:
        net = (
            "network ON" if sandbox_allow_network else "no network"
        )
        console.print(
            Text(
                f"  bash runs in an OS sandbox — writes limited to "
                f"this repo, {net}",
                style="dim green",
            )
        )
    console.print()


def print_code(text: str, lexer: str = "python") -> None:
    """Render a code block with syntax highlighting (used by the
    diff-approval UI in Phase 2)."""
    console.print(Syntax(text, lexer, theme="ansi_dark"))
