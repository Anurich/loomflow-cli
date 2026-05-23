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
        if chunk.get("kind") != "text":
            return
        text = chunk.get("text") or ""
        if not text:
            return
        if not self._in_text:
            # First chunk of a prose burst — pause the spinner (it
            # owns the cursor line) + drop a blank line so the
            # streamed text starts clean.
            self._pause_status()
            console.print()
            self._in_text = True
        self._any_text = True
        self._text_buffer.append(text)
        # Stream the raw chunk inline. ``markup=False`` /
        # ``highlight=False`` so a stray ``[`` in the model's text
        # isn't parsed as a Rich style tag mid-stream.
        console.print(text, end="", markup=False, highlight=False)

    def _end_text(self) -> None:
        if not self._in_text:
            return
        # Close the streamed line. The streamed plain text IS the
        # final output — no re-render (re-rendering as Markdown
        # would print the whole response a SECOND time).
        self._text_buffer.clear()
        self._in_text = False
        console.print()  # newline to end the inline stream
        # Prose burst over — bring the spinner back; the next
        # tool_call will overwrite this label with something
        # more specific.
        self._set_status("thinking...")

    # ---- architecture events --------------------------------------------

    def _on_architecture_event(self, payload: dict[str, Any]) -> None:
        """Surface the bits the user actually cares about. Most
        architecture events are framework-internal progress signals
        the renderer correctly hides — but ``router.dispatched``
        tells the user WHICH route the classifier picked
        ('simple' vs 'complex' for loom-code), which fixes the
        observability asymmetry where COMPLEX turns showed
        delegate+worker activity but SIMPLE turns showed nothing
        but a spinner. Unrelated event names are ignored."""
        name = payload.get("name")
        if name != "router.dispatched":
            return
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
