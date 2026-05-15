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

from typing import Any

from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text

console = Console()

# Tools whose results are worth showing in full-ish; others get a
# one-line summary so the terminal doesn't flood.
_VERBOSE_RESULT_TOOLS = {"read", "grep", "ls", "find"}
_RESULT_PREVIEW_CHARS = 600


class StreamRenderer:
    """Stateful renderer for one ``Agent.stream`` run.

    Tracks whether we're mid-model-chunk so tool calls don't
    interleave awkwardly with streamed assistant text. Also
    captures the last living-plan render + the final result dict
    so the REPL can power ``/plan`` and ``/cost`` without
    re-parsing the stream.
    """

    def __init__(self) -> None:
        self._in_text = False
        # Whether ANY assistant prose has been shown this run — drives
        # the _on_completed fallback (print result["output"] only if
        # nothing streamed, so we never double-print).
        self._any_text = False
        # call_id -> tool name. loomflow's `tool_result` event only
        # carries `call_id` (ToolResult has no `tool` field) — the
        # tool NAME is only on the `tool_call` event. We bridge the
        # two so result rendering can tell which tool ran.
        self._call_names: dict[str, str] = {}
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
        chunk = payload.get("chunk") or {}
        if chunk.get("kind") != "text":
            return
        text = chunk.get("text") or ""
        if not text:
            return
        if not self._in_text:
            console.print()  # blank line before a fresh assistant turn
            self._in_text = True
        self._any_text = True
        console.print(text, end="", markup=False, highlight=False)

    def _end_text(self) -> None:
        if self._in_text:
            console.print()
            self._in_text = False

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

    def _on_tool_result(self, payload: dict[str, Any]) -> None:
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
        cap = _RESULT_PREVIEW_CHARS * (3 if is_verbose else 1)
        preview = text if len(text) <= cap else (
            text[:cap] + f"\n    … (+{len(text) - cap} chars)"
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
        turns = result.get("turns", "?")
        cost = result.get("cost_usd", 0.0)
        tin = result.get("tokens_in", 0)
        cached = result.get("cached_tokens_in", 0)
        tout = result.get("tokens_out", 0)
        console.print()
        console.rule(style="dim")
        console.print(
            Text.assemble(
                ("  ", ""),
                (f"{turns} turns", "dim"),
                ("  ·  ", "dim"),
                (f"{tin:,}+{cached:,} in / {tout:,} out", "dim"),
                ("  ·  ", "dim"),
                (f"${cost:.4f}", "dim green"),
            )
        )


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
