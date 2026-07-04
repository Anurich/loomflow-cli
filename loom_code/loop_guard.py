"""Doom-loop detection + environment-failure hints — native
post-tool middleware on every tool-executing agent.

Terminal-Bench 2.0 evidence behind both rules:

* A LoopDetectionMiddleware tracking per-file edit counts was one of
  the harness changes behind deepagents-cli's +13.7-point jump —
  agents stuck re-editing the same file are the dominant "doom loop".
* The single most common agent command failure is invoking an
  executable that isn't installed / not on PATH (24.1% of all command
  failures) — a failure mode a one-line hint fixes.

Both interventions are steering TEXT appended to the tool RESULT.
loomflow runs post-tool hooks on the live ``ToolResult`` *before* the
loop serialises it into the conversation, so the model sees the hint
exactly where the failure happened — no prompt-bloat, no extra turn.
Weak free-tier models (loom-code's differentiator) benefit most: the
research shows good scaffolding helps cheap models the hardest
(+10.1pp for deepseek-v4-flash class models).

Module-level state on purpose, mirroring ``consent.py``: the guard is
registered at agent construction (long before any turn exists) and
reset by the REPL at each turn start. A plain module beats threading a
handle through the build stack. Lifetime is the process.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

# Thresholds. Edits to one file get a warning ON the Nth edit (and
# each after); identical bash commands get one on the Nth run. Chosen
# to fire on genuine loops but never on the normal edit → test → edit
# rhythm (2 edits to a file in a turn is routine; 3+ starts to smell).
EDIT_REPEAT_THRESHOLD = 3
CMD_REPEAT_THRESHOLD = 4

_EDIT_TOOLS = frozenset({"edit", "multi_edit", "write"})

# "binary not found" shapes across shells/OSes. Group 1 (when present)
# is the binary name.
_NOT_FOUND_PATTERNS = (
    re.compile(r"(?:bash|sh|zsh):(?: line \d+:)? ([^\s:]+): command not found"),
    re.compile(r"^([^\s:]+): command not found", re.MULTILINE),
    re.compile(r"'([^']+)' is not recognized as an internal or external"),
    re.compile(r"([^\s:]+): No such file or directory", re.MULTILINE),
)

_edit_counts: Counter[str] = Counter()
_cmd_counts: Counter[str] = Counter()
_hinted_binaries: set[str] = set()


def reset() -> None:
    """Drop all per-turn counters (REPL calls this at turn start)."""
    _edit_counts.clear()
    _cmd_counts.clear()
    _hinted_binaries.clear()


def _missing_binary(output: str) -> str | None:
    """The binary name from a command-not-found error, or None."""
    for pat in _NOT_FOUND_PATTERNS:
        m = pat.search(output)
        if m:
            name = m.group(1).strip()
            # Paths aren't "missing binaries" — a missing FILE arg
            # hits the same errno text; only bare command names get
            # the PATH hint.
            if name and "/" not in name and "\\" not in name:
                return name
    return None


def hint_for(tool: str, args: dict[str, Any], output: str) -> str | None:
    """The steering hint for one completed tool call, or None.

    Pure: counters advance as a side effect, but no I/O and no
    knowledge of loomflow types — trivially testable."""
    if tool in _EDIT_TOOLS:
        path = str(args.get("path", "")).strip()
        if not path:
            return None
        _edit_counts[path] += 1
        n = _edit_counts[path]
        if n >= EDIT_REPEAT_THRESHOLD:
            return (
                f"[loop-guard] edit #{n} to {path} this turn. If the "
                "previous edits didn't fix the problem, STOP editing "
                "this file — re-read it in full and reconsider the "
                "approach before changing it again."
            )
        return None

    if tool == "bash":
        command = str(args.get("command", "")).strip()
        if not command:
            return None
        binary = _missing_binary(output)
        if binary and binary not in _hinted_binaries:
            _hinted_binaries.add(binary)
            return (
                f"[env] '{binary}' isn't installed or isn't on PATH. "
                f"Check with `which {binary}` / `command -v {binary}` "
                "or install it before retrying — don't re-run the "
                "same command unchanged."
            )
        _cmd_counts[command] += 1
        n = _cmd_counts[command]
        if n >= CMD_REPEAT_THRESHOLD:
            return (
                f"[loop-guard] this exact command has now run {n}× "
                "this turn. If it keeps failing the same way, change "
                "the approach — read the error carefully, try a "
                "narrower diagnostic, or re-read the relevant file — "
                "instead of re-running it."
            )
    return None


async def post_tool(call: Any, result: Any) -> None:
    """The loomflow post-tool hook: append the steering hint to the
    result's output so the model sees it inline with the failure.

    Appends only to STRING outputs (the universal case for the file +
    bash kernel); anything else is left untouched. Never raises —
    loomflow absorbs hook errors, but don't rely on it."""
    try:
        hint = hint_for(
            str(getattr(call, "tool", "")),
            dict(getattr(call, "args", {}) or {}),
            str(getattr(result, "output", "") or ""),
        )
        if hint and isinstance(result.output, str):
            result.output = f"{result.output}\n\n{hint}"
    except Exception:  # noqa: BLE001 — steering must never break a tool
        return


def attach(agent: Any) -> None:
    """Register the guard on ``agent``'s hook registry. Safe to call
    on any loomflow Agent; silently no-ops if the registry shape is
    unexpected (custom builds)."""
    try:
        agent._hooks.register_post_tool(post_tool)  # noqa: SLF001
    except Exception:  # noqa: BLE001 — guard is best-effort
        return
