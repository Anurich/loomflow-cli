"""Shell-command hooks for loom-code (the ``.loom`` ``settings.toml``).

Turns the ``[[hooks]]`` entries discovered by
:mod:`loom_code.extensions` into runnable behaviour. There are two
worlds, because loomflow only exposes hook points for some events:

* **Tool-lifecycle** (``PreToolUse`` / ``PostToolUse``) become loomflow
  ``HookRegistry`` callbacks attached to the tool-executing agents
  (the workers + the simple coder — NOT the coordinator, which only
  delegates). ``PreToolUse`` can BLOCK a tool call or REWRITE its
  input; ``PostToolUse`` observes the result.
* **REPL-lifecycle** (``SessionStart`` / ``UserPromptSubmit`` /
  ``SessionEnd``) the REPL fires itself — loomflow has no hook point
  for them. ``UserPromptSubmit`` can inject extra context into the
  prompt or block the turn.

Each hook is a shell command. We run it with ``anyio.run_process``
(a *string* command runs through the real shell, so pipes / ``&&``
work), pass a JSON event payload on stdin, and interpret the result:

    exit 2            -> BLOCK; reason from stderr
    exit 0 + JSON out -> read ``decision`` / ``reason`` /
                         ``additionalContext`` / ``updatedInput``
    exit 0 + text out -> treated as ``additionalContext``
    any other exit    -> non-blocking error; ignored

A hook must never crash a run: a command that fails to launch, times
out, or emits garbage degrades to "no opinion."

``Stop`` hooks are intentionally NOT wired here yet — they re-enable
the framework's bounded continuation loop (``max_stop_hook_iterations``,
deliberately 0 in loom-code) and need their interaction with that cap
designed carefully. Tracked as a follow-up.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
from loomflow.core.types import PermissionDecision, ToolCall, ToolResult

from .extensions import HookSpec


def matches(matcher: str, tool_name: str) -> bool:
    """Does ``matcher`` select ``tool_name``?

    ``"*"`` / ``""`` match everything. A token of only word chars and
    ``|`` is a pipe-separated exact list (``"bash|edit"``). Anything
    else is treated as a regex searched against the tool name; an
    invalid regex matches nothing (fail closed for the matcher, not the
    run)."""
    matcher = matcher.strip()
    if matcher in ("", "*"):
        return True
    if re.fullmatch(r"[\w|]+", matcher):
        return tool_name in matcher.split("|")
    try:
        return re.search(matcher, tool_name) is not None
    except re.error:
        return False


@dataclass
class HookOutcome:
    """Normalised result of running one hook command."""

    block: bool = False
    reason: str | None = None
    additional_context: str | None = None
    updated_input: dict[str, Any] | None = None


# Strong references to in-flight background hook tasks — asyncio only
# keeps weak refs to tasks, so without this set a fire-and-forget hook
# could be garbage-collected mid-run. Done-callback discards.
_background_tasks: set[Any] = set()


def _fire_and_forget(
    spec: HookSpec, payload: dict[str, Any], *, cwd: Path
) -> None:
    """Schedule ``_run`` without awaiting it (``background = true``
    hooks): the turn continues immediately; exit code and stdout are
    ignored by design. Falls back to a silent no-op when there's no
    running loop (sync callers in tests)."""
    import asyncio

    try:
        task = asyncio.get_running_loop().create_task(
            _run(spec, payload, cwd=cwd)
        )
    except RuntimeError:
        return
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _run(
    spec: HookSpec, payload: dict[str, Any], *, cwd: Path
) -> HookOutcome:
    """Run one hook command with ``payload`` on stdin; never raises.

    Honours ``spec.timeout`` via ``anyio.move_on_after`` — a hung hook
    is cancelled (which terminates the subprocess) and degrades to "no
    opinion."""
    stdin = json.dumps(payload).encode("utf-8")
    result: Any = None
    # Pipe-race errors to swallow: a hook that exits without reading
    # stdin (``touch marker``, ``true``, …) closes the pipe while
    # run_process is still writing the payload — the write raises
    # BrokenResourceError (anyio's wrap of BrokenPipeError /
    # ConnectionResetError). The hook RAN; only payload delivery
    # failed, so it degrades to "no opinion". anyio's internal task
    # group may deliver it bare OR wrapped in an ExceptionGroup —
    # handle both, and re-raise groups holding anything else.
    _pipe_errors = (OSError, ValueError, anyio.BrokenResourceError)
    with anyio.move_on_after(spec.timeout):
        try:
            result = await anyio.run_process(
                spec.command, input=stdin, cwd=str(cwd), check=False
            )
        except _pipe_errors:
            return HookOutcome()
        except BaseExceptionGroup as eg:
            # subgroup() tests GROUP nodes too — exclude them so only
            # leaf exceptions decide (a group is never itself a pipe
            # error, and matching it would re-raise everything).
            real = eg.subgroup(
                lambda e: not isinstance(
                    e, (BaseExceptionGroup, *_pipe_errors)
                )
            )
            if real is not None:
                raise  # a real error is in there — don't mask it
            return HookOutcome()
    if result is None:
        # Timed out (cancelled) or failed to launch.
        return HookOutcome()

    code = result.returncode
    out = result.stdout.decode("utf-8", "replace").strip()
    err = result.stderr.decode("utf-8", "replace").strip()

    if code == 2:
        return HookOutcome(
            block=True, reason=err or f"blocked by {spec.source} hook"
        )
    if code != 0:
        return HookOutcome()  # non-blocking error
    if not out:
        return HookOutcome()
    try:
        data = json.loads(out)
    except (json.JSONDecodeError, ValueError):
        # Plain text on stdout — treat as context the hook wants added.
        return HookOutcome(additional_context=out)
    if not isinstance(data, dict):
        return HookOutcome()

    decision = str(data.get("decision", "")).strip().lower()
    upd = data.get("updatedInput")
    ctx = data.get("additionalContext")
    reason = data.get("reason")
    return HookOutcome(
        block=decision in ("block", "deny"),
        reason=(str(reason) if reason is not None else None),
        additional_context=(str(ctx) if ctx else None),
        updated_input=upd if isinstance(upd, dict) else None,
    )


# ---- tool-lifecycle hooks (framework HookRegistry) ------------------


def _make_pre_tool_hook(spec: HookSpec, *, cwd: Path) -> Any:
    """Build a loomflow ``PreToolHook`` from a PreToolUse spec.

    Returns a ``PermissionDecision.deny_`` to block, or ``None`` to
    pass (loomflow's registry only acts on a deny). ``updatedInput``
    mutates the live ``ToolCall.args`` in place — the loop executes the
    same object, so the rewrite takes effect."""

    async def hook(call: ToolCall) -> PermissionDecision | None:
        if not matches(spec.matcher, call.tool):
            return None
        payload = {
            "hook_event_name": "PreToolUse",
            "tool_name": call.tool,
            "tool_input": dict(call.args),
            "cwd": str(cwd),
        }
        if spec.background:
            # Fire-and-forget: cannot block or rewrite the call —
            # documented contract of ``background = true``.
            _fire_and_forget(spec, payload, cwd=cwd)
            return None
        outcome = await _run(spec, payload, cwd=cwd)
        if outcome.updated_input is not None:
            call.args.clear()
            call.args.update(outcome.updated_input)
        if outcome.block:
            return PermissionDecision.deny_(
                outcome.reason or f"blocked by {spec.source} hook"
            )
        return None

    return hook


def _make_post_tool_hook(spec: HookSpec, *, cwd: Path) -> Any:
    """Build a loomflow ``PostToolHook`` from a PostToolUse spec.

    PostToolUse is observe-only in loomflow (the callback returns
    ``None`` and cannot alter the result), matching its common use —
    formatting, logging, notifications."""

    async def hook(call: ToolCall, result: ToolResult) -> None:
        if not matches(spec.matcher, call.tool):
            return
        payload = {
            "hook_event_name": "PostToolUse",
            "tool_name": call.tool,
            "tool_input": dict(call.args),
            "tool_output": _truncate(_safe_str(result.output)),
            "tool_error": result.error,
            "ok": result.ok,
            "cwd": str(cwd),
        }
        if spec.background:
            _fire_and_forget(spec, payload, cwd=cwd)
            return
        await _run(spec, payload, cwd=cwd)

    return hook


def attach_tool_hooks(agent: Any, specs: list[HookSpec], *, cwd: Path) -> None:
    """Register the PreToolUse/PostToolUse hooks in ``specs`` onto
    ``agent``'s loomflow ``HookRegistry`` (``agent._hooks``).

    No-op when ``specs`` has no tool-lifecycle entries, so the agent's
    fast-hooks path stays intact when the user defined none. Bumps the
    registry's per-hook timeout to cover the slowest spec — loomflow's
    default cap (5s) would otherwise cancel a longer hook early."""
    pre = [s for s in specs if s.event == "PreToolUse"]
    post = [s for s in specs if s.event == "PostToolUse"]
    if not pre and not post:
        return
    registry = agent._hooks  # noqa: SLF001 — the agent's HookRegistry
    longest = max(s.timeout for s in (*pre, *post))
    registry.hook_timeout_s = max(registry.hook_timeout_s, longest + 1.0)
    for s in pre:
        registry.register_pre_tool(_make_pre_tool_hook(s, cwd=cwd))
    for s in post:
        registry.register_post_tool(_make_post_tool_hook(s, cwd=cwd))


# ---- REPL-lifecycle hooks (fired by the REPL) -----------------------


@dataclass
class ReplHookResult:
    """Aggregate of all REPL hooks fired for one event."""

    blocked: bool = False
    reason: str | None = None
    contexts: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    @property
    def added_context(self) -> str:
        """The combined ``additionalContext`` to fold into the prompt
        (empty string when none)."""
        return "\n".join(c for c in self.contexts if c).strip()


async def run_repl_hooks(
    specs: list[HookSpec],
    event: str,
    *,
    cwd: Path,
    prompt: str | None = None,
) -> ReplHookResult:
    """Run every hook registered for a REPL-lifecycle ``event``.

    ``event`` is one of ``SessionStart`` / ``UserPromptSubmit`` /
    ``SessionEnd``. For ``UserPromptSubmit`` the user's ``prompt`` is
    included in the payload and a hook may block the turn (exit 2) or
    return ``additionalContext`` to append. Hooks run in declaration
    order (user scope before project scope, per discovery)."""
    out = ReplHookResult()
    for spec in specs:
        if spec.event != event:
            continue
        payload: dict[str, Any] = {"hook_event_name": event, "cwd": str(cwd)}
        if prompt is not None:
            payload["prompt"] = prompt
        if spec.background:
            # Fire-and-forget: can't block the turn or add context.
            _fire_and_forget(spec, payload, cwd=cwd)
            continue
        outcome = await _run(spec, payload, cwd=cwd)
        if outcome.block:
            out.blocked = True
            out.reason = outcome.reason
        if outcome.additional_context:
            out.contexts.append(outcome.additional_context)
    return out


# ---- helpers --------------------------------------------------------


def _safe_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _truncate(text: str, limit: int = 4000) -> str:
    """Cap tool output piped to a hook — a hook doesn't need a 1MB
    file dump, and a giant stdin can stall the subprocess."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"
