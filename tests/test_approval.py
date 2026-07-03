"""Tests for the approval handlers.

Both have a *silent* failure mode that's worth locking down:

* ``auto_approve`` denying anything would silently break the
  ``--yes`` flag — every destructive call would be denied with no
  prompt to recover, and the agent would just stop changing files.
* ``ApprovalGate`` losing its session-allow-all short-circuit
  would re-prompt the user on EVERY destructive call after they
  already said "allow all" — the REPL becomes unusable.

Neither is hit by the structural test suite. Lock them down here.
"""

from __future__ import annotations

import pytest
from loomflow.core.types import ToolCall

from loom_code.approval import ApprovalGate, auto_approve

# Approval handlers are async — anyio's pytest plugin ships with
# anyio (already a transitive dep via loomflow); the
# ``anyio_backend`` fixture in conftest.py pins it to asyncio.
pytestmark = pytest.mark.anyio


async def test_auto_approve_allows_a_destructive_call() -> None:
    # The whole point of --yes: rm -rf in a sandbox MUST go
    # through. If auto_approve ever denies, --yes silently breaks.
    call = ToolCall(tool="bash", args={"command": "rm -rf /tmp/x"})
    assert await auto_approve(call) is True


async def test_auto_approve_ignores_the_user_id() -> None:
    # The signature accepts user_id for handler-protocol compat;
    # auto_approve must not start gating on it.
    call = ToolCall(
        tool="edit",
        args={"path": "x.py", "old_string": "a", "new_string": "b"},
    )
    assert await auto_approve(call, user_id="loom-code") is True


async def test_approval_gate_starts_locked() -> None:
    # Fresh gate must NOT auto-approve — the user hasn't said
    # "allow all" yet. If this regresses, --yes-equivalent
    # behavior leaks into the REPL with no user consent.
    gate = ApprovalGate()
    assert gate._allow_all is False


async def test_approval_gate_allow_all_short_circuits() -> None:
    # Once the user picks 'a', subsequent calls must allow without
    # prompting (no Prompt.ask reached — that would block on stdin
    # and hang the test). The handler returns True via the
    # ``if self._allow_all`` early return at the top.
    gate = ApprovalGate()
    gate._allow_all = True
    call = ToolCall(tool="bash", args={"command": "rm file.txt"})
    assert await gate.handler(call) is True


async def test_approval_gate_allow_all_works_for_every_tool() -> None:
    # Allow-all is gate-wide, not per-tool. If anyone ever scopes
    # it (e.g. "allow all bash but re-ask on edit"), this catches
    # the regression — the property is currently universal.
    gate = ApprovalGate()
    gate._allow_all = True
    for tool in ("bash", "edit", "write"):
        call = ToolCall(tool=tool, args={})
        assert await gate.handler(call) is True


# ---- Windows key reader (msvcrt) -------------------------------------
# termios doesn't exist on Windows; the selector dispatches to
# _read_key_msvcrt there. We can't run real Windows in CI, but the
# reader's decode logic is pure — drive it with a fake msvcrt module.


def _fake_msvcrt(keys: list[str]):
    """A stand-in msvcrt whose getwch() pops from ``keys``."""
    import types

    mod = types.ModuleType("msvcrt")
    seq = iter(keys)
    mod.getwch = lambda: next(seq)
    return mod


def test_msvcrt_reader_decodes_logical_keys(monkeypatch) -> None:
    import sys as _sys

    from loom_code.approval import _read_key_msvcrt

    cases = [
        (["\r"], "enter"),
        (["\n"], "enter"),
        (["\x03"], "esc"),   # Ctrl-C → SAFE cancel
        (["\x1b"], "esc"),
        (["\xe0", "H"], "up"),
        (["\xe0", "P"], "down"),
        (["\x00", "H"], "up"),     # alternate extended prefix
        (["\xe0", "K"], "esc"),    # unknown extended key → safe
        (["3"], "3"),
        (["Y"], "y"),
    ]
    for keys, expected in cases:
        monkeypatch.setitem(
            _sys.modules, "msvcrt", _fake_msvcrt(keys)
        )
        assert _read_key_msvcrt() == expected, (keys, expected)


def test_select_option_survives_missing_termios(monkeypatch) -> None:
    """Regression: on Windows there is no termios — the selector's
    bare ``import termios`` crashed /set_model with
    ModuleNotFoundError for every pipx-on-Windows user. With the
    dispatch fix, a missing termios falls through to the msvcrt
    reader (faked here): ↓ then Enter must select option 2 and
    never raise."""
    import builtins
    import sys as _sys

    from loom_code import approval

    real_import = builtins.__import__

    def _no_termios(name, *args, **kwargs):
        if name in ("termios", "tty"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_termios)
    monkeypatch.setitem(
        _sys.modules, "msvcrt", _fake_msvcrt(["\xe0", "P", "\r"])
    )
    monkeypatch.delitem(_sys.modules, "termios", raising=False)
    monkeypatch.delitem(_sys.modules, "tty", raising=False)
    # stdin must look like a TTY to reach the interactive path
    monkeypatch.setattr(
        approval.sys.stdin, "isatty", lambda: True, raising=False
    )
    result = approval._select_option(
        [("a", "first"), ("b", "second"), ("c", "third")]
    )
    assert result == "b"  # ↓ moved 0→1, Enter picked it
