"""Liveness guards on the REPL's stream consumer.

Two ways a turn used to run away, both observed live with weak
free-tier models:

* the stream simply stops producing events (hung provider) — the
  spinner spun forever while the process burned CPU;
* the model repeats the SAME tool call with identical args in a loop
  and grinds toward ``max_turns`` (loomflow's no-progress hook only
  arms under ``/goal``).

``_consume_agent_stream`` now runs an idle watchdog + a consecutive-
repeat stall detector. These tests drive the real method with a stub
``self`` and fake agents, so the async plumbing (task group, cancel
scopes, exception routing) is exercised for real.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anyio
import anyio.lowlevel
import pytest

from loom_code.render import StreamRenderer
from loom_code.repl import Repl


def _stub_repl(idle_timeout: float) -> Any:
    """The minimal attribute surface ``_consume_agent_stream`` touches,
    without constructing a full Repl (which needs a project + agent)."""
    stub = SimpleNamespace(
        _gate_active=False,
        _idle_timeout=idle_timeout,
        session_id="test-session",
        total_summaries=0,
        total_compacts=0,
        total_snips=0,
        _print_turn_error=lambda exc: None,
    )
    # Borrow the real implementation.
    stub._consume_agent_stream = (
        Repl._consume_agent_stream.__get__(stub)
    )
    return stub


def _event(kind: str, payload: dict[str, Any]) -> Any:
    return SimpleNamespace(kind=kind, payload=payload)


class _HangingAgent:
    """Emits one event, then goes silent forever."""

    async def stream(self, prompt: str, **kwargs: Any):
        yield _event("model_chunk", {"chunk": {"kind": "text", "text": "x"}})
        await anyio.sleep(3600)


class _LoopingAgent:
    """Repeats the identical tool call endlessly."""

    async def stream(self, prompt: str, **kwargs: Any):
        for _ in range(50):
            yield _event(
                "tool_call",
                {"call": {"tool": "read", "args": {"path": "a.py"}}},
            )
            await anyio.lowlevel.checkpoint()


class _CleanAgent:
    """A normal short run: two different tool calls, then done."""

    async def stream(self, prompt: str, **kwargs: Any):
        yield _event(
            "tool_call",
            {"call": {"tool": "read", "args": {"path": "a.py"}}},
        )
        yield _event(
            "tool_call",
            {"call": {"tool": "read", "args": {"path": "b.py"}}},
        )
        yield _event("completed", {"result": {"output": "done"}})


def _renderer() -> StreamRenderer:
    return StreamRenderer()


@pytest.mark.anyio
async def test_idle_watchdog_aborts_hung_stream() -> None:
    stub = _stub_repl(idle_timeout=0.3)
    ok = await stub._consume_agent_stream(
        _HangingAgent(), "task", _renderer(), lambda: None
    )
    assert ok is False  # aborted, not hung


@pytest.mark.anyio
async def test_stall_detector_aborts_identical_repeats() -> None:
    stub = _stub_repl(idle_timeout=0)  # watchdog disabled
    ok = await stub._consume_agent_stream(
        _LoopingAgent(), "task", _renderer(), lambda: None
    )
    assert ok is False  # stalled, aborted early


@pytest.mark.anyio
async def test_slow_tool_is_not_killed_as_hung() -> None:
    # A tool_call followed by a long silence (a running test-suite /
    # build) must NOT trip the idle watchdog — the tool is working,
    # not the stream hanging. Only the model going quiet counts.
    stub = _stub_repl(idle_timeout=0.3)

    class _SlowTool:
        async def stream(self, prompt: str, **kwargs: Any):
            yield _event(
                "tool_call",
                {"call": {"tool": "bash", "args": {"command": "pytest"}}},
            )
            await anyio.sleep(0.9)  # 3x idle_timeout, but tool is running
            yield _event("tool_result", {"call_id": "1"})
            yield _event("completed", {"result": {"output": "done"}})

    ok = await stub._consume_agent_stream(
        _SlowTool(), "task", _renderer(), lambda: None
    )
    assert ok is True  # survived — not aborted as hung


@pytest.mark.anyio
async def test_clean_run_passes_both_guards() -> None:
    stub = _stub_repl(idle_timeout=5)
    ok = await stub._consume_agent_stream(
        _CleanAgent(), "task", _renderer(), lambda: None
    )
    assert ok is True


@pytest.mark.anyio
async def test_gate_wait_does_not_count_as_idle() -> None:
    # With the approval gate active, silence is the USER's thinking
    # time — the watchdog must not fire even past the timeout.
    stub = _stub_repl(idle_timeout=0.3)
    stub._gate_active = True

    class _SlowThenDone:
        async def stream(self, prompt: str, **kwargs: Any):
            yield _event("started", {})
            await anyio.sleep(0.8)  # > idle_timeout, but gate is up
            yield _event("completed", {"result": {"output": "ok"}})

    ok = await stub._consume_agent_stream(
        _SlowThenDone(), "task", _renderer(), lambda: None
    )
    assert ok is True
