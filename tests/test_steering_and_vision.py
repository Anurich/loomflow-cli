"""Steering + vision wiring in the REPL's stream consumption.

- ``_SteeringQueue``: the metadata object loomflow's ReAct loop drains
  (``pop_all`` before each model call, loomflow >= 0.10.32).
- ``_consume_agent_stream`` passes ``metadata`` carrying the steering
  queue always, and staged clipboard images exactly once.
- ``clipboard_image.to_loom_image`` produces the dict shape loomflow's
  ``_loom_images`` coercion accepts.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from loom_code.render import StreamRenderer
from loom_code.repl import Repl, _SteeringQueue

pytestmark = pytest.mark.anyio


def _stub_repl(**extra: Any) -> Any:
    stub = SimpleNamespace(
        _gate_active=False,
        _idle_timeout=0,
        session_id="test-session",
        total_summaries=0,
        total_compacts=0,
        total_snips=0,
        _print_turn_error=lambda exc: None,
        **extra,
    )
    stub._consume_agent_stream = Repl._consume_agent_stream.__get__(stub)
    return stub


class _RecordingAgent:
    """Records the kwargs of the stream call, finishes instantly."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def stream(self, prompt: str, **kwargs: Any):
        self.kwargs = kwargs
        yield SimpleNamespace(
            kind="model_chunk",
            payload={"chunk": {"kind": "text", "text": "ok"}},
        )


def _renderer() -> StreamRenderer:
    return StreamRenderer()


# ---- _SteeringQueue -------------------------------------------------------


def test_steering_queue_pop_all_drains() -> None:
    q = _SteeringQueue()
    q.push("first")
    q.push("second")
    assert q.pop_all() == ["first", "second"]
    assert q.pop_all() == []


# ---- metadata wiring ------------------------------------------------------


async def test_stream_receives_steering_queue() -> None:
    agent = _RecordingAgent()
    stub = _stub_repl()
    ok = await stub._consume_agent_stream(
        agent, "hi", _renderer(), lambda: None
    )
    assert ok is True
    meta = agent.kwargs.get("metadata") or {}
    assert isinstance(meta.get("_loom_steering"), _SteeringQueue)


async def test_staged_images_ride_metadata_once() -> None:
    agent = _RecordingAgent()
    img = {"data": "aGk=", "media_type": "image/png"}
    stub = _stub_repl(_pending_images=[img])
    await stub._consume_agent_stream(
        agent, "what is in this image? [image-1]", _renderer(),
        lambda: None,
    )
    meta = agent.kwargs.get("metadata") or {}
    assert meta.get("_loom_images") == [img]
    # consumed — the NEXT run must not resend them
    assert stub._pending_images == []
    agent2 = _RecordingAgent()
    await stub._consume_agent_stream(
        agent2, "and now?", _renderer(), lambda: None
    )
    assert "_loom_images" not in (agent2.kwargs.get("metadata") or {})


# ---- clipboard shapes -----------------------------------------------------


def test_to_loom_image_shape() -> None:
    from loom_code.clipboard_image import to_loom_image

    out = to_loom_image(b"\x89PNG\r\n\x1a\nrest", "image/png")
    assert set(out) == {"data", "media_type"}
    assert out["media_type"] == "image/png"
    import base64

    assert base64.b64decode(out["data"]).startswith(b"\x89PNG")


def test_macos_hex_parse() -> None:
    """The osascript «data PNGf<hex>» shape decodes to raw bytes."""
    from loom_code import clipboard_image as ci

    png = b"\x89PNG\r\n\x1a\nDATA"
    fake_out = f"«data PNGf{png.hex().upper()}»\n"

    class _P:
        returncode = 0
        stdout = fake_out
        stderr = ""

    ci_run = ci.subprocess.run
    try:
        ci.subprocess.run = lambda *a, **k: _P()  # type: ignore[assignment]
        got = ci._grab_macos()
    finally:
        ci.subprocess.run = ci_run
    assert got is not None
    data, media = got
    assert data == png and media == "image/png"
