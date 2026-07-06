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


# ---- drag-and-drop image paths --------------------------------------------


def _paths_stub() -> Any:
    stub = SimpleNamespace(_pending_images=[])
    stub._stage_image_paths = Repl._stage_image_paths.__get__(stub)
    stub._IMAGE_PATH_RE = Repl._IMAGE_PATH_RE
    stub._IMAGE_MIME = Repl._IMAGE_MIME
    stub._IMAGE_MAX_BYTES = Repl._IMAGE_MAX_BYTES
    return stub


def test_dropped_path_is_staged_and_replaced(tmp_path) -> None:
    png = tmp_path / "shot.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nDATA")
    stub = _paths_stub()
    out = stub._stage_image_paths(f"what is in {png} here?")
    assert out == "what is in [image-1] here?"
    assert len(stub._pending_images) == 1
    assert stub._pending_images[0]["media_type"] == "image/png"


def test_quoted_and_escaped_space_paths(tmp_path) -> None:
    img = tmp_path / "my shot.jpg"
    img.write_bytes(b"\xff\xd8\xffJPEG")
    stub = _paths_stub()
    # terminals quote dropped paths with spaces …
    out = stub._stage_image_paths(f"look at '{img}'")
    assert out == "look at [image-1]"
    # … or escape the spaces
    escaped = str(img).replace(" ", "\\ ")
    out2 = stub._stage_image_paths(f"and {escaped} again")
    assert out2 == "and [image-2] again"
    assert [i["media_type"] for i in stub._pending_images] == [
        "image/jpeg",
        "image/jpeg",
    ]


def test_absent_or_non_image_paths_pass_through(tmp_path) -> None:
    stub = _paths_stub()
    line = "rename old.png to new.png and check /no/such/file.png"
    assert stub._stage_image_paths(line) == line
    assert stub._pending_images == []
    # real file but not an image extension → untouched
    txt = tmp_path / "notes.txt"
    txt.write_text("hi")
    line2 = f"read {txt} please"
    assert stub._stage_image_paths(line2) == line2


def test_oversized_image_left_as_text(tmp_path) -> None:
    big = tmp_path / "huge.png"
    big.write_bytes(b"\x89PNG" + b"\x00" * (10 * 1024 * 1024 + 1))
    stub = _paths_stub()
    line = f"see {big}"
    assert stub._stage_image_paths(line) == line
    assert stub._pending_images == []
