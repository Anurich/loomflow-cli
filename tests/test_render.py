"""Tests for StreamRenderer — event payload-shape handling.

These lock down the renderer against the exact bug class that bit
loom-code before: the renderer guessing loomflow's event payload
shapes instead of matching them. Two real bugs this catches:

* ``model_chunk`` carries text at ``payload["chunk"]["text"]`` only
  when ``kind == "text"`` — an earlier renderer looked for
  ``payload["text"]`` and silently dropped every streamed token.
* ``ToolResult`` has NO ``tool`` field — only ``call_id``. The
  renderer must bridge id -> name from the ``tool_call`` event, or
  it can never tell which result was the living plan.

The tests build REAL loomflow ``Event``s, so a payload-shape
change in loomflow breaks them here instead of in production.
"""

from __future__ import annotations

from loomflow.core.types import Event, ModelChunk, ToolCall, ToolResult

from loom_code.render import StreamRenderer


def test_text_chunk_marks_text_shown() -> None:
    r = StreamRenderer()
    r.handle(
        Event.model_chunk("s", ModelChunk(kind="text", text="hello"))
    )
    assert r._any_text is True


def test_non_text_chunk_is_ignored() -> None:
    r = StreamRenderer()
    r.handle(
        Event.model_chunk(
            "s", ModelChunk(kind="finish", finish_reason="stop")
        )
    )
    assert r._any_text is False


def test_tool_call_name_remembered_for_results() -> None:
    # ToolResult has no `tool` field — only call_id. The renderer
    # bridges id -> name from the tool_call event.
    r = StreamRenderer()
    r.handle(
        Event.tool_call(
            "s", ToolCall(id="c1", tool="grep", args={"pattern": "x"})
        )
    )
    assert r._call_names["c1"] == "grep"


def test_plan_result_captured_via_call_id() -> None:
    r = StreamRenderer()
    r.handle(
        Event.tool_call("s", ToolCall(id="c9", tool="plan_write", args={}))
    )
    plan_text = "**GOAL:** do the thing\n\n| 1 | todo | step |"
    r.handle(
        Event.tool_result(
            "s", ToolResult(call_id="c9", ok=True, output=plan_text)
        )
    )
    assert r.last_plan == plan_text


def test_non_plan_result_does_not_set_last_plan() -> None:
    r = StreamRenderer()
    r.handle(
        Event.tool_call("s", ToolCall(id="c2", tool="read", args={}))
    )
    r.handle(
        Event.tool_result(
            "s", ToolResult(call_id="c2", ok=True, output="file contents")
        )
    )
    assert r.last_plan is None


def test_completed_captures_result_dict() -> None:
    r = StreamRenderer()
    result = {
        "output": "all done",
        "turns": 3,
        "cost_usd": 0.012,
        "tokens_in": 100,
        "cached_tokens_in": 0,
        "tokens_out": 20,
    }
    r.handle(Event.completed("s", result))
    assert r.last_result == result
