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


def test_streamed_text_is_buffered_not_printed_per_chunk() -> None:
    """Each chunk should accumulate into the buffer; nothing is
    rendered until ``_end_text`` flushes it. We can't easily assert
    "nothing was printed" but we can assert the buffer grew and
    didn't reset between chunks."""
    r = StreamRenderer()
    r.handle(
        Event.model_chunk("s", ModelChunk(kind="text", text="# Title\n"))
    )
    r.handle(
        Event.model_chunk(
            "s", ModelChunk(kind="text", text="and some text\n")
        )
    )
    assert "".join(r._text_buffer) == "# Title\nand some text\n"
    assert r._in_text is True


def test_end_text_drains_buffer() -> None:
    """``_end_text`` (triggered by a tool_call after prose, or by
    a non-text chunk burst end) must flush the buffer + reset the
    in_text flag so the next prose burst starts cleanly."""
    r = StreamRenderer()
    r.handle(
        Event.model_chunk("s", ModelChunk(kind="text", text="hello"))
    )
    assert r._text_buffer == ["hello"]
    r._end_text()
    assert r._text_buffer == []
    assert r._in_text is False


def test_tool_call_after_text_flushes_buffer() -> None:
    """A tool_call event coming after a text burst must trigger
    `_end_text` automatically — otherwise the tool-call's print
    line would interleave with un-rendered markdown."""
    r = StreamRenderer()
    r.handle(
        Event.model_chunk(
            "s", ModelChunk(kind="text", text="Let me grep.")
        )
    )
    r.handle(
        Event.tool_call(
            "s", ToolCall(id="c1", tool="grep", args={"pattern": "x"})
        )
    )
    assert r._text_buffer == []
    assert r._in_text is False


def test_truncate_preview_returns_short_unchanged() -> None:
    from loom_code.render import _truncate_preview

    assert _truncate_preview("short", char_cap=100, line_cap=10) == "short"


def test_truncate_preview_caps_lines_first_when_smaller() -> None:
    from loom_code.render import _truncate_preview

    text = "\n".join(f"line{i}" for i in range(20))
    out = _truncate_preview(text, char_cap=10_000, line_cap=5)
    # Should have at most 5 source lines + the trailer
    assert "+15 lines" in out
    assert out.count("\n") <= 6  # 5 content + 1 trailer line


def test_truncate_preview_caps_chars_when_smaller() -> None:
    from loom_code.render import _truncate_preview

    out = _truncate_preview("a" * 500, char_cap=50, line_cap=100)
    assert "+450 chars" in out
    # Trailer is on a new line; content body itself is ≤50 chars
    body = out.split("\n")[0]
    assert len(body) == 50


def test_truncate_preview_empty_input() -> None:
    from loom_code.render import _truncate_preview

    assert _truncate_preview("", char_cap=10, line_cap=2) == ""


# ---- glyph plan rendering -------------------------------------------


def test_extract_plan_goal() -> None:
    from loom_code.render import _extract_plan_goal

    md = "**GOAL:** Fix all the bugs\n\n| # | Status |\n"
    assert _extract_plan_goal(md) == "Fix all the bugs"
    # No goal line → empty string (header falls back to "Plan").
    assert _extract_plan_goal("no goal here") == ""


def test_render_plan_glyphs_header_counts() -> None:
    from loom_code.render import _render_plan_glyphs

    steps = [
        {"description": "a", "status": "done", "finding": "did it"},
        {"description": "b", "status": "doing", "finding": ""},
        {"description": "c", "status": "todo", "finding": ""},
        {"description": "d", "status": "blocked", "finding": "stuck"},
    ]
    plain = _render_plan_glyphs(steps, goal="My goal").plain
    # Header shows goal + done/total + blocked count.
    assert "My goal" in plain
    assert "1/4 done" in plain
    assert "1 blocked" in plain


def test_render_plan_glyphs_uses_status_glyphs() -> None:
    from loom_code.render import _render_plan_glyphs

    steps = [
        {"description": "done step", "status": "done", "finding": "f"},
        {"description": "doing step", "status": "doing", "finding": ""},
        {"description": "todo step", "status": "todo", "finding": ""},
        {"description": "skip step", "status": "skipped", "finding": "why"},
        {"description": "block step", "status": "blocked", "finding": "why"},
    ]
    plain = _render_plan_glyphs(steps).plain
    assert "■ done step" in plain
    assert "▸ doing step" in plain
    assert "□ todo step" in plain
    assert "⊘ skip step" in plain
    assert "✗ block step" in plain
    # Doing gets the (doing) marker.
    assert "(doing)" in plain
    # Done shows finding inline; skipped/blocked show the reason flag.
    assert "— f" in plain
    assert "› why" in plain


def test_render_plan_glyphs_no_blocked_omits_blocked_count() -> None:
    from loom_code.render import _render_plan_glyphs

    steps = [
        {"description": "a", "status": "done", "finding": ""},
        {"description": "b", "status": "todo", "finding": ""},
    ]
    plain = _render_plan_glyphs(steps, goal="g").plain
    assert "1/2 done" in plain
    assert "blocked" not in plain


def test_render_plan_glyphs_description_with_brackets_safe() -> None:
    """Descriptions containing ``[`` must NOT be mis-parsed as Rich
    style markup — we build with Text.append, not markup, so this
    renders literally."""
    from loom_code.render import _render_plan_glyphs

    steps = [
        {
            "description": "handle list[str] type hints",
            "status": "todo",
            "finding": "",
        }
    ]
    plain = _render_plan_glyphs(steps).plain
    assert "list[str]" in plain


def test_plan_result_renders_glyphs_when_steps_captured() -> None:
    """End-to-end: a plan_write tool_call (captures structured
    steps) followed by its tool_result should render the glyph
    view, not the raw markdown table."""
    from loomflow.core.types import Event, ToolCall

    r = StreamRenderer()
    # tool_call carrying structured steps.
    r.handle(
        Event.tool_call(
            "s",
            ToolCall(
                id="c1",
                tool="plan_write",
                args={
                    "goal": "g",
                    "steps": [
                        {"description": "step one", "status": "done",
                         "finding": "ok"},
                        {"description": "step two", "status": "todo"},
                    ],
                },
            ),
        )
    )
    assert r.last_plan_steps is not None
    assert len(r.last_plan_steps) == 2
