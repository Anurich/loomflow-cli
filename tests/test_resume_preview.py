"""Tests for the /resume history-preview helpers.

After /resume, the REPL surfaces the last N turn groups so the user
sees what they're inheriting. This file pins the structural shape
of the grouping + truncation logic — the actual REPL invocation
(``_render_resumed_history_preview``) is harder to test in isolation
(needs a live agent + memory) but the helpers it builds on are
pure and easy to lock down.
"""

from __future__ import annotations

from loom_code.repl import (
    _collapse_consecutive_duplicate_turns,
    _group_messages_into_turns,
    _truncate_one_line,
)


class _FakeMessage:
    """Minimal duck-typed Message stand-in. The real loomflow
    ``Message`` is a pydantic model with the same .role / .content /
    .tool_calls attrs the grouper reads."""

    def __init__(
        self,
        role: str,
        content: str = "",
        tool_calls: tuple = (),
    ) -> None:
        self.role = role
        self.content = content
        self.tool_calls = tool_calls


def test_truncate_one_line_short_passthrough() -> None:
    assert _truncate_one_line("hello world", 100) == "hello world"


def test_truncate_one_line_collapses_whitespace_and_caps() -> None:
    text = "line one\n  line two\n\nline three with much more content"
    out = _truncate_one_line(text, 30)
    # No newlines; collapsed whitespace.
    assert "\n" not in out
    # Ends with ellipsis when capped.
    assert out.endswith("…")
    assert len(out) <= 30


def test_truncate_one_line_empty_returns_empty() -> None:
    assert _truncate_one_line("", 50) == ""


def test_group_messages_into_turns_basic_pair() -> None:
    msgs = [
        _FakeMessage("user", "what is X?"),
        _FakeMessage("assistant", "X is the answer."),
    ]
    groups = _group_messages_into_turns(msgs)
    assert groups == [("what is X?", "X is the answer.", 0)]


def test_group_messages_counts_tool_calls() -> None:
    """Tool calls are counted from the assistant message's
    ``tool_calls`` attribute; tool RESULT messages don't double-
    count (they're the other half of the call/result pair)."""
    msgs = [
        _FakeMessage("user", "do stuff"),
        _FakeMessage(
            "assistant",
            "I'll do that.",
            tool_calls=("c1", "c2", "c3"),
        ),
        _FakeMessage("tool", "result of c1"),
        _FakeMessage("tool", "result of c2"),
        _FakeMessage("tool", "result of c3"),
        _FakeMessage("assistant", "Done."),
    ]
    groups = _group_messages_into_turns(msgs)
    assert len(groups) == 1
    user, asst, n_calls = groups[0]
    assert user == "do stuff"
    assert "I'll do that." in asst
    assert "Done." in asst
    assert n_calls == 3  # NOT 6 — tool results don't double-count


def test_group_messages_multiple_turn_groups() -> None:
    """Each USER message opens a new group. Assistant text is
    folded into the currently-open group only."""
    msgs = [
        _FakeMessage("user", "first prompt"),
        _FakeMessage("assistant", "first reply"),
        _FakeMessage("user", "second prompt"),
        _FakeMessage("assistant", "second reply"),
        _FakeMessage("user", "third prompt"),
        _FakeMessage("assistant", "third reply"),
    ]
    groups = _group_messages_into_turns(msgs)
    assert [g[0] for g in groups] == [
        "first prompt", "second prompt", "third prompt",
    ]
    assert [g[1] for g in groups] == [
        "first reply", "second reply", "third reply",
    ]


def test_group_messages_drops_system_messages() -> None:
    """SYSTEM messages are framework context, not conversation —
    they don't open or contribute to a turn group."""
    msgs = [
        _FakeMessage("system", "you are loom-code"),
        _FakeMessage("user", "hi"),
        _FakeMessage("system", "memory block injected"),
        _FakeMessage("assistant", "hello"),
    ]
    groups = _group_messages_into_turns(msgs)
    assert groups == [("hi", "hello", 0)]


def test_group_messages_empty_input() -> None:
    assert _group_messages_into_turns([]) == []


def test_group_messages_no_user_returns_empty() -> None:
    """No USER message means no group opens — even if there's an
    orphan assistant message (shouldn't happen in practice, but
    we degrade gracefully)."""
    msgs = [
        _FakeMessage("system", "system only"),
        _FakeMessage("assistant", "orphan reply"),
    ]
    assert _group_messages_into_turns(msgs) == []


def test_group_messages_user_with_no_assistant_response() -> None:
    """A USER turn with no following ASSISTANT (e.g. the agent
    crashed mid-run) still appears as a group with empty
    response text."""
    msgs = [
        _FakeMessage("user", "what happened?"),
    ]
    groups = _group_messages_into_turns(msgs)
    assert groups == [("what happened?", "", 0)]


# ---------------------------------------------------------------------------
# _collapse_consecutive_duplicate_turns — dedup runs of identical
# (user_prompt, assistant_text) pairs so the preview doesn't show
# noise like the same exchange three times in a row.
# ---------------------------------------------------------------------------


def test_collapse_empty_returns_empty() -> None:
    assert _collapse_consecutive_duplicate_turns([]) == []


def test_collapse_no_duplicates_preserves_groups() -> None:
    """Distinct consecutive groups → each gets repeats=1."""
    groups = [
        ("a", "reply a", 0),
        ("b", "reply b", 0),
        ("c", "reply c", 0),
    ]
    out = _collapse_consecutive_duplicate_turns(groups)
    assert out == [
        ("a", "reply a", 0, 1),
        ("b", "reply b", 0, 1),
        ("c", "reply c", 0, 1),
    ]


def test_collapse_run_of_identical_turns_into_one_with_count() -> None:
    """Three consecutive identical (user, assistant) pairs become
    one row annotated with the count. The headline case driving
    this whole change."""
    groups = [
        ("check what is this about?", "Please specify a file.", 0),
        ("check what is this about?", "Please specify a file.", 0),
        ("check what is this about?", "Please specify a file.", 0),
    ]
    out = _collapse_consecutive_duplicate_turns(groups)
    assert out == [
        ("check what is this about?", "Please specify a file.", 0, 3),
    ]


def test_collapse_non_consecutive_duplicates_stay_separate() -> None:
    """A → B → A should stay 3 rows even though A repeats —
    non-adjacency means they were different points in the
    conversation. Only collapsing CONSECUTIVE runs."""
    groups = [
        ("a", "reply a", 0),
        ("b", "reply b", 0),
        ("a", "reply a", 0),
    ]
    out = _collapse_consecutive_duplicate_turns(groups)
    assert out == [
        ("a", "reply a", 0, 1),
        ("b", "reply b", 0, 1),
        ("a", "reply a", 0, 1),
    ]


def test_collapse_distinguishes_by_both_user_and_assistant() -> None:
    """Same user prompt, DIFFERENT assistant replies = separate
    rows. Different exchanges that happened to share a prompt
    shouldn't merge."""
    groups = [
        ("status?", "all green", 0),
        ("status?", "now red", 0),
    ]
    out = _collapse_consecutive_duplicate_turns(groups)
    assert out == [
        ("status?", "all green", 0, 1),
        ("status?", "now red", 0, 1),
    ]


def test_collapse_preserves_first_tool_count_in_run() -> None:
    """When collapsing, the n_tool_calls of the FIRST instance
    is preserved — collapsed copies are assumed structurally
    identical (they have the same assistant text)."""
    groups = [
        ("x", "y", 3),
        ("x", "y", 3),
        ("x", "y", 3),
    ]
    out = _collapse_consecutive_duplicate_turns(groups)
    assert out == [("x", "y", 3, 3)]


def test_collapse_mixed_runs() -> None:
    """A realistic case: a run of dups, then a distinct turn,
    then another run."""
    groups = [
        ("dup1", "r1", 0),
        ("dup1", "r1", 0),
        ("middle", "rm", 1),
        ("dup2", "r2", 0),
        ("dup2", "r2", 0),
        ("dup2", "r2", 0),
    ]
    out = _collapse_consecutive_duplicate_turns(groups)
    assert out == [
        ("dup1", "r1", 0, 2),
        ("middle", "rm", 1, 1),
        ("dup2", "r2", 0, 3),
    ]
