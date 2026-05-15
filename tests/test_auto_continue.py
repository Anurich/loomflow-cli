"""Tests for the auto-continue decision + plan-progress parsing.

Two layers under test:

* :func:`_count_plan_remaining` — accepts structured steps OR
  rendered markdown. Structured wins; markdown is the regex
  fallback for cases where the renderer didn't observe a
  structured plan_write this iteration.
* :func:`should_auto_continue` — the Ralph-loop decision
  function: continue / plan_drained / cap_reached / stalled.

The actual loop wiring in ``_turn`` is exercised via integration
where possible; pure-function tests below cover the decision math
so a regression doesn't slip past unit coverage.
"""

from __future__ import annotations

from loom_code.repl import (
    _AUTO_CONTINUE_LIMIT,
    _count_plan_remaining,
    should_auto_continue,
)

# ---- _count_plan_remaining (structured input) ---------------------


def test_count_remaining_from_structured_steps() -> None:
    steps = [
        {"description": "a", "status": "done"},
        {"description": "b", "status": "doing"},
        {"description": "c", "status": "todo"},
        {"description": "d", "status": "todo"},
    ]
    assert _count_plan_remaining(plan_steps=steps) == 3


def test_count_remaining_treats_skipped_and_blocked_as_done() -> None:
    """A blocked or skipped step is "not currently remaining work" —
    we should not auto-continue on its behalf."""
    steps = [
        {"description": "a", "status": "done"},
        {"description": "b", "status": "skipped"},
        {"description": "c", "status": "blocked"},
        {"description": "d", "status": "todo"},
    ]
    assert _count_plan_remaining(plan_steps=steps) == 1


def test_count_remaining_all_done() -> None:
    """Fully drained plan → 0 remaining → no auto-continue."""
    steps = [{"description": "x", "status": "done"}]
    assert _count_plan_remaining(plan_steps=steps) == 0


def test_count_remaining_empty_list() -> None:
    """Empty plan → 0 (no work to do, no auto-continue)."""
    assert _count_plan_remaining(plan_steps=[]) == 0


def test_count_remaining_structured_preferred_over_text() -> None:
    """When both inputs are provided the structured one wins —
    the text might be from a stale render."""
    steps = [{"description": "a", "status": "done"}]
    text = "**Progress:** 0/5 done"  # would say 5 remaining
    assert _count_plan_remaining(plan_steps=steps, plan_text=text) == 0


# ---- _count_plan_remaining (markdown fallback) -------------------


def test_count_remaining_from_markdown_progress_line() -> None:
    """Regex fallback path — when structured steps aren't
    captured we parse the rendered markdown."""
    text = "**Progress:** 2/6 done"
    assert _count_plan_remaining(plan_text=text) == 4


def test_count_remaining_markdown_no_progress_line() -> None:
    """Markdown without a Progress line → 0. Better to under-
    fire auto-continue than to fire on garbage."""
    assert _count_plan_remaining(plan_text="just some text") == 0


def test_count_remaining_no_input_returns_zero() -> None:
    """Neither structured nor text → 0. No plan = no auto-continue."""
    assert _count_plan_remaining() == 0


# ---- should_auto_continue --------------------------------------


def test_continue_when_remaining_and_under_cap() -> None:
    cont, reason = should_auto_continue(
        remaining=3,
        previous_remaining=None,
        iterations_used=0,
        limit=5,
    )
    assert cont is True
    assert reason == ""


def test_stop_when_plan_drained() -> None:
    cont, reason = should_auto_continue(
        remaining=0,
        previous_remaining=2,
        iterations_used=1,
        limit=5,
    )
    assert cont is False
    assert reason == "plan_drained"


def test_stop_when_cap_reached() -> None:
    cont, reason = should_auto_continue(
        remaining=2,
        previous_remaining=3,
        iterations_used=5,
        limit=5,
    )
    assert cont is False
    assert reason == "cap_reached"


def test_stop_when_stalled() -> None:
    """Two consecutive iterations with the same remaining count
    → stall → bail. Model is talking but not making bookkeeping
    moves; further iterations would burn cost for nothing."""
    cont, reason = should_auto_continue(
        remaining=3,
        previous_remaining=3,
        iterations_used=2,
        limit=5,
    )
    assert cont is False
    assert reason == "stalled"


def test_stop_when_remaining_increased() -> None:
    """Pathological: the agent ADDED steps mid-iteration. Treat
    as stalled (no progress reduction)."""
    cont, reason = should_auto_continue(
        remaining=5,
        previous_remaining=3,
        iterations_used=1,
        limit=5,
    )
    assert cont is False
    assert reason == "stalled"


def test_no_stall_check_on_first_iteration() -> None:
    """First iteration has no previous count to compare against.
    Must continue (don't bail immediately just because we have
    no baseline)."""
    cont, reason = should_auto_continue(
        remaining=3,
        previous_remaining=None,
        iterations_used=0,
        limit=5,
    )
    assert cont is True


def test_progress_keeps_loop_going() -> None:
    """remaining went 5 → 4 between iterations → keep going."""
    cont, _ = should_auto_continue(
        remaining=4,
        previous_remaining=5,
        iterations_used=1,
        limit=5,
    )
    assert cont is True


def test_default_limit_is_reasonable() -> None:
    """Sanity check: the default cap should cover a typical 6-step
    scaffold task (1 initial run + 5 continues = 6 steps)."""
    assert _AUTO_CONTINUE_LIMIT == 5
