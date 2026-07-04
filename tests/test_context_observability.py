"""Context observability (loom_code.context_report + the /context
and /prompt surfaces).

The positioning claim these lock down: what /context reports IS the
model's context — window, the same used-tokens figure the compactor
keys on, and every injected block with its size. Rendering is pure
functions so no agent/console machinery is needed.
"""

from __future__ import annotations

from loom_code.context_report import (
    context_percent,
    context_report,
    estimate_tokens,
    prompt_dump,
)

# ---- estimate_tokens -------------------------------------------------


def test_estimate_tokens_empty_is_zero() -> None:
    assert estimate_tokens("") == 0


def test_estimate_tokens_short_floors_at_one() -> None:
    assert estimate_tokens("ab") == 1


def test_estimate_tokens_chars_over_four() -> None:
    assert estimate_tokens("x" * 4000) == 1000


# ---- context_percent -------------------------------------------------


def test_percent_normal() -> None:
    assert context_percent(50_000, 200_000) == 25


def test_percent_zero_window_is_zero() -> None:
    # Unknown model → window 0 → don't divide, don't crash.
    assert context_percent(50_000, 0) == 0


def test_percent_clamps_at_100() -> None:
    assert context_percent(999_999, 100_000) == 100


# ---- context_report --------------------------------------------------


def _report(**over):
    kw = dict(
        model="gpt-4.1-mini",
        window=200_000,
        used_tokens=50_000,
        threshold=160_000,
        blocks=[("loom_index", "x" * 4000), ("project_rules", "y" * 400)],
        n_exchanges=3,
    )
    kw.update(over)
    return context_report(**kw)


def test_report_shows_window_used_and_percent() -> None:
    r = _report()
    assert "200,000" in r
    assert "50,000" in r
    assert "25%" in r


def test_report_lists_every_block_with_token_estimate() -> None:
    r = _report()
    assert "loom_index" in r and "1,000" in r
    assert "project_rules" in r and "100" in r
    assert "total" in r


def test_report_shows_compaction_threshold() -> None:
    assert "160,000" in _report()


def test_report_compaction_off() -> None:
    assert "off" in _report(threshold=0)


def test_report_no_blocks() -> None:
    assert "none" in _report(blocks=[])


def test_report_transparency_claim_present() -> None:
    # The line that makes the promise — if it goes, the positioning
    # goes with it.
    assert "entire context" in _report()


# ---- prompt_dump -----------------------------------------------------


def test_prompt_dump_includes_instructions_verbatim() -> None:
    d = prompt_dump(
        instructions="You are loom-code. NEVER lie.",
        blocks=[("session_summary", "we fixed the parser")],
    )
    assert "You are loom-code. NEVER lie." in d
    assert "session_summary" in d
    assert "we fixed the parser" in d


def test_prompt_dump_handles_missing_instructions() -> None:
    d = prompt_dump(instructions=None, blocks=[])
    assert "not exposed" in d


def test_prompt_dump_marks_empty_blocks() -> None:
    d = prompt_dump(instructions="i", blocks=[("learned_notes", "")])
    assert "(empty)" in d
