"""Tests for the paste-collapse helper.

The bracketed-paste keybinding itself isn't unit-tested (it needs
a live prompt_toolkit Buffer + KeyPressEvent). What we DO lock
down here is the expansion logic + the stash state — both pure
functions, both load-bearing for "the agent sees what the user
actually pasted."

Silent failure mode worth guarding: a typo or refactor in
``expand_pastes`` could drop placeholders and the agent would get
``[paste-1: 50 lines, ...]`` literally in its task instructions
instead of the actual content. Catastrophic for usefulness.
"""

from __future__ import annotations

import pytest

from loom_code import paste as paste_mod
from loom_code.paste import (
    _pastes,
    expand_pastes,
    reset_paste_stash,
    stash_size,
)


@pytest.fixture(autouse=True)
def _isolated_stash() -> None:
    """Module-level state — wipe before AND after each test so
    one test can't leak pastes into another and so a failure
    can't poison the fixture state."""
    _pastes.clear()
    yield
    _pastes.clear()


def test_expand_pastes_substitutes_stashed_content() -> None:
    _pastes.extend(["hello world", "goodbye world"])
    line = "look at [paste-1: 1 lines, 11 chars] then [paste-2: ...]"
    out = expand_pastes(line)
    assert "hello world" in out
    assert "goodbye world" in out
    assert "[paste-1" not in out
    assert "[paste-2" not in out


def test_expand_pastes_preserves_surrounding_text() -> None:
    # The non-placeholder parts of the line MUST survive
    # untouched — agent needs the surrounding words.
    _pastes.append("BIG_CONTENT_HERE")
    line = "please refactor [paste-1: ...] to use async"
    out = expand_pastes(line)
    assert out == "please refactor BIG_CONTENT_HERE to use async"


def test_expand_pastes_with_no_placeholders_is_noop() -> None:
    # Normal short prompts shouldn't be touched at all.
    line = "fix the bug in auth.py"
    assert expand_pastes(line) == line


def test_expand_pastes_handles_unknown_index_gracefully() -> None:
    # User typed a placeholder by hand referring to a paste we
    # never stashed. We leave it as-is rather than dropping or
    # erroring — most helpful to the agent who can ignore an
    # unresolved marker but would be confused by a missing one.
    _pastes.append("real-paste")
    line = "look at [paste-1: ...] vs [paste-99: ...]"
    out = expand_pastes(line)
    assert "real-paste" in out
    assert "[paste-99: ...]" in out


def test_expand_pastes_preserves_order_across_multiple() -> None:
    # Multiple placeholders in order — each maps to its own
    # stashed content, no cross-contamination.
    _pastes.extend(["AAA", "BBB", "CCC"])
    line = "first [paste-1: a] then [paste-3: c] and [paste-2: b]"
    out = expand_pastes(line)
    assert out == "first AAA then CCC and BBB"


def test_reset_paste_stash_clears_state() -> None:
    _pastes.extend(["a", "b", "c"])
    assert stash_size() == 3
    reset_paste_stash()
    assert stash_size() == 0
    # And expand becomes a no-op on the same placeholder.
    assert expand_pastes("[paste-1: ...]") == "[paste-1: ...]"


def test_thresholds_are_module_constants() -> None:
    # The thresholds shouldn't drift accidentally — they govern
    # whether the user sees a wall of text or a clean placeholder.
    # Pin them so a refactor has to acknowledge the change.
    assert paste_mod._PASTE_CHAR_THRESHOLD == 500
    assert paste_mod._PASTE_LINE_THRESHOLD == 4
